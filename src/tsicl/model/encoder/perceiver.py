from typing import Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..inr import (
    MultiScaleNeRFEncoding,
    NeRFEncoding,
)
from .blocks import (
    Attention,
    CrossAttention,
    FeedForward,
    MultiScaleAttention,
    PreNorm,
    PreNormCross,
)

# ACKNOWLEDGEMENT: code adapted from Aroma implementation https://github.com/LouisSerrano/aroma,
# itself adapted from perceiver implementation by lucidrains

class PerceiverEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 1,
        num_channels: int = 1,
        num_latents: int = 64,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        depth: int = 3,
        latent_heads: int = 8,
        latent_dim_head: int = 64,
        cross_heads: int = 8,
        cross_dim_head: int = 64,
        max_pos_encoding_freq: int = 4,
        num_freq: int = 12,
        scales: Sequence[int] = [3, 4, 5],
        mlp_feature_dim: int = 16,
    ):
        """
        Modified AROMA encoder based on a Perceiver architecture.

        Args:
            input_dim (int): number of coordinates
            num_channels (int): number of channels in values tensors
            num_latents (int): number of query tokens (*encode*)
            hidden_dim: (int): dim of the query tokens (*encode*)
            latent_dim: (int): dim of the tokens in latent space (*process*)
            depth (int): number of SA layers in latent space (*process*)
            latent_heads (int): number of latent self-attn heads (*process*)
            latent_dim_head (int): dim of each latent self-attn head (*process*)
            cross_heads: (int): number of cross-attn heads (*encode* and *decode*)
            cross_dim_head (int): dim of each cross-attn head (*encode* and *decode*)
            max_pos_encoding_freq (int): max frequ (log2 scale) for Fourier feature pos embeddings of the geo encoder (*encode*)
            num_freq (int): number of frequ bands to sample for Fourier features (*encode* and *decode*)
            scales (Sequence[int]): in log2 scale, Fourier features bounds of each band (multi-band, *decode*)
            mlp_feature_dim: (int): feature dimensions before the MLP (*decode*)
        """
        
        super().__init__()

        self.depth                = depth
        self.num_latents          = num_latents

        self.pos_encoding = NeRFEncoding(
            num_freq      = num_freq,
            max_freq_log2 = max_pos_encoding_freq,
            input_dim     = input_dim,
            base_freq     = 2,
            log_sampling  = True,
            include_input = True,
            min_freq_log2 = 0
        )
        key_dim = self.pos_encoding.out_dim

        # initialize latent tokens:
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # get QK dim:
        value_dim = hidden_dim

        # a. project (scalar or 2D) values to higher dim space:
        self.lift_values = nn.Linear(num_channels, hidden_dim)

        # b. cross attend the coordinates:
        self.cross_attend_geo = nn.ModuleList(
            [
                PreNormCross(
                    hidden_dim,
                    CrossAttention(
                        query_dim = hidden_dim,
                        key_dim   = key_dim,
                        value_dim = key_dim,
                        heads     = cross_heads,
                        dim_head  = cross_dim_head,
                    ),
                    k_dim = key_dim,
                    v_dim = key_dim,
                ),
                PreNorm(hidden_dim, FeedForward(hidden_dim)),
            ]
        )

        # c. cross attend the pixels:
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNormCross(
                    hidden_dim,
                    CrossAttention(
                        query_dim = hidden_dim,
                        key_dim   = key_dim,
                        value_dim = value_dim,
                        heads     = cross_heads,
                        dim_head  = cross_dim_head,
                    ),
                    k_dim = key_dim,
                    v_dim = value_dim,
                ),
                PreNorm(hidden_dim, FeedForward(hidden_dim)),
            ]
        )

        # d. self-attention in latent space:
        # Un seul bloc SA + FFN partagé sur les `depth` passes avant ET après le channel mixing
        shared_latent_block = nn.ModuleList([
            PreNorm(hidden_dim, Attention(query_dim=hidden_dim, heads=latent_heads, dim_head=latent_dim_head)),
            PreNorm(hidden_dim, FeedForward(hidden_dim)),
        ])
        self.layers_before_mixing = nn.ModuleList([shared_latent_block for _ in range(depth)])
        self.layers_after_mixing  = nn.ModuleList([shared_latent_block for _ in range(depth)])

        # e. cross attend channels:
        
        # Un seul bloc Cross-Attention + FFN partagé sur les `depth` passes de channel mixing
        shared_channel_block = nn.ModuleList([
            PreNorm(hidden_dim, CrossAttention(query_dim=hidden_dim, key_dim=hidden_dim,
                                               value_dim=hidden_dim, heads=cross_heads,
                                               dim_head=cross_dim_head)),
            PreNorm(hidden_dim, FeedForward(hidden_dim)),
        ])
        self.cross_attend_channels = nn.ModuleList([shared_channel_block for _ in range(depth)])

        # f. multi-scale cross-attention decoder:
        self.pos_query = MultiScaleNeRFEncoding(
            num_freqs_per_scale = num_freq,
            log_sampling        = True,
            include_input       = True,
            input_dim           = input_dim,
            base_freq           = 2,
            scales              = scales,
            use_pi              = True,
            disjoint            = True
        )
        queries_dim = self.pos_query.out_dim_per_scale

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            MultiScaleAttention(
                query_dim   = queries_dim,
                context_dim = hidden_dim,
                out_dim     = mlp_feature_dim,
                heads       = cross_heads,
                dim_head    = cross_dim_head,
            ),
            context_dim = hidden_dim,
        )

        self.mean_fc   = nn.Linear(hidden_dim, latent_dim)
        self.lift_z    = nn.Linear(latent_dim, hidden_dim)
        self.out_dim   = len(scales) * mlp_feature_dim

    def _pad(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        series_covar: torch.Tensor,
        coords_covar: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:

        b = series.shape[0]
        missing_len = series_covar.shape[2] - series.shape[1]

        # handle nans in covar:
        mask_covar   = series_covar.isnan().sum(0).squeeze(-1) == 0 # (C-1, covar_seq_len)
        series_covar = torch.nan_to_num(series_covar, nan = 0.0)

        if missing_len > 0:

            mask = torch.cat([
                torch.cat(
                    [
                        torch.ones(1, series.shape[1]),
                        torch.zeros(1, missing_len)
                    ], dim = 1
                ).to(series.device),
                mask_covar
            ], dim = 0).bool() # (C, seq_len)
            # XXX merge with existing `mask`?

            series = torch.cat([
                series,
                torch.zeros((b, missing_len, 1)).to(series.device)
            ], dim = 1) # (bs, seq_len, 1)

            coords = torch.cat([
                coords,
                torch.zeros((b, missing_len, 1)).to(series.device)
            ], dim = 1) # (bs, seq_len, 1)

        
        elif missing_len < 0:

            mask = torch.cat([
                torch.ones(1, series.shape[1]).to(series.device),
                torch.cat(
                    [
                        mask_covar,
                        torch.zeros(series_covar.shape[1], abs(missing_len)).to(series.device)
                    ], dim = 1
                )
            ], dim = 0).bool().to(series.device) # (C, seq_len)
            # XXX merge with existing `mask`?

            series_covar = torch.cat([
                series_covar,
                torch.zeros((series.shape[0], coords_covar.shape[1], abs(missing_len), 1)).to(series.device)
            ], dim = 2)
            
            coords_covar = torch.cat([
                coords_covar,
                torch.zeros((series.shape[0], coords_covar.shape[1], abs(missing_len), 1)).to(series.device)
            ], dim = 2) # (bs, C-1, seq_len, 1)

        assert series.shape[1] == series_covar.shape[2] == coords.shape[1] == coords_covar.shape[2]

        return series, coords, series_covar, coords_covar, mask

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        series_covar: torch.Tensor | None = None,
        coords_covar: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_coords: torch.Tensor | None = None,
        return_latent_only: bool = False
    ) -> torch.Tensor:
        
        '''
        series is shape (B, T_shared, 1)
        coords is shape (B, T_shared, 1)
        mask is shape (B, T_shared, 1)
        series_covar is shape (B, C-1, T_shared, 1)
        coords_covar is shape (B, C-1, T_shared, 1)
        '''

        if series_covar is None:
            series_all = series.unsqueeze(1) # (bs, 1, seq_len, 1)
            coords_all = coords.unsqueeze(1) # (bs, 1, seq_len, 1)
        else:
            assert coords_covar is not None
            if coords_covar.ndim == 3:
                coords_covar = repeat(coords_covar, 'b t 1 -> b c t 1', c = series_covar.shape[1])

            # pad to max_seq_len:
            series, coords, series_covar, coords_covar, mask = self._pad(
                series       = series,
                coords       = coords,
                series_covar = series_covar,
                coords_covar = coords_covar,
                mask         = mask
            )

            series_all = torch.cat([series.unsqueeze(1), series_covar], dim=1) # (bs, C, seq_len, 1)
            coords_all = torch.cat([coords.unsqueeze(1), coords_covar], dim=1) # (bs, C, seq_len, 1)

            if mask is not None:
                assert mask.ndim == 2
                mask = repeat(mask, 'c t -> (b c) t', b=series.shape[0])

        b, c, t, _ = series_all.shape

        series_all = rearrange(series_all, 'b c t 1 -> (b c) t 1') # (bs x C, seq_len, 1)
        coords_all = rearrange(coords_all, 'b c t 1 -> (b c) t 1') # (bs x C, seq_len, 1)


        # 1/5 Geometry encoder

        # prepare queries of the geo encoder:
        x = repeat(self.latents, "n d -> b n d", b=b*c)   # (bs x C, num_latents, hidden_dim)

        # prepare keys and values of the geo encoder:
        k = self.pos_encoding(coords_all)                 # (bs x C, seq_len, )
        v = self.lift_values(series_all)                  # (bs x C, seq_len, hidden_dim)

        # cross attend to the other pixel locations
        cross_attn, cross_ff = self.cross_attend_geo
        x = cross_attn(x, k=k, v=k, mask=mask) + x    # (bs x C, num_latents, hidden_dim)
        x = cross_ff(x) + x                           # (bs x C, num_latents, hidden_dim)


        # 2/5 Value encoder

        # cross attend the coordinates:
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, k=k, v=v, mask=mask) + x    # (bs x C, num_latents, hidden_dim)
        x = cross_ff(x) + x                                                                 # (bs x C, num_latents, hidden_dim)


        # 3/5 Process in latent space

        # L layers of self-attention on latent tokens:
        x = self.lift_z(self.mean_fc(x))
        for self_attn, self_ff in self.layers_before_mixing:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # 4/5 cross-attend the channels
        
        x = rearrange(x, '(b c) m d -> (b m) c d', b=b, c=c)            # (bs x num_latents, C, hidden_dim)

        # select target time series dimension:
        x_auto_reg = x[:, 0, :].unsqueeze(1)                            # (bs x num_latents, 1, hidden_dim)

        for cross_attn, cross_ff in self.cross_attend_channels: # pyright: ignore[reportGeneralTypeIssues]

            # do cross-token attention:
            x_auto_reg = cross_attn(x_auto_reg, k=x, v=x, mask=None) + x_auto_reg    # (bs x num_latents, 1, hidden_dim)
            x_auto_reg = cross_ff(x_auto_reg) + x_auto_reg                           # (bs x num_latents, 1, hidden_dim)
            
        # rearrange and self attention block:
        x = rearrange(x_auto_reg, '(b m) 1 d -> b m d', b=b, m=self.num_latents) # (bs, num_latents, hidden_dim)

        for self_attn, self_ff in self.layers_after_mixing: # pyright: ignore[reportGeneralTypeIssues]

            # do self-attention
            x = self_attn(x) + x    # (bs, num_latents, hidden_dim)
            x = self_ff(x) + x      # (bs, num_latents, hidden_dim)

        if return_latent_only:
            return x
        
        # 5/5 Cross-attn with target coords to get decoder features
        
        # get queries of the cross-attn decoder:
        if target_coords is None:
            queries = self.pos_query(coords)        # (bs, target_seq_len, num_bandwidth, pos_embed_dim)
        else:
            queries = self.pos_query(target_coords) # (bs, target_seq_len, num_bandwidth, pos_embed_dim)

        # cross attend from decoder queries to latents:
        latents = self.decoder_cross_attn(queries, context=x)   # (bs, target_seq_len, num_bandwidth, mlp_feature_dim)

        return latents
