from __future__ import annotations

from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ..encoder.blocks.attention import CrossAttention, FeedForward
from ..encoder.blocks.utils import PreNorm, PreNormCross
from .layers import Encoder

# ACKNOWLEDGEMENT: code adapted from TabICL implementation:
# https://github.com/soda-inria/tabicl/tree/main

class ICLearningCrossAttn(nn.Module):
    """In-context quantile regressor with cross-attention over target values and covariates.

    1. Embeds context target values and optional covariates into each timestep via cross-attention.
    2. Conditions target timesteps on context timesteps via split self-attention (`Encoder`).
    3. Predicts `nb_quantiles` quantiles for each target timestep.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_blocks : int
        Number of attention blocks in the ICL encoder.
    nhead : int
        Number of attention heads in the ICL encoder.
    dim_feedforward : int
        Hidden dimension of the feedforward network in the ICL encoder.
    cross_heads : int, default=4
        Number of cross-attention heads used to embed values and covariates.
    cross_dim_head : int, default=64
        Dimension of each cross-attention head.
    dropout : float, default=0.0
        Dropout probability.
    activation : str or callable, default="gelu"
        Activation function used in the feedforward network.
    start_quantile : float, default=0.05
        Lowest predicted quantile level.
    end_quantile : float, default=0.95
        Highest predicted quantile level.
    nb_quantiles : int, default=19
        Number of quantile levels predicted by the output head.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        cross_heads: int = 4,
        cross_dim_head: int = 64,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        start_quantile: float = 0.05,
        end_quantile: float = 0.95,
        nb_quantiles: int = 19,
    ):
        super().__init__()

        self.tf_icl = Encoder(
            num_blocks      = num_blocks,
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = activation,
        )
        self.ln = nn.LayerNorm(d_model)

        # value projector:
        self.y_encoder   = nn.Linear(1, d_model)

        # covar projector:
        self.cov_encoder = nn.Linear(1, d_model)

        # learnable tokens:
        self.target_token = nn.Parameter(torch.randn(1, d_model) * .05)
        self.sep_token    = nn.Parameter(torch.randn(2, d_model) * .05)

        # cross-attn layer:
        self.cross_attn   = nn.ModuleList(
                [
                    PreNormCross(
                        d_model,
                        CrossAttention(
                            query_dim = d_model,
                            key_dim   = d_model,
                            value_dim = d_model,
                            heads     = cross_heads,
                            dim_head  = cross_dim_head,
                        ),
                        k_dim = d_model,
                        v_dim = d_model,
                    ),
                    PreNorm(d_model, FeedForward(d_model)),
                ]
            )
        # quantile head:
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, nb_quantiles)
        )

        self.start_quantile  = start_quantile
        self.end_quantile    = end_quantile
        self.nb_quantiles    = nb_quantiles
        self.d_model         = d_model

    def _icl_predictions(
        self,
        R: Tensor,
        y_train: Tensor,
        y_cov: Tensor | None = None,
    ) -> Tensor:
        """Compute quantile predictions for all timesteps (context + target).

        Parameters
        ----------
        R : Tensor of shape (B, T, D)
            Per-timestep representations from the encoder, with context first (`[:train_size]`)
            and target timesteps after (`[train_size:]`).
        y_train : Tensor of shape (B, train_size, 1)
            Observed context values.
        y_cov : Tensor of shape (B, C, T_cov, 1), optional
            Optional covariates (`T_cov == train_size` or `T_cov == T`), may contain NaNs.

        Returns
        -------
        Tensor of shape (B, T, nb_quantiles)
            Predicted quantiles for all timesteps.
        """

        train_size = y_train.shape[1]
        
        bs = len(y_train)

        # prepare queries:
        x = repeat(self.target_token, "1 d -> (b T) 1 d", b=bs, T = R.shape[1]) # (bs x T, 1, d)
        
        # prepare keys, values (on context):
        y = self.y_encoder(y_train.float())     # (bs, T_ctx, d)
        train_seq = [
            rearrange(y, 'b T d -> (b T) 1 d'),                                 # (bs x T_ctx, 1, d)
            repeat(self.sep_token, "m d -> (b T) m d", b=bs, T = train_size)    # (bs x T_ctx, 2, d)
        ]

        # prepare keys, values (on target):
        test_seq = [
            repeat(self.sep_token, "m d -> (b T) m d", b=bs, T = R.shape[1] - train_size)   # (bs x T_mis, 2, d)
        ]

        attn_mask_train, attn_mask_test = None, None

        # append covar to keys, values:
        if y_cov is not None:
            assert y_cov.shape[-2] >= train_size

            # get mask of observed covariates:
            is_observed_mask = ~torch.isnan(y_cov) # (bs, c, T, 1)

            # replace nan (missing covars) by zeroes:
            # (the replacing value does not matter, will be zeroed-out by attn mask)
            y_cov = torch.nan_to_num(y_cov, nan=0.0)

            # project covariates to latent space:
            z = self.cov_encoder(y_cov)     # (bs, C, T, d)

            # complete TF-ICL train sequence:
            train_seq.append(
                rearrange(z[:,:,:train_size,:] , 'b c t d -> (b t) c d', b = bs, t = train_size)
            )

            # if missing covar in train sequence, build attention mask:
            if (~is_observed_mask[...,:train_size,:]).sum() > 0:
                attn_mask_train = torch.cat([
                    torch.ones( len(train_seq[0]),1+train_seq[1].shape[1], 1).bool().to(z.device), # (bs x T_ctx, 1+2, 1)
                    rearrange( is_observed_mask[..., :train_size, :], 'b c T 1 -> (b T) c 1' )     # (bs x T_ctx, C, 1)
                ], dim = 1) # (bs x T_ctx, 1+2+C, 1)
            
            if z.shape[-2] > train_size:

                # complete TF-ICL test sequence:
                test_seq.append( rearrange(z[:,:,train_size:,:] , 'b c t d -> (b t) c d', b = bs, t = R.shape[1] - train_size))

                # if missing covar in test sequence, build attention mask:
                if (~is_observed_mask[...,train_size:,:]).sum() > 0:
                    attn_mask_test = torch.cat([
                        torch.ones( len(test_seq[0]),test_seq[0].shape[1], 1).bool().to(z.device),     # (bs x T_mis, 2, 1)
                        rearrange( is_observed_mask[..., train_size:, :], 'b c T 1 -> (b T) c 1' )     # (bs x T_mis, C, 1)
                    ], dim = 1) # (bs x T_mis, 2+C, 1)

        # build sequence of KV:
        train_seq = torch.cat(train_seq, dim=1) # (bs x T_ctx, 1+2+C, d)
        test_seq  = torch.cat(test_seq, dim=1)  # type: ignore # (bs x T_mis, 2+C, d)

        # do cross-attn on train and test separately:
        cross_attn, cross_ff = self.cross_attn
        # query_tr: x [(b T_tr) 1 d]
        # kv_tr: [(b T_tr) 1+2+C d]
        # mask: (b T_tr) 1+2+C
        x_tr = cross_attn(
            x[:len(train_seq)],
            k    = train_seq,
            v    = train_seq,
            mask = attn_mask_train
        ) + x[:len(train_seq)] # (bs x T_ctx, 1, d)
        x_tr = cross_ff(x_tr) + x_tr
        x_tr = rearrange(x_tr, '(b T) 1 d -> b T d', b=bs)

        x_te = cross_attn(
            x[:len(test_seq)],
            k    = test_seq,
            v    = test_seq,
            mask = attn_mask_test
        ) + x[:len(test_seq)] # (bs x T_mis, 1, d)
        x_te = cross_ff(x_te) + x_te
        x_te = rearrange(x_te, '(b T) 1 d -> b T d', b=bs)

        # update feature tensor:
        R = R + torch.cat([x_tr, x_te], dim=1)

        # to tf icl:
        src = self.tf_icl(R, train_size=train_size)
        src = self.ln(src)
        
        # quantile head:
        out = self.decoder(src)  # (B, T, nb_quantiles)

        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        y_cov: Tensor | None = None,
    ) -> Tensor:
        """Predict target quantiles conditioned on context observations and optional covariates.

        Parameters
        ----------
        R : Tensor of shape (B, T, D)
            Per-timestep representations from the encoder (`T = train_size + target_len`).
        y_train : Tensor of shape (B, train_size, 1)
            Observed context values.
        y_cov : Tensor of shape (B, C, T_cov, 1), optional
            Optional covariates (`T_cov == train_size` or `T_cov == T`), may contain NaNs.

        Returns
        -------
        Tensor of shape (B, T - train_size, nb_quantiles)
            Predicted quantiles at target timesteps.
        """

        train_size = y_train.shape[1]
        if y_cov is not None:
            assert (y_cov.shape[2] == train_size) or (y_cov.shape[2] == R.shape[1])
        out = self._icl_predictions(R, y_train, y_cov)
        out = out[:, train_size:]

        return out

