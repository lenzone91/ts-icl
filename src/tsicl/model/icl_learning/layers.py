# ACKNOWLEDGEMENT:
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/layers.py

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """Enhanced multi-head attention.

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined)

    num_heads : int
        Number of attention heads

    dropout : float, default=0.0
        Dropout probability applied to attention weights
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: Tensor, train_size: int) -> Tensor:
        """Compute multi-head attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., tgt_len, embed_dim)

        train_size : int
            Creates a split attention pattern where:
              - The first `train_size` tokens perform self-attention only (attend to themselves)
              - The remaining tokens attend only to the first `train_size` tokens

        Returns
        -------
        Tensor
            Attention output of shape (..., tgt_len, embed_dim)
        """

        batch_size, seq_len, embed_dim = x.shape

        # Projection conjointe Q, K, V : (B, T, E) -> 3 x (B, nh, T, hs)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0

        # Le contexte ([:train_size]) s'observe lui-même
        k_ctx = k[:, :, :train_size, :]
        v_ctx = v[:, :, :train_size, :]
        attn_ctx = F.scaled_dot_product_attention(q[:, :, :train_size, :], k_ctx, v_ctx, dropout_p=dropout_p)

        # Les cibles ([train_size:]) observent uniquement le contexte
        if train_size < seq_len:
            attn_tgt = F.scaled_dot_product_attention(q[:, :, train_size:, :], k_ctx, v_ctx, dropout_p=dropout_p)
            attn = torch.cat([attn_ctx, attn_tgt], dim=2)
        else:
            attn = attn_ctx

        # Fusion des têtes + projection de sortie
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn)

class MultiheadAttentionBlock(nn.Module):
    """Attention block supporting rotary positional encoding.

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
       Dimension of the feedforward network

    dropout : float
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
    ):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activations = {"relu": F.relu, "gelu": F.gelu}
            if activation not in activations:
                raise ValueError(f"Unknown activation: {activation}")
            self.activation = activations[activation]
        else:
            self.activation = activation

        self.init_weights()

    def init_weights(self):
        """Initialize residual output projections to zero for stable training."""
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor, train_size: int) -> Tensor:
        """Process input through attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., tgt_len, d_model)

        train_size : int, default=None
            Creates a split attention pattern where:
              - The first `train_size` tokens perform self-attention only (attend to themselves)
              - The remaining tokens attend only to the first `train_size` tokens

        Returns
        -------
        Tensor
            Output tensor of shape (..., tgt_len, d_model)
        """

        # Pre-norm: normalize before attention and FFN
        x = x + self._attn_block(self.norm1(x), train_size)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _attn_block(self, x: Tensor, train_size: int) -> Tensor:
        return self.dropout1(self.attn(x, train_size))

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
        Number of multihead attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model         = d_model,
                    nhead           = nhead,
                    dim_feedforward = dim_feedforward,
                    dropout         = dropout,
                    activation      = activation,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, src: Tensor, train_size: int) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : int
            Creates a split attention pattern where:
              - The first `train_size` tokens perform self-attention only (attend to themselves)
              - The remaining tokens attend only to the first `train_size` tokens

        Returns
        -------
        Tensor
            Output tensor of shape (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(x=out, train_size=train_size)

        return out
