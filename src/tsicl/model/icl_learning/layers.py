# ACKNOWLEDGEMENT:
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/layers.py

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import multi_head_attention_forward

class MultiheadAttention(nn.Module):
    """Enhanced multi-head attention with rotary positional embedding support.

    This extends PyTorch's MultiheadAttention to support rotary position embeddings (RoPE)
    and specialized attention masking when `train_size` is an integer. The implementation always
    uses `batch_first=True`, meaning all input tensors have shape (..., seq_len, embed_dim).

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined)

    num_heads : int
        Number of attention heads

    dropout : float, default=0.0
        Dropout probability applied to attention weights

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: Tensor, train_size: int) -> Tensor:
        """Compute multi-head attention with support for rotary positional encoding.

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

        return multi_head_attention_forward(
            x,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            train_size=train_size,
        )


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

    dropout : float, default=0.0
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
        """Process input through attention with optional rotary positional encoding.

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
