from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """q: (B, H, N_q, D), k/v: (B, H, N_kv, D), mask: bool (B, N_kv) or (B, N_kv, 1) with True = keep."""
    attn_mask = None
    if mask is not None:
        # Reshape (B, ...) -> (B, 1, 1, N_kv) pour broadcast sur H et N_q
        attn_mask = mask.bool().reshape(mask.shape[0], 1, 1, -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

class FeedForward(nn.Module):
    """ A two-layer MLP with GeLU intermediate activation."""

    def __init__(
        self,
        dim: int,
        mult: int = 4,
    ) -> None:
        """
        A two-layer MLP with GeLU intermediate activation.

        Args:
            dim (int): input and output dim
            mult (int): multiplicative factor for the hidden layer dim
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttention(nn.Module):
    """ An implementation of multi-head cross-attention with arbitrary q, k, v."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.,
    ) -> None:
        """
        Multi-head cross-attention with arbitrary q, k, v.

        Linear projection to Q and K, V -> one-layer attention -> feedforward.

        Args:
            query_dim (int): input dim of the query
            key_dim (int): input dim of the key
            value_dim (int): input dim of the values
            heads (int): number of heads
            dim_head (int): dim of one attention head
            dropout (float): dropout frac
        """

        super().__init__()
        
        inner_dim      = dim_head * heads
        self.heads     = heads

        # input projectors:
        self.to_q   = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k   = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        # attention and residual connection dropouts:
        self.dropout  = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        q, k, v = self.to_q(x), self.to_k(k), self.to_v(v)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = _sdpa(q, k, v, mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.resid_drop(self.to_out(out))

class Attention(nn.Module):
    """ Multi-head self-attention, or cross-attention if context is provided for KV tensors."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.,
    ) -> None:
        """
        Multi-head self-attention or cross-attention if context is provided for KV tensors.
        
        Linear projection to Q and K, V (jointly from context) -> one-layer attention -> feedforward.

        Args:
            query_dim (int): input dim of the query
            context_dim (int): input dim of the context, defaults to `query_dim`
            out_dim (int): dim of the output, defaults to `query_dim`
            heads (int): number of attention heads
            dim_head (int): dim of attention head
            dropout (float): attention and ff dropout  
        """
        
        super().__init__()

        inner_dim      = dim_head * heads
        self.heads     = heads

        # input projectors:
        self.to_q   = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv  = nn.Linear(context_dim if context_dim is not None else query_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim if out_dim is not None else query_dim)

        # attention and residual connection dropouts:
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        context = context if context is not None else x
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (k, v))
        out = _sdpa(q, k, v, mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.resid_drop(self.to_out(out))

class MultiScaleAttention(Attention):
    """
    Multi-head multi-scale (queries at different frequency bandwidths) self-attention,
    or cross-attention if context is provided for KV tensors.
    """

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        s = x.shape[2]
        context = context if context is not None else x
        q = rearrange(self.to_q(x), "b n s (h d) -> b h (s n) d", h=self.heads)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (k, v))
        out = _sdpa(q, k, v, mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out, "b h (s n) d -> b n s (h d)", s=s)
        return self.resid_drop(self.to_out(out))
