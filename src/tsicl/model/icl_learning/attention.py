# ACKNOWLEDGEMENT:
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/attention.py

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

def multi_head_attention_forward(
    x: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    train_size: int,
    training: bool = True,
) -> Tensor:
    """Multi-head attention with support for rotary position embeddings
    as well as specialized processing when attn_mask is an integer.

    Parameters
    ----------
    x : Tensor
        Tensor of shape (..., tgt_len, embed_dim)

    num_heads : int
        Number of attention heads

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections

    in_proj_bias : Tensor
        Combined bias vector for input projections

    dropout_p : float
        Dropout probability applied to attention weights

    out_proj_weight : Tensor
        Output projection weight matrix

    out_proj_bias : Tensor
        Output projection bias vector

    train_size : int
        Creates a split attention pattern where:
            - The first `train_size` tokens perform self-attention only (attend to themselves)
            - The remaining tokens attend only to the first `train_size` tokens
        
    training : bool, default=True
        Whether the model is in training mode (affects dropout)

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., tgt_len, embed_dim)
    """

    # Extract shape information, supporting arbitrary batch dimensions
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    # Joint projection of query, key, value
    q, k, v = F.linear(x, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    # Process left segment (self-attention within first cut_pos tokens)
    k_ctx = k[:, :, :train_size, :]
    v_ctx = v[:, :, :train_size, :]
    attn_ctx = F.scaled_dot_product_attention(q[:, :, :train_size, :], k_ctx, v_ctx, dropout_p=dropout_p)

    # Process right segment (tokens after train_size attending to tokens before train_size)
    if train_size < seq_len:
        attn_tgt = F.scaled_dot_product_attention(q[:, :, train_size:, :], k_ctx, v_ctx, dropout_p=dropout_p)
        attn = torch.cat([attn_ctx, attn_tgt], dim=2)
    else:
        attn = attn_ctx

    # heads merging + output projection : (B, nh, T, hs) -> (B, T, E)
    attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    return F.linear(attn, out_proj_weight, out_proj_bias)
