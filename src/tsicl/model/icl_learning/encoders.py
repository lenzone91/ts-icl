# ACKNOWLEDGEMENT:
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/encoders.py

from __future__ import annotations

from typing import Callable

from torch import Tensor, nn

from .layers import MultiheadAttentionBlock

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
