import torch
import torch.nn as nn

class PreNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        fn: nn.Module,
        context_dim: int | None = None
    ) -> None:
        """
        z-norm over the last dim of the input (LayerNorm) before applying a given module.

        Args:
            dim (int): last dimension of the input (queries)
            fn (nn.Module): the module before which LayerNorm is applied,
                typically `Attention`, `MultiScaleAttention` or `FeedForward`
            context_dim (int): last dimension of the context inputs, if provided
        """

        super().__init__()

        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None # type: ignore

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        # normalize inputs:
        x = self.norm(x)

        # normalize context if provided:
        if self.norm_context is not None:
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        # apply module on normalized inputs:
        return self.fn(x, **kwargs)


class PreNormCross(nn.Module):

    def __init__(
        self,
        dim: int,
        fn: nn.Module,
        k_dim: int | None = None,
        v_dim: int | None = None
    ):
        """
        z-norm over the last dim of the input (LayerNorm) before applying a given module.

        Args:
            dim (int): last dimension of the input (queries)
            fn (nn.Module): the module before which LayerNorm is applied,
                typically `CrossAttention`
            k_dim (int): last dimension of the keys, if provided
            v_dim (int): last dimension of the values, if provided
        """

        super().__init__()

        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.norm_k = nn.LayerNorm(k_dim) if k_dim is not None else None # type: ignore
        self.norm_v = nn.LayerNorm(v_dim) if v_dim is not None else None # type: ignore
        assert not ( (k_dim is not None) ^ (v_dim is not None) ) # not xor

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        # normalize queries:
        x = self.norm(x)

        # normalize keys and values:
        if self.norm_v is not None:
            k = kwargs["k"]
            v = kwargs["v"]
            normed_k = self.norm_k(k)
            normed_v = self.norm_v(v)
            kwargs.update(k=normed_k, v=normed_v)

        # apply module over normalized inputs:
        return self.fn(x, **kwargs)


