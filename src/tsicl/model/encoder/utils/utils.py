from typing import Sequence

import numpy as np
import torch

class DiagonalGaussianDistribution(object):

    def __init__(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False
    ) -> None:

        self.mean = mean
        self.deterministic = deterministic
        
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )
        else:
            self.logvar = logvar
            self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, K: int = 1) -> torch.Tensor:
        """ Draw `K`samples from the Diag Gaussian distrib."""

        if K == 1:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(
                device=self.mean.device
            )
            return x
        else:
            # XXX check dims in repeat for time series
            x = self.mean[None, ...].repeat([K, 1, 1, 1]) + self.std[None, ...].repeat(
                K, 1, 1, 1
            ) * torch.randn([K, *self.mean.shape]).to(device=self.mean.device)
            return x

    def nll(
        self,
        sample: torch.Tensor,
        dims: Sequence = [1, 2]
    ) -> torch.Tensor:
        
        if self.deterministic:
            return torch.Tensor([0.0])
        
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
