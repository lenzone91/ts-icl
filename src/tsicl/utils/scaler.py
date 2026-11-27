from typing import Union

import torch

class CustomStandardScaler:

    def __init__(self, dim: Union[int, tuple] = 1, epsilon: float = 1e-5) -> None:
        self.mean    = torch.empty((1,))
        self.std     = torch.empty((1,))
        self.dim     = dim
        self.epsilon = epsilon
        
    def fit(self, X: torch.Tensor) -> None:
        # X has shape (batch_size, grid_size, d) (in general: d=1)        
        x = X
        self.mean: torch.Tensor = torch.nan_to_num( torch.nanmean(x, dim=self.dim, keepdim=True), nan=0.0)
        scale = torch.nan_to_num( (x - self.mean).square().nanmean(dim=self.dim, keepdim=True).sqrt(), nan=1.0)
        self.std: torch.Tensor  = torch.where(scale==0, self.epsilon, scale)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X_transform = (X - self.mean) / self.std

        return X_transform
    
    def inv_transform(self, X: torch.Tensor) -> torch.Tensor:
        std  = self.std
        mean = self.mean

        X_inv_transform = X * std + mean

        return X_inv_transform
