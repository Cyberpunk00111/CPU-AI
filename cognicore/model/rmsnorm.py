"""RMSNorm implementation for CPU-efficient transformer blocks."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation without mean subtraction.

    Args:
        dim: Size of the final hidden dimension.
        eps: Numerical stability constant added before the square root.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x by its root mean square over the final dimension.

        Args:
            x: Tensor with hidden dimension at the final axis.

        Returns:
            RMS-normalised tensor with the same shape as x.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
