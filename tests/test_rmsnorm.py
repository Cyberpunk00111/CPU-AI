"""Tests for RMSNorm."""

import torch

from cognicore.model.rmsnorm import RMSNorm


def test_rmsnorm_matches_formula() -> None:
    """RMSNorm should divide by RMS without subtracting the mean."""
    layer = RMSNorm(dim=4, eps=1.0e-6)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    assert torch.allclose(layer(x), expected)
