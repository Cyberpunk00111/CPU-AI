"""Tests for BitLinear quantisation."""

import torch

from cognicore.model.bitlinear import BitLinear


def test_quantized_weight_is_ternary() -> None:
    """BitLinear should produce {-1, 0, +1} forward weights."""
    layer = BitLinear(4, 3, activation_bits=8, bias=False, eps=1.0e-6)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor([[0.1, 0.8, -0.6, 0.0], [2.0, -2.0, 0.1, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
    unique_values = set(layer.quantized_weight().detach().flatten().tolist())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


def test_bitlinear_forward_shape() -> None:
    """BitLinear should preserve leading dimensions and change only the final axis."""
    layer = BitLinear(4, 6, activation_bits=8, bias=True, eps=1.0e-6)
    output = layer(torch.randn(2, 5, 4))
    assert output.shape == (2, 5, 6)
