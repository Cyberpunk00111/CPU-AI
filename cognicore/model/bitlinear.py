"""BitLinear layer using ternary weights and INT8-style activation quantisation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass while preserving identity gradients.

    Args:
        x: Tensor to round.

    Returns:
        Rounded tensor with straight-through estimator gradients.
    """
    return (torch.round(x) - x).detach() + x


class BitLinear(nn.Module):
    """Linear layer with BitNet-style absmean ternary weight quantisation.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation_bits: Activation quantisation bit width.
        bias: Whether to include an additive bias.
        eps: Numerical stability constant for scaling factors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_bits: int,
        bias: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise parameters using the default PyTorch linear-layer scheme."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def activation_qmax(self) -> int:
        """Maximum positive integer representable by the activation bit width."""
        return (2 ** (self.activation_bits - 1)) - 1

    @property
    def activation_qmin(self) -> int:
        """Minimum signed integer representable by the activation bit width."""
        return -(2 ** (self.activation_bits - 1))

    def quantized_weight(self) -> torch.Tensor:
        """Return STE ternary weights in {-1, 0, +1}.

        Returns:
            Tensor with the same shape as weight and ternary forward values.
        """
        alpha = self.weight.abs().mean().clamp_min(self.eps)
        return ste_round(self.weight / alpha).clamp(-1, 1)

    def quantized_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise activations per token along the final dimension.

        Args:
            x: Input activation tensor.

        Returns:
            Tuple of quantised activations and their per-token scaling gamma.
        """
        gamma = (x.abs().amax(dim=-1, keepdim=True) / self.activation_qmax).clamp_min(self.eps)
        x_q = ste_round(x / gamma).clamp(self.activation_qmin, self.activation_qmax)
        return x_q, gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantised linear projection and rescale to floating point.

        Args:
            x: Input tensor ending with in_features.

        Returns:
            Output tensor ending with out_features.
        """
        alpha = self.weight.abs().mean().clamp_min(self.eps)
        weight_q = self.quantized_weight()
        x_q, gamma = self.quantized_activation(x)
        out = F.linear(x_q, weight_q, self.bias)
        return out * alpha * gamma
