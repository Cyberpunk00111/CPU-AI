"""SwiGLU feed-forward network used in CogniCore transformer blocks."""

from __future__ import annotations

import torch
from torch import nn

from cognicore.model.bitlinear import BitLinear


class SwiGLUFeedForward(nn.Module):
    """SwiGLU MLP block with BitLinear projections.

    Args:
        embedding_dim: Input and output hidden size.
        hidden_dim: Intermediate feed-forward width.
        activation_bits: Activation quantisation bit width for BitLinear.
        dropout: Dropout probability after the down projection.
        eps: Numerical stability constant for BitLinear scaling.
        beta: SwiGLU beta multiplier.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation_bits: int,
        dropout: float,
        eps: float,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gate_proj = BitLinear(embedding_dim, hidden_dim, activation_bits, False, eps)
        self.up_proj = BitLinear(embedding_dim, hidden_dim, activation_bits, False, eps)
        self.down_proj = BitLinear(hidden_dim, embedding_dim, activation_bits, False, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SwiGLU projection.

        Args:
            x: Hidden states shaped (batch, sequence, embedding_dim).

        Returns:
            Tensor with the same shape as x.
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = gate * torch.sigmoid(self.beta * gate) * up
        return self.dropout(self.down_proj(hidden))
