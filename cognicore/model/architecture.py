"""Decoder-only CogniCore transformer architecture."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cognicore.model.attention import GroupedQueryAttention
from cognicore.model.config import CogniCoreConfig
from cognicore.model.ffn import SwiGLUFeedForward
from cognicore.model.rmsnorm import RMSNorm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CogniCoreOutput:
    """Output bundle returned by CogniCore forward passes."""

    logits: torch.Tensor
    loss: torch.Tensor | None


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with GQA attention and SwiGLU feed-forward layers."""

    def __init__(self, config: CogniCoreConfig) -> None:
        super().__init__()
        model_config = config.model
        self.attn_norm = RMSNorm(model_config.embedding_dim, model_config.norm_eps)
        self.attn = GroupedQueryAttention(
            embedding_dim=model_config.embedding_dim,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            max_seq_length=model_config.max_seq_length,
            activation_bits=model_config.activation_bits,
            dropout=model_config.dropout,
            eps=model_config.norm_eps,
        )
        self.ffn_norm = RMSNorm(model_config.embedding_dim, model_config.norm_eps)
        self.ffn = SwiGLUFeedForward(
            embedding_dim=model_config.embedding_dim,
            hidden_dim=model_config.ffn_hidden_dim,
            activation_bits=model_config.activation_bits,
            dropout=model_config.dropout,
            eps=model_config.norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a transformer block with residual connections."""
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class CogniCoreModel(nn.Module):
    """CPU-native decoder-only language model using BitLinear transformer blocks.

    Args:
        config: Validated CogniCore configuration.
    """

    def __init__(self, config: CogniCoreConfig) -> None:
        super().__init__()
        self.config = config
        model_config = config.model
        self.token_embedding = nn.Embedding(model_config.vocab_size, model_config.embedding_dim)
        self.dropout = nn.Dropout(model_config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(model_config.num_layers)]
        )
        self.final_norm = RMSNorm(model_config.embedding_dim, model_config.norm_eps)
        self.lm_head = nn.Linear(model_config.embedding_dim, model_config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise trainable parameters with GPT-style small normal weights."""
        std = 0.02
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> CogniCoreOutput:
        """Run next-token language-model forward pass.

        Args:
            input_ids: Token IDs shaped (batch, sequence).
            targets: Optional target token IDs shaped (batch, sequence).

        Returns:
            CogniCoreOutput containing logits and optional cross-entropy loss.
        """
        if input_ids.size(1) > self.config.model.max_seq_length:
            raise ValueError("input sequence exceeds configured max_seq_length")
        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return CogniCoreOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """Autoregressively sample tokens from the model.

        Args:
            input_ids: Prompt token IDs shaped (batch, sequence).
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature.
            top_k: Number of highest-probability tokens to sample from.

        Returns:
            Token IDs containing prompt and generated continuation.
        """
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.model.max_seq_length :]
            logits = self(context).logits[:, -1, :] / max(temperature, 1.0e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    def parameter_count(self) -> int:
        """Return the total number of model parameters."""
        return sum(parameter.numel() for parameter in self.parameters())
