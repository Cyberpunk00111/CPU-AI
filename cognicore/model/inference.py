"""Streaming inference engine for CogniCore."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import torch

from cognicore.model.architecture import CogniCoreModel
from cognicore.model.config import CogniCoreConfig, load_config
from cognicore.model.tokenizer.tokenizer_utils import CogniCoreTokenizer, load_tokenizer

logger = logging.getLogger(__name__)


class CogniCoreInference:
    """Load a trained CogniCore model and stream decoded tokens.

    Args:
        config: Validated CogniCore config.
        tokenizer: Loaded tokenizer wrapper.
        checkpoint_path: Optional checkpoint to restore.
    """

    def __init__(
        self,
        config: CogniCoreConfig,
        tokenizer: CogniCoreTokenizer,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = CogniCoreModel(config)
        self.model_loaded = False
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    @classmethod
    def from_paths(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
    ) -> CogniCoreInference:
        """Create an inference engine from filesystem paths."""
        config = load_config(config_path)
        tokenizer = load_tokenizer(config.data.tokenizer_path)
        return cls(config, tokenizer, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore model weights from a training checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model_loaded = True
        logger.info("Loaded inference checkpoint from %s", path)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> Iterator[str]:
        """Yield decoded text fragments token by token.

        Args:
            prompt: User prompt text.
            max_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature.
            top_k: Top-k sampling cutoff.

        Yields:
            Decoded token fragments.
        """
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        generated = input_ids
        for _ in range(max_tokens):
            context = generated[:, -self.config.model.max_seq_length :]
            logits = self.model(context).logits[:, -1, :] / max(temperature, 1.0e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            if token_id == self.tokenizer.eos_id:
                break
            generated = torch.cat((generated, next_token), dim=1)
            yield self.tokenizer.decode([token_id])
