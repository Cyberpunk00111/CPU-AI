"""Tokenizer helpers shared by preprocessing and inference."""

from __future__ import annotations

import logging
from pathlib import Path

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class CogniCoreTokenizer:
    """Small wrapper around HuggingFace tokenizers for consistent IDs.

    Args:
        tokenizer: Loaded HuggingFace tokenizer instance.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token = "<pad>"

    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""
        token_id = self.tokenizer.token_to_id(self.eos_token)
        if token_id is None:
            raise ValueError("Tokenizer is missing <eos> special token")
        return token_id

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        token_id = self.tokenizer.token_to_id(self.bos_token)
        if token_id is None:
            raise ValueError("Tokenizer is missing <bos> special token")
        return token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: Text to encode.
            add_special_tokens: Whether to wrap the text with BOS/EOS.

        Returns:
            Token ID list.
        """
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""
        return self.tokenizer.decode(token_ids)


def load_tokenizer(tokenizer_path: str | Path) -> CogniCoreTokenizer:
    """Load a tokenizer from tokenizer.json in the configured directory.

    Args:
        tokenizer_path: Directory containing tokenizer.json.

    Returns:
        Loaded CogniCoreTokenizer.
    """
    path = Path(tokenizer_path) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    logger.info("Loading tokenizer from %s", path)
    return CogniCoreTokenizer(Tokenizer.from_file(str(path)))
