"""Train CogniCore's Byte-Level BPE tokenizer from a corpus file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from cognicore.model.config import load_config

logger = logging.getLogger(__name__)


def train_tokenizer(config_path: str | Path) -> Path:
    """Train and save the configured BPE tokenizer.

    Args:
        config_path: Path to CogniCore YAML config.

    Returns:
        Directory where tokenizer files were written.
    """
    config = load_config(config_path)
    corpus_path = config.data.corpus_path
    output_dir = config.data.tokenizer_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=config.model.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<eos>", "<bos>", "<unk>", "<sep>"],
    )
    tokenizer.save_model(str(output_dir))
    tokenizer.save(str(output_dir / "tokenizer.json"))
    logger.info("Saved tokenizer to %s", output_dir)
    return output_dir


def main() -> None:
    """CLI entry point for tokenizer training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to CogniCore YAML config.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_tokenizer(args.config)


if __name__ == "__main__":
    main()
