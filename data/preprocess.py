"""Tokenise the CogniCore corpus into train/validation binary shards.

Processes the corpus in streaming chunks to avoid loading 2 GB of text into
memory all at once.  Each <eos>-delimited sample is tokenised independently and
appended to an in-memory token list which is flushed to disk once all samples
are consumed.  On a 10 M-parameter model the uint16 shard typically fits in
RAM (< 4 GB even for 500 K Wikipedia articles).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from cognicore.model.config import load_config
from cognicore.model.tokenizer.tokenizer_utils import load_tokenizer

logger = logging.getLogger(__name__)

# Number of text samples to batch before logging progress.
_LOG_EVERY = 10_000


def preprocess(config_path: str | Path) -> tuple[Path, Path]:
    """Encode corpus text and write 90/10 train/validation .bin files.

    Reads the corpus line by line, groups samples by the <eos> sentinel, and
    encodes them with the trained BPE tokenizer.  Writes two uint16 binary
    shards: ``train.bin`` (90 %) and ``val.bin`` (10 %).

    Args:
        config_path: Path to CogniCore YAML config.

    Returns:
        Tuple of (train_path, val_path).

    Raises:
        FileNotFoundError: If corpus or tokenizer is missing.
        ValueError: If vocab_size exceeds uint16 range.
    """
    config = load_config(config_path)
    corpus_path = config.data.corpus_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    if config.model.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("Configured vocab_size exceeds uint16 shard capacity (max 65535)")

    logger.info("Streaming corpus from %s", corpus_path)

    all_tokens: list[int] = []
    sample_buf: list[str] = []
    sample_count = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line == "<eos>":
                # End of one training sample — encode the accumulated text.
                text = " ".join(sample_buf).strip()
                if text:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(ids)
                    all_tokens.append(tokenizer.eos_id)
                sample_buf = []
                sample_count += 1
                if sample_count % _LOG_EVERY == 0:
                    logger.info(
                        "Processed %d samples | %d tokens accumulated",
                        sample_count,
                        len(all_tokens),
                    )
            else:
                sample_buf.append(line)

    # Flush any trailing sample that had no closing <eos>.
    if sample_buf:
        text = " ".join(sample_buf).strip()
        if text:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)

    total_tokens = len(all_tokens)
    logger.info(
        "Corpus fully tokenised: %d samples → %d tokens", sample_count, total_tokens
    )
    if total_tokens == 0:
        raise ValueError("No tokens produced — check corpus format and tokenizer.")

    # 90/10 train/validation split by token index.
    split_index = int(total_tokens * 0.9)
    train_arr = np.asarray(all_tokens[:split_index], dtype=np.uint16)
    val_arr = np.asarray(all_tokens[split_index:], dtype=np.uint16)

    config.data.train_path.parent.mkdir(parents=True, exist_ok=True)
    config.data.val_path.parent.mkdir(parents=True, exist_ok=True)
    train_arr.tofile(config.data.train_path)
    val_arr.tofile(config.data.val_path)

    logger.info(
        "Wrote %d train tokens to %s", len(train_arr), config.data.train_path
    )
    logger.info(
        "Wrote %d val tokens to %s", len(val_arr), config.data.val_path
    )
    return config.data.train_path, config.data.val_path


def main() -> None:
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to CogniCore YAML config.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    preprocess(args.config)


if __name__ == "__main__":
    main()
