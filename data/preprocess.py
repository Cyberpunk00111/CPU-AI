"""Tokenise the CogniCore corpus into train/validation binary shards.

Processes the corpus in streaming chunks to avoid loading 2 GB of text into
memory all at once.  Each <eos>-delimited sample is tokenised independently and
appended to an in-memory token list which is flushed to disk once all samples
are consumed.  On a 10 M-parameter model the uint16 shard typically fits in
RAM (< 4 GB even for 500 K Wikipedia articles).
"""

from __future__ import annotations

import argparse
import array
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

    # Use a temporary file to store token IDs incrementally on disk.
    # This prevents Out Of Memory (OOM) errors in limited-RAM environments.
    temp_path = config.data.train_path.parent / "temp_tokens.bin"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        temp_path.unlink()

    token_buffer = array.array("H")  # 'H' is unsigned 16-bit short (uint16)
    sample_buf: list[str] = []
    sample_count = 0
    total_tokens = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line == "<eos>":
                # End of one training sample — encode the accumulated text.
                text = " ".join(sample_buf).strip()
                if text:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(ids)
                    token_buffer.append(tokenizer.eos_id)
                    total_tokens += len(ids) + 1
                sample_buf = []
                sample_count += 1

                # Flush token buffer to disk periodically (every 1,000,000 tokens)
                # to keep RAM usage extremely low (~50MB overhead).
                if len(token_buffer) >= 1_000_000:
                    with temp_path.open("ab") as tfh:
                        token_buffer.tofile(tfh)
                    token_buffer = array.array("H")

                if sample_count % _LOG_EVERY == 0:
                    logger.info(
                        "Processed %d samples | %d tokens accumulated",
                        sample_count,
                        total_tokens,
                    )
            else:
                sample_buf.append(line)

    # Flush any trailing sample that had no closing <eos>.
    if sample_buf:
        text = " ".join(sample_buf).strip()
        if text:
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids)
            total_tokens += len(ids)

    if len(token_buffer) > 0:
        with temp_path.open("ab") as tfh:
            token_buffer.tofile(tfh)

    logger.info(
        "Corpus fully tokenised: %d samples → %d tokens", sample_count, total_tokens
    )
    if total_tokens == 0:
        if temp_path.exists():
            temp_path.unlink()
        raise ValueError("No tokens produced — check corpus format and tokenizer.")

    # 90/10 train/validation split by token index.
    split_index = int(total_tokens * 0.9)
    split_byte_index = split_index * 2  # 2 bytes per uint16 token

    config.data.train_path.parent.mkdir(parents=True, exist_ok=True)
    config.data.val_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the temporary binary file in chunks and write them to train.bin and val.bin.
    chunk_size = 64 * 1024 * 1024  # 64 MB chunk size
    bytes_written = 0

    with temp_path.open("rb") as sfh, \
         config.data.train_path.open("wb") as trfh, \
         config.data.val_path.open("wb") as vafh:

        while True:
            data = sfh.read(chunk_size)
            if not data:
                break

            if bytes_written + len(data) <= split_byte_index:
                trfh.write(data)
            elif bytes_written >= split_byte_index:
                vafh.write(data)
            else:
                # Chunk spans the 90/10 split boundary.
                split_offset = split_byte_index - bytes_written
                trfh.write(data[:split_offset])
                vafh.write(data[split_offset:])

            bytes_written += len(data)

    # Clean up the temporary file.
    if temp_path.exists():
        temp_path.unlink()

    logger.info(
        "Wrote %d train tokens to %s", split_index, config.data.train_path
    )
    logger.info(
        "Wrote %d val tokens to %s", total_tokens - split_index, config.data.val_path
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
