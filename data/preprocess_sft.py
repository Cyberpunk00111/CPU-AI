"""Tokenise the SFT corpus into train/validation binary shards using the Phase 2 tokenizer."""

from __future__ import annotations

import argparse
import array
import logging
import sys
from pathlib import Path

# Add current working directory to path for imports
sys.path.append(str(Path.cwd()))

import numpy as np

from cognicore.model.config import load_config
from cognicore.model.tokenizer.tokenizer_utils import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cognicore.sft.preprocess")

_LOG_EVERY = 5000


def preprocess_sft(config_path: str | Path) -> tuple[Path, Path]:
    """Encode SFT corpus text and write 90/10 train/validation .bin files."""
    config = load_config(config_path)
    corpus_path = config.data.corpus_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    if config.model.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("Configured vocab_size exceeds uint16 shard capacity (max 65535)")

    logger.info("Streaming SFT corpus from %s", corpus_path)

    temp_path = config.data.train_path.parent / "temp_sft_tokens.bin"
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
                text = " ".join(sample_buf).strip()
                if text:
                    # Tokenise using add_special_tokens=False (we append eos_id manually)
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(ids)
                    token_buffer.append(tokenizer.eos_id)
                    total_tokens += len(ids) + 1
                sample_buf = []
                sample_count += 1

                # Flush token buffer to disk periodically
                if len(token_buffer) >= 500_000:
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

    # Flush any trailing sample
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
        "SFT Corpus fully tokenised: %d samples → %d tokens", sample_count, total_tokens
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

    chunk_size = 16 * 1024 * 1024  # 16 MB chunk size
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
                split_offset = split_byte_index - bytes_written
                trfh.write(data[:split_offset])
                vafh.write(data[split_offset:])

            bytes_written += len(data)

    if temp_path.exists():
        temp_path.unlink()

    logger.info(
        "Wrote %d SFT train tokens to %s", split_index, config.data.train_path
    )
    logger.info(
        "Wrote %d SFT val tokens to %s", total_tokens - split_index, config.data.val_path
    )
    return config.data.train_path, config.data.val_path


def main() -> None:
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to SFT YAML config.")
    args = parser.parse_args()
    preprocess_sft(args.config)


if __name__ == "__main__":
    main()
