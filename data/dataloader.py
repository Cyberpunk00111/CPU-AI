"""CPU-friendly memory-mapped next-token dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped token dataset for autoregressive training.

    Args:
        path: Binary uint16 token shard.
        seq_length: Number of input tokens per item.
    """

    def __init__(self, path: str | Path, seq_length: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token shard not found: {self.path}")
        self.seq_length = seq_length
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= seq_length:
            raise ValueError("Token shard must contain more tokens than seq_length")

    def __len__(self) -> int:
        """Return number of next-token windows in the shard."""
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input and target token windows."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk = np.asarray(self.tokens[index : index + self.seq_length + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y
