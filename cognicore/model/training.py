"""CPU-optimised training loop for CogniCore."""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cognicore.model.architecture import CogniCoreModel
from cognicore.model.config import CogniCoreConfig, load_config
from data.dataloader import TokenShardDataset

logger = logging.getLogger(__name__)


def set_reproducibility(seed: int) -> None:
    """Set all local random seeds used by the training path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cpu(config: CogniCoreConfig) -> None:
    """Apply configured CPU threading preferences."""
    if config.cpu.num_threads > 0:
        torch.set_num_threads(config.cpu.num_threads)
    logger.info("PyTorch CPU threads: %d", torch.get_num_threads())


def learning_rate_for_step(step: int, config: CogniCoreConfig) -> float:
    """Cosine learning-rate schedule with linear warmup."""
    base_lr = config.training.learning_rate
    if step < config.training.warmup_steps:
        return base_lr * float(step + 1) / float(max(config.training.warmup_steps, 1))
    progress = (step - config.training.warmup_steps) / max(
        config.training.max_steps - config.training.warmup_steps,
        1,
    )
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Owns model, optimiser, dataloaders, and checkpointing.

    Args:
        config: Validated CogniCore configuration.
    """

    def __init__(self, config: CogniCoreConfig) -> None:
        self.config = config
        set_reproducibility(config.training.seed)
        configure_cpu(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using training device: %s", self.device)
        self.model = CogniCoreModel(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )
        self.train_loader = self._make_loader(config.data.train_path, shuffle=True)
        self.val_loader = self._make_loader(config.data.val_path, shuffle=False)
        self.checkpoint_dir = Path(config.logging.log_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _make_loader(
        self, path: Path, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create a CPU DataLoader for a token shard."""
        dataset = TokenShardDataset(path, self.config.model.max_seq_length)
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def save_checkpoint(self, step: int, loss: float) -> Path:
        """Persist a training checkpoint."""
        path = self.checkpoint_dir / f"cognicore_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "loss": loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.model_dump(mode="json"),
            },
            path,
        )
        logger.info("Saved checkpoint to %s", path)
        return path

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute mean validation loss over a subset of validation passes."""
        self.model.eval()
        losses: list[float] = []
        for i, (input_ids, targets) in enumerate(self.val_loader):
            if i >= self.config.training.eval_iters:
                break
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            output = self.model(input_ids, targets)
            if output.loss is not None:
                losses.append(float(output.loss.item()))
        self.model.train()
        if not losses:
            raise RuntimeError("Validation loader produced no batches")
        return float(sum(losses) / len(losses))

    def train(self) -> None:
        """Run the configured training loop."""
        self.model.train()
        logger.info("Training hyperparameters: %s", self.config.training.model_dump())
        step = 0
        while step < self.config.training.max_steps:
            for input_ids, targets in self.train_loader:
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                lr = learning_rate_for_step(step, self.config)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(input_ids, targets)
                if output.loss is None:
                    raise RuntimeError("Training forward pass did not return a loss")
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )
                self.optimizer.step()

                if step % self.config.logging.log_every == 0:
                    logger.info("step=%d loss=%.4f lr=%.8f", step, output.loss.item(), lr)
                if step > 0 and step % self.config.training.eval_every == 0:
                    val_loss = self.evaluate()
                    logger.info(
                        "step=%d val_loss=%.4f perplexity=%.4f",
                        step,
                        val_loss,
                        math.exp(val_loss),
                    )
                if step > 0 and step % self.config.training.save_every == 0:
                    self.save_checkpoint(step, float(output.loss.item()))

                step += 1
                if step >= self.config.training.max_steps:
                    break


def main() -> None:
    """CLI entry point for model training."""
    import multiprocessing

    # Required for Windows: prevents re-execution of the __main__ block in
    # spawned DataLoader worker processes (Windows uses 'spawn' not 'fork').
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to CogniCore YAML config.")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write training logs to a file.",
    )
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        from pathlib import Path as _Path

        _Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    config = load_config(args.config)
    Trainer(config).train()


if __name__ == "__main__":
    main()
