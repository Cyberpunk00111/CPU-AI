"""Supervised Fine-Tuning (SFT) runner for CogniCore.

Loads a pre-trained base checkpoint and fine-tunes it on the SFT instruction dataset.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cognicore.model.architecture import CogniCoreModel
from cognicore.model.config import CogniCoreConfig, load_config
from data.dataloader import TokenShardDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cognicore.sft.train")


def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class SFTTrainer:
    """Trainer class for SFT/Fine-tuning."""

    def __init__(self, config: CogniCoreConfig, pretrained_checkpoint_path: Path) -> None:
        self.config = config
        set_reproducibility(config.training.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using SFT training device: %s", self.device)
        
        # Instantiate model
        self.model = CogniCoreModel(config).to(self.device)
        
        # Load weights from pre-trained checkpoint
        logger.info("Loading pre-trained base model weights from: %s", pretrained_checkpoint_path)
        try:
            checkpoint = torch.load(pretrained_checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint)
            # Remove head weight if shape differs, though they match in Phase 2
            self.model.load_state_dict(state_dict)
            logger.info("Successfully loaded pre-trained model weights.")
        except Exception as exc:
            logger.error("Failed to load pre-trained checkpoint: %s", exc)
            raise exc

        # Initialize fresh SFT optimizer (reset pre-training optimizer state)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )
        
        # Data Loaders
        self.train_loader = self._make_loader(config.data.train_path, shuffle=True)
        self.val_loader = self._make_loader(config.data.val_path, shuffle=False)
        
        self.checkpoint_dir = Path(config.logging.log_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _make_loader(
        self, path: Path, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create dataloader for token shards."""
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
        """Persist fine-tuned model checkpoint."""
        path = self.checkpoint_dir / f"cognicore_sft_step_{step}.pt"
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
        logger.info("Saved SFT checkpoint to %s", path)
        return path

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute validation loss on SFT validation set."""
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
            return 0.0
        return float(sum(losses) / len(losses))

    def train(self) -> None:
        """Run SFT training loop starting from step 0."""
        self.model.train()
        logger.info("SFT Hyperparameters: %s", self.config.training.model_dump())
        
        step = 0
        max_steps = self.config.training.max_steps
        
        while step < max_steps:
            for input_ids, targets in self.train_loader:
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Apply cosine learning rate schedule
                lr = learning_rate_for_step(step, self.config)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(input_ids, targets)
                
                if output.loss is None:
                    raise RuntimeError("SFT forward pass did not return a loss")
                
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )
                self.optimizer.step()

                if step % self.config.logging.log_every == 0:
                    logger.info("step=%d SFT_loss=%.4f lr=%.8f", step, output.loss.item(), lr)
                    
                if step > 0 and step % self.config.training.eval_every == 0:
                    val_loss = self.evaluate()
                    logger.info(
                        "step=%d SFT_val_loss=%.4f SFT_perplexity=%.4f",
                        step,
                        val_loss,
                        math.exp(val_loss) if val_loss < 50 else float('inf'),
                    )
                    
                if step > 0 and step % self.config.training.save_every == 0:
                    self.save_checkpoint(step, float(output.loss.item()))

                step += 1
                if step >= max_steps:
                    break
        
        # Save final checkpoint
        self.save_checkpoint(max_steps, float(output.loss.item()))
        logger.info("SFT Training complete.")


def main() -> None:
    """CLI entry point."""
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to SFT YAML config.")
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        help="Path to pre-trained base model checkpoint file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pretrained_path = Path(args.pretrained_checkpoint)
    if not pretrained_path.exists():
        logger.error("Pretrained checkpoint file not found: %s", pretrained_path)
        sys.exit(1)

    SFTTrainer(config, pretrained_path).train()


if __name__ == "__main__":
    main()
