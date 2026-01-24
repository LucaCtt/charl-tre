"""Checkpoint management for saving and loading model states."""

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """Save model and optimizer state dicts as a checkpoint."""
        path = self.checkpoint_dir / f"cp-{epoch:04d}.safetensors"
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_file(state_dict, path)
