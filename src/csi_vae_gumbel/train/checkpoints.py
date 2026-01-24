from pathlib import Path

import torch


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize the CheckpointManager with a directory to save checkpoints.

        Arguments:
            checkpoint_dir: Directory where checkpoints will be saved.

        """
        self.checkpoint_dir = checkpoint_dir

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model_state: dict, optimizer_state: dict, epoch: int) -> None:
        """Save model and optimizer state dicts as a checkpoint."""
        path = self.checkpoint_dir / f"cp-{epoch:04d}.pt"
        state_dict = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }
        torch.save(state_dict, path)

    def load_latest_checkpoint(self) -> tuple[dict, dict, int] | None:
        """Load the latest checkpoint if available.

        Returns:
            A tuple containing the model state dict, optimizer state dict, and epoch number.
            If no checkpoint is found, returns None.

        """
        checkpoints = list(self.checkpoint_dir.glob("cp-*.pt"))
        if not checkpoints:
            return None

        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(latest_checkpoint)

        model_state = checkpoint["model_state"]
        optimizer_state = checkpoint["optimizer_state"]
        epoch = checkpoint["epoch"]

        return model_state, optimizer_state, epoch
