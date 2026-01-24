import logging
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from csi_vae_gumbel.model.vae import VAE, vae_loss
from csi_vae_gumbel.train.checkpoints import CheckpointManager
from csi_vae_gumbel.train.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: VAE,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        early_stopping: EarlyStopping,
        checkpoint_manager: CheckpointManager,
        gpu_id: int,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for training.
            early_stopping: EarlyStopping instance to monitor training.
            checkpoint_manager: CheckpointManager to save model checkpoints.
            gpu_id: GPU ID for Distributed Data Parallel.

        """
        self.model = DistributedDataParallel(model, device_ids=[gpu_id])
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager
        self.gpu_id = gpu_id

    def __run_batch(self, x_true: torch.Tensor) -> tuple[float, float, float]:
        self.optimizer.zero_grad()
        x_recon, z = self.model(x_true, 0.1)
        loss, recon_loss, kl_loss, _, _ = vae_loss(x_recon, x_true, z)

        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), kl_loss.item()

    def __run_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()

        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.dataloader.sampler, DistributedSampler):
            self.dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        with tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            disable=(self.gpu_id != 0),
        ) as progress_bar:
            for x_true, _ in progress_bar:
                loss, recon_loss, kl_loss = self.__run_batch(x_true.to(self.gpu_id))

                epoch_loss += loss
                epoch_recon += recon_loss
                epoch_kl += kl_loss

                progress_bar.set_postfix(
                    OrderedDict(
                        [
                            ("loss", loss),
                            ("recon_loss", recon_loss),
                            ("kl_loss", kl_loss),
                        ],
                    ),
                )
        epoch_loss /= len(self.dataloader)
        epoch_recon /= len(self.dataloader)
        epoch_kl /= len(self.dataloader)

        return epoch_loss, epoch_recon, epoch_kl

    def train(self, n_epochs: int) -> None:
        """Train the VAE model for a specified number of epochs."""
        for epoch in range(n_epochs):
            epoch_loss, epoch_recon, epoch_kl = self.__run_epoch(epoch)

            if self.gpu_id == 0:
                self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch)
                logger.info([epoch, epoch_loss, epoch_recon, epoch_kl])

            if self.early_stopping.step(epoch_loss):
                break
