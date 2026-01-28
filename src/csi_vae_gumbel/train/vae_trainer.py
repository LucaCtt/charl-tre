from collections.abc import Callable

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.model.loss import vae_loss
from csi_vae_gumbel.model.vae import MultiViewCategoricalVAE
from csi_vae_gumbel.train.async_callback_worker import AsyncCallbackWorker
from csi_vae_gumbel.train.checkpoints import CheckpointManager
from csi_vae_gumbel.train.early_stopping import EarlyStopping


class VAETrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: MultiViewCategoricalVAE,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        early_stopping: EarlyStopping,
        checkpoint_manager: CheckpointManager,
        gpu_id: int,
        batch_callback: Callable | None = None,
        epoch_callback: Callable | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for training.
            early_stopping: EarlyStopping instance to monitor training.
            checkpoint_manager: CheckpointManager to save model checkpoints.
            gpu_id: GPU ID for Distributed Data Parallel.
            batch_callback: Optional callback function called at the end of each batch.
            epoch_callback: Optional callback function called at the end of each epoch.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = optimizer
        self.__early_stopping = early_stopping
        self.__checkpoint_manager = checkpoint_manager
        self.__gpu_id = gpu_id
        self.__batch_callback = batch_callback
        self.__epoch_callback = epoch_callback

        self.__callback_worker = AsyncCallbackWorker()

    def __run_batch(self, x_true: torch.Tensor) -> tuple[float, float, float]:
        self.__optimizer.zero_grad()
        x_recon, z = self.__model(x_true, 0.9)

        loss, recon_loss, kl_loss, _, _ = vae_loss(x_recon, x_true, z)

        loss.backward()
        self.__optimizer.step()

        return loss.item(), recon_loss.item(), kl_loss.item()

    def __run_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.__model.train()

        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for x_true, _ in self.__dataloader:
            loss, recon_loss, kl_loss = self.__run_batch(x_true.to(self.__gpu_id))

            if self.__gpu_id == 0 and self.__batch_callback is not None:
                self.__callback_worker.submit(self.__batch_callback, epoch, loss, recon_loss, kl_loss)

            epoch_loss += loss
            epoch_recon += recon_loss
            epoch_kl += kl_loss

        epoch_loss /= len(self.__dataloader)
        epoch_recon /= len(self.__dataloader)
        epoch_kl /= len(self.__dataloader)

        return epoch_loss, epoch_recon, epoch_kl

    def train(self, max_epochs: int) -> None:
        """Train the VAE model for a specified number of epochs."""
        latest_checkpoint = self.__checkpoint_manager.load_latest_checkpoint()
        if latest_checkpoint is not None:
            model_state, optimizer_state, start_epoch = latest_checkpoint
            self.__model.module.load_state_dict(model_state)
            self.__optimizer.load_state_dict(optimizer_state)
        else:
            start_epoch = 0

        for epoch in range(start_epoch, max_epochs):
            epoch_loss, epoch_recon, epoch_kl = self.__run_epoch(epoch)

            if self.__gpu_id == 0:
                self.__checkpoint_manager.save_checkpoint(
                    self.__model.module.state_dict(),
                    self.__optimizer.state_dict(),
                    epoch,
                )
                if self.__epoch_callback is not None:
                    self.__callback_worker.submit(self.__epoch_callback, epoch, epoch_loss, epoch_recon, epoch_kl)

            if self.__early_stopping.step(epoch_loss):
                break
