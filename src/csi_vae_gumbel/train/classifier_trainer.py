from collections.abc import Callable

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from csi_vae_gumbel.train.async_callback_worker import AsyncCallbackWorker


class ClassifierTrainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        vae: nn.Module,
        optimizer: optim.Optimizer,
        batch_callback: Callable | None,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__vae = vae.to(gpu_id)  # No need to DDP the VAE as it's frozen
        self.__optimizer = optimizer
        self.__batch_callback = batch_callback
        self.__gpu_id = gpu_id
        self.__criterion = nn.CrossEntropyLoss()

        self.__callback_worker = AsyncCallbackWorker()

    def __run_epoch(self, epoch: int) -> tuple[float, float]:
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i, (x, y) in enumerate(self.__dataloader):
            self.__optimizer.zero_grad()

            with torch.no_grad():
                _, z_hard_vae, _ = self.__vae(x.to(self.__gpu_id))
                z_hard_vae = z_hard_vae.view(z_hard_vae.size(0), -1)

            logits = self.__model(z_hard_vae)
            loss = self.__criterion(logits, y.to(self.__gpu_id))
            accuracy = (logits.argmax(dim=1) == y.to(self.__gpu_id)).float().mean()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

            loss.backward()
            self.__optimizer.step()

            if self.__gpu_id == 0 and self.__batch_callback is not None:
                n_batches = i + 1

                self.__callback_worker.submit(
                    self.__batch_callback,
                    epoch,
                    epoch_loss / n_batches,
                    epoch_accuracy / n_batches,
                )

        epoch_loss /= len(self.__dataloader)
        epoch_accuracy /= len(self.__dataloader)

        return epoch_loss, epoch_accuracy

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()
        self.__vae.eval()

        self.__callback_worker.start()

        total_loss = 0.0
        total_accuracy = 0.0

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch(epoch)

            total_loss += epoch_loss
            total_accuracy += epoch_accuracy

        total_loss /= epochs
        total_accuracy /= epochs

        self.__callback_worker.stop()
        return total_loss, total_accuracy
