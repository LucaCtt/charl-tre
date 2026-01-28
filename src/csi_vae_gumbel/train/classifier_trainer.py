from collections.abc import Callable

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
        optimizer: optim.Optimizer,
        batch_callback: Callable | None,
        epoch_callback: Callable | None,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = optimizer
        self.__batch_callback = batch_callback
        self.__epoch_callback = epoch_callback
        self.__gpu_id = gpu_id

        self.__callback_worker = AsyncCallbackWorker()

    def train(self, epochs: int) -> float:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()

        total_loss = 0.0

        for epoch in range(epochs):
            if isinstance(self.__dataloader.sampler, DistributedSampler):
                self.__dataloader.sampler.set_epoch(epoch)

            epoch_loss = 0.0

            for x, y in self.__dataloader:
                self.__optimizer.zero_grad()
                logits = self.__model(x)
                loss = nn.CrossEntropyLoss()(logits, y)

                loss.backward()
                self.__optimizer.step()

                if self.__gpu_id == 0 and self.__batch_callback is not None:
                    self.__callback_worker.submit(self.__batch_callback, epoch, loss)

                epoch_loss += loss.item()

            epoch_loss /= len(self.__dataloader)

            if self.__gpu_id == 0 and self.__epoch_callback is not None:
                self.__callback_worker.submit(self.__epoch_callback, epoch, epoch_loss)

            total_loss += epoch_loss

        total_loss /= epochs

        return total_loss
