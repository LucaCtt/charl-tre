from collections.abc import Callable

from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from csi_vae_gumbel.models.vae import MultiViewCategoricalVAE
from csi_vae_gumbel.train.async_callback_worker import AsyncCallbackWorker


class ClassifierTrainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        vae: MultiViewCategoricalVAE,
        optimizer: optim.Optimizer,
        batch_callback: Callable | None,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__vae = DistributedDataParallel(vae.to(gpu_id), device_ids=[gpu_id])
        self.__optimizer = optimizer
        self.__batch_callback = batch_callback
        self.__gpu_id = gpu_id

        self.__callback_worker = AsyncCallbackWorker()

    def train(self, epochs: int) -> None:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()

        for epoch in range(epochs):
            if isinstance(self.__dataloader.sampler, DistributedSampler):
                self.__dataloader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_accuracy = 0.0

            for x, y in self.__dataloader:
                self.__optimizer.zero_grad()

                _, logits_vae = self.__vae(x)
                # Flatten the VAE logits for the classifier input
                logits_vae = logits_vae.view(logits_vae.size(0), -1)
                # Detach to avoid backprop through VAE
                logits_vae = logits_vae.detach()

                logits = self.__model(logits_vae)
                loss = nn.CrossEntropyLoss()(logits, y.to(self.__gpu_id))
                accuracy = (logits.argmax(dim=1) == y.to(self.__gpu_id)).float().mean()

                loss.backward()
                self.__optimizer.step()

                if self.__gpu_id == 0 and self.__batch_callback is not None:
                    self.__callback_worker.submit(self.__batch_callback, epoch, loss.item(), accuracy.item())

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()

            epoch_loss /= len(self.__dataloader)
            epoch_accuracy /= len(self.__dataloader)
