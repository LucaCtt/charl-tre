import torch
from torch import distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


class ClassifierTrainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        vae: nn.Module,
        optimizer: optim.Optimizer,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__vae = vae.to(gpu_id)  # No need to DDP the VAE as it's frozen
        self.__optimizer = optimizer
        self.__gpu_id = gpu_id
        self.__criterion = nn.CrossEntropyLoss()

    def __run_epoch(self, epoch: int) -> tuple[float, float]:
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for x, y in self.__dataloader:
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

        epoch_loss /= len(self.__dataloader)
        epoch_accuracy /= len(self.__dataloader)

        return epoch_loss, epoch_accuracy

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()
        self.__vae.eval()

        total_loss = 0.0
        total_accuracy = 0.0

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch(epoch)

            metrics = torch.tensor([epoch_loss, epoch_accuracy], device=self.__gpu_id)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= dist.get_world_size()

            total_loss += metrics[0].item()
            total_accuracy += metrics[1].item()

        total_loss /= epochs
        total_accuracy /= epochs

        return total_loss, total_accuracy
