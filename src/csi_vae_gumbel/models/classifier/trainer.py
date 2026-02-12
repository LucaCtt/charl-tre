import torch
from torch import distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class Trainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        vae: nn.Module,
        test_window_factor: int,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__vae = vae.to(gpu_id)
        self.__test_window_factor = test_window_factor
        self.__gpu_id = gpu_id
        self.__criterion = nn.CrossEntropyLoss()

        self.__optimizer = optim.Adam(self.__model.parameters())

    def __run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single training batch."""
        self.__optimizer.zero_grad()

        batch_size = x.shape[0]
        window_size = x.shape[2] // self.__test_window_factor

        with torch.no_grad():
            # Split every x along the window size dimension into separate samples,
            # so that we can feed them into the VAE.
            xs = []
            for i in range(self.__test_window_factor):
                xi = x[:, :, window_size * i : window_size * (i + 1), :]

                _, z_hard, _ = self.__vae(xi.to(self.__gpu_id))

                # (B, latent_dim, n_categories) → (B, latent_dim * n_categories)
                z_hard = z_hard.view(batch_size, -1)

                xs.append(z_hard)

            xs = torch.cat(xs, dim=1)  # (B, latent_dim * n_categories * test_window_factor)

        logits = self.__model(xs)
        loss = self.__criterion(logits, y.to(self.__gpu_id))

        loss.backward()
        self.__optimizer.step()

        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == y.to(self.__gpu_id)).float().mean()

        return loss.detach(), accuracy.detach()

    def __run_epoch(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        metrics = torch.zeros(2, device=self.__gpu_id)

        for x, y in self.__dataloader:
            loss, accuracy = self.__run_batch(x.to(self.__gpu_id), y.to(self.__gpu_id))

            metrics += torch.tensor([loss.detach(), accuracy], device=self.__gpu_id)

        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        # Get total number of batches across all processes to account for batch size differences
        total_batches = torch.tensor(len(self.__dataloader), device=self.__gpu_id)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        metrics /= total_batches

        return metrics[0], metrics[1]

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()
        self.__vae.eval()

        total_metrics = torch.zeros(2, device=self.__gpu_id)

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch(epoch)
            total_metrics += torch.tensor([epoch_loss, epoch_accuracy], device=self.__gpu_id)

        total_metrics /= epochs

        return tuple(total_metrics.tolist())
