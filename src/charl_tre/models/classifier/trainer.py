import torch
from torch import distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from charl_tre.util import split_test_window


class Trainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        vae: nn.Module,
        sample_window_size: int,
        overlap_size: int,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer.

        Arguments:
            model: The classifier model to be trained.
            dataloader: DataLoader for the training dataset.
            vae: The pre-trained VAE model used for feature extraction.
            sample_window_size: The size of the window to split the input samples into.
            overlap_size: The number of frames to overlap between windows.
            gpu_id: The GPU ID to use for training.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__vae = vae.to(gpu_id)
        self.__sample_window_size = sample_window_size
        self.__overlap_size = overlap_size

        self.__gpu_id = gpu_id
        self.__criterion = nn.CrossEntropyLoss()

        self.__optimizer = optim.Adam(self.__model.parameters(), lr=1e-3)

    def __run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single training batch."""
        self.__optimizer.zero_grad()

        original_batch_size = x.shape[0]

        with torch.no_grad():
            # Split every x along the window size dimension into separate samples,
            # so that we can feed them into the VAE.
            x_r = split_test_window(x, self.__sample_window_size, self.__overlap_size)

            _, z_hard, _ = self.__vae(x_r)

            # (B * n_windows, latent_dim, n_categories) → (B, latent_dim * n_categories * n_windows)
            n_windows = x_r.shape[0] // original_batch_size
            z_hard = z_hard.view(original_batch_size, n_windows, -1)
            z_hard = z_hard.reshape(original_batch_size, -1)

        logits = self.__model(z_hard)
        loss = self.__criterion(logits, y)

        loss.backward()
        self.__optimizer.step()

        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == y).float().mean()

        return loss.detach(), accuracy.detach()

    def __run_epoch(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        metrics = torch.zeros(2, device=self.__gpu_id)

        for x, y in self.__dataloader:
            loss, accuracy = self.__run_batch(x.to(self.__gpu_id), y.to(self.__gpu_id))

            metrics[0] += loss
            metrics[1] += accuracy

        # Sum metrics across all processes
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        # Get total number of batches across all processes to account for batch size differences
        total_batches = torch.tensor(len(self.__dataloader), dtype=torch.int64, device=self.__gpu_id)
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

        dist.all_reduce(total_metrics, op=dist.ReduceOp.SUM)
        total_metrics /= dist.get_world_size()

        return tuple(total_metrics.tolist())
