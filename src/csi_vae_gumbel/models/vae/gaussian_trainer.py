import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.models.vae.loss import gaussian_vae_loss


class GaussianTrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        gpu_id: int,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            parameters: VAE training parameters.
            gpu_id: GPU ID for Distributed Data Parallel.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = torch.optim.AdamW(
            self.__model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        self.__lr_annealer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.__optimizer,
            mode="min",
        )

        self.__gpu_id = gpu_id

    def __run_batch(
        self,
        x_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training batch.

        Arguments:
            x_true: Ground truth input tensor.

        Returns:
            Tuple containing total loss, reconstruction loss, KL divergence loss, and logits.

        """
        self.__optimizer.zero_grad()

        x_recon, logits, mu, logvar = self.__model(x_true)

        loss, recon_loss, kl_loss = gaussian_vae_loss(x_recon, x_true, mu, logvar)

        loss.backward()
        self.__optimizer.step()

        return loss, recon_loss, kl_loss, logits

    def __run_epoch(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training epoch.

        Arguments:
            epoch: Current epoch number.

        Returns:
            Tuple containing average total loss, reconstruction loss,
            KL divergence loss, and average latent entropy for the epoch.

        """
        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        metrics = torch.zeros(4, device=self.__gpu_id)
        n_latents = torch.tensor(0, device=self.__gpu_id)

        for x_true, _ in self.__dataloader:
            loss, recon_loss, kl_loss, _ = self.__run_batch(x_true.to(self.__gpu_id))

            metrics[0] += loss.detach()
            metrics[1] += recon_loss.detach()
            metrics[2] += kl_loss.detach()

        # Synchronize metrics across all processes
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_latents, op=dist.ReduceOp.SUM)

        # Get total number of batches across all processes to account for batch size differences
        total_batches = torch.tensor(len(self.__dataloader), device=self.__gpu_id)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        # Average the loss and other metrics
        mean_metrics = metrics[:3] / total_batches
        mean_entropy = metrics[3] / n_latents

        # This has to be called after each epoch
        self.__lr_annealer.step(mean_metrics[0])

        return mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_entropy.unsqueeze(0)

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Number of epochs to train.
            max_epochs_zero_kl: Maximum number of consecutive epochs with near-zero KL divergence before pruning.
            eps: Threshold to consider KL divergence as near-zero.

        Returns:
            Tuple containing average total loss, reconstruction loss,
            and KL divergence loss over all epochs.

        """
        self.__model.train()

        total_metrics = torch.zeros(4, device=self.__gpu_id)

        for epoch in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss, epoch_entropy = self.__run_epoch(epoch)

            # Distributed averaging of metrics
            total_metrics += torch.tensor(
                [epoch_loss, epoch_recon_loss, epoch_kl_loss, epoch_entropy],
                device=self.__gpu_id,
            )

        total_metrics /= epochs

        return tuple(total_metrics[:3].tolist())
