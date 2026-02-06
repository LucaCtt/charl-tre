import optuna
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.train.annealers import (
    CapacityAnnealer,
    GumbelTemperatureAnnealer,
    KLWeightAnnealer,
)
from csi_vae_gumbel.train.vae_loss import vae_loss
from csi_vae_gumbel.train.vae_parameters import VAEParameters


class VAETrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        parameters: VAEParameters,
        gpu_id: int,
        trial: optuna.integration.TorchDistributedTrial | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            parameters: VAE training parameters.
            gpu_id: GPU ID for Distributed Data Parallel.
            trial: Optuna trial for hyperparameter optimization (optional).
            callback: Optional callback function to be called after each epoch.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        self.__lr_annealer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.__optimizer,
            mode="min",
        )
        self.__capacity_annealer = CapacityAnnealer(
            final_capacity=parameters.final_cap,
        )
        self.__temperature_annealer = GumbelTemperatureAnnealer(
            start_tau=parameters.start_gumbel_temp,
        )
        self.__kl_weight_annealer = KLWeightAnnealer(
            max_weight=parameters.final_kl_weight,
        )

        self.__gpu_id = gpu_id
        self.__trial = trial

    def __run_batch(
        self,
        x_true: torch.Tensor,
        tau: float,
        kl_weight: float,
        capacity: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__optimizer.zero_grad()

        x_recon, _, logits = self.__model(x_true, tau)

        loss, recon_loss, kl_loss = vae_loss(
            x_recon,
            x_true,
            logits,
            kl_weight=kl_weight,
            capacity=capacity,
        )

        loss.backward()
        self.__optimizer.step()

        return loss, recon_loss, kl_loss

    def __run_epoch(self, epoch: int) -> tuple[float, float, float]:
        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        metrics = torch.tensor([0.0, 0.0, 0.0], device=self.__gpu_id)

        tau = self.__temperature_annealer.step(epoch)
        kl_weight = self.__kl_weight_annealer.step(epoch)
        capacity = self.__capacity_annealer.step(epoch)

        for x_true, _ in self.__dataloader:
            loss, recon_loss, kl_loss = self.__run_batch(
                x_true.to(self.__gpu_id),
                tau,
                kl_weight,
                capacity,
            )

            metrics += torch.tensor(
                [loss.detach(), recon_loss.detach(), kl_loss.detach()],
                device=self.__gpu_id,
            )

        # Synchronize metrics across all processes so the lr_annealer gets the correct value
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        # Get total number of batches across all processes to account for batch size differences
        total_batches = torch.tensor(len(self.__dataloader), device=self.__gpu_id)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
        metrics /= total_batches

        # This has to be called after each epoch
        self.__lr_annealer.step(metrics[0])

        return tuple(metrics.tolist())

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

        total_metrics = torch.tensor([0.0, 0.0, 0.0], device=self.__gpu_id)

        for epoch in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self.__run_epoch(epoch)

            # Distributed averaging of metrics
            total_metrics += torch.tensor(
                [epoch_loss, epoch_recon_loss, epoch_kl_loss],
                device=self.__gpu_id,
            )

            if self.__trial is not None:
                self.__trial.report(epoch_loss, step=epoch)
                kl_history = self.__trial.user_attrs.get("kl_history", [])
                kl_history.append(epoch_kl_loss)
                self.__trial.set_user_attr("kl_history", kl_history)

                if self.__trial.should_prune():
                    msg = f"Pruned at epoch {epoch}."
                    raise optuna.TrialPruned(msg)

        total_metrics /= epochs

        return tuple(total_metrics.tolist())
