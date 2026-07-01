import contextlib
from typing import NotRequired, TypedDict

import optuna
import torch
from optuna_integration.pytorch_distributed import TorchDistributedTrial
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from charl_tre.models.early_stopping import EarlyStopping
from charl_tre.models.vae.collapse_detector import CollapseDetector
from charl_tre.models.vae.free_bits_annealer import FreeBitsAnnealer
from charl_tre.models.vae.kl_annealer import KLAnnealer
from charl_tre.models.vae.loss import dirichlet_loss


def _is_dead(tensor: torch.Tensor) -> bool:
    """Check if a tensor contains NaN or infinite values."""
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


class DeadLossError(Exception):
    """Raised when the loss becomes NaN or infinite during training."""

    def __init__(self) -> None:
        """Initialize the error with the problematic loss value."""
        super().__init__("Loss became NaN or infinite.")


class PosteriorCollapseError(Exception):
    """Raised when the VAE posterior collapses during training."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Posterior collapse detected.")


class TrainerParams(TypedDict):
    """Parameters for configuring the VAE trainer."""

    lr: float
    """Learning rate for the optimizer."""
    early_stop_patience: int
    """Patience for early stopping."""
    early_stop_warmup_epochs: int
    """Number of epochs to warm up before starting early stopping."""
    collapse_patience: int
    """Patience for detecting posterior collapse."""
    kl_max: float
    """Maximum KL divergence weight."""
    prior_alpha: float
    """Prior alpha for the Dirichlet distribution."""
    free_bits_start: NotRequired[float]
    """Initial free-bits floor for the KL divergence term."""
    free_bits_end: NotRequired[float]
    """Final free-bits floor for the KL divergence term."""
    free_bits: NotRequired[float]
    """Backward-compatible constant free-bits floor."""


class Trainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        params: TrainerParams,
        gpu_id: int,
        trial: TorchDistributedTrial | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            train_dl: DataLoader for training data.
            val_dl: DataLoader for validation data.
            params: VAE training parameters.
            gpu_id: GPU ID for Distributed Data Parallel.
            trial: Optuna trial for hyperparameter optimization (optional).

        """
        self._device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        if self._device.type == "cuda":
            self._model = DistributedDataParallel(model.to(self._device), device_ids=[gpu_id], output_device=gpu_id)
        else:
            self._model = DistributedDataParallel(model.to(self._device))

        self._train_dl = train_dl
        self._val_dl = val_dl
        self._params = params
        self._prior_alpha = params["prior_alpha"]
        self._trial = trial
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=params["lr"],
        )
        self._scaler = torch.GradScaler(device=self._device.type)
        self._early_stopping = EarlyStopping(
            self._model,
            params["early_stop_patience"],
            params["early_stop_warmup_epochs"],
        )
        self._collapse_detector = CollapseDetector(params["collapse_patience"])

        self._len_train = len(train_dl)
        self._len_val = len(val_dl)

    def _run_batch(
        self,
        x_true: torch.Tensor,
        kl_weight: float,
        free_bits: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training batch.

        Arguments:
            x_true: Ground truth input tensor.
            kl_weight: Current KL divergence weight.
            free_bits: Current free-bits floor for KL divergence.

        Returns:
            Tuple containing total loss, reconstruction loss, KL divergence loss, and alpha.

        """
        self._optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            x_recon, alpha = self._model(x_true)
            loss, recon_loss, kl_loss = dirichlet_loss(
                x_recon,
                x_true,
                alpha,
                kl_weight=kl_weight,
                free_bits=free_bits,
                prior_alpha=self._prior_alpha,
            )

        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._scaler.step(self._optimizer)
        self._scaler.update()

        return loss, recon_loss, kl_loss, alpha

    def _run_epoch(
        self,
        epoch: int,
        kl_weight: float,
        free_bits: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training epoch.

        Arguments:
            epoch: Current epoch number.
            kl_weight: Current KL divergence weight.
            free_bits: Current free-bits floor for KL divergence.

        Returns:
            Tuple containing average total loss, reconstruction loss, and KL divergence loss.

        """
        self._model.train()

        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self._train_dl.sampler, DistributedSampler):
            self._train_dl.sampler.set_epoch(epoch)

        metrics = torch.zeros(3, device=self._device)

        for x_true, _ in self._train_dl:
            loss, recon_loss, kl_loss, _ = self._run_batch(
                x_true.to(self._device, non_blocking=True),
                kl_weight,
                free_bits,
            )

            metrics[0] += loss.detach()
            metrics[1] += recon_loss.detach()
            metrics[2] += kl_loss.detach()

        # Synchronize metrics across all processes
        if dist.is_initialized():
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        # Get total number of batches across all processes to account for batch size differences
        total_batches = torch.tensor(self._len_train, device=self._device)
        if dist.is_initialized():
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        # Average the loss and other metrics
        mean_metrics = metrics / total_batches

        return mean_metrics[0], mean_metrics[1], mean_metrics[2]

    @torch.no_grad()
    def _run_val_epoch(self, kl_weight: float, free_bits: float) -> torch.Tensor:
        """Run a single validation epoch.

        Arguments:
            kl_weight: Current KL divergence weight.
            free_bits: Current free-bits floor for KL divergence.

        Returns:
            Average validation loss over the validation dataset.

        """
        self._model.eval()

        total_loss = torch.tensor(0.0, device=self._device)

        for x_true_cpu, _ in self._val_dl:
            x_true = x_true_cpu.to(self._device, non_blocking=True)

            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                x_recon, alpha = self._model(x_true)
                loss, _, _ = dirichlet_loss(
                    x_recon,
                    x_true,
                    alpha,
                    kl_weight=kl_weight,
                    free_bits=free_bits,
                    prior_alpha=self._prior_alpha,
                )

            total_loss += loss.detach()

        total_batches = torch.tensor(self._len_val, device=self._device)
        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        return total_loss / total_batches

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Number of epochs to train.

        Returns:
            Tuple containing average total loss, reconstruction loss, and KL divergence loss over all epochs.

        """
        total_metrics = torch.zeros(3, device=self._device)
        annealer = KLAnnealer(epochs, kl_max=self._params["kl_max"])

        # Backward compatibility: if only free_bits is provided, use a flat schedule.
        start_free_bits = self._params.get("free_bits_start", self._params.get("free_bits", 0.0))
        end_free_bits = self._params.get("free_bits_end", self._params.get("free_bits", 0.0))
        free_bits_annealer = FreeBitsAnnealer(
            total_epochs=epochs,
            start_value=start_free_bits,
            end_value=end_free_bits,
        )
        epochs_run = 0

        for epoch in range(epochs):
            annealer.step()
            free_bits_annealer.step()
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self._run_epoch(
                epoch,
                annealer.weight,
                free_bits_annealer.value,
            )

            if _is_dead(torch.tensor([epoch_loss, epoch_recon_loss, epoch_kl_loss])):
                raise DeadLossError

            self._collapse_detector.step(epoch_kl_loss)
            if self._collapse_detector.is_collapsed():
                raise PosteriorCollapseError

            # Distributed averaging of metrics
            total_metrics += torch.stack([epoch_loss, epoch_recon_loss, epoch_kl_loss])
            epochs_run += 1

            val_loss = self._run_val_epoch(annealer.weight, free_bits_annealer.value)
            if _is_dead(val_loss):
                raise DeadLossError

            self._early_stopping.step(val_loss)
            if self._early_stopping.should_stop:
                break

            if self._trial is not None:
                self._trial.report(epoch_loss.item(), step=epoch)

                if self._trial.should_prune():
                    raise optuna.TrialPruned

        with contextlib.suppress(RuntimeError):
            self._early_stopping.restore_best_weights()

        total_metrics /= epochs_run

        return total_metrics[0].item(), total_metrics[1].item(), total_metrics[2].item()
