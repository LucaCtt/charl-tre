import contextlib
from typing import TypedDict

import optuna
import torch
from optuna_integration.pytorch_distributed import TorchDistributedTrial
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from charl_tre.models import util
from charl_tre.models.hierarchical import annealers, errors
from charl_tre.models.hierarchical.collapse_detector import CollapseDetector
from charl_tre.models.hierarchical.early_stopping import EarlyStopping
from charl_tre.models.hierarchical.loss import hierarchical_loss


class TrainerParams(TypedDict):
    """Parameters for HierarchicalTrainer."""

    lr: float
    early_stop_patience: int
    early_stop_warmup_epochs: int
    collapse_patience: int
    kl_dirichlet_final: float
    kl_gaussian_final: float
    free_bits_start: float
    free_bits_end: float


class Trainer:
    """Trainer architecture tracking separate regularization schedules for multi-tier VAEs."""

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        params: TrainerParams,
        gpu_id: int,
        trial: TorchDistributedTrial | None = None,
    ) -> None:
        """Initialize the HierarchicalTrainer.

        Arguments:
            model (nn.Module): The hierarchical model to train.
            train_dl (DataLoader): The training data loader.
            val_dl (DataLoader): The validation data loader.
            params (TrainerParams): The training parameters.
            gpu_id (int): The ID of the GPU to use.
            trial (TorchDistributedTrial | None): The Optuna trial for distributed training.

        """
        self._device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        if self._device.type == "cuda":
            self._model = DistributedDataParallel(model.to(self._device), device_ids=[gpu_id], output_device=gpu_id)
        else:
            self._model = DistributedDataParallel(model.to(self._device))

        self._train_dl = train_dl
        self._val_dl = val_dl
        self._params = params
        self._trial = trial
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=params["lr"])
        self._scaler = torch.GradScaler(device=self._device.type)

        self._early_stopping = EarlyStopping(
            self._model,
            params["early_stop_patience"],
            params["early_stop_warmup_epochs"],
        )

        # Level-isolated monitors to mitigate localized posterior collapses
        self._dirichlet_collapse_detector = CollapseDetector(params["collapse_patience"])
        self._gaussian_collapse_detector = CollapseDetector(params["collapse_patience"])

        self._len_train = len(train_dl)
        self._len_val = len(val_dl)

    def _run_batch(
        self,
        x_true: torch.Tensor,
        kl_weight_dirichlet: float,
        kl_weight_gaussian: float,
        free_bits_dirichlet: float,
        free_bits_gaussian: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            recon, mix_logits, alpha, mu_q, logvar_q, mu_p, logvar_p = self._model(x_true)

            loss, recon_loss, kl_dir, kl_gauss = hierarchical_loss(
                recon,
                x_true,
                mix_logits,
                alpha,
                mu_q,
                logvar_q,
                mu_p,
                logvar_p,
                kl_weight_dirichlet=kl_weight_dirichlet,
                kl_weight_gaussian=kl_weight_gaussian,
                free_bits_dirichlet=free_bits_dirichlet,
                free_bits_gaussian=free_bits_gaussian,
            )

        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._scaler.step(self._optimizer)
        self._scaler.update()

        return loss, recon_loss, kl_dir, kl_gauss

    def _run_epoch(
        self,
        epoch: int,
        kl_weight_dirichlet: float,
        kl_weight_gaussian: float,
        free_bits_dirichlet: float,
        free_bits_gaussian: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._model.train()
        if isinstance(self._train_dl.sampler, DistributedSampler):
            self._train_dl.sampler.set_epoch(epoch)

        metrics = torch.zeros(4, device=self._device)

        for x_true, _ in self._train_dl:
            loss, recon_loss, kl_dir, kl_gauss = self._run_batch(
                x_true.to(self._device, non_blocking=True),
                kl_weight_dirichlet,
                kl_weight_gaussian,
                free_bits_dirichlet,
                free_bits_gaussian,
            )
            metrics[0] += loss.detach()
            metrics[1] += recon_loss.detach()
            metrics[2] += kl_dir.detach()
            metrics[3] += kl_gauss.detach()

        if dist.is_initialized():
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        total_batches = torch.tensor(self._len_train, device=self._device)
        if dist.is_initialized():
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        mean_metrics = metrics / total_batches
        return mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_metrics[3]

    @torch.no_grad()
    def _run_val_epoch(
        self,
        kl_weight_dirichlet: float,
        kl_weight_gaussian: float,
        free_bits_dirichlet: float,
        free_bits_gaussian: float,
    ) -> torch.Tensor:
        self._model.eval()
        total_loss = torch.tensor(0.0, device=self._device)

        for x_true_cpu, _ in self._val_dl:
            x_true = x_true_cpu.to(self._device, non_blocking=True)
            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                recon, mix_logits, alpha, mu_q, logvar_q, mu_p, logvar_p = self._model(x_true)
                loss, _, _, _ = hierarchical_loss(
                    recon,
                    x_true,
                    mix_logits,
                    alpha,
                    mu_q,
                    logvar_q,
                    mu_p,
                    logvar_p,
                    kl_weight_dirichlet=kl_weight_dirichlet,
                    kl_weight_gaussian=kl_weight_gaussian,
                    free_bits_dirichlet=free_bits_dirichlet,
                    free_bits_gaussian=free_bits_gaussian,
                )
            total_loss += loss.detach()

        total_batches = torch.tensor(self._len_val, device=self._device)
        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        return total_loss / total_batches

    def train(self, epochs: int) -> tuple[float, float, float, float]:
        """Train the model for a specified number of epochs.

        Arguments:
            epochs (int): The number of epochs to train the model.

        Returns:
            tuple[float, float, float, float]:
                - Average loss over the training epochs.
                - Average reconstruction loss over the training epochs.
                - Average KL divergence for the Dirichlet distribution over the training epochs.
                - Average KL divergence for the Gaussian distribution over the training epochs.

        """
        total_metrics = torch.zeros(4, device=self._device)

        dirichlet_annealer = annealers.KL(epochs, kl_final=self._params["kl_dirichlet_final"])
        gaussian_annealer = annealers.KL(epochs, kl_final=self._params["kl_gaussian_final"])

        dirichlet_fb_annealer = annealers.FreeBits(
            total_epochs=epochs,
            start_value=self._params["free_bits_start"],
            end_value=self._params["free_bits_end"],
        )
        gaussian_fb_annealer = annealers.FreeBits(
            total_epochs=epochs,
            start_value=self._params["free_bits_start"],
            end_value=self._params["free_bits_end"],
        )
        epochs_run = 0

        for epoch in range(epochs):
            dirichlet_annealer.step()
            gaussian_annealer.step()
            dirichlet_fb_annealer.step()
            gaussian_fb_annealer.step()

            epoch_loss, epoch_recon, epoch_kl_dir, epoch_kl_gauss = self._run_epoch(
                epoch,
                dirichlet_annealer.weight,
                gaussian_annealer.weight,
                dirichlet_fb_annealer.value,
                gaussian_fb_annealer.value,
            )

            if util.is_dead(torch.stack([epoch_loss, epoch_recon, epoch_kl_dir, epoch_kl_gauss])):
                raise errors.DeadLossError

            # Evaluate split collapse criteria independently
            self._dirichlet_collapse_detector.step(epoch_kl_dir)
            self._gaussian_collapse_detector.step(epoch_kl_gauss)

            if self._dirichlet_collapse_detector.is_collapsed() or self._gaussian_collapse_detector.is_collapsed():
                raise errors.PosteriorCollapseError

            total_metrics += torch.stack([epoch_loss, epoch_recon, epoch_kl_dir, epoch_kl_gauss])
            epochs_run += 1

            val_loss = self._run_val_epoch(
                dirichlet_annealer.weight,
                gaussian_annealer.weight,
                dirichlet_fb_annealer.value,
                gaussian_fb_annealer.value,
            )
            if util.is_dead(val_loss):
                raise errors.DeadLossError

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
        return tuple(total_metrics.tolist())
