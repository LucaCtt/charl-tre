import logging
import os
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import optuna
import torch
from optuna.trial import BaseTrial, TrialState
from rich.logging import RichHandler
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.multiprocessing.spawn import spawn

from csi_vae_gumbel.dataset import get_splits
from csi_vae_gumbel.loss import CapacityAnnealer, EntropyAnnealer, GumbelTemperatureAnnealer, KLWeightAnnealer
from csi_vae_gumbel.models import CategoricalVAE
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train import VAETrainer

settings = Settings()

level = logging.DEBUG if settings.debug else logging.INFO
handler = RichHandler(level=level, show_path=False)
logging.basicConfig(level=level, handlers=[handler])
logger = logging.getLogger("rich")

warnings.filterwarnings("ignore", module="optuna_integration.pytorch_distributed")


@dataclass
class TrialParameters:
    """Parameters for a single Optuna trial."""

    learning_rate: float
    kl_weight: float
    entropy_weight: float
    n_categories: int
    latent_dim: int
    capacity: float
    gumbel_temp: float


def _ddp_setup(rank: int, world_size: int) -> None:
    """Initialize the distributed environment.

    Arguments:
        rank: Unique identifier of each process
        world_size: Total number of processes

    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    acc = torch.accelerator.current_accelerator()
    if acc is None:
        msg = "No accelerator found for DDP setup."
        raise RuntimeError(msg)

    backend = torch.distributed.get_default_backend_for_device(acc)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=rank)


def _objective(single_trial: BaseTrial | None, rank: int, world_size: int) -> float:
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    train_dataloader, _ = get_splits(
        dataset_path=Path(settings.dataset_path),
        batch_size=settings.batch_size // world_size,
        window_size=settings.window_size,
        overlap_size=settings.overlap_size,
        n_activities=settings.n_activities,
        n_samples=settings.n_samples,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
    )

    params = TrialParameters(
        learning_rate=trial.suggest_float(
            "learning_rate",
            settings.min_learning_rate,
            settings.max_learning_rate,
            log=True,
        ),
        kl_weight=trial.suggest_float(
            "kl_weight",
            settings.min_kl_weight,
            settings.max_kl_weight,
            log=True,
        ),
        entropy_weight=trial.suggest_float(
            "entropy_weight",
            settings.min_entropy_weight,
            settings.max_entropy_weight,
            log=True,
        ),
        n_categories=trial.suggest_int(
            "n_categories",
            settings.min_n_categories,
            settings.max_n_categories,
            step=1,
        ),
        latent_dim=trial.suggest_int(
            "latent_dim",
            settings.min_latent_dim,
            settings.max_latent_dim,
            step=1,
        ),
        capacity=trial.suggest_float(
            "final_capacity",
            settings.min_final_capacity,
            settings.max_final_capacity,
        ),
        gumbel_temp=trial.suggest_float(
            "gumbel_temp",
            settings.min_gumbel_temp,
            settings.max_gumbel_temp,
        ),
    )

    vae = CategoricalVAE(
        window_size=settings.window_size,
        n_categories=params.n_categories,
        latent_dim=params.latent_dim,
    )

    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=params.learning_rate,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
    )

    capacity_scheduler = CapacityAnnealer(
        max_capacity=params.capacity,
        ramp_epochs=settings.n_epochs // 3,
        start_epoch=settings.n_epochs // 10,
    )
    entropy_scheduler = EntropyAnnealer(
        max_weight=params.entropy_weight,
        n_epochs=settings.n_epochs,
        switch_epoch=settings.n_epochs // 2,
    )
    temperature_scheduler = GumbelTemperatureAnnealer(
        start_tau=params.gumbel_temp,
        min_tau=params.gumbel_temp / 10,
    )
    kl_weight_scheduler = KLWeightAnnealer(
        max_weight=params.kl_weight,
        start_epoch=settings.n_epochs // 10,
        ramp_epochs=settings.n_epochs // 3,
    )

    vae_trainer = VAETrainer(
        model=vae,
        dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        capacity_annealer=capacity_scheduler,
        entropy_annealer=entropy_scheduler,
        temperature_annealer=temperature_scheduler,
        kl_weight_annealer=kl_weight_scheduler,
        loss_type="bce",
        gpu_id=rank,
        trial=trial,
    )

    total_loss, _, _ = vae_trainer.train(settings.n_epochs)

    return total_loss


def _run_optimize(rank: int, world_size: int, return_dict: dict) -> None:
    _ddp_setup(rank, world_size)

    if rank == 0:
        study = optuna.create_study(direction="minimize")
        study.optimize(partial(_objective, rank=rank, world_size=world_size), n_trials=settings.n_trials)
        return_dict["study"] = study
    else:
        for _ in range(settings.n_trials):
            try:
                _objective(None, rank, world_size)
            except optuna.TrialPruned:
                logger.exception("Trial pruned.")

    dist.destroy_process_group()


def main() -> None:
    """Run the hyperparameter optimization using Optuna."""
    if settings.debug:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    return_dict = mp.Manager().dict()
    spawn(_run_optimize, args=(world_size, return_dict), nprocs=world_size, join=True)

    study = return_dict["study"]
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Number of finished trials: %d", len(study.trials))
    logger.info("Number of pruned trials: %d", len(pruned_trials))
    logger.info("Number of complete trials: %d", len(complete_trials))
    logger.info("Best trial params: %s", study.best_trial.params)


if __name__ == "__main__":
    main()
