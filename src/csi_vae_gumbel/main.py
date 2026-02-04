import contextlib
import logging
import os
import random
import warnings
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
from csi_vae_gumbel.evaluator import Evaluator
from csi_vae_gumbel.models import CategoricalVAE
from csi_vae_gumbel.models.classifier import Classifier
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train import KLCollapsePruner, VAEParameters, VAETrainer
from csi_vae_gumbel.train.classifier_trainer import ClassifierTrainer

settings = Settings()

level = logging.DEBUG if settings.debug else logging.INFO
handler = RichHandler(level=level, show_path=False)
logging.basicConfig(level=level, handlers=[handler], format="%(message)s")
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
logger = logging.getLogger("rich")

# Suppress Optuna warnings
warnings.filterwarnings("ignore", module="optuna_integration.pytorch_distributed")
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Reproducible seeds
os.environ.setdefault("PYTHONHASHSEED", str(settings.seed))
random.seed(settings.seed)
torch.manual_seed(settings.seed)


def _ddp_setup(rank: int, world_size: int) -> None:
    """Initialize the distributed environment. Must be called by every process.

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
    """Objective function for Optuna hyperparameter optimization."""
    trial = optuna.integration.TorchDistributedTrial(single_trial)
    if rank == 0:
        logger.info("Starting trial %s", trial.number)

    train_dl, _ = get_splits(
        dataset_path=Path(settings.dataset_path),
        batch_size=settings.batch_size // world_size,
        window_size=settings.window_size,
        overlap_size=settings.overlap_size,
        n_activities=settings.n_activities,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
    )
    if rank == 0:
        logger.info("Data loaders created for trial %s", trial.number)

    parameters = VAEParameters(
        start_lr=trial.suggest_float(
            "start_lr",
            settings.start_lr_min,
            settings.start_lr_max,
            log=True,
        ),
        final_kl_weight=trial.suggest_float(
            "final_kl_weight",
            settings.final_kl_weight_min,
            settings.final_kl_weight_max,
            log=True,
        ),
        latent_dim=trial.suggest_int(
            "latent_dim",
            settings.latent_dim_min,
            settings.latent_dim_max,
            step=1,
        ),
        final_cap=trial.suggest_float(
            "final_cap",
            settings.final_cap_min,
            settings.final_cap_max,
        ),
        gumbel_temp=trial.suggest_float(
            "gumbel_temp",
            settings.gumbel_temp_min,
            settings.gumbel_temp_max,
        ),
    )

    # Build and train VAE
    vae = CategoricalVAE(
        window_size=settings.window_size,
        n_categories=settings.n_categories,
        latent_dim=parameters.latent_dim,
    )
    vae_trainer = VAETrainer(
        model=vae,
        dataloader=train_dl,
        parameters=parameters,
        gpu_id=rank,
        trial=trial,
    )
    loss, recon_loss, kl_loss = vae_trainer.train(settings.n_epochs)
    if rank == 0:
        logger.info(
            "VAE training completed for trial %s with loss %.4f, recon_loss %.4f, kl_loss %.4f",
            trial.number,
            loss,
            recon_loss,
            kl_loss,
        )
        save_path = Path(settings.study_dir) / f"trial_{trial.number}"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(vae.state_dict(), save_path / "model.pt")

    return recon_loss


def _eval_best_model(rank: int, world_size: int, return_dict: dict) -> None:
    """Evaluate the best model found by Optuna on the test set."""
    _ddp_setup(rank, world_size)

    best_trial = return_dict["study"].best_trial

    if rank == 0:
        logger.info("Evaluating best trial %d", best_trial.number)

    _, test_dl = get_splits(
        dataset_path=Path(settings.dataset_path),
        batch_size=settings.batch_size // world_size,
        window_size=settings.window_size * 3,
        overlap_size=settings.overlap_size,
        n_activities=settings.n_activities,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
    )

    params = VAEParameters(**best_trial.params)

    vae = CategoricalVAE(
        window_size=settings.window_size,
        n_categories=settings.n_categories,
        latent_dim=params.latent_dim,
    )
    load_path = Path(settings.study_dir) / f"trial_{best_trial.number}" / "model.pt"
    vae.load_state_dict(torch.load(load_path))
    if rank == 0:
        logger.info("Loaded best VAE model for evaluation.")

    classifier = Classifier(
        params.latent_dim * 3,
        settings.n_activities,
        params.latent_dim * settings.n_activities // 2,
    )
    classifier_trainer = ClassifierTrainer(
        model=classifier,
        dataloader=test_dl,
        vae=vae,
        gpu_id=rank,
    )
    classifier_trainer.train(settings.n_epochs)
    if rank == 0:
        logger.info("Classifier training completed for evaluation.")

    evaluator = Evaluator(vae, classifier, test_dl, settings.activities_labels, rank, Path(settings.study_dir))
    evaluator.evaluate()

    dist.destroy_process_group()


def _run_optimize(rank: int, world_size: int, return_dict: dict) -> None:
    """Run the Optuna hyperparameter optimization in a distributed manner."""
    _ddp_setup(rank, world_size)

    try:
        if rank == 0:
            study = optuna.create_study(
                direction="minimize",
                study_name=settings.study_name,
                sampler=optuna.samplers.TPESampler(seed=settings.seed),
                pruner=KLCollapsePruner(),
            )
            study.optimize(partial(_objective, rank=rank, world_size=world_size), n_trials=settings.n_trials)
            return_dict["study"] = study
        else:
            for _ in range(settings.n_trials):
                with contextlib.suppress(optuna.TrialPruned):
                    _objective(None, rank, world_size)

    finally:
        dist.destroy_process_group()


def main() -> None:
    """Run the hyperparameter optimization using Optuna."""
    if settings.debug:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Multiprocessing manager to collect results from different processes
    return_dict = mp.Manager().dict()

    # Run Optuna optimization
    spawn(_run_optimize, args=(world_size, return_dict), nprocs=world_size, join=True)

    # Evaluate the best model
    spawn(_eval_best_model, args=(world_size, return_dict), nprocs=world_size, join=True)

    study = return_dict["study"]
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info(
        "Study done with %d trials: %d pruned, %d complete.",
        len(study.trials),
        len(pruned_trials),
        len(complete_trials),
    )
    logger.info(
        "Best trial #%d: value=%.4f, params=%s",
        study.best_trial.number,
        study.best_trial.value,
        study.best_trial.params,
    )


if __name__ == "__main__":
    main()
