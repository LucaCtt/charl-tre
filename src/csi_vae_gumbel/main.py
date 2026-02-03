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
from csi_vae_gumbel.models import CategoricalVAE, Classifier
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train import ClassifierTrainer, KLCollapsePruner, VAEParameters, VAETrainer

settings = Settings()

handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
logger = logging.getLogger("rich")
# Set log level to INFO for external libraries, DEBUG for local if in debug mode
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

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

    def log_progress(msg: str, *args: float | str) -> None:
        """Log message only from the main process."""
        if rank == 0:
            logger.debug(msg, *args)

    trial = optuna.integration.TorchDistributedTrial(single_trial)
    log_progress("Starting trial %s", trial.number)

    train_dataloader, val_dataloader = get_splits(
        dataset_path=Path(settings.dataset_path),
        batch_size=settings.batch_size // world_size,
        window_size=settings.window_size,
        overlap_size=settings.overlap_size,
        n_activities=settings.n_activities,
        n_samples=settings.n_samples,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
    )
    log_progress("Data loaders created for trial %s", trial.number)

    parameters = VAEParameters(
        start_lr=trial.suggest_float(
            "lr",
            settings.start_lr_min,
            settings.start_lr_max,
            log=True,
        ),
        final_kl_weight=trial.suggest_float(
            "kl_weight",
            settings.final_kl_weight_min,
            settings.final_kl_weight_max,
            log=True,
        ),
        final_entr_weight=trial.suggest_float(
            "entr_weight",
            settings.final_entr_weight_min,
            settings.final_entr_weight_max,
            log=True,
        ),
        n_cats=trial.suggest_int(
            "n_cats",
            settings.n_cats_min,
            settings.n_cats_max,
            step=1,
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
        loss_type=trial.suggest_categorical(
            "loss_type",
            ["bce", "mse"],
        ),  # pyright: ignore[reportArgumentType]
    )

    # Build and train VAE
    vae = CategoricalVAE(
        window_size=settings.window_size,
        n_categories=parameters.n_cats,
        latent_dim=parameters.latent_dim,
    )
    vae_trainer = VAETrainer(
        model=vae,
        dataloader=train_dataloader,
        parameters=parameters,
        gpu_id=rank,
        trial=trial,
    )
    loss, recon_loss, kl_loss, entropy_loss = vae_trainer.train(settings.n_epochs)
    log_progress(
        "VAE training completed for trial %s with loss %.4f, recon_loss %.4f, kl_loss %.4f, entropy_loss %.4f",
        trial.number,
        loss,
        recon_loss,
        kl_loss,
        entropy_loss,
    )

    # Build and train classifier on frozen VAE latent space
    classifier = Classifier(
        input_dim=parameters.n_cats * parameters.latent_dim,
        output_dim=settings.n_activities,
        hidden_dim=parameters.n_cats * settings.n_activities // 2,
    )
    classifier_trainer = ClassifierTrainer(
        model=classifier,
        dataloader=train_dataloader,
        vae=vae,
        optimizer=torch.optim.Adam(classifier.parameters()),
        gpu_id=rank,
    )
    loss, accuracy = classifier_trainer.train(settings.n_epochs)
    log_progress(
        "Classifier training completed for trial %s with loss %.4f, accuracy %.4f",
        trial.number,
        loss,
        accuracy,
    )

    trial_dir = Path(settings.study_dir) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on validation set
    evaluator = Evaluator(
        vae=vae,
        classifier=classifier,
        dataloader=val_dataloader,
        classes=settings.activities_labels,
        gpu_id=rank,
        out_dir=trial_dir,
    )
    accuracy = evaluator.evaluate()
    log_progress(
        "Trial %s completed with accuracy %.4f",
        trial.number,
        accuracy,
    )

    return accuracy


def _run_optimize(rank: int, world_size: int, return_dict: dict) -> None:
    """Run the Optuna hyperparameter optimization in a distributed manner."""
    _ddp_setup(rank, world_size)

    try:
        if rank == 0:
            study = optuna.create_study(
                direction="maximize",  # Optimize for classification accuracy
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
    # Create study directory
    Path(settings.study_dir).mkdir(parents=True, exist_ok=True)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Multiprocessing manager to collect results from different processes
    return_dict = mp.Manager().dict()
    spawn(_run_optimize, args=(world_size, return_dict), nprocs=world_size, join=True)

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
