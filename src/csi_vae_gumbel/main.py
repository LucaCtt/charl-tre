import contextlib
import json
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
from torch.utils.data import DataLoader, DistributedSampler

from csi_vae_gumbel.collapse_pruner import CollapsePruner
from csi_vae_gumbel.dataset import CSIDataset, load_datasets
from csi_vae_gumbel.evaluator import Evaluator
from csi_vae_gumbel.models import classifier, vae
from csi_vae_gumbel.settings import Settings

settings = Settings()

# Configure logging
level = logging.DEBUG if settings.debug else logging.INFO
handler = RichHandler(level=level, show_path=False)
logging.basicConfig(level=level, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# Suppress Optuna warnings
warnings.filterwarnings("ignore", module="optuna_integration.pytorch_distributed")
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Reproducible seeds
os.environ.setdefault("PYTHONHASHSEED", str(settings.seed))
random.seed(settings.seed)
torch.manual_seed(settings.seed)


def _ddp_setup(rank: int, world_size: int) -> None:
    """Initialize the distributed environment. Must be called by every distributed process.

    Arguments:
        rank: Unique identifier of each distributed process
        world_size: Total number of distributed processes

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


def _objective(single_trial: BaseTrial | None, rank: int, train_dl: DataLoader) -> float:
    """Objective function for Optuna hyperparameter optimization.

    Arguments:
        single_trial: The Optuna trial object
        rank: Unique identifier of each process
        train_dl: DataLoader for training data

    Returns:
        The reconstruction loss after training

    """
    trial = optuna.integration.TorchDistributedTrial(single_trial)
    if rank == 0:
        logger.info("Starting trial %s", trial.number)

    parameters = vae.Parameters(
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
            step=0.2,
        ),
        start_gumbel_temp=trial.suggest_float(
            "start_gumbel_temp",
            settings.start_gumbel_temp_min,
            settings.start_gumbel_temp_max,
        ),
    )

    # Build and train VAE
    vae_model = vae.SingleAntennaVAE(
        settings.train_window_size,
        settings.n_subcarriers,
        settings.n_categories,
        parameters.latent_dim,
    )
    vae_trainer = vae.Trainer(vae_model, train_dl, parameters, rank, trial)
    loss, recon_loss, kl_loss = vae_trainer.train(settings.vae_n_epochs)

    if rank == 0:
        logger.info(
            "VAE training completed for trial %s with loss %.4f, recon_loss %.4f, kl_loss %.4f",
            trial.number,
            loss,
            recon_loss,
            kl_loss,
        )
        save_path = Path(settings.study_path) / f"trial_{trial.number}"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(vae_model.state_dict(), save_path / "model.pt")

        results = {
            "trial_number": trial.number,
            "final_kl_weight": parameters.final_kl_weight,
            "latent_dim": parameters.latent_dim,
            "final_cap": parameters.final_cap,
            "start_gumbel_temp": parameters.start_gumbel_temp,
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
        with Path.open(save_path / "results.json", "w") as f:
            json.dump(results, f)

    return recon_loss


def _run_optimize(rank: int, world_size: int, shared_dict: dict, train_ds: CSIDataset) -> None:
    _ddp_setup(rank, world_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=settings.train_batch_size,
        sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
    )

    if rank == 0:
        study = optuna.create_study(
            direction="minimize",
            study_name=settings.study_name,
            sampler=optuna.samplers.TPESampler(seed=settings.seed),
            pruner=CollapsePruner(n_categories=settings.n_categories),
        )
        study.optimize(
            partial(_objective, rank=rank, train_dl=train_dl),
            n_trials=settings.n_trials,
            gc_after_trial=True,
        )
        shared_dict["study"] = study
    else:
        for _ in range(settings.n_trials):
            with contextlib.suppress(optuna.TrialPruned):
                _objective(None, rank, train_dl)

    dist.barrier()

    if rank == 0:
        study = shared_dict["study"]

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

        results = {
            "best_trial": [study.best_trial.number],
            "best_value": [study.best_trial.value],
            "pruned_trials": len(pruned_trials),
            "complete_trials": len(complete_trials),
        }
        with Path.open(Path(settings.study_path) / "study_results.json", "w") as f:
            json.dump(results, f)

    dist.destroy_process_group()


def opt() -> None:
    """Run optuna optimization and evaluation for the CSI VAE model."""
    if settings.debug:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Create datasets once and share across VAE training processes
    logger.info("Loading datasets from %s", settings.dataset_path)
    train_ds, test_ds = load_datasets(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.train_window_size,
        test_ratio=settings.test_ratio,
        n_activities=settings.n_activities,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
        seed=settings.seed,
    )
    logger.info("Datasets loaded with %d training samples and %d testing samples", len(train_ds), len(test_ds))

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Multiprocessing manager to share study across different processes.
    # Note: datasets are not shared via this dict to avoid serialization.
    shared_dict = mp.Manager().dict()

    # Run Optuna optimization
    spawn(_run_optimize, args=(world_size, shared_dict, train_ds), nprocs=world_size, join=True)


def _run_classifier_train(
    rank: int,
    world_size: int,
    vae_model: vae.SingleAntennaVAE,
    classifier_model: classifier.BasicNNClassifier,
    train_ds: CSIDataset,
) -> None:
    _ddp_setup(rank, world_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=settings.train_batch_size,
        sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
    )

    classifier_trainer = classifier.Trainer(
        model=classifier_model,
        dataloader=train_dl,
        vae=vae_model,
        sample_window_size=settings.train_window_size,
        overlap_size=settings.test_overlap_size,
        gpu_id=rank,
    )
    loss, accuracy = classifier_trainer.train(settings.classifier_n_epochs)

    if rank == 0:
        logger.info("Classifier training completed with loss %.4f and accuracy %.4f", loss, accuracy)
        torch.save(classifier_model.state_dict(), Path(settings.study_path) / "classifier.pt")

    dist.destroy_process_group()


def test() -> None:
    """Evaluate the best VAE model on the test set using a classifier."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    with Path(f"{settings.study_path}/study_results.json").open("r") as f:
        study_info = json.load(f)
        best_trial_number = study_info["best_trial"][0]

    best_model_path = Path(settings.study_path) / f"trial_{best_trial_number}"

    with Path(best_model_path / "results.json").open("r") as f:
        info = json.load(f)

    params = vae.Parameters(
        final_cap=info["final_cap"],
        start_gumbel_temp=info["start_gumbel_temp"],
        final_kl_weight=info["final_kl_weight"],
        latent_dim=info["latent_dim"],
    )

    logger.info("Loading best VAE model for evaluation...")
    vae_model = vae.SingleAntennaVAE(
        settings.train_window_size,
        settings.n_subcarriers,
        settings.n_categories,
        params.latent_dim,
    )
    best_model_weights = torch.load(best_model_path / "model.pt")
    vae_model.load_state_dict(best_model_weights)

    logger.info("Loading datasets...")
    train_ds, test_ds = load_datasets(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.test_window_size,
        test_ratio=settings.test_ratio,
        n_activities=settings.n_activities,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
        seed=settings.seed,
    )

    classifier_model = classifier.BasicNNClassifier(
        params.latent_dim * settings.n_categories * settings.n_train_windows_in_test,
        settings.n_activities,
        int(1.5 * params.latent_dim * settings.n_categories * settings.n_train_windows_in_test),
    )

    if not Path(settings.study_path).joinpath("classifier.pt").exists():
        logger.info("Training classifier...")
        spawn(
            _run_classifier_train,
            args=(world_size, vae_model, classifier_model, train_ds),
            nprocs=world_size,
            join=True,
        )

    classifier_weights = torch.load(Path(settings.study_path) / "classifier.pt")
    classifier_model.load_state_dict(classifier_weights)

    logger.info("Evaluating on test set...")
    test_dl = DataLoader(test_ds, batch_size=len(test_ds) // (world_size * 12), shuffle=False, pin_memory=True)
    evaluator = Evaluator(
        vae=vae_model,
        classifier=classifier_model,
        dataloader=test_dl,
        sample_window_size=settings.train_window_size,
        overlap_size=settings.test_overlap_size,
        classes=settings.activities,
        gpu_id=0,
        out_dir=Path(settings.study_path),
    )
    accuracy = evaluator.evaluate()
    logger.info("Evaluation completed with test accuracy %.4f", accuracy)


if __name__ == "__main__":
    opt()
    test()
