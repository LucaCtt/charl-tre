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
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader, DistributedSampler

from charl_tre import util
from charl_tre.collapse_pruner import CollapsePruner
from charl_tre.dataset import CSIDataset, load_datasets
from charl_tre.models import Evaluator, classifier, vae
from charl_tre.settings import Settings

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

# DDP debug settings
if settings.debug:
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
        settings.latent_dim,
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


def _run_optimize(rank: int, world_size: int, train_ds: CSIDataset) -> None:
    util.setup_ddp(rank, world_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
    )

    study = None
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
    else:
        for _ in range(settings.n_trials):
            with contextlib.suppress(optuna.TrialPruned):
                _objective(None, rank, train_dl)

    dist.barrier()

    if rank == 0 and study is not None:
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
            "best_trial": study.best_trial.number,
            "best_value": study.best_trial.value,
            "pruned_trials": len(pruned_trials),
            "complete_trials": len(complete_trials),
        }
        with Path.open(Path(settings.study_path) / "study_results.json", "w") as f:
            json.dump(results, f)

    dist.destroy_process_group()


def _run_test(
    rank: int,
    world_size: int,
    train_ds: CSIDataset,
    test_ds: CSIDataset,
) -> None:
    util.setup_ddp(rank, world_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        sampler=DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
    )

    best_model_path = util.get_best_model_path(Path(settings.study_path))
    best_params = vae.Parameters(**util.get_vae_params(best_model_path))

    vae_model = vae.SingleAntennaVAE(
        settings.train_window_size,
        settings.n_subcarriers,
        settings.n_categories,
        best_params.latent_dim,
    )
    best_model_weights = torch.load(best_model_path / "model.pt", weights_only=True)
    vae_model.load_state_dict(best_model_weights)

    classifier_model = classifier.BasicNNClassifier(
        best_params.latent_dim * settings.n_categories * settings.n_train_windows_in_test,
        settings.n_activities,
        int(1.5 * best_params.latent_dim * settings.n_categories * settings.n_train_windows_in_test),
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

        logger.info("Evaluating on test set...")

    dist.barrier()

    evaluator = Evaluator(
        vae=vae_model,
        classifier=classifier_model,
        dataloader=test_dl,
        sample_window_size=settings.train_window_size,
        overlap_size=settings.test_overlap_size,
        classes=settings.activities,
        gpu_id=rank,
        out_dir=Path(settings.study_path),
    )
    accuracy = evaluator.evaluate()

    dist.barrier()

    if rank == 0:
        logger.info("Evaluation completed with test accuracy %.4f", accuracy)

    dist.destroy_process_group()


def opt() -> None:
    """Run optuna optimization and evaluation for the CSI VAE model."""
    # Create datasets once and share across VAE training processes
    logger.info("Loading dataset from %s...", settings.dataset_path)
    train_ds, _ = load_datasets(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.train_window_size,
        test_ratio=settings.test_ratio,
        n_activities=settings.n_activities,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
        seed=settings.seed,
        stride=settings.stride,
    )
    logger.info("Dataset loaded with %d training samples", len(train_ds))

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Run Optuna optimization
    spawn(_run_optimize, args=(world_size, train_ds), nprocs=world_size, join=True)


def test() -> None:
    """Evaluate the best VAE model on the test set using a classifier."""
    logger.info("Loading datasets from %s...", settings.dataset_path)
    train_ds, test_ds = load_datasets(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.test_window_size,  # Window size here is (usually) different from VAE training
        test_ratio=settings.test_ratio,
        n_activities=settings.n_activities,
        n_antennas=settings.n_antennas,
        antenna_select=settings.antenna_select,
        seed=settings.seed,
        stride=settings.stride,
    )
    logger.info("Datasets loaded with %d training samples and %d testing samples", len(train_ds), len(test_ds))

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    logger.info("Training classifier...")
    spawn(
        _run_test,
        args=(world_size, train_ds, test_ds),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    opt()
    test()
