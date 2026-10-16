import contextlib
import json
import logging
import os
import random
import statistics
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna_integration.pytorch_distributed import TorchDistributedTrial
from rich.logging import RichHandler
from torch import distributed as dist
from torch.multiprocessing.spawn import spawn

from charl_tre import dataset, util
from charl_tre.hyperparams import HyperParams
from charl_tre.models import classifier, errors, hierarchical
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.DEBUG, show_path=False)
logging.basicConfig(level=logging.DEBUG, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# Suppress Optuna warnings
warnings.filterwarnings("ignore", module="optuna_integration.pytorch_distributed")
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Enable cuDNN auto-tuner for better performancwe
torch.backends.cudnn.benchmark = True


def _generate_seeds(starter_seed: int, n_seeds: int) -> list[int]:
    """Generate a fixed list of seeds from a starter seed.

    Arguments:
        starter_seed: The seed to use for the random number generator.
        n_seeds: The number of seeds to generate.

    Returns:
        A list of n_seeds integers to use as seeds for the trials.

    """
    rng = random.SystemRandom(starter_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_seeds)]


def _train_and_evaluate_seed(
    seed: int,
    rank: int,
    trial: TorchDistributedTrial,
    lr: float,
    kl_dirichlet_final: float,
    kl_gaussian_final: float,
    n_components: int,
    batch_size: int,
    n_fusion_layers: int,
    fusion_dropout: float,
    n_mixtures: int,
    vae_train_ds: dataset.MultiAntenna,
    vae_val_ds: dataset.MultiAntenna,
    classifier_train_ds: dataset.MultiAntenna,
    classifier_val_ds: dataset.MultiAntenna,
    save_path: Path,
) -> float:
    """Train and evaluate a single seed's VAE and classifier, returning the classifier accuracy."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    h_vae = hierarchical.Autoencoder(
        window_size=settings.vae_window_size,
        n_subcarriers=settings.n_subcarriers,
        n_dirichlet_components=n_components,
        n_mixtures=n_mixtures,
        n_fusion_layers=n_fusion_layers,
        fusion_dropout=fusion_dropout,
        n_antennas=settings.n_antennas,
    )

    # Create data loaders for VAE training
    vae_train_dl = util.make_dl(
        vae_train_ds,
        batch_size,
        shuffle=True,
        n_workers=settings.n_workers,
        seed=seed,
    )
    vae_val_dl = util.make_dl(
        vae_val_ds,
        batch_size,
        shuffle=False,
        n_workers=settings.n_workers,
        seed=seed,
    )

    # Train the early fusion VAE
    h_vae_trainer = hierarchical.Trainer(
        h_vae,
        vae_train_dl,
        vae_val_dl,
        hierarchical.TrainerParams(
            lr=lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            collapse_patience=settings.collapse_patience,
            kl_dirichlet_final=kl_dirichlet_final,
            kl_gaussian_final=kl_gaussian_final,
            free_bits_start=settings.free_bits_start,
            free_bits_end=settings.free_bits_end,
        ),
        rank,
        trial,
    )
    try:
        h_vae_trainer.train(settings.n_epochs)
    except (errors.PosteriorCollapseError, errors.DeadLossError) as e:
        raise optuna.TrialPruned(str(e)) from e

    # Create data loaders for classifier training
    classifier_train_dl = util.make_dl(
        classifier_train_ds,
        batch_size,
        shuffle=True,
        n_workers=settings.n_workers,
        seed=seed,
    )
    classifier_val_dl = util.make_dl(
        classifier_val_ds,
        batch_size,
        shuffle=False,
        n_workers=settings.n_workers,
        seed=seed,
    )

    classifier_model = classifier.Classifier(
        antennas=[h_vae],
        n_components=n_components * n_mixtures,
        n_activities=settings.n_activities,
        sample_window_size=settings.vae_window_size,
        overlap_size=settings.overlap_size,
        n_layers=1,
        dropout=0,
    )

    # Train the classifier on top of the frozen VAE
    classifier_trainer = classifier.Trainer(
        classifier_model,
        classifier_train_dl,
        classifier_val_dl,
        classifier.TrainerParams(
            lr=lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        ),
        rank,
    )
    try:
        classifier_loss, classifier_accuracy = classifier_trainer.train(settings.n_epochs)
    except errors.DeadLossError as e:
        raise optuna.TrialPruned(str(e)) from e

    if rank == 0:
        logger.debug(
            "Trial %s seed %d completed: loss=%.4f, accuracy=%.4f",
            trial.number,
            seed,
            classifier_loss,
            classifier_accuracy,
        )
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(h_vae.state_dict(), save_path / f"h_vae_{seed}.pt")
        torch.save(classifier_model.state_dict(), save_path / f"classifier_{seed}.pt")

    return classifier_accuracy


def _sync_seed_failure(rank: int, seed_failed: bool) -> bool:
    """Agree across all DDP ranks on whether the current seed failed.

    A rank whose seed raised an error must not move on to the next seed (and start
    building/broadcasting a new DDP model) while other ranks are still mid-training on the
    current one: the collective calls would then be issued out of step and abort the process
    group. All ranks therefore vote via all_reduce and then wait at a barrier before continuing.
    """
    if not dist.is_initialized():
        return seed_failed

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    failure_flag = torch.tensor(1.0 if seed_failed else 0.0, device=device)
    dist.all_reduce(failure_flag, op=dist.ReduceOp.MAX)
    dist.barrier()
    return bool(failure_flag.item())


def _run_seeds(
    seeds: list[int],
    rank: int,
    trial: TorchDistributedTrial,
    lr: float,
    kl_dirichlet_final: float,
    kl_gaussian_final: float,
    n_components: int,
    batch_size: int,
    n_fusion_layers: int,
    fusion_dropout: float,
    n_mixtures: int,
    vae_train_ds: dataset.MultiAntenna,
    vae_val_ds: dataset.MultiAntenna,
    classifier_train_ds: dataset.MultiAntenna,
    classifier_val_ds: dataset.MultiAntenna,
    save_path: Path,
) -> dict[int, float]:
    """Train and evaluate every seed, tolerating up to `settings.max_seed_failure_rate` failures.

    Raises:
        optuna.TrialPruned: If more than `settings.max_seed_failure_rate` of the seeds raise an error.

    """
    accuracies: dict[int, float] = {}
    n_failed = 0

    for seed in seeds:
        seed_failed = False
        accuracy = 0.0
        try:
            accuracy = _train_and_evaluate_seed(
                seed,
                rank,
                trial,
                lr,
                kl_dirichlet_final,
                kl_gaussian_final,
                n_components,
                batch_size,
                n_fusion_layers,
                fusion_dropout,
                n_mixtures,
                vae_train_ds,
                vae_val_ds,
                classifier_train_ds,
                classifier_val_ds,
                save_path,
            )
        except (optuna.TrialPruned, RuntimeError) as e:
            seed_failed = True
            if rank == 0:
                logger.debug("Trial %s seed %d failed: %s", trial.number, seed, e)

        seed_failed = _sync_seed_failure(rank, seed_failed)
        if seed_failed:
            n_failed += 1
            continue
        accuracies[seed] = accuracy

    if n_failed > settings.max_seeds_failed:
        msg = f"Trial {trial.number} failed: {n_failed} seeds raised errors."
        raise optuna.TrialPruned(msg)

    return accuracies


def _objective(
    single_trial: optuna.trial.BaseTrial | None,
    rank: int,
    vae_train_ds: dataset.MultiAntenna,
    vae_val_ds: dataset.MultiAntenna,
    classifier_train_ds: dataset.MultiAntenna,
    classifier_val_ds: dataset.MultiAntenna,
) -> float:
    """Objective function for hierarchical fusion hyperparameter optimization.

    Trains all 4 Dirichlet VAEs jointly with a shared latent space,
    then trains a classifier on top of the shared latent representation.
    This is repeated sequentially for `settings.n_seeds_per_trial` seeds, and
    the per-seed accuracies (and their 25th/75th percentiles) are recorded as
    trial user attributes.

    Arguments:
        single_trial: The Optuna trial object
        rank: Unique identifier of each process
        vae_train_ds: Training dataset for the VAE
        vae_val_ds: Validation dataset for the VAE
        classifier_train_ds: Training dataset for the classifier
        classifier_val_ds: Validation dataset for the classifier

    Returns:
        The median classifier accuracy across seeds, penalized by model size

    """
    trial = TorchDistributedTrial(single_trial)
    hyperparams = HyperParams.from_settings(settings)
    lr = trial.suggest_float(
        "lr",
        **hyperparams.lr.to_dict(),
    )
    kl_dirichlet_final = trial.suggest_float(
        "kl_final",
        **hyperparams.kl_final.to_dict(),
    )
    kl_gaussian_final = trial.suggest_float(
        "kl_gaussian_final",
        **hyperparams.kl_final.to_dict(),
    )
    n_components = trial.suggest_int(
        "n_components",
        **hyperparams.n_components.to_dict(),
    )
    batch_size = trial.suggest_int(
        "batch_size",
        **hyperparams.batch_size.to_dict(),
    )
    n_fusion_layers = trial.suggest_int(
        "n_fusion_layers",
        **hyperparams.n_fusion_layers.to_dict(),
    )
    fusion_dropout = trial.suggest_float(
        "fusion_dropout",
        **hyperparams.fusion_dropout.to_dict(),
    )
    n_mixtures = trial.suggest_int(
        "n_mixtures",
        **hyperparams.n_mixtures.to_dict(),
    )

    if rank == 0:
        logger.info("Starting trial %s with parameters: %s", trial.number, trial.params)

    seeds = _generate_seeds(settings.starter_seed, settings.n_seeds_per_trial)
    save_path = Path(settings.study_path) / f"trial_{trial.number}"
    accuracies = _run_seeds(
        seeds,
        rank,
        trial,
        lr,
        kl_dirichlet_final,
        kl_gaussian_final,
        n_components,
        batch_size,
        n_fusion_layers,
        fusion_dropout,
        n_mixtures,
        vae_train_ds,
        vae_val_ds,
        classifier_train_ds,
        classifier_val_ds,
        save_path,
    )

    accuracy_values = list(accuracies.values())
    median_accuracy = statistics.median(accuracy_values)
    accuracy_p25 = float(np.percentile(accuracy_values, 25))
    accuracy_p75 = float(np.percentile(accuracy_values, 75))
    objective_score = median_accuracy - (settings.n_components_penalty_weight * n_components * n_mixtures)

    trial.set_user_attr("accuracies", accuracies)
    trial.set_user_attr("accuracy_p25", accuracy_p25)
    trial.set_user_attr("accuracy_p75", accuracy_p75)
    trial.set_user_attr("median_accuracy", median_accuracy)

    if rank == 0:
        logger.info(
            "Trial %s completed: median_accuracy=%.4f (p25=%.4f, p75=%.4f), objective=%.4f",
            trial.number,
            median_accuracy,
            accuracy_p25,
            accuracy_p75,
            objective_score,
        )

        results = {
            "trial_number": trial.number,
            "accuracies": accuracies,
            "median_accuracy": median_accuracy,
            "accuracy_p25": accuracy_p25,
            "accuracy_p75": accuracy_p75,
            "objective_score": objective_score,
            "params": trial.params,
        }
        with Path.open(save_path / "results.json", "w") as f:
            json.dump(results, f)

    return objective_score


def _run_optimize(
    rank: int,
    world_size: int,
    vae_train_ds: dataset.MultiAntenna,
    vae_val_ds: dataset.MultiAntenna,
    classifier_train_ds: dataset.MultiAntenna,
    classifier_val_ds: dataset.MultiAntenna,
) -> None:
    """Run the Optuna optimization process in a distributed manner."""
    # Initialize DDP, standard boilerplate stuff
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        msg = "No accelerator found for DDP setup."
        raise RuntimeError(msg)
    backend = torch.distributed.get_default_backend_for_device(accelerator)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=rank)

    study = None
    try:
        if rank == 0:
            study = util.make_study(settings.study_name, settings.study_path, settings.starter_seed)
            study.optimize(
                lambda trial: _objective(
                    trial,
                    rank,
                    vae_train_ds,
                    vae_val_ds,
                    classifier_train_ds,
                    classifier_val_ds,
                ),
                n_trials=settings.n_trials,
                gc_after_trial=True,
            )
        else:
            for _ in range(settings.n_trials):
                with contextlib.suppress(optuna.TrialPruned):
                    _objective(None, rank, vae_train_ds, vae_val_ds, classifier_train_ds, classifier_val_ds)

        # Wait for all ranks to finish before proceeding to avoid race conditions
        torch.distributed.barrier()

        if rank == 0 and study is not None:
            pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

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
                "completed_trials": len(complete_trials),
            }
            with Path.open(Path(settings.study_path) / "study_results.json", "w") as f:
                json.dump(results, f)

    finally:
        torch.distributed.destroy_process_group()


def opt() -> None:
    """Run optuna optimization and evaluation for the CSI VAE model."""
    logger.info("Loading scenario dataset from %s...", settings.dataset_path)

    # Load the datasets for VAE and classifier training and validation once at the start.
    # We will build data loaders for each seed later, but the underlying datasets are the same.
    vae_train_ds, vae_val_ds, _ = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.vae_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    classifier_train_ds, classifier_val_ds, _ = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.classifier_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    logger.info("Scenario dataset loaded.")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info("Using %d GPUs for distributed training.", world_size)

    # Run Optuna optimization
    spawn(
        _run_optimize,
        args=(world_size, vae_train_ds, vae_val_ds, classifier_train_ds, classifier_val_ds),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    opt()
