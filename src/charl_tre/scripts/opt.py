import contextlib
import json
import logging
import warnings
from pathlib import Path

import optuna
import torch
from optuna_integration.pytorch_distributed import TorchDistributedTrial
from rich.logging import RichHandler
from torch.multiprocessing.spawn import spawn

from charl_tre import dataset, util
from charl_tre.hyperparams import HyperParams
from charl_tre.models import classifier, errors, hierarchical
from charl_tre.settings import Settings
from charl_tre.studies import make_study

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# Suppress Optuna warnings
warnings.filterwarnings("ignore", module="optuna_integration.pytorch_distributed")
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat-32 for faster matrix multiplications
torch.backends.cudnn.allow_tf32 = True  # Allow TensorFloat-32 for faster convolutions
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance


def _objective(
    single_trial: optuna.trial.BaseTrial | None,
    rank: int,
    vae_train_ds: dataset.MultiAntenna,
    vae_val_ds: dataset.MultiAntenna,
    fusion_train_ds: dataset.MultiAntenna,
    fusion_val_ds: dataset.MultiAntenna,
) -> float:
    """Objective function for hierarchical fusion hyperparameter optimization.

    Trains all 4 Dirichlet VAEs jointly with a shared latent space,
    then trains a classifier on top of the shared latent representation.

    Arguments:
        single_trial: The Optuna trial object
        rank: Unique identifier of each process
        vae_train_ds: Training dataset for the VAE
        vae_val_ds: Validation dataset for the VAE
        fusion_train_ds: Training dataset for the fusion classifier
        fusion_val_ds: Validation dataset for the fusion classifier

    Returns:
        The early fusion classifier accuracy after training

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
        low=1,
        high=8,
        step=1,
    )

    if rank == 0:
        logger.info("Starting trial %s with parameters: %s", trial.number, trial.params)

    early_fusion = hierarchical.Fusion(
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
        num_workers=settings.num_workers,
        seed=settings.seed,
    )
    vae_val_dl = util.make_dl(
        vae_val_ds,
        batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )

    # Create early fusion VAE trainer parameters
    fusion_params = hierarchical.TrainerParams(
        lr=lr,
        early_stop_patience=settings.early_stop_patience,
        early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        collapse_patience=settings.collapse_patience,
        kl_dirichlet_final=kl_dirichlet_final,
        kl_gaussian_final=kl_gaussian_final,
        free_bits_start=settings.free_bits_start,
        free_bits_end=settings.free_bits_end,
    )

    # Train the early fusion VAE
    fusion_trainer = hierarchical.Trainer(
        early_fusion,
        vae_train_dl,
        vae_val_dl,
        fusion_params,
        rank,
        trial,
    )
    try:
        fusion_trainer.train(settings.n_epochs)
    except (errors.PosteriorCollapseError, errors.DeadLossError) as e:
        raise optuna.TrialPruned(str(e)) from e

    # Create data loaders for classifier training
    fusion_train_dl = util.make_dl(
        fusion_train_ds,
        batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )
    fusion_val_dl = util.make_dl(
        fusion_val_ds,
        batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )

    classifier_model = classifier.Classifier(
        antennas=[early_fusion],
        n_components=n_components * n_mixtures,
        n_activities=settings.n_activities,
        sample_window_size=settings.vae_window_size,
        overlap_size=settings.overlap_size,
        n_layers=n_fusion_layers,
        dropout=fusion_dropout,
    )

    # Train the classifier on top of the frozen VAE
    classifier_trainer = classifier.Trainer(
        classifier_model,
        fusion_train_dl,
        fusion_val_dl,
        classifier.TrainerParams(
            lr=lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        ),
        rank,
    )
    classifier_loss, classifier_accuracy = classifier_trainer.train(settings.n_epochs)
    objective_score = classifier_accuracy - (settings.n_components_penalty_weight * n_components)

    if rank == 0:
        logger.info(
            "Early fusion trial %s completed: loss=%.4f, accuracy=%.4f, objective=%.4f",
            trial.number,
            classifier_loss,
            classifier_accuracy,
            objective_score,
        )
        save_path = Path(settings.study_path) / f"early_trial_{trial.number}"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(early_fusion.state_dict(), save_path / "early_fusion.pt")

        results = {
            "trial_number": trial.number,
            "classifier_loss": classifier_loss,
            "classifier_accuracy": classifier_accuracy,
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
    fusion_train_ds: dataset.MultiAntenna,
    fusion_val_ds: dataset.MultiAntenna,
) -> None:
    util.setup_ddp(rank, world_size)

    study = None
    if rank == 0:
        study = make_study(settings.study_name, settings.study_path, settings.seed)
        study.optimize(
            lambda trial: _objective(
                trial,
                rank,
                vae_train_ds,
                vae_val_ds,
                fusion_train_ds,
                fusion_val_ds,
            ),
            n_trials=settings.n_trials,
            gc_after_trial=True,
        )
    else:
        for _ in range(settings.n_trials):
            with contextlib.suppress(optuna.TrialPruned):
                _objective(None, rank, vae_train_ds, vae_val_ds, fusion_train_ds, fusion_val_ds)

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

    torch.distributed.destroy_process_group()


def opt() -> None:
    """Run optuna optimization and evaluation for the CSI VAE model."""
    util.init_rng(settings.seed)

    # Create datasets once and share across VAE training processes
    logger.info("Loading scenario from %s...", settings.dataset_path)
    vae_train_ds, vae_val_ds, _ = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.vae_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    fusion_train_ds, fusion_val_ds, _ = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.fusion_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    logger.info("Scenario loaded.")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Run Optuna optimization
    spawn(
        _run_optimize,
        args=(world_size, vae_train_ds, vae_val_ds, fusion_train_ds, fusion_val_ds),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    opt()
