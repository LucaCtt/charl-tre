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
from optuna_integration.pytorch_distributed import TorchDistributedTrial
from rich.logging import RichHandler
from torch import distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from charl_tre import dataset, util
from charl_tre.models import fusion, vae
from charl_tre.models.vae.dirichlet import CONV_SPECS
from charl_tre.settings import ParamRange, Settings

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


def _init_rng(seed: int) -> None:
    """Initialize random seeds for reproducibility.

    Arguments:
        seed (int): The random seed to use for all random number generators.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_to_props(param: ParamRange) -> dict:
    """Convert a ParamRange to a dictionary of properties for Optuna suggest methods."""
    props = {"low": param.min, "high": param.max}
    if param.step == "log":
        props["log"] = True
    elif param.step is not None:
        props["step"] = param.step
    return props


def _make_dl(ds: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Create a DataLoader with common settings.

    Arguments:
        ds: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.
        seed: Random seed for reproducibility of shuffling.

    Returns:
        A DataLoader instance for the given dataset and settings.

    """
    # In spawned DDP+Optuna runs, worker subprocess shutdown can race with trial pruning
    # and emit QueueFeederThread bad-file-descriptor errors. Keep loading in-process there.
    num_workers = 0 if dist.is_initialized() else settings.num_workers

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if not dist.is_initialized() else False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        sampler=DistributedSampler(ds, shuffle=shuffle, seed=seed) if dist.is_initialized() else None,
    )


def _objective(
    single_trial: BaseTrial | None,
    rank: int,
    train_ds: dataset.MultiAntenna,
    val_ds: dataset.MultiAntenna,
) -> float:
    """Objective function for Optuna hyperparameter optimization.

    Arguments:
        single_trial: The Optuna trial object
        rank: Unique identifier of each process
        train_ds: MultiAntenna training dataset
        val_ds: MultiAntenna validation dataset

    Returns:
        The fusion model reconstruction loss after training

    """
    trial = TorchDistributedTrial(single_trial)

    lr = trial.suggest_float(
        "lr",
        **_param_to_props(settings.lr),
    )

    vae_params = vae.TrainerParams(
        early_stop_patience=settings.early_stop_patience,
        early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        collapse_patience=settings.collapse_patience,
        lr=lr,
        kl_max=trial.suggest_float(
            "kl_max",
            **_param_to_props(settings.kl_max),
        ),
        prior_alpha=trial.suggest_float(
            "prior_alpha",
            **_param_to_props(settings.prior_alpha),
        ),
        free_bits_start=settings.free_bits_start,
        free_bits_end=settings.free_bits_end,
    )

    n_components = trial.suggest_int(
        "n_components",
        **_param_to_props(settings.n_components),
    )
    conv_layers_spec = trial.suggest_categorical(
        "conv_layers_spec",
        settings.conv_layers_spec.values,
    )
    batch_size = trial.suggest_int(
        "batch_size",
        **_param_to_props(settings.batch_size),
    )
    n_fusion_layers = trial.suggest_int(
        "n_fusion_layers",
        **_param_to_props(settings.n_fusion_layers),
    )
    fusion_dropout = trial.suggest_float(
        "fusion_dropout",
        **_param_to_props(settings.fusion_dropout),
    )

    if rank == 0:
        logger.info("Starting trial %s with parameters: %s", trial.number, trial.params)

    # Train one VAE per antenna
    vaes = []
    for antenna_idx in range(settings.n_antennas):
        antenna_train_ds = dataset.SingleAntenna(train_ds, antenna_idx)
        antenna_val_ds = dataset.SingleAntenna(val_ds, antenna_idx)
        antenna_train_dl = _make_dl(
            antenna_train_ds,
            batch_size,
            shuffle=True,
            seed=settings.seed,
        )
        antenna_val_dl = _make_dl(
            antenna_val_ds,
            batch_size,
            shuffle=False,
            seed=settings.seed,
        )

        vae_model = vae.SingleAntenna(
            settings.train_window_size,
            settings.n_subcarriers,
            n_components,
            CONV_SPECS[conv_layers_spec],
        )

        vae_trainer = vae.Trainer(
            vae_model,
            antenna_train_dl,
            antenna_val_dl,
            vae_params,
            rank,
            trial,
        )
        try:
            vae_trainer.train(settings.n_epochs)
        except (vae.PosteriorCollapseError, vae.DeadLossError) as e:
            raise optuna.TrialPruned(str(e)) from e

        vaes.append(vae_model)

    train_ds, val_ds, _ = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.test_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )

    # Train fusion model on pre-trained VAEs
    fusion_train_dl = _make_dl(train_ds, batch_size, shuffle=True, seed=settings.seed)
    fusion_val_dl = _make_dl(val_ds, batch_size, shuffle=False, seed=settings.seed)

    delayed_fusion = fusion.Delayed(
        vaes,
        n_components,
        settings.n_activities,
        n_fusion_layers,
        fusion_dropout,
    )

    fusion_trainer = fusion.Trainer(
        delayed_fusion,
        fusion_train_dl,
        fusion_val_dl,
        fusion.TrainerParams(
            lr=lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            sample_window_size=settings.train_window_size,
            overlap_size=settings.overlap_size,
        ),
        rank,
    )
    fusion_loss, fusion_accuracy = fusion_trainer.train(settings.n_epochs)

    if rank == 0:
        logger.info(
            "Trial %s completed: train loss=%.4f, train accuracy=%.4f",
            trial.number,
            fusion_loss,
            fusion_accuracy,
        )
        save_path = Path(settings.study_path) / f"trial_{trial.number}"
        save_path.mkdir(parents=True, exist_ok=True)
        for i, vae_m in enumerate(vaes):
            torch.save(vae_m.state_dict(), save_path / f"vae_{i}.pt")
        torch.save(delayed_fusion.state_dict(), save_path / "fusion.pt")

        results = {
            "trial_number": trial.number,
            "n_components": n_components,
            "conv_layers_spec": conv_layers_spec,
            "fusion_loss": fusion_loss,
            "fusion_accuracy": fusion_accuracy,
        }
        with Path.open(save_path / "results.json", "w") as f:
            json.dump(results, f)

    return fusion_loss


def _run_optimize(rank: int, world_size: int, train_ds: dataset.MultiAntenna, val_ds: dataset.MultiAntenna) -> None:
    util.setup_ddp(rank, world_size)

    study = None
    if rank == 0:
        study = optuna.create_study(
            direction="minimize",
            study_name=settings.study_name,
            sampler=optuna.samplers.TPESampler(seed=settings.seed),
        )
        study.optimize(
            partial(_objective, rank=rank, train_ds=train_ds, val_ds=val_ds),
            n_trials=settings.n_trials,
            gc_after_trial=True,
        )
    else:
        for _ in range(settings.n_trials):
            with contextlib.suppress(optuna.TrialPruned):
                _objective(None, rank, train_ds, val_ds)

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


def opt() -> None:
    """Run optuna optimization and evaluation for the CSI VAE model."""
    _init_rng(settings.seed)

    # Create datasets once and share across VAE training processes
    logger.info("Loading dataset from %s...", settings.dataset_path)
    train_ds, val_ds, _ = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.train_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    logger.info("Dataset loaded with %d training samples", len(train_ds))

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Run Optuna optimization
    spawn(_run_optimize, args=(world_size, train_ds, val_ds), nprocs=world_size, join=True)


if __name__ == "__main__":
    opt()
