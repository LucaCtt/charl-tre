import copy
import json
import logging
import os
import random
from pathlib import Path

import torch
from rich.logging import RichHandler
from torch import distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader, DistributedSampler

from charl_tre import dataset, util
from charl_tre.models import fusion, vae
from charl_tre.models.fusion.evaluator import Evaluator
from charl_tre.models.vae.dirichlet import CONV_SPECS
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Reproducible seeds
os.environ.setdefault("PYTHONHASHSEED", str(settings.seed))
random.seed(settings.seed)
torch.manual_seed(settings.seed)


def _run_test(
    rank: int,
    world_size: int,
    train_ds: dataset.MultiAntenna,
    val_ds: dataset.MultiAntenna,
    test_ds: dataset.MultiAntenna,
) -> None:
    util.setup_ddp(rank, world_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=settings.batch_size.min,
        sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        num_workers=settings.num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=settings.batch_size.min,
        sampler=DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        num_workers=settings.num_workers,
    )

    best_model_path = util.get_best_model_path(Path(settings.study_path))

    with (best_model_path / "results.json").open("r") as f:
        info = json.load(f)

    vae_model = vae.SingleAntenna(
        settings.train_window_size,
        settings.n_subcarriers,
        info["n_components"],
        CONV_SPECS[info["conv_layers_spec"]],
    )
    best_model_weights = torch.load(best_model_path / "model.pt", weights_only=True)
    vae_model.load_state_dict(best_model_weights)

    # One VAE per antenna; deepcopy to give each antenna its own parameter set
    vaes = [copy.deepcopy(vae_model) for _ in range(settings.n_antennas)]

    delayed_fusion = fusion.Delayed(
        vaes,
        info["n_components"],
        settings.n_activities,
        settings.n_fusion_layers.min,
        settings.fusion_dropout.min,
    )

    trainer = fusion.Trainer(
        delayed_fusion,
        train_dl,
        val_dl,
        fusion.TrainerParams(
            lr=settings.lr.min,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            sample_window_size=settings.train_window_size,
            overlap_size=settings.overlap_size,
        ),
        rank,
    )
    loss, accuracy = trainer.train(settings.n_epochs)

    if rank == 0:
        logger.info("Fusion training completed with loss %.4f and accuracy %.4f", loss, accuracy)
        torch.save(delayed_fusion.state_dict(), Path(settings.study_path) / "fusion.pt")

        logger.info("Evaluating on test set...")

        eval_dl = DataLoader(
            test_ds,
            batch_size=settings.batch_size.min,
            shuffle=False,
            num_workers=settings.num_workers,
            pin_memory=True,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        evaluator = Evaluator(model=delayed_fusion, dataloader=eval_dl, device=device)
        accuracy = evaluator.evaluate()

    dist.barrier()

    if rank == 0:
        logger.info("Evaluation completed with test accuracy %.4f", accuracy)

    dist.destroy_process_group()


def test() -> None:
    """Evaluate the best VAE model on the test set using a fusion classifier."""
    logger.info("Loading datasets from %s...", settings.dataset_path)
    train_ds, val_ds, test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.test_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    logger.info(
        "Datasets loaded with %d training, %d validation, and %d testing samples",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    logger.info("Training fusion classifier...")
    spawn(
        _run_test,
        args=(world_size, train_ds, val_ds, test_ds),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    test()
