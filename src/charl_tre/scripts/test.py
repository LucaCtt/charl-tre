import logging
import os
import random
from pathlib import Path

import torch
from rich.logging import RichHandler
from torch import distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader, DistributedSampler

from charl_tre import util
from charl_tre.dataset import CSIDataset, load_datasets
from charl_tre.models import Evaluator, fusion, vae
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
level = logging.DEBUG if settings.debug else logging.INFO
handler = RichHandler(level=level, show_path=False)
logging.basicConfig(level=level, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Reproducible seeds
os.environ.setdefault("PYTHONHASHSEED", str(settings.seed))
random.seed(settings.seed)
torch.manual_seed(settings.seed)

# DDP debug settings
if settings.debug:
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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

    classifier_model = fusion.BasicNNClassifier(
        best_params.latent_dim * settings.n_categories * settings.n_train_windows_in_test,
        settings.n_activities,
        int(1.5 * best_params.latent_dim * settings.n_categories * settings.n_train_windows_in_test),
    )
    classifier_trainer = fusion.Trainer(
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
    test()
