import json
import logging
from pathlib import Path

import torch
from rich.logging import RichHandler
from torch import distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader

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

def _run_test(
    rank: int,
    world_size: int,
    test_ds: dataset.MultiAntenna,
) -> None:
    util.setup_ddp(rank, world_size)

    best_model_path = util.get_best_model_path(Path(settings.study_path))

    with (best_model_path / "results.json").open("r") as f:
        info = json.load(f)

    vaes: list[vae.SingleAntenna] = []
    for antenna_idx in range(settings.n_antennas):
        vae_model = vae.SingleAntenna(
            settings.vae_window_size,
            settings.n_subcarriers,
            info["n_components"],
            CONV_SPECS[info["conv_layers_spec"]],
        )

        antenna_model_path = best_model_path / f"vae_{antenna_idx}.pt"
        best_model_weights = torch.load(antenna_model_path, weights_only=True)
        vae_model.load_state_dict(best_model_weights)
        vaes.append(vae_model)

    delayed_fusion = fusion.Delayed(
        vaes,
        info["n_components"],
        settings.n_activities,
        settings.n_fusion_layers.min,
        settings.fusion_dropout.min,
    )

    fusion_weights = torch.load(best_model_path / "fusion.pt", weights_only=True)
    delayed_fusion.load_state_dict(fusion_weights)

    accuracy = 0.0
    if rank == 0:
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
    """Evaluate the best saved multi-antenna fusion model on the test set."""
    logger.info("Loading datasets from %s...", settings.dataset_path)
    _, _, test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.fusion_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    logger.info("Test dataset loaded with %d samples", len(test_ds))

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    logger.info("Evaluating best saved fusion model...")
    spawn(
        _run_test,
        args=(world_size, test_ds),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    test()
