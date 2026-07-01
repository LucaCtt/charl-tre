import json
import logging
from pathlib import Path

import numpy as np
import torch
from rich.logging import RichHandler

from charl_tre import dataset
from charl_tre.models.vae import SingleAntenna
from charl_tre.models.vae.dirichlet import CONV_SPECS
from charl_tre.settings import Settings
from charl_tre.util import split_test_window

settings = Settings()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def _get_best_model() -> SingleAntenna:
    """Load the best model and its parameters from the study results.

    Returns:
        The best VAE model, with the weights loaded.

    """
    with Path(f"../{settings.study_path}/study_results.json").open("r") as f:
        study_info = json.load(f)

    best_model_path = Path(f"../{settings.study_path}/trial_{study_info['best_trial']}")

    with Path(best_model_path / "results.json").open("r") as f:
        info = json.load(f)

    vae_model = SingleAntenna(
        settings.vae_window_size,
        settings.n_subcarriers,
        info["n_components"],
        CONV_SPECS[info["conv_layers_spec"]],
    )
    best_model_weights = torch.load(best_model_path / "model.pt", map_location=device, weights_only=True)
    vae_model.load_state_dict(best_model_weights)

    return vae_model


def _get_full_dataloader() -> torch.utils.data.DataLoader:
    """Load the full dataset (all splits) for latent extraction, without shuffling."""
    train_ds, val_ds, test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.fusion_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )

    antenna_train = dataset.SingleAntenna(train_ds, settings.antenna_select)
    antenna_val = dataset.SingleAntenna(val_ds, settings.antenna_select)
    antenna_test = dataset.SingleAntenna(test_ds, settings.antenna_select)

    full_ds = torch.utils.data.ConcatDataset([antenna_train, antenna_val, antenna_test])

    return torch.utils.data.DataLoader(
        full_ds,
        batch_size=settings.batch_size.min,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=True,
    )


def latents() -> None:
    """Extract and save the latent representations for the entire dataset using the best VAE model."""
    vae_model = _get_best_model().to(device)
    vae_model.eval()

    dataloader = _get_full_dataloader()

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            batch_size = x.shape[0]

            # x: (batch_size, test_window_size, n_subcarriers) from SingleAntenna dataset
            # split_test_window expects (batch_size, n_antennas, window_size, n_subcarriers)
            x_r = x.unsqueeze(1).to(device)
            x_r = split_test_window(x_r, settings.vae_window_size, 0)

            n_windows = x_r.shape[0] // batch_size

            # Repeat labels to match the expanded batch size (batch_size * n_windows,)
            y_r = y.repeat_interleave(n_windows, dim=0).to(device)

            # Squeeze antenna dim: (batch_size * n_windows, train_window_size, n_subcarriers)
            x_r = x_r.squeeze(1)

            _, alpha = vae_model(x_r)

            all_latents.append(alpha.cpu())
            all_labels.append(y_r.cpu())

    latents_tensor = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)

    output_path = Path(f"../{settings.study_path}/latents")
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "latents.npy", latents_tensor.numpy())
    np.save(output_path / "labels.npy", labels.numpy())

    logger.info("Saved %d latent vectors to %s", len(latents_tensor), output_path)


if __name__ == "__main__":
    latents()
