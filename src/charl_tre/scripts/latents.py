import json
import logging
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import scipy.io as sio
import torch
from rich.logging import RichHandler

from charl_tre.dataset import CSIDataset
from charl_tre.models.vae import Parameters, SingleAntennaVAE
from charl_tre.settings import Settings
from charl_tre.util import split_test_window

settings = Settings()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
level = logging.DEBUG if settings.debug else logging.INFO
handler = RichHandler(level=level, show_path=False)
logging.basicConfig(level=level, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def _get_best_model() -> SingleAntennaVAE:
    """Load the best model and its parameters from the study results.

    Returns:
        The best VAE model, with the weights loaded.

    """
    with Path(f"../{settings.study_path}/study_results.json").open("r") as f:
        study_info = json.load(f)

    best_model_path = Path(f"../{settings.study_path}/trial_{study_info['best_trial']}")

    with Path(best_model_path / "results.json").open("r") as f:
        info = json.load(f)

    params = Parameters(
        final_cap=info["final_cap"],
        start_gumbel_temp=info["start_gumbel_temp"],
        final_kl_weight=info["final_kl_weight"],
        latent_dim=info["latent_dim"],
    )

    vae_model = SingleAntennaVAE(
        settings.train_window_size,
        settings.n_subcarriers,
        settings.n_categories,
        params.latent_dim,
    )
    best_model_weights = torch.load(best_model_path / "model.pt")
    vae_model.load_state_dict(best_model_weights)

    return vae_model


def _get_full_dataloader() -> torch.utils.data.DataLoader:
    """Load the full dataset for evaluation, without any train/test split."""
    files = [Path(f"../{settings.dataset_path}") / f"S1a_{x}.mat" for x in ascii_uppercase[: settings.n_activities]]
    mats = [np.array(sio.loadmat(file)["csi"]) for file in files]

    dataset = CSIDataset(
        mats,
        settings.test_window_size,
        n_antennas=1,
        antenna_select=0,
        augment_probability=0,
        seed=settings.seed,
        stride=settings.stride,
    )

    return torch.utils.data.DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)


def latents() -> None:
    """Extract and save the latent representations for the entire dataset using the best VAE model."""
    vae = _get_best_model().to(device)
    vae.eval()

    dataloader = _get_full_dataloader()

    all_latents_hard = []
    all_latents_soft = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            batch_size = x.shape[0]

            x_r = x.to(device)
            x_r = split_test_window(x_r, settings.train_window_size, 0)

            n_windows = x_r.shape[0] // batch_size

            # Repeat labels to match the new batch size after reshaping (B * test_window_ratio,)
            y_r = y.repeat_interleave(n_windows, dim=0).to(device)

            # Latents will have shape (B * test_window_ratio, latent_dim, n_categories) for both hard and soft
            _, z_hard, z_soft = vae(x_r)

            all_latents_hard.append(z_hard.cpu())
            all_latents_soft.append(z_soft.cpu())
            all_labels.append(y_r.cpu())

    latents_hard = torch.cat(all_latents_hard, dim=0)
    latents_soft = torch.cat(all_latents_soft, dim=0)
    labels = torch.cat(all_labels, dim=0)

    output_path = Path(f"../{settings.study_path}/latents")
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "latents_hard.npy", latents_hard.numpy())
    np.save(output_path / "latents_soft.npy", latents_soft.numpy())
    np.save(output_path / "labels.npy", labels.numpy())


if __name__ == "__main__":
    latents()
