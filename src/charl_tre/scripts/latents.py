import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as func
from rich.logging import RichHandler

from charl_tre import dataset, util
from charl_tre.models import hierarchical
from charl_tre.settings import Settings
from charl_tre.studies import get_best_trial, read_study

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def _alpha_to_hard_onehot(alpha: np.ndarray) -> np.ndarray:
    """Convert per-mixture Dirichlet concentrations to one-hot component assignments."""
    component_idx = np.argmax(alpha, axis=2)
    n_components = alpha.shape[2]
    return np.eye(n_components, dtype=np.int8)[component_idx]


def latents() -> None:
    """Extract and save mixture-Dirichlet latents for the full dataset."""
    util.init_rng(settings.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading scenario from %s...", settings.dataset_path)
    train_ds, val_ds, test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.vae_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    full_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])
    logger.info("Scenario loaded. Total windows: %d", len(full_ds))

    best_trial = get_best_trial(read_study(settings.study_path))
    best_trial_path = Path(settings.study_path) / f"trial_{best_trial.trial_number}"

    n_components = int(best_trial.params["n_components"])
    n_mixtures = int(best_trial.params["n_mixtures"])
    n_fusion_layers = int(best_trial.params["n_fusion_layers"])
    fusion_dropout = float(best_trial.params["fusion_dropout"])
    batch_size = int(best_trial.params["batch_size"])

    logger.info(
        "Best trial parameters: n_components=%d, n_mixtures=%d, n_fusion_layers=%d, fusion_dropout=%.4f, batch_size=%d",
        n_components,
        n_mixtures,
        n_fusion_layers,
        fusion_dropout,
        batch_size,
    )

    full_dl = util.make_dl(
        full_ds,
        batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )

    h_vae = hierarchical.Fusion(
        window_size=settings.vae_window_size,
        n_subcarriers=settings.n_subcarriers,
        n_dirichlet_components=n_components,
        n_mixtures=n_mixtures,
        n_fusion_layers=n_fusion_layers,
        fusion_dropout=fusion_dropout,
        n_antennas=settings.n_antennas,
    ).to(device)
    h_vae_weights = torch.load(best_trial_path / "h_vae.pt", weights_only=True)
    h_vae.load_state_dict(h_vae_weights, strict=True)
    h_vae.eval()

    alpha_list: list[np.ndarray] = []
    mix_probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    logger.info("Extracting latents...")
    with torch.no_grad():
        for x, y in full_dl:
            _, mix_logits, alpha, *_ = h_vae(x.to(device, dtype=torch.float32))
            alpha_list.append(alpha.cpu().numpy())
            mix_probs_list.append(func.softmax(mix_logits, dim=-1).cpu().numpy())
            labels_list.append(y.numpy())

    alpha_arr = np.concatenate(alpha_list, axis=0)  # (T, M, C)
    mix_probs_arr = np.concatenate(mix_probs_list, axis=0)  # (T, M)
    labels_arr = np.concatenate(labels_list, axis=0)  # (T,)
    latents_hard_arr = _alpha_to_hard_onehot(alpha_arr)  # (T, M, C)

    out_dir = Path(settings.study_path) / "latents"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "latents_hard.npy", latents_hard_arr)
    np.save(out_dir / "alpha.npy", alpha_arr)
    np.save(out_dir / "mix_probs.npy", mix_probs_arr)
    np.save(out_dir / "labels.npy", labels_arr)

    logger.info(
        "Saved latents to %s: latents_hard=%s, alpha=%s, mix_probs=%s, labels=%s",
        out_dir,
        latents_hard_arr.shape,
        alpha_arr.shape,
        mix_probs_arr.shape,
        labels_arr.shape,
    )


if __name__ == "__main__":
    latents()
