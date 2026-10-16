import logging
from pathlib import Path

import torch
from rich.logging import RichHandler

from charl_tre import dataset, util
from charl_tre.metrics import compute_metrics, metrics_summary
from charl_tre.models import classifier, hierarchical
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


@torch.no_grad()
def test() -> None:
    """Evaluate the best saved multi-antenna fusion model on the test set."""
    util.init_rng(settings.starter_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading scenario from %s...", settings.dataset_path)
    _, _, test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.classifier_window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )
    logger.info("Scenario loaded.")

    best_trial = util.read_study(settings.study_path).best_trial
    best_trial_path = Path(settings.study_path) / f"trial_{best_trial.number}"

    n_components = int(best_trial.params["n_components"])
    n_mixtures = int(best_trial.params["n_mixtures"])
    n_fusion_layers = int(best_trial.params["n_fusion_layers"])
    fusion_dropout = float(best_trial.params["fusion_dropout"])
    batch_size = int(best_trial.params["batch_size"])
    accuracies = best_trial.user_attrs["accuracies"]

    best_seed = int(max(accuracies, key=accuracies.get))
    best_seed_accuracy = accuracies[str(best_seed)]

    logger.info(
        "Best trial parameters: %s (best seed %d, accuracy=%.4f)\n",
        best_trial.params,
        best_seed,
        best_seed_accuracy,
    )

    test_dl = util.make_dl(
        test_ds,
        batch_size,
        shuffle=False,
        n_workers=settings.n_workers,
        seed=settings.starter_seed,
    )

    h_vae = hierarchical.Autoencoder(
        window_size=settings.vae_window_size,
        n_subcarriers=settings.n_subcarriers,
        n_dirichlet_components=n_components,
        n_mixtures=n_mixtures,
        n_fusion_layers=n_fusion_layers,
        fusion_dropout=fusion_dropout,
        n_antennas=settings.n_antennas,
    ).to(device)
    h_vae_weights = torch.load(best_trial_path / f"h_vae_{best_seed}.pt", weights_only=True)
    h_vae.load_state_dict(h_vae_weights, strict=True)

    classifier_model = classifier.Classifier(
        antennas=[h_vae],
        n_components=n_components * n_mixtures,
        n_activities=settings.n_activities,
        sample_window_size=settings.vae_window_size,
        overlap_size=settings.overlap_size,
        n_layers=1,
        dropout=0,
    ).to(device)
    classifier_model_weights = torch.load(best_trial_path / f"classifier_{best_seed}.pt", weights_only=True)
    classifier_model.load_state_dict(classifier_model_weights, strict=True)

    logger.info("Evaluating classifier on test set...")
    classifier_model.eval()
    y_preds = []
    y_true = []

    for batch_x, batch_y in test_dl:
        x, y = batch_x.to(device), batch_y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            batch_preds = classifier_model(x).argmax(dim=1)
        y_preds.extend(batch_preds.cpu().numpy())
        y_true.extend(y.cpu().numpy())

    metrics = compute_metrics(y_true, y_preds)
    logger.info("Evaluation metrics:\n%s", metrics_summary(metrics, labels=settings.activities))


if __name__ == "__main__":
    test()
