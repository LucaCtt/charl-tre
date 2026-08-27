import logging
import tempfile
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from charl_tre import util
from charl_tre.causal.evaluator import evaluate
from charl_tre.causal.lpcmci import LPCMCIParams, LPCMCIVariable, run_lpcmci_batch
from charl_tre.causal.rules import RuleBuilder
from charl_tre.metrics import metrics_summary
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.DEBUG, show_path=False)
logging.basicConfig(level=logging.DEBUG, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def _dirichlet_clr(latents: np.ndarray) -> np.ndarray:
    """Apply the centered log-ratio (CLR) transformation to Dirichlet-distributed latents.

    Arguments:
        latents (np.ndarray): Latents of shape (n_activities, n_windows, n_mixtures, n_components).

    Returns:
        np.ndarray: CLR-transformed latents of the same shape as the input.

    """
    n_components = latents.shape[-1]
    total = latents.sum(axis=-1, keepdims=True) + n_components
    p = (latents + 1) / total
    log_p = np.log(p)

    return log_p - log_p.mean(axis=-1, keepdims=True)


def _load_latents(latents_dir: Path, n_activities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load latents and labels from the specified directory.

    Arguments:
        latents_dir (Path): Directory containing the latents and labels files.
        n_activities (int): Number of activities to load.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the latents and labels arrays.

    """
    alphas = np.load(latents_dir / "alphas.npz")
    labels = np.load(latents_dir / "labels.npz")
    latents: dict[str, np.ndarray] = {}

    for ds_name in ["train", "val", "test"]:
        activity_latents = []
        for activity in range(n_activities):
            activity_alpha = alphas[ds_name][labels[ds_name] == activity]
            activity_latents.append(activity_alpha)

        # Create a 4D array of shape (n_activities, n_windows, n_mixtures, n_components)
        activity_latents = np.stack(activity_latents, axis=0)
        activity_latents = _dirichlet_clr(activity_latents)

        # Remove the last component to avoid redundancy in the transformation
        latents[ds_name] = activity_latents

    return latents["train"], latents["val"], latents["test"]


def _causal_discovery(
    train_latents: np.ndarray,
    variables: list[LPCMCIVariable],
    cache_dir: str | None = None,
) -> np.ndarray:
    """Run LPCMCI discovery on the training latents.

    Arguments:
        train_latents (np.ndarray): Training latents of shape (n_activities, n_windows, n_mixtures, n_components).
        variables (list[LPCMCIVariable]): List of LPCMCIVariable instances representing the variables.
        cache_dir (str | None): Directory to store temporary files for the tests.

    Returns:
        np.ndarray: Adjacency matrix of shape (n_vars, n_vars, tau_max + 1).

    """
    params = LPCMCIParams(variables=variables, pc_alpha=0.03, max_p_global=10)
    return run_lpcmci_batch(train_latents, params=params, max_workers=6, cache_dir=cache_dir)


def causal() -> None:
    """Extract and save mixture-Dirichlet latents for the full dataset."""
    util.init_rng(settings.seed)
    causal_dir = Path(settings.study_path) / "causal"

    logger.info("Loading latents...")
    latents_dir = Path(settings.study_path) / "latents"
    train_latents, _, test_latents = _load_latents(latents_dir, settings.n_activities)

    _, _, n_mixtures, n_components = train_latents.shape
    variables = [
        LPCMCIVariable(mixture=mixture, component=component)
        for mixture in range(n_mixtures)
        for component in range(n_components)
    ]

    if (causal_dir / "adjacency_matrix.npy").exists():
        logger.info("Adjacency matrix already exists. Skipping causal discovery.")

        adjacency_matrix = np.load(causal_dir / "adjacency_matrix.npy")
    else:
        logger.info("Running LPCMCI discovery...")

        with tempfile.TemporaryDirectory() as cache_dir:
            adjacency_matrix = _causal_discovery(train_latents, variables, cache_dir)

        Path(causal_dir).mkdir(parents=True, exist_ok=True)
        np.save(
            causal_dir / "adjacency_matrix.npy",
            adjacency_matrix,
        )

        logger.info("Done.")

    rule_builder = RuleBuilder()
    rules = rule_builder.build(adjacency_matrix, variables, train_latents)

    for activity in range(settings.n_activities):
        rules_strs = [
            f"□(z_{rule.edge.source.component} → ○{rule.edge.lag} z_{rule.edge.target.component})"
            for rule in rules.get(activity, [])
        ]
        logger.debug("%s: %s", settings.activities[activity], rules_strs)

    metrics = evaluate(rules, test_latents, 22)
    logger.info("Evaluation metrics:\n%s", metrics_summary(metrics, settings.activities))


if __name__ == "__main__":
    causal()
