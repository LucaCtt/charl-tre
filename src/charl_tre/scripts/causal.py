import logging
import tempfile
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from charl_tre import util
from charl_tre.causal.evaluator import classify, evaluate
from charl_tre.causal.lpcmci import LPCMCIParams, LPCMCIVariable, run_lpcmci_batch
from charl_tre.causal.rules import CausalEdge, RuleBuilder
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def _load_latents(latents_dir: Path, n_activities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load latents and labels from the specified directory.

    Args:
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

        latents[ds_name] = np.stack(activity_latents, axis=0)

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
    params = LPCMCIParams(variables=variables)
    return run_lpcmci_batch(train_latents, params=params, max_workers=6, cache_dir=cache_dir)


def _edges_from_adjacency(
    adjacency_matrix: np.ndarray,
    variables: list[LPCMCIVariable],
) -> dict[int, list[CausalEdge]]:
    """Convert an adjacency matrix to a dictionary of edges by activity.

    Arguments:
        adjacency_matrix (np.ndarray): The adjacency matrix of shape (n_activities, n_vars, n_vars, tau_max+1).
        variables (list[LPCMCIVariable]): List of LPCMCIVariable instances representing the variables.

    Returns:
        dict[int, list[CausalEdge]]: A dictionary mapping activity IDs to their causal edges.

    """
    edges_by_activity: dict[int, list[CausalEdge]] = {}

    for activity in range(adjacency_matrix.shape[0]):
        edges: list[CausalEdge] = []
        marked = np.argwhere(adjacency_matrix[activity]["mark"])
        for source, target, lag in marked:
            edges.append(
                CausalEdge(
                    source=variables[source],
                    target=variables[target],
                    lag=int(lag),
                    value=float(adjacency_matrix[activity, source, target, lag]["value"]),
                ),
            )
        edges_by_activity[activity] = edges

    return edges_by_activity


def causal() -> None:
    """Extract and save mixture-Dirichlet latents for the full dataset."""
    util.init_rng(settings.seed)
    causal_dir = Path(settings.study_path) / "causal"

    logger.info("Loading latents...")
    latents_dir = Path(settings.study_path) / "latents"
    train_latents, _, _ = _load_latents(latents_dir, settings.n_activities)

    _, _, n_mixtures, n_components = train_latents.shape
    variables = [
        LPCMCIVariable(mixture=mixture, component=component)
        for mixture in range(n_mixtures)
        for component in range(n_components)
    ]

    if (causal_dir / "adjacency_matrix.npz").exists():
        logger.info("Adjacency matrix already exists. Skipping causal discovery.")

        adjacency_matrix = np.load(causal_dir / "adjacency_matrix.npz")["adjacency_matrix"]
    else:
        logger.info("Running LPCMCI discovery...")

        with tempfile.TemporaryDirectory() as cache_dir:
            adjacency_matrix = _causal_discovery(train_latents, variables, cache_dir)

        Path(causal_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            causal_dir / "adjacency_matrix.npz",
            adjacency_matrix=adjacency_matrix,
        )

        logger.info("Done.")

    rule_builder = RuleBuilder()
    edges_by_activity = _edges_from_adjacency(adjacency_matrix, variables)
    rules = rule_builder.build(edges_by_activity, train_latents)

    metrics = evaluate(rules, train_latents, 5)
    print(metrics)


if __name__ == "__main__":
    causal()
