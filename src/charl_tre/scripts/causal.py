import logging
import tempfile
from pathlib import Path

import numpy as np
import optuna
from rich.logging import RichHandler

from charl_tre import util
from charl_tre.causal.evaluator import evaluate
from charl_tre.causal.lpcmci import LPCMCIParams, LPCMCIVariable, run_lpcmci_batch
from charl_tre.causal.rules import RuleBuilder
from charl_tre.metrics import metrics_summary
from charl_tre.settings import Settings

settings = Settings()

# Configure logging
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")


def _clr(
    alphas: np.ndarray,  # (n_activities, n_windows, n_mixtures, n_components)
) -> np.ndarray:
    """Decompose raw Dirichlet alphas into [log_concentration | CLR_reduced].

    Returns shape (n_activities, n_windows, n_mixtures, n_components)
    where the last axis contains:
      - index 0: log(sum of alphas) -- the log-concentration scalar
      - indices 1..K-1: CLR-transformed composition, last component dropped

    The returned array has the same shape as the input -- log-concentration
    replaces the dropped CLR component rather than expanding the dimension,
    keeping downstream code shape-compatible.
    """
    n_components = alphas.shape[-1]

    # Smoothed Dirichlet mean: avoids log(0) for near-zero alphas
    total = alphas.sum(axis=-1, keepdims=True) + n_components
    p = (alphas + 1.0) / total

    # CLR: log(p_i) - mean_j(log(p_j)), then drop last component
    log_p = np.log(p)
    clr = log_p - log_p.mean(axis=-1, keepdims=True)
    return clr[..., :-1]  # drop redundant last component, (K-1) dims


def load_latents(latents_dir: Path, n_activities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        latents[ds_name] = _clr(activity_latents)

    return latents["train"], latents["val"], latents["test"]


def _build_rules(
    trial: optuna.Trial,
    adjacency: np.ndarray,
    variables: list[LPCMCIVariable],
    train_latents: np.ndarray,
    val_latents: np.ndarray,
) -> float:
    """Build rules from the adjacency matrix and training latents.

    Arguments:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.
        adjacency (np.ndarray): Adjacency matrix of shape (n_vars, n_vars, tau_max + 1).
        variables (list[LPCMCIVariable]): List of LPCMCIVariable instances representing the variables.
        train_latents (np.ndarray): Training latents of shape (n_activities, n_windows, n_mixtures, n_components).
        val_latents (np.ndarray): Validation latents of shape (n_activities, n_windows, n_mixtures, n_components).

    Returns:
        float: The evaluation metric to optimize.

    """
    min_mci_strength = trial.suggest_float("min_mci_strength", 0.1, 0.4, step=0.1)
    min_samples = trial.suggest_int("min_samples", 10, 30, step=5)
    top_rules_per_activity = trial.suggest_int("top_rules_per_activity", 2, 12, step=2)
    drop_threshold = trial.suggest_float("drop_threshold", 0.1, 0.5, step=0.05)
    null_clip_sigma = trial.suggest_float("null_clip_sigma", 1.0, 5.0, step=0.5)
    segment_length = trial.suggest_int("segment_length", 12, 30, step=2)

    rule_builder = RuleBuilder(min_mci_strength, min_samples, top_rules_per_activity, drop_threshold)
    rules = rule_builder.build(adjacency, variables, train_latents)
    metrics = evaluate(rules, val_latents, segment_length, null_clip_sigma)

    total_rules = sum(len(rules_list) for rules_list in rules.values())
    return metrics.multi_class.accuracy - 1e-3 * total_rules  # Penalize for too many rules


def causal() -> None:
    """Extract and save mixture-Dirichlet latents for the full dataset."""
    util.init_rng(settings.starter_seed)
    causal_dir = Path(settings.study_path) / "causal"

    logger.info("Loading latents...")
    latents_dir = Path(settings.study_path) / "latents"
    train_latents, val_latents, test_latents = load_latents(latents_dir, settings.n_activities)

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

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
            params = LPCMCIParams(variables=variables, tau_min=1, tau_max=11, max_p_global=3)
            adjacency_matrix = run_lpcmci_batch(
                train_latents,
                params=params,
                max_workers=settings.n_workers,
                cache_dir=cache_dir,
            )

        Path(causal_dir).mkdir(parents=True, exist_ok=True)
        np.save(
            causal_dir / "adjacency_matrix.npy",
            adjacency_matrix,
        )

        logger.info("Done.")

    # Hyperparameter optimization for rule building on the validation set
    study = util.make_study(
        study_name="causal_rule_discovery",
        storage_dir=f"{settings.study_path}/causal",
        seed=settings.starter_seed,
    )
    study.optimize(
        lambda trial: _build_rules(trial, adjacency_matrix, variables, train_latents, val_latents),
        n_trials=settings.n_trials,
    )

    logger.info("Best trial value: %s", study.best_value)

    # Final evaluation on the test set using the best hyperparameters
    params = study.best_params
    rule_builder = RuleBuilder(
        min_mci_strength=params["min_mci_strength"],
        min_samples=params["min_samples"],
        top_rules_per_activity=params["top_rules_per_activity"],
        drop_threshold=params["drop_threshold"],
    )
    rules = rule_builder.build(adjacency_matrix, variables, train_latents)

    for activity in range(settings.n_activities):
        rules_strs = [
            (
                f"□(z_{rule.edge.source.component} → ○{rule.edge.lag} "
                f"{'' if rule.edge.value > 0 else '-'}z_{rule.edge.target.component})"
            )
            for rule in rules.get(activity, [])
        ]
        logger.debug("%s: %s", settings.activities[activity], rules_strs)

    metrics = evaluate(rules, test_latents, params["segment_length"], params["null_clip_sigma"])
    logger.info("Evaluation metrics:\n%s", metrics_summary(metrics, settings.activities))


if __name__ == "__main__":
    causal()
