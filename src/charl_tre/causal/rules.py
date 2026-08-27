from dataclasses import dataclass

import numpy as np

from charl_tre.causal.lpcmci import LPCMCIVariable


@dataclass(frozen=True)
class CausalEdge:
    """Represents a causal edge between two latent variables with a specified lag and strength."""

    source: LPCMCIVariable
    target: LPCMCIVariable
    lag: int
    value: float

    @property
    def strength(self) -> float:
        """Return absolute MCI edge strength weighted by link confidence."""
        return abs(self.value)


@dataclass(frozen=True)
class Rule:
    """Represents an exact probabilistic causal rule based on Gaussian Log-Likelihood Ratios."""

    edge: CausalEdge
    source_threshold: float
    class_distribution: tuple[float, float]
    """Gaussian distribution parameters (mean, std) for the target variable under the rule"""

    bg_distribution: tuple[float, float]
    """Gaussian distribution parameters (mean, std) for the target variable under background conditions"""

    kl_divergence: float

    def llr(self, target_values: np.ndarray) -> float:
        """Compute robust clipped LLR sum: log[P_class(y) / P_bg(y)].

        Arguments:
            target_values (np.ndarray): Target variable values to evaluate.

        Returns:
            float: The sum of the clipped log-likelihood ratios for the provided target values.

        """
        if target_values.size == 0:
            return 0.0

        var_class = self.class_distribution[1] ** 2
        var_bg = self.bg_distribution[1] ** 2

        log_std_ratio = np.log(self.bg_distribution[1] / self.class_distribution[1])
        quad_class = ((target_values - self.class_distribution[0]) ** 2) / (2.0 * var_class)
        quad_bg = ((target_values - self.bg_distribution[0]) ** 2) / (2.0 * var_bg)

        per_step_llr = log_std_ratio - quad_class + quad_bg

        # Cap the LLR values to avoid extreme values that could dominate the sum.
        # We use the 95th percentile of the absolute LLR values for clipping.
        max_clip = float(np.percentile(np.abs(per_step_llr), 95.0))
        clipped_llr = np.clip(per_step_llr, -max_clip, max_clip)

        return float(np.sum(clipped_llr))


def _build_edges_from_adjacency(
    adjacency: np.ndarray,
    sorted_variables: list[LPCMCIVariable],
    min_strength: float,
) -> dict[int, list[CausalEdge]]:
    """Convert an LPCMCI adjacency array into filtered CausalEdge objects per activity."""
    n_activities = adjacency.shape[0]
    edges_by_activity: dict[int, list[CausalEdge]] = {}

    for activity_id in range(n_activities):
        cell = adjacency[activity_id]
        source_idx, target_idx, lag_idx = np.nonzero(cell["mark"])
        values = cell["value"][source_idx, target_idx, lag_idx]

        valid_edges = []
        for s, t, lag, val in zip(source_idx, target_idx, lag_idx, values, strict=True):
            if abs(val) >= min_strength:
                valid_edges.append(
                    CausalEdge(
                        source=sorted_variables[s],
                        target=sorted_variables[t],
                        lag=int(lag),
                        value=float(val),
                    ),
                )

        edges_by_activity[activity_id] = valid_edges

    return edges_by_activity


class RuleBuilder:
    """Builds a diverse, noise-filtered library of causal rules ranked by KL divergence."""

    def __init__(
        self,
        min_mci_strength: float = 0.10,
        min_samples: int = 5,
        top_rules_per_activity: int = 2,
    ) -> None:
        """Initialize the RuleBuilder."""
        self._min_mci_strength = min_mci_strength
        self._min_samples = min_samples
        self._top_rules_per_activity = top_rules_per_activity

    def build(
        self,
        adjacency: np.ndarray,
        sorted_variables: list[LPCMCIVariable],
        train_data: np.ndarray,
    ) -> dict[int, list[Rule]]:
        """Build library with MCI thresholding and target diversity constraints.

        Arguments:
            adjacency: A 3D numpy array representing the LPCMCI adjacency matrix.
            sorted_variables: A list of LPCMCIVariable objects corresponding to the adjacency matrix.
            train_data: A 4D numpy array of shape (n_activities, n_windows, n_mixtures, n_components)
                representing the training sequences for each activity.

        Returns:
            A dictionary mapping activity indices to lists of Rule objects.

        """
        n_activities, n_windows, _, _ = train_data.shape

        # ParCorr outputs values in [-1, 1], we use Fisher's z-transform to the full real line
        edges_by_activity = _build_edges_from_adjacency(adjacency, sorted_variables, self._min_mci_strength)

        # Array of activity IDs for indexing
        activity_ids = np.arange(n_activities)[:, None]

        rules: dict[int, list[Rule]] = {}

        for activity in range(n_activities):
            edges = edges_by_activity.get(activity, [])
            candidates: list[Rule] = []

            for edge in edges:
                source, target, lag = edge.source, edge.target, edge.lag

                source_all = train_data[:, : n_windows - lag, source.mixture, source.component]
                target_all = train_data[:, lag:, target.mixture, target.component]

                source_activity = source_all[activity]
                target_activity = target_all[activity]

                if source_activity.size < self._min_samples:
                    continue

                source_threshold = float(np.median(source_activity))
                y_activity = target_activity[source_activity >= source_threshold]
                if y_activity.size < self._min_samples:
                    continue

                bg_mask = (source_all >= source_threshold) & (activity_ids != activity)
                y_bg = target_all[bg_mask]
                if y_bg.size < self._min_samples:
                    continue

                # Compute Gaussian params for both activity and background samples
                mu_a, std_a = (float(np.mean(y_activity)), float(np.sqrt(np.var(y_activity))))
                mu_bg, std_bg = float(np.mean(y_bg)), float(np.sqrt(np.var(y_bg)))

                # Compute KL divergence between the two Gaussian distributions
                var_a, var_bg = std_a**2, std_bg**2
                kl_div = np.log(std_bg / std_a) + (var_a + (mu_a - mu_bg) ** 2) / (2.0 * var_bg) - 0.5
                kl_div = max(kl_div, 0.0)

                candidates.append(
                    Rule(
                        edge,
                        source_threshold,
                        class_distribution=(mu_a, std_a),
                        bg_distribution=(mu_bg, std_bg),
                        kl_divergence=kl_div,
                    ),
                )

            candidates.sort(key=lambda r: r.kl_divergence, reverse=True)
            rules[activity] = candidates[: self._top_rules_per_activity]

        return rules
