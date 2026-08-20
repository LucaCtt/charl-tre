from dataclasses import dataclass, replace

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
        """Return the absolute value of the edge's strength."""
        return abs(self.value)

    @property
    def is_positive(self) -> bool:
        """Return True if the edge's value is non-negative, indicating a positive causal effect."""
        return self.value >= 0.0


@dataclass(frozen=True)
class Rule:
    """Represents a continuous causal rule derived from a CausalEdge."""

    edge: CausalEdge
    target_mean_train: float
    target_mean_other: float
    weight: float
    source_threshold: float
    target_direction: int
    selection_score: float | None
    is_core: bool
    normalized_weight: float | None = None
    normalized_core_weight: float | None = None
    normalized_tie_weight: float | None = None

    @property
    def delta(self) -> float:
        """Directional difference between conditional target means."""
        return self.target_mean_train - self.target_mean_other


@dataclass(frozen=True)
class RuleBudgetConfig:
    """Configuration for allocating budgets for rule selection."""

    top_rules_per_activity: int = 50
    min_rules_per_activity: int = 6
    max_rules_scale: float = 1.6
    low_edge_threshold: int = 3
    low_edge_budget_ratio: float = 0.3


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for selecting rules from candidates."""

    core_fraction: float = 0.6
    min_delta: float = 0.0


def build_edges_from_adjacency(
    adjacency: np.ndarray,
    sorted_variables: list[LPCMCIVariable],
) -> dict[int, list[CausalEdge]]:
    """Convert an LPCMCI adjacency array into CausalEdge objects per activity.

    Arguments:
        adjacency (np.ndarray): An LPCMCI adjacency array of shape (n_activities, n_variables, n_variables, n_lags).
        sorted_variables (list[LPCMCIVariable]): A list of LPCMCIVariable objects
            corresponding to the indices in the adjacency array.

    Returns:
        dict[int, list[CausalEdge]]: A dictionary mapping activity IDs to lists of CausalEdge objects
            representing the causal relationships for that activity.

    """
    n_activities = adjacency.shape[0]
    edges_by_activity: dict[int, list[CausalEdge]] = {}

    for activity_id in range(n_activities):
        cell = adjacency[activity_id]
        source_idx, target_idx, lag_idx = np.nonzero(cell["mark"])
        values = cell["value"][source_idx, target_idx, lag_idx]

        edges_by_activity[activity_id] = [
            CausalEdge(
                source=sorted_variables[s],
                target=sorted_variables[t],
                lag=int(lag),
                value=float(val),
            )
            for s, t, lag, val in zip(source_idx, target_idx, lag_idx, values, strict=True)
        ]

    return edges_by_activity


def _compute_source_threshold(
    train_data: np.ndarray,
    source: LPCMCIVariable,
) -> float:
    """Compute the global median threshold across all activities and time windows.

    Arguments:
        train_data (np.ndarray): The training data array of shape (..., n_mixtures, n_components).
        source (LPCMCIVariable): The source variable for which to compute the threshold.

    Returns:
        float: The median value of the source variable across all activities and time windows.

    """
    values = train_data[..., source.mixture, source.component].ravel()
    return float(np.median(values))


def _compute_unconditional_target_means(
    train_data: np.ndarray,
    target: LPCMCIVariable,
) -> np.ndarray:
    """Compute unconditional target means for all activities simultaneously.

    Arguments:
        train_data (np.ndarray): The training data array of shape (..., n_mixtures, n_components).
        target (LPCMCIVariable): The target variable for which to compute the unconditional means.

    Returns:
        np.ndarray: An array of shape (n_activities,) containing the unconditional means
            of the target variable across all time windows.

    """
    data = train_data[..., target.mixture, target.component]
    return data.mean(axis=1)


def _conditional_target_means(
    train_data: np.ndarray,
    edge: CausalEdge,
    source_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute conditional target means and sample counts across all activities simultaneously via vectorized slicing.

    Arguments:
        train_data (np.ndarray): The training data array of shape (n_activities, n_windows, n_mixtures, n_components).
        edge (CausalEdge): The causal edge for which to compute the conditional means.
        source_threshold (float): The threshold value for the source variable to determine valid samples.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - An array of shape (n_activities,) with the conditional means of the target variable
              given the source variable exceeds the threshold.
            - An array of shape (n_activities,) with the counts of valid samples
                used to compute the conditional means for each activity.

    """
    n_activities, n_windows = train_data.shape[:2]

    source_seq = train_data[:, : n_windows - edge.lag, edge.source.mixture, edge.source.component]
    target_seq = train_data[:, edge.lag :, edge.target.mixture, edge.target.component]

    valid_mask = source_seq >= source_threshold
    counts = valid_mask.sum(axis=1)

    sums = np.where(valid_mask, target_seq, 0.0).sum(axis=1)
    means = np.divide(sums, counts, out=np.zeros(n_activities), where=counts > 0)

    return means, counts


def _annotate_hierarchy(selected: list[Rule], core_fraction: float, eps: float = 1e-8) -> list[Rule]:
    """Annotate selected rules with hierarchy and normalized weight information."""
    if not selected:
        return []

    selected.sort(key=lambda r: r.selection_score if r.selection_score is not None else r.weight, reverse=True)

    n_selected = len(selected)
    core_count = max(1, min(n_selected, round(n_selected * core_fraction)))

    weights = np.maximum([r.weight for r in selected], eps)
    total_weight = float(weights.sum())
    core_weight = float(weights[:core_count].sum())
    tie_weight = float(weights[core_count:].sum())

    annotated = []
    for idx, rule in enumerate(selected):
        w = float(weights[idx])
        is_core = idx < core_count

        annotated.append(
            replace(
                rule,
                is_core=is_core,
                normalized_weight=w / total_weight if total_weight > 0.0 else 0.0,
                normalized_core_weight=w / core_weight if is_core and core_weight > 0.0 else 0.0,
                normalized_tie_weight=w / tie_weight if not is_core and tie_weight > 0.0 else 0.0,
            ),
        )

    return annotated


class RuleBuilder:
    """Builds a library of causal rules from causal edges and training data with budget allocation and selection."""

    def __init__(
        self,
        budget_config: RuleBudgetConfig | None = None,
        selection_config: SelectionConfig | None = None,
    ) -> None:
        """Initialize the RuleBuilder with optional budget and selection configurations."""
        self._budget_config = budget_config or RuleBudgetConfig()
        self._selection_config = selection_config or SelectionConfig()

    def build(
        self,
        adjacency: np.ndarray,
        sorted_variables: list[LPCMCIVariable],
        train_data: np.ndarray,
    ) -> dict[int, list[Rule]]:
        """Build a library of causal rules mapped by activity ID."""
        edges_by_activity = build_edges_from_adjacency(adjacency, sorted_variables)
        candidates_by_activity = self._build_candidates(edges_by_activity, train_data)
        budgets = self._allocate_budgets(candidates_by_activity, edges_by_activity)

        library: dict[int, list[Rule]] = {}
        for activity, candidates in candidates_by_activity.items():
            budget = budgets.get(activity, 0)
            library[activity] = self._select_with_pruning(candidates, budget)

        return library

    def _build_candidates(
        self,
        edges_by_activity: dict[int, list[CausalEdge]],
        train_data: np.ndarray,
    ) -> dict[int, list[Rule]]:
        n_activities = train_data.shape[0]

        # Collect unique domain objects/configurations across all activities
        all_edges = [edge for edges in edges_by_activity.values() for edge in edges]
        unique_sources = {edge.source for edge in all_edges}
        unique_targets = {edge.target for edge in all_edges}
        unique_edge_keys = {(e.source, e.target, e.lag): e for e in all_edges}

        # Precompute statistics passing domain objects directly
        threshold_cache = {src: _compute_source_threshold(train_data, src) for src in unique_sources}
        uncond_cache = {tgt: _compute_unconditional_target_means(train_data, tgt) for tgt in unique_targets}
        cond_cache = {
            key: _conditional_target_means(train_data, edge, threshold_cache[edge.source])
            for key, edge in unique_edge_keys.items()
        }

        # Construct candidates with cached lookups
        result: dict[int, list[Rule]] = {}
        denom = float(n_activities - 1) if n_activities > 1 else 1.0

        for activity, edges in edges_by_activity.items():
            candidates: list[Rule] = []

            for edge in edges:
                source_threshold = threshold_cache[edge.source]
                uncond_means = uncond_cache[edge.target]
                cond_means, cond_counts = cond_cache[(edge.source, edge.target, edge.lag)]

                own_mean = float(cond_means[activity])
                own_count = int(cond_counts[activity])

                if n_activities > 1:
                    effective = np.where(cond_counts > 0, cond_means, uncond_means)
                    other_mean = float((effective.sum() - effective[activity]) / denom)
                else:
                    other_mean = float(uncond_means[activity])

                target_direction = 1 if edge.is_positive else -1
                target_mean_train = target_direction * own_mean
                target_mean_other = target_direction * other_mean
                delta = target_mean_train - target_mean_other
                weight = float(edge.strength * max(delta, 0.0) * np.log1p(own_count))

                candidates.append(
                    Rule(
                        edge,
                        target_mean_train,
                        target_mean_other,
                        weight,
                        source_threshold,
                        target_direction,
                        selection_score=None,
                        is_core=False,
                    ),
                )

            result[activity] = candidates

        return result

    def _allocate_budgets(
        self,
        candidates_by_activity: dict[int, list[Rule]],
        edges_by_activity: dict[int, list[CausalEdge]],
    ) -> dict[int, int]:
        counts = [len(c) for c in candidates_by_activity.values() if len(c) > 0]
        median_ref = float(np.median(counts)) if counts else 1.0

        cfg = self._budget_config
        max_budget = max(cfg.top_rules_per_activity * cfg.max_rules_scale, cfg.top_rules_per_activity)

        budgets: dict[int, int] = {}
        for activity, candidates in candidates_by_activity.items():
            count = len(candidates)
            scale = (count / median_ref) ** 0.5 if median_ref > 0 else 1.0
            raw_budget = cfg.top_rules_per_activity * scale
            budget = min(max(raw_budget, cfg.min_rules_per_activity), max_budget, count)

            if len(edges_by_activity.get(activity, [])) <= cfg.low_edge_threshold:
                low_cap = max(
                    cfg.top_rules_per_activity * cfg.low_edge_budget_ratio,
                    cfg.min_rules_per_activity,
                )
                budget = min(budget, low_cap)

            budgets[activity] = round(budget)

        return budgets

    def _select_with_pruning(self, candidates: list[Rule], budget: int) -> list[Rule]:
        if budget == 0 or not candidates:
            return []

        best_by_key: dict[tuple[LPCMCIVariable, LPCMCIVariable, int, int], Rule] = {}
        for rule in candidates:
            key = (rule.edge.source, rule.edge.target, rule.edge.lag, rule.target_direction)
            if key not in best_by_key or best_by_key[key].weight < rule.weight:
                best_by_key[key] = rule

        deduped = list(best_by_key.values())
        positive = [r for r in deduped if r.delta >= self._selection_config.min_delta]

        pool = positive if len(positive) >= max(3, budget // 2) else deduped
        pool.sort(key=lambda r: (r.weight, r.delta, r.edge.strength), reverse=True)

        selected: list[Rule] = []
        selected_keys: set[tuple[LPCMCIVariable, LPCMCIVariable, int, int]] = set()
        source_lag_counts: dict[tuple[LPCMCIVariable, int], int] = {}
        pair_counts: dict[tuple[LPCMCIVariable, LPCMCIVariable], int] = {}

        while pool and len(selected) < budget:
            best_idx = None
            best_score = float("-inf")

            for idx, rule in enumerate(pool):
                key = (rule.edge.source, rule.edge.target, rule.edge.lag, rule.target_direction)
                opposite_key = (rule.edge.source, rule.edge.target, rule.edge.lag, -rule.target_direction)
                if key in selected_keys or opposite_key in selected_keys:
                    continue

                base_score = (
                    rule.weight
                    if rule.weight != 0.0
                    else 1e-6 + 0.15 * max(rule.delta, 0.0) + 0.05 * rule.edge.strength
                )
                sl_count = source_lag_counts.get((rule.edge.source, rule.edge.lag), 0)
                pc_count = pair_counts.get((rule.edge.source, rule.edge.target), 0)

                diversity_factor = 1.0 / (1.0 + 0.65 * sl_count + 0.85 * pc_count)
                relaxed_factor = 1.0 if rule.delta >= self._selection_config.min_delta else 0.6

                score = base_score * diversity_factor * relaxed_factor

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            chosen_rule = pool.pop(best_idx)
            chosen_rule = replace(chosen_rule, selection_score=best_score)

            sl_key = (chosen_rule.edge.source, chosen_rule.edge.lag)
            pc_key = (chosen_rule.edge.source, chosen_rule.edge.target)
            source_lag_counts[sl_key] = source_lag_counts.get(sl_key, 0) + 1
            pair_counts[pc_key] = pair_counts.get(pc_key, 0) + 1

            selected_keys.add((
                chosen_rule.edge.source,
                chosen_rule.edge.target,
                chosen_rule.edge.lag,
                chosen_rule.target_direction,
            ))
            selected.append(chosen_rule)

        if not selected:
            return []

        return _annotate_hierarchy(selected, self._selection_config.core_fraction)
