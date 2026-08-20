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


def _conditional_target_mean(
    data: np.ndarray,
    edge: CausalEdge,
    source_threshold: float,
) -> tuple[float, int]:
    """Compute a conditional target mean and sample count for a continuous rule.

    Arguments:
        data (np.ndarray): 3D array of shape (n_windows, n_mixtures, n_components).
        edge (CausalEdge): The causal edge containing source, target, lag, and other properties.
        source_threshold (float): Source value threshold for activating the rule condition.

    Returns:
        tuple[float, int]: Conditional target mean and number of source conditions.

    """
    n_windows = data.shape[0]
    lag = edge.lag

    target_values: list[float] = []
    for t in range(n_windows - lag):
        source_value = data[t, edge.source.mixture, edge.source.component]
        target_value = data[t + lag, edge.target.mixture, edge.target.component]

        if np.isfinite(source_value) and np.isfinite(target_value) and source_value >= source_threshold:
            target_values.append(float(target_value))

    return (float(np.mean(target_values)) if target_values else 0.0), len(target_values)


def _source_threshold(train_data: np.ndarray, edge: CausalEdge) -> float:
    """Return a common source threshold so activities are evaluated on the same condition."""
    source_values = train_data[:, :, edge.source.mixture, edge.source.component]
    finite_values = source_values[np.isfinite(source_values)]
    return float(np.median(finite_values)) if finite_values.size else 0.0


def _target_mean(data: np.ndarray, edge: CausalEdge) -> float:
    """Return the unconditional mean of an edge target."""
    values = data[:, edge.target.mixture, edge.target.component]
    finite_values = values[np.isfinite(values)]
    return float(np.mean(finite_values)) if finite_values.size else 0.0


def _median(values: list[float]) -> float | None:
    """Compute the median of a list of float values.

    Arguments:
        values (list[float]): A list of float values.

    Returns:
        float | None: The median value, or None if the list is empty.

    """
    n = len(values)
    if n == 0:
        return None
    sorted_values = sorted(values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

    return sorted_values[mid]


def _annotate_hierarchy(selected: list[Rule], core_fraction: float, eps: float = 1e-8) -> list[Rule]:
    """Annotate the selected rules with hierarchy information based on their weights.

    Arguments:
        selected (list[Rule]): A list of selected Rule objects.
        core_fraction (float): The fraction of rules to be considered as core.
        eps (float): A small value to avoid division by zero.

    Returns:
        list[Rule]: The annotated list of Rule objects with hierarchy information.

    """
    selected.sort(key=lambda r: r.selection_score if r.selection_score is not None else r.weight, reverse=True)

    core_count = max(1, min(len(selected), round(len(selected) * core_fraction)))
    total_w = sum(max(r.weight, eps) for r in selected)
    core_w = sum(max(r.weight, eps) for r in selected[:core_count])
    tie_w = sum(max(r.weight, eps) for r in selected[core_count:])

    annotated = []
    for idx, rule in enumerate(selected):
        w = max(rule.weight, eps)
        is_core = idx < core_count

        annotated_rule = replace(
            rule,
            is_core=is_core,
            normalized_weight=w / total_w if total_w > 0.0 else 0.0,
            normalized_core_weight=w / core_w if is_core and core_w > 0.0 else 0.0,
            normalized_tie_weight=w / tie_w if not is_core and tie_w > 0.0 else 0.0,
        )
        annotated.append(annotated_rule)

    return annotated


class RuleBuilder:
    """Builds a library of causal rules from causal edges and training data, with budget allocation and selection."""

    def __init__(
        self,
        budget_config: RuleBudgetConfig | None = None,
        selection_config: SelectionConfig | None = None,
    ) -> None:
        """Initialize the RuleBuilder with optional configuration parameters."""
        self._budget_config = budget_config or RuleBudgetConfig()
        self._selection_config = selection_config or SelectionConfig()

    def build(
        self,
        edges_by_activity: dict[int, list[CausalEdge]],
        train_data: np.ndarray,
    ) -> dict[int, list[Rule]]:
        """Build a library of causal rules from an LPCMCI adjacency matrix.

        Arguments:
            edges_by_activity (dict[int, list[CausalEdge]]): A dictionary mapping activity IDs to their causal edges.
            train_data (np.ndarray): A 4D array of shape (n_activities, n_windows, n_mixtures, n_components)
                representing the training data for each activity.

        Returns:
            dict[int, list[Rule]]: A dictionary mapping activity IDs to their selected causal rules
                after budget allocation and selection.

        """
        candidates_by_activity = self._build_candidates(edges_by_activity, train_data)
        budgets = self._allocate_budgets(candidates_by_activity, edges_by_activity)

        library: dict[int, list[Rule]] = {}
        for activity, candidates in candidates_by_activity.items():
            budget = budgets.get(activity, 0)
            selected = self._select_with_pruning(candidates.copy(), budget)

            library[activity] = selected

        return library

    def _build_candidates(
        self,
        edges_by_activity: dict[int, list[CausalEdge]],
        train_data: np.ndarray,
    ) -> dict[int, list[Rule]]:
        activities = list(edges_by_activity.keys())
        result: dict[int, list[Rule]] = {}

        for activity in activities:
            edges = edges_by_activity.get(activity, [])
            candidates: list[Rule] = []

            for edge in edges:
                candidate = self._build_candidate(
                    activity,
                    activities,
                    train_data,
                    edge,
                )
                if candidate is not None:
                    candidates.append(candidate)

            result[activity] = candidates

        return result

    def _build_candidate(
        self,
        activity: int,
        activities: list[int],
        train_data: np.ndarray,
        edge: CausalEdge,
    ) -> Rule | None:
        own_data = train_data[activity]
        source_threshold = _source_threshold(train_data, edge)

        own_mean, own_count = _conditional_target_mean(own_data, edge, source_threshold)

        other_means: list[float] = []
        for other_activity in activities:
            if other_activity == activity:
                continue
            other_data = train_data[other_activity]
            other_mean, other_count = _conditional_target_mean(other_data, edge, source_threshold)
            other_means.append(other_mean if other_count > 0 else _target_mean(other_data, edge))

        other_mean = sum(other_means) / len(other_means) if other_means else _target_mean(own_data, edge)

        target_train, target_other = (own_mean, other_mean) if edge.is_positive else (-own_mean, -other_mean)

        delta = target_train - target_other
        weight = edge.strength * max(delta, 0.0) * np.log1p(own_count)

        return Rule(
            edge=edge,
            target_mean_train=target_train,
            target_mean_other=target_other,
            weight=weight,
            source_threshold=source_threshold,
            target_direction=1 if edge.is_positive else -1,
            selection_score=None,
            is_core=False,
        )

    def _allocate_budgets(
        self,
        candidates_by_activity: dict[int, list[Rule]],
        edges_by_activity: dict[int, list[CausalEdge]],
    ) -> dict[int, int]:
        """Allocate budgets for each activity based on the number of candidate rules and the configuration parameters.

        Arguments:
            candidates_by_activity (dict[int, list[Rule]]): A dictionary mapping activity IDs to their candidate rules.
            edges_by_activity (dict[int, list[CausalEdge]]): A dictionary mapping activity IDs to their causal edges.

        Returns:
            dict[int, int]: A dictionary mapping activity IDs to their allocated budgets (maximum number of rules).

        """
        non_zero_candidates = [float(len(c)) for c in candidates_by_activity.values() if len(c) > 0]
        median_references = _median(non_zero_candidates) or 1.0

        max_budget = max(
            self._budget_config.top_rules_per_activity * self._budget_config.max_rules_scale,
            self._budget_config.top_rules_per_activity,
        )

        budgets: dict[int, int] = {}
        for activity, candidates in candidates_by_activity.items():
            count = len(candidates)

            scale = (count / median_references) ** 0.5 if median_references > 0 else 1.0
            raw_budget = self._budget_config.top_rules_per_activity * scale

            budget = min(max(raw_budget, self._budget_config.min_rules_per_activity), max_budget, count)

            if len(edges_by_activity.get(activity, [])) <= self._budget_config.low_edge_threshold:
                low_cap = max(
                    self._budget_config.top_rules_per_activity * self._budget_config.low_edge_budget_ratio,
                    self._budget_config.min_rules_per_activity,
                )
                budget = min(budget, low_cap)

            budgets[activity] = round(budget)

        return budgets

    def _select_with_pruning(self, candidates: list[Rule], budget: int) -> list[Rule]:
        """Select rules from candidates with pruning based on the selection configuration and budget.

        Arguments:
            candidates (list[Rule]): A list of candidate Rule objects.
            budget (int): The maximum number of rules to select.

        Returns:
            list[Rule]: A list of selected Rule objects after pruning and selection.

        """
        if budget == 0 or not candidates:
            return []

        best_by_key: dict[tuple[LPCMCIVariable, LPCMCIVariable, int, int], Rule] = {}
        for rule in candidates:
            key = (rule.edge.source, rule.edge.target, rule.edge.lag, rule.target_direction)
            existing = best_by_key.get(key)
            if existing is None or existing.weight < rule.weight:
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

            source_lag_counts[(chosen_rule.edge.source, chosen_rule.edge.lag)] = (
                source_lag_counts.get((chosen_rule.edge.source, chosen_rule.edge.lag), 0) + 1
            )
            pair_counts[(chosen_rule.edge.source, chosen_rule.edge.target)] = (
                pair_counts.get((chosen_rule.edge.source, chosen_rule.edge.target), 0) + 1
            )
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
