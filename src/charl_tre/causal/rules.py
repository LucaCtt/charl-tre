"""Symbolic rule mining from causal edges."""

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

import numpy as np

from charl_tre.causal.types import CausalEdge


@dataclass(frozen=True)
class Rule:
    """Symbolic IF-THEN rule with associated statistics."""

    activity_id: int
    activity_name: str
    src: int
    src_var: str
    dst: int
    dst_var: str
    lag: int
    sign: Literal[1, -1]
    strength: float
    target_dst_state: int
    p_target_train: float
    p_target_other: float
    delta: float
    n_condition_train: int
    weight: float
    rule_origin: str
    rule_text: str
    # Selection metadata (populated during pruning)
    selection_score: float | None = None
    is_core: bool = False
    hierarchy: str = "core"
    normalized_weight: float = 0.0
    normalized_core_weight: float = 0.0
    normalized_tie_weight: float = 0.0


@dataclass(frozen=True)
class ActivityModel:
    """Metadata and prototype information for a specific activity."""

    activity_id: int
    activity_name: str
    rule_budget: int
    n_rules: int
    n_graph_rules: int
    n_fallback_rules: int
    core_rule_count: int
    tie_rule_count: int
    edge_count: int
    prior: float
    log_prior: float
    score_bias: float
    min_margin: float
    tie_breaker_weight: float
    coverage_weight: float
    prior_weight: float
    prototype_weight: float
    use_prototype_fallback: bool
    prototype_true_rate: list[float]


@dataclass(frozen=True)
class RuleBudgetConfig:
    """Configuration for allocating rule budgets to activities based on the number of candidate rules and edges."""

    top_rules_per_activity: int = 50
    min_rules_per_activity: int = 6
    max_rules_scale: float = 1.6
    low_edge_threshold: int = 3
    low_edge_budget_ratio: float = 0.3


@dataclass(frozen=True)
class FallbackMiningConfig:
    """Configuration for mining fallback candidates when there are few graph edges."""

    tau_max: int = 5
    top_source_vars: int = 8
    top_target_vars: int = 12
    min_condition_count: int = 8
    min_delta: float = 0.05
    pool_size: int = 300


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for selecting rules from candidates, including pruning and hierarchy annotation."""

    core_fraction: float = 0.6
    min_delta: float = 0.0
    tie_breaker_weight: float = 0.25
    coverage_weight: float = 0.2
    prior_weight: float = 0.1
    prototype_weight: float = 0.25
    min_margin: float = 0.02
    calibration_clip: float = 2.0


def _rule_hits(data: np.ndarray, src: int, dst: int, lag: int, target_state: int) -> tuple[int, int]:
    """Count the number of times the source variable is active at t-lag and the target variable is in target state at t.

    Arguments:
        data: A 2D numpy array of shape (n_samples, n_variables) containing the time series data.
        src: The index of the source variable.
        dst: The index of the target variable.
        lag: The time lag to consider between the source and target variables.
        target_state: The target state of the destination variable to count hits for (e.g., 1 for true, 0 for false).

    Returns:
        A tuple containing:
        - hits: The number of samples where the source variable is active at t-lag
            and the target variable is in the target state at t.
        - count: The total number of samples where the source variable is active at t-lag,
            regardless of the target variable's state.

    """
    if lag <= 0 or len(data) <= lag:
        return 0, 0
    mask = data[:-lag, src] == 1
    count = int(np.sum(mask))
    if count == 0:
        return 0, 0
    return int(np.sum(data[lag:, dst][mask] == target_state)), count


def _laplace(hits: int, count: int, alpha: float = 1.0) -> float:
    """Compute the Laplace-smoothed probability of hits given count with smoothing parameter alpha.

    Arguments:
        hits: The number of successful hits (e.g., source active and target in target state).
        count: The total number of opportunities for hits (e.g., source active).
        alpha: The smoothing parameter for Laplace smoothing (default is 1.0).

    Returns:
        The Laplace-smoothed probability of hits given count, calculated as (hits + alpha) / (count + 2 * alpha).

    """
    return float((hits + alpha) / (count + 2 * alpha))


class RuleBuilder:
    """Build and select symbolic IF-THEN rules from causal graphs and training data."""

    def __init__(
        self,
        activity_names: dict[int, str],
        var_names: list[str],
        budget_cfg: RuleBudgetConfig | None = None,
        fallback_cfg: FallbackMiningConfig | None = None,
        selection_cfg: SelectionConfig | None = None,
    ) -> None:
        """Initialize the builder."""
        self.__activity_names = activity_names
        self.__var_names = var_names
        self.__budget_cfg = budget_cfg or RuleBudgetConfig()
        self.__fallback_cfg = fallback_cfg or FallbackMiningConfig()
        self.__selection_cfg = selection_cfg or SelectionConfig()

    def build(
        self,
        edges_by_activity: dict[int, list[CausalEdge]],
        train_data: dict[int, np.ndarray],
    ) -> tuple[dict[int, list[Rule]], dict[int, ActivityModel]]:
        """Build and select rules for each activity based on the provided causal edges and training data.

        Arguments:
            edges_by_activity: A mapping from activity ID to a list of causal edges inferred for that activity.
            train_data: A mapping from activity ID to a 2D numpy array of shape (n_samples, n_variables)
                containing the training data for that activity.

        Returns:
            A tuple containing:
            - A dictionary mapping each activity ID to a list of selected rules for that activity.
            - A dictionary mapping each activity ID to a corresponding activity model
                containing metadata and prototype information.

        """
        activity_ids = sorted(edges_by_activity)
        total_train = max(1.0, float(sum(len(train_data[aid]) for aid in activity_ids)))

        candidates_by_activity = self.__build_all_candidates(activity_ids, edges_by_activity, train_data)
        budgets = self.__allocate_budgets(candidates_by_activity, edges_by_activity)

        library: dict[int, list[Rule]] = {}
        models: dict[int, ActivityModel] = {}

        for aid in activity_ids:
            candidates = candidates_by_activity[aid]
            has_fallback = any(r.rule_origin != "graph" for r in candidates)

            if has_fallback:
                selected = self.__select_with_pruning(candidates, budgets[aid])
            else:
                budget = min(len(candidates), self.__budget_cfg.top_rules_per_activity)
                selected = self.__select_baseline(candidates, budget)

            library[aid] = selected
            models[aid] = self.__build_model(
                aid,
                selected,
                edges_by_activity[aid],
                train_data,
                total_train,
                budgets[aid],
            )
        return library, models

    def __build_all_candidates(
        self,
        activity_ids: list[int],
        edges_by_activity: dict[int, list[CausalEdge]],
        train_data: dict[int, np.ndarray],
    ) -> dict[int, list[Rule]]:
        """Build candidate rules for each activity based on the provided edges and training data.

        Arguments:
            activity_ids: A list of activity IDs to build candidates for.
            edges_by_activity: A mapping from activity ID to a list of causal edges inferred for that activity.
            train_data: A mapping from activity ID to a 2D numpy array of shape (n_samples, n_variables)
                containing the training data for that activity.

        Returns:
            A dictionary mapping each activity ID to a list of candidate rules for that activity.

        """
        result: dict[int, list[Rule]] = {}
        for activity_id in activity_ids:
            graph_candidates = [
                c
                for edge in edges_by_activity[activity_id]
                if (
                    c := self.__build_candidate(
                        activity_id=activity_id,
                        activity_ids=activity_ids,
                        train_data=train_data,
                        src=edge.src,
                        dst=edge.dst,
                        lag=edge.lag,
                        strength=edge.strength,
                        sign=edge.sign,
                        rule_origin="graph",
                        min_condition_count=1,
                    )
                )
                is not None
            ]
            fallback_candidates: list[Rule] = []
            if len(edges_by_activity[activity_id]) <= self.__budget_cfg.low_edge_threshold:
                fallback_candidates = self.__mine_fallback_candidates(
                    activity_id=activity_id,
                    activity_ids=activity_ids,
                    train_data=train_data,
                )
            result[activity_id] = graph_candidates + fallback_candidates
        return result

    def __mine_fallback_candidates(
        self,
        activity_id: int,
        activity_ids: list[int],
        train_data: dict[int, np.ndarray],
    ) -> list[Rule]:
        """Mine fallback candidate rules for an activity when there are few graph edges.

        Arguments:
            activity_id: The ID of the activity to mine candidates for.
            activity_ids: A list of all activity IDs.
            train_data: A mapping from activity ID to a 2D numpy array of shape (n_samples, n_variables)
                containing the training data for each activity.

        Returns:
            A list of candidate rules mined for the specified activity.

        """
        cfg = self.__fallback_cfg
        own_data = train_data[activity_id]
        other_data = np.concatenate([train_data[oid] for oid in activity_ids if oid != activity_id], axis=0)
        own_rate = np.mean(own_data, axis=0, dtype=np.float64)
        other_rate = np.mean(other_data, axis=0, dtype=np.float64)
        discriminative = np.abs(own_rate - other_rate)

        src_counts = np.sum(own_data, axis=0)
        sources = [int(i) for i in np.argsort(-src_counts)[: max(1, cfg.top_source_vars)]]
        targets = [int(i) for i in np.argsort(-discriminative)[: max(1, cfg.top_target_vars)]]

        candidates: list[Rule] = []
        for lag, src, dst in product(range(1, cfg.tau_max + 1), sources, targets):
            if src_counts[src] < cfg.min_condition_count:
                continue
            strength_proxy = 0.08 + float(discriminative[dst])
            candidate = self.__build_candidate(
                activity_id=activity_id,
                activity_ids=activity_ids,
                train_data=train_data,
                src=src,
                dst=dst,
                lag=lag,
                strength=strength_proxy,
                sign=None,
                rule_origin="fallback_mined",
                min_condition_count=cfg.min_condition_count,
            )
            if candidate is not None and candidate.delta > cfg.min_delta:
                candidates.append(candidate)

        ranked = sorted(candidates, key=lambda x: (x.weight, x.delta, x.strength), reverse=True)
        return ranked[: cfg.pool_size]

    def __build_candidate(
        self,
        activity_id: int,
        activity_ids: list[int],
        train_data: dict[int, np.ndarray],
        src: int,
        dst: int,
        lag: int,
        strength: float,
        sign: int | None,
        rule_origin: str,
        min_condition_count: int,
    ) -> Rule | None:
        """Build a candidate rule for a given source, target, and lag based on training data statistics.

        Arguments:
            activity_id: The ID of the activity to build the candidate for.
            activity_ids: A list of all activity IDs.
            train_data: A mapping from activity ID to a 2D numpy array of shape (n_samples, n_variables)
                containing the training data for each activity.
            src: The index of the source variable.
            dst: The index of the target variable.
            lag: The time lag between the source and target variables.
            strength: A proxy for the strength of the candidate rule, used for ranking.
            sign: The expected sign of the relationship (1 for positive, -1 for negative, None for unknown).
            rule_origin: A string indicating the origin of the rule (e.g., "graph" or "fallback_mined").
            min_condition_count: The minimum number of samples where the source variable
                is active to consider the candidate valid.

        Returns:
            A dictionary representing the candidate rule with various statistics, or None if the candidate is invalid.

        """
        own_hits, own_count = _rule_hits(train_data[activity_id], src, dst, lag, target_state=1)
        if own_count < min_condition_count:
            return None

        p_true_train = _laplace(own_hits, own_count)
        other_ps = [
            _laplace(*_rule_hits(train_data[oid], src, dst, lag, target_state=1))
            for oid in activity_ids
            if oid != activity_id
        ]
        p_true_other = float(np.mean(other_ps)) if other_ps else 0.5

        if sign is None:
            target_state = 1 if p_true_train >= p_true_other else 0
            derived_sign = 1 if target_state == 1 else -1
        else:
            target_state = int(sign >= 0)
            derived_sign = int(sign if sign != 0 else 1)

        p_train = p_true_train if target_state == 1 else 1.0 - p_true_train
        p_other = p_true_other if target_state == 1 else 1.0 - p_true_other
        delta = p_train - p_other
        base_strength = max(0.03, float(strength))
        weight = base_strength * max(0.0, delta) * math.log1p(own_count)

        direction = "true" if target_state == 1 else "false"
        sign_char = "+" if derived_sign >= 0 else "-"

        return Rule(
            activity_id=activity_id,
            activity_name=self.__activity_names[activity_id],
            src=src,
            src_var=self.__var_names[src],
            dst=dst,
            dst_var=self.__var_names[dst],
            lag=lag,
            sign=1 if derived_sign >= 0 else -1,
            strength=base_strength,
            target_dst_state=target_state,
            p_target_train=p_train,
            p_target_other=p_other,
            delta=delta,
            n_condition_train=own_count,
            weight=weight,
            rule_origin=rule_origin,
            rule_text=(
                f"IF {self.__var_names[src]}[t-{lag}] is true THEN "
                f"{self.__var_names[dst]}[t] is likely {direction} "
                f"(sign={sign_char}, p_train={p_train:.3f}, "
                f"p_other={p_other:.3f}, strength={base_strength:.3f})"
            ),
        )

    def __allocate_budgets(
        self,
        candidates_by_activity: dict[int, list[Rule]],
        edges_by_activity: dict[int, list[CausalEdge]],
    ) -> dict[int, int]:
        """Allocate a rule budget for each activity based on the number of candidates and graph edges.

        Arguments:
            candidates_by_activity: A mapping from activity ID to a list of candidate rules for that activity.
            edges_by_activity: A mapping from activity ID to a list of causal edges inferred for that activity.

        Returns:
            A dictionary mapping each activity ID to an allocated rule budget (integer) for that activity.

        """
        config = self.__budget_cfg
        counts = np.array([len(c) for c in candidates_by_activity.values()], dtype=np.float64)
        non_zero = counts[counts > 0]
        ref_count = float(np.median(non_zero)) if non_zero.size else 1.0
        max_budget = max(config.top_rules_per_activity, round(config.top_rules_per_activity * config.max_rules_scale))

        budgets: dict[int, int] = {}
        for activity_id, candidates in candidates_by_activity.items():
            count = len(candidates)
            if count <= 0:
                budgets[activity_id] = 0
                continue
            scale = math.sqrt(count / max(ref_count, 1.0))
            raw = round(config.top_rules_per_activity * scale)
            budget = max(config.min_rules_per_activity, min(max_budget, raw))
            budget = min(count, budget)

            if len(edges_by_activity[activity_id]) <= config.low_edge_threshold:
                low_cap = max(
                    config.min_rules_per_activity,
                    round(config.top_rules_per_activity * config.low_edge_budget_ratio),
                )
                budget = min(budget, low_cap)
            budgets[activity_id] = budget
        return budgets

    def __select_with_pruning(self, candidates: list[Rule], budget: int) -> list[Rule]:
        """Select rules from candidates using a pruning strategy that balances strength, diversity, and origin.

        Arguments:
            candidates: A list of candidate rules to select from.
            budget: The maximum number of rules to select.

        Returns:
            A list of selected rules, annotated with hierarchy and normalized scores.

        """
        config = self.__selection_cfg
        if budget <= 0 or not candidates:
            return []

        # Deduplication: keep best rule per (src, dst, lag, target) key.
        best_by_key: dict[tuple, Rule] = {}
        for rule in candidates:
            key = (rule.src, rule.dst, rule.lag, rule.target_dst_state)
            if key not in best_by_key or rule.weight > best_by_key[key].weight:
                best_by_key[key] = rule
        deduped = list(best_by_key.values())

        positive = [r for r in deduped if r.delta >= config.min_delta]
        pool = sorted(
            positive if len(positive) >= max(3, budget // 2) else deduped,
            key=lambda x: (x.weight, x.delta, x.strength),
            reverse=True,
        )

        selected: list[Rule] = []
        selected_keys: set[tuple] = set()
        source_lag_counts: dict[tuple, int] = defaultdict(int)
        pair_counts: dict[tuple, int] = defaultdict(int)

        while pool and len(selected) < budget:
            best_idx, best_score = None, float("-inf")
            for idx, rule in enumerate(pool):
                src, dst, lag = rule.src, rule.dst, rule.lag
                target = rule.target_dst_state
                key = (src, dst, lag, target)
                if key in selected_keys or (src, dst, lag, 1 - target) in selected_keys:
                    continue
                base = float(rule.weight) or 1e-6 + 0.15 * max(0.0, rule.delta) + 0.05 * rule.strength
                diversity = 1.0 / (1.0 + 0.65 * source_lag_counts[(src, lag)] + 0.85 * pair_counts[(src, dst)])
                origin_f = 1.0 if rule.rule_origin == "graph" else 0.93
                relaxed_f = 1.0 if rule.delta >= config.min_delta else 0.6
                score = base * diversity * origin_f * relaxed_f
                if score > best_score:
                    best_score, best_idx = score, idx

            if best_idx is None:
                break

            chosen = replace(pool.pop(best_idx), selection_score=best_score)
            selected.append(chosen)

            src, dst, lag, target = chosen.src, chosen.dst, chosen.lag, chosen.target_dst_state
            selected_keys.add((src, dst, lag, target))
            source_lag_counts[(src, lag)] += 1
            pair_counts[(src, dst)] += 1

        if not selected:
            return []

        return self._annotate_hierarchy(selected, config.core_fraction)

    def __select_baseline(self, candidates: list[Rule], budget: int) -> list[Rule]:
        """Select rules from candidates using a simple baseline strategy based on strength and origin.

        Arguments:
            candidates: A list of candidate rules to select from.
            budget: The maximum number of rules to select.

        Returns:
            A list of selected rules, annotated with hierarchy and normalized scores.

        """
        config = self.__selection_cfg
        if budget <= 0 or not candidates:
            return []
        positive = sorted(
            [r for r in candidates if r.delta >= config.min_delta],
            key=lambda x: x.weight,
            reverse=True,
        )
        selected = (
            positive[:budget]
            if len(positive) >= min(3, budget)
            else sorted(candidates, key=lambda x: (x.weight, x.strength), reverse=True)[:budget]
        )

        total = float(sum(max(1e-8, r.weight) for r in selected))
        updated = []
        for rule in selected:
            w = max(1e-8, float(rule.weight))
            updated.append(
                replace(
                    rule,
                    selection_score=w,
                    is_core=True,
                    hierarchy="core",
                    normalized_weight=w / total if total > 0 else 0.0,
                    normalized_core_weight=w / total if total > 0 else 0.0,
                    normalized_tie_weight=0.0,
                ),
            )
        return updated

    @staticmethod
    def _annotate_hierarchy(selected: list[Rule], core_fraction: float) -> list[Rule]:
        selected.sort(
            key=lambda x: float(x.selection_score if x.selection_score is not None else x.weight),
            reverse=True,
        )
        core_count = max(1, min(len(selected), round(len(selected) * core_fraction)))
        total_w = float(sum(max(1e-8, r.weight) for r in selected))
        core_w = float(sum(max(1e-8, r.weight) for r in selected[:core_count]))
        tie_w = float(sum(max(1e-8, r.weight) for r in selected[core_count:]))

        updated = []
        for idx, rule in enumerate(selected):
            w = max(1e-8, float(rule.weight))
            is_core = idx < core_count
            updated.append(
                replace(
                    rule,
                    is_core=is_core,
                    hierarchy="core" if is_core else "tie_breaker",
                    normalized_weight=w / total_w if total_w > 0 else 0.0,
                    normalized_core_weight=w / core_w if is_core and core_w > 0 else 0.0,
                    normalized_tie_weight=w / tie_w if not is_core and tie_w > 0 else 0.0,
                ),
            )
        return updated

    def __build_model(
        self,
        activity_id: int,
        selected: list[Rule],
        edges: list[CausalEdge],
        train_data: dict[int, np.ndarray],
        total_train: float,
        budget: int,
    ) -> ActivityModel:
        """Build activity model containing metadata and prototype information based on selected rules and training data.

        Arguments:
            activity_id: The ID of the activity to build the model for.
            selected: A list of selected rules for the activity.
            edges: A list of causal edges inferred for the activity.
            train_data: A mapping from activity ID to a 2D numpy array of shape (n_samples, n_variables)
                containing the training data for each activity.
            total_train: The total number of training samples across all activities, used for calculating priors.
            budget: The allocated rule budget for the activity, used for metadata.

        Returns:
            A dictionary representing the activity model with metadata and prototype information.

        """
        config = self.__selection_cfg
        prior = len(train_data[activity_id]) / total_train
        prototype_array = np.asarray(np.mean(train_data[activity_id], axis=0, dtype=np.float64), dtype=np.float64)
        prototype_true_rate: list[float] = [float(x) for x in np.ravel(prototype_array)]
        n_graph = sum(1 for r in selected if r.rule_origin == "graph")
        use_proto = len(selected) == 0 or len(edges) <= self.__budget_cfg.low_edge_threshold

        return ActivityModel(
            activity_id=activity_id,
            activity_name=self.__activity_names[activity_id],
            rule_budget=budget,
            n_rules=len(selected),
            n_graph_rules=n_graph,
            n_fallback_rules=len(selected) - n_graph,
            core_rule_count=sum(1 for r in selected if r.is_core),
            tie_rule_count=sum(1 for r in selected if not r.is_core),
            edge_count=len(edges),
            prior=prior,
            log_prior=math.log(max(prior, 1e-12)),
            score_bias=0.0,
            min_margin=config.min_margin,
            tie_breaker_weight=config.tie_breaker_weight,
            coverage_weight=config.coverage_weight,
            prior_weight=config.prior_weight,
            prototype_weight=config.prototype_weight,
            use_prototype_fallback=use_proto,
            prototype_true_rate=prototype_true_rate,
        )
