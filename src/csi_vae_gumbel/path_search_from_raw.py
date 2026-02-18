from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tigramite import plotting as tp

from csi_vae_gumbel.causal_analysis_full import (
    CausalEdge,
    build_activity_signatures,
    edge_strength_map,
    enumerate_best_paths,
    extract_edges,
    filter_graph,
    load_latent_hards,
    path_to_strings,
    save_confusion_matrix_plot,
    select_unique_paths,
    split_by_activity,
    split_train_test_by_activity,
)
from csi_vae_gumbel.settings import Settings


def save_edges_csv(edges: list[CausalEdge], output_path: Path) -> None:
    """Write edge list to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["src", "dst", "lag", "strength", "sign", "mark"],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow(asdict(edge))


def rule_target_state(sign: int) -> int:
    """Map edge sign to expected destination state when source is true."""
    return 1 if sign >= 0 else 0


def rule_hits_and_count(
    data: np.ndarray,
    src: int,
    dst: int,
    lag: int,
    target_state: int,
) -> tuple[int, int]:
    """Count target hits under condition src[t-lag] == 1."""
    if lag <= 0 or len(data) <= lag:
        return 0, 0
    src_values = data[:-lag, src]
    dst_values = data[lag:, dst]
    mask = src_values == 1
    count = int(np.sum(mask))
    if count == 0:
        return 0, 0
    hits = int(np.sum(dst_values[mask] == target_state))
    return hits, count


def smoothed_probability(hits: int, count: int, alpha: float = 1.0) -> float:
    """Laplace-smoothed Bernoulli probability."""
    return float((hits + alpha) / (count + 2 * alpha))


def build_rule_candidate(
    *,
    settings: Settings,
    var_names: list[str],
    activity_id: int,
    activity_ids: list[int],
    train_data_by_activity: dict[int, np.ndarray],
    src: int,
    dst: int,
    lag: int,
    strength: float,
    sign: int | None,
    rule_origin: str,
    min_condition_count: int,
) -> dict[str, Any] | None:
    """Create one candidate symbolic rule from either a causal edge or fallback relation."""
    own_data = train_data_by_activity[activity_id]
    own_hits_true, own_count = rule_hits_and_count(
        data=own_data,
        src=src,
        dst=dst,
        lag=lag,
        target_state=1,
    )
    if own_count < min_condition_count:
        return None

    p_true_train = smoothed_probability(own_hits_true, own_count)
    other_true_ps: list[float] = []
    for other_id in activity_ids:
        if other_id == activity_id:
            continue
        hits_o, count_o = rule_hits_and_count(
            data=train_data_by_activity[other_id],
            src=src,
            dst=dst,
            lag=lag,
            target_state=1,
        )
        other_true_ps.append(smoothed_probability(hits_o, count_o))
    p_true_other = float(np.mean(other_true_ps)) if other_true_ps else 0.5

    if sign is None:
        target_state = 1 if p_true_train >= p_true_other else 0
        derived_sign = 1 if target_state == 1 else -1
    else:
        target_state = rule_target_state(sign)
        derived_sign = int(sign if sign != 0 else 1)

    p_train = p_true_train if target_state == 1 else 1.0 - p_true_train
    p_other = p_true_other if target_state == 1 else 1.0 - p_true_other
    delta = p_train - p_other

    # Keep scores positive for ranking while still retaining signed deltas in metadata.
    delta_gain = max(0.0, delta)
    support_gain = math.log1p(own_count)
    base_strength = max(0.03, float(strength))
    weight = base_strength * delta_gain * support_gain
    direction = "true" if target_state == 1 else "false"

    return {
        "activity_id": activity_id,
        "activity_name": settings.activities[activity_id],
        "src": int(src),
        "src_var": var_names[src],
        "dst": int(dst),
        "dst_var": var_names[dst],
        "lag": int(lag),
        "sign": int(1 if derived_sign >= 0 else -1),
        "strength": float(base_strength),
        "target_dst_state": int(target_state),
        "p_target_train": float(p_train),
        "p_target_other": float(p_other),
        "delta": float(delta),
        "n_condition_train": int(own_count),
        "weight": float(weight),
        "rule_origin": str(rule_origin),
        "rule_text": (
            f"IF {var_names[src]}[t-{lag}] is true THEN "
            f"{var_names[dst]}[t] is likely {direction} "
            f"(sign={'+' if derived_sign >= 0 else '-'}, "
            f"p_train={p_train:.3f}, p_other={p_other:.3f}, strength={base_strength:.3f})"
        ),
    }


def mine_low_edge_rule_candidates(
    *,
    settings: Settings,
    var_names: list[str],
    activity_id: int,
    activity_ids: list[int],
    train_data_by_activity: dict[int, np.ndarray],
    tau_max: int,
    top_source_vars: int,
    top_target_vars: int,
    min_condition_count: int,
    min_delta: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Mine symbolic fallback rules directly from train dynamics for low-edge activities."""
    own_data = train_data_by_activity[activity_id]
    other_data = np.concatenate(
        [train_data_by_activity[other_id] for other_id in activity_ids if other_id != activity_id],
        axis=0,
    )

    own_true_rate = np.mean(own_data, axis=0, dtype=np.float64)
    other_true_rate = np.mean(other_data, axis=0, dtype=np.float64)
    discriminative_target_strength = np.abs(own_true_rate - other_true_rate)

    src_counts = np.sum(own_data, axis=0)
    source_order = np.argsort(-src_counts)
    target_order = np.argsort(-discriminative_target_strength)

    selected_sources = [int(idx) for idx in source_order[: max(1, top_source_vars)]]
    selected_targets = [int(idx) for idx in target_order[: max(1, top_target_vars)]]

    candidates: list[dict[str, Any]] = []
    for lag, src, dst in product(range(1, tau_max + 1), selected_sources, selected_targets):
        if src_counts[src] < min_condition_count:
            continue

        # Proxy strength rewards variables whose marginals differ across activities.
        strength_proxy = 0.08 + float(discriminative_target_strength[dst])
        candidate = build_rule_candidate(
            settings=settings,
            var_names=var_names,
            activity_id=activity_id,
            activity_ids=activity_ids,
            train_data_by_activity=train_data_by_activity,
            src=src,
            dst=dst,
            lag=lag,
            strength=strength_proxy,
            sign=None,
            rule_origin="fallback_mined",
            min_condition_count=min_condition_count,
        )
        if candidate is None:
            continue
        if candidate["delta"] <= min_delta:
            continue
        candidates.append(candidate)

    ranked = sorted(candidates, key=lambda x: (x["weight"], x["delta"], x["strength"]), reverse=True)
    return ranked[:max_candidates]


def allocate_activity_rule_budgets(
    *,
    candidates_by_activity: dict[int, list[dict[str, Any]]],
    top_rules_per_activity: int,
    min_rules_per_activity: int,
    max_rules_scale: float,
) -> dict[int, int]:
    """Allocate per-activity rule budgets from candidate pool sizes."""
    counts = np.array([len(candidates) for candidates in candidates_by_activity.values()], dtype=np.float64)
    non_zero_counts = counts[counts > 0]
    ref_count = float(np.median(non_zero_counts)) if non_zero_counts.size else 1.0

    max_rules = max(top_rules_per_activity, int(round(top_rules_per_activity * max_rules_scale)))
    budgets: dict[int, int] = {}
    for activity_id, candidates in candidates_by_activity.items():
        count = len(candidates)
        if count <= 0:
            budgets[activity_id] = 0
            continue

        scale = math.sqrt(count / max(ref_count, 1.0))
        scaled_budget = int(round(top_rules_per_activity * scale))
        clipped_budget = max(min_rules_per_activity, min(max_rules, scaled_budget))
        budgets[activity_id] = min(count, clipped_budget)
    return budgets


def select_rules_with_pruning(
    *,
    candidates: list[dict[str, Any]],
    budget: int,
    min_delta: float,
    core_fraction: float,
) -> list[dict[str, Any]]:
    """Select rules with dedupe, conflict pruning, and diversity-aware ranking."""
    if budget <= 0 or not candidates:
        return []

    # Deduplicate exact semantic rules first.
    best_by_key: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for rule in candidates:
        key = (int(rule["src"]), int(rule["dst"]), int(rule["lag"]), int(rule["target_dst_state"]))
        current = best_by_key.get(key)
        if current is None or float(rule["weight"]) > float(current["weight"]):
            best_by_key[key] = rule

    deduped = list(best_by_key.values())
    positive = [rule for rule in deduped if float(rule["delta"]) >= min_delta]
    if len(positive) >= max(3, budget // 2):
        pool = sorted(positive, key=lambda x: (x["weight"], x["delta"], x["strength"]), reverse=True)
    else:
        pool = sorted(deduped, key=lambda x: (x["weight"], x["delta"], x["strength"]), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int, int, int]] = set()
    source_lag_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    while pool and len(selected) < budget:
        best_idx: int | None = None
        best_score = float("-inf")

        for idx, rule in enumerate(pool):
            src = int(rule["src"])
            dst = int(rule["dst"])
            lag = int(rule["lag"])
            target = int(rule["target_dst_state"])
            key = (src, dst, lag, target)
            conflict_key = (src, dst, lag, 1 - target)

            if key in selected_keys or conflict_key in selected_keys:
                continue

            base = float(rule["weight"])
            if base <= 0:
                base = 1e-6 + 0.15 * max(0.0, float(rule["delta"])) + 0.05 * float(rule["strength"])

            diversity_penalty = 1.0 / (1.0 + 0.65 * source_lag_counts[(src, lag)] + 0.85 * pair_counts[(src, dst)])
            origin_factor = 1.0 if rule.get("rule_origin") == "graph" else 0.93
            relaxed_factor = 1.0 if float(rule["delta"]) >= min_delta else 0.6
            selection_score = base * diversity_penalty * origin_factor * relaxed_factor

            if selection_score > best_score:
                best_score = selection_score
                best_idx = idx

        if best_idx is None:
            break

        chosen = pool.pop(best_idx)
        chosen = dict(chosen)
        chosen["selection_score"] = float(best_score)
        selected.append(chosen)

        src = int(chosen["src"])
        dst = int(chosen["dst"])
        lag = int(chosen["lag"])
        target = int(chosen["target_dst_state"])
        selected_keys.add((src, dst, lag, target))
        source_lag_counts[(src, lag)] += 1
        pair_counts[(src, dst)] += 1

    if not selected:
        return []

    selected.sort(key=lambda x: float(x.get("selection_score", x["weight"])), reverse=True)
    core_count = max(1, min(len(selected), int(round(len(selected) * core_fraction))))

    total_weight = float(sum(max(1e-8, float(rule["weight"])) for rule in selected))
    core_weight_sum = float(sum(max(1e-8, float(rule["weight"])) for rule in selected[:core_count]))
    tie_weight_sum = float(sum(max(1e-8, float(rule["weight"])) for rule in selected[core_count:]))

    for idx, rule in enumerate(selected):
        weight = max(1e-8, float(rule["weight"]))
        is_core = idx < core_count
        rule["is_core"] = bool(is_core)
        rule["hierarchy"] = "core" if is_core else "tie_breaker"
        rule["normalized_weight"] = float(weight / total_weight) if total_weight > 0 else 0.0
        rule["normalized_core_weight"] = float(weight / core_weight_sum) if is_core and core_weight_sum > 0 else 0.0
        rule["normalized_tie_weight"] = float(weight / tie_weight_sum) if (not is_core and tie_weight_sum > 0) else 0.0
    return selected


def select_rules_baseline(
    *,
    candidates: list[dict[str, Any]],
    budget: int,
    min_delta: float,
) -> list[dict[str, Any]]:
    """Replicate the original top-weight rule selection for graph-only activities."""
    if budget <= 0 or not candidates:
        return []

    positive = [rule for rule in candidates if float(rule["delta"]) >= min_delta]
    selected = sorted(positive, key=lambda x: float(x["weight"]), reverse=True)
    if len(selected) < min(3, budget):
        fallback = sorted(
            candidates,
            key=lambda x: (float(x["weight"]), float(x["strength"])),
            reverse=True,
        )
        selected = fallback[:budget]
    else:
        selected = selected[:budget]

    if not selected:
        return []

    total_weight = float(sum(max(1e-8, float(rule["weight"])) for rule in selected))
    for rule in selected:
        weight = max(1e-8, float(rule["weight"]))
        rule["selection_score"] = float(weight)
        rule["is_core"] = True
        rule["hierarchy"] = "core"
        rule["normalized_weight"] = float(weight / total_weight) if total_weight > 0 else 0.0
        rule["normalized_core_weight"] = float(weight / total_weight) if total_weight > 0 else 0.0
        rule["normalized_tie_weight"] = 0.0
    return selected


def build_rule_library(
    *,
    settings: Settings,
    var_names: list[str],
    edges_by_activity: dict[int, list[CausalEdge]],
    train_data_by_activity: dict[int, np.ndarray],
    top_rules_per_activity: int,
    min_delta: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    """Create deterministic symbolic rules and per-activity classifier metadata."""
    return build_rule_library_advanced(
        settings=settings,
        var_names=var_names,
        edges_by_activity=edges_by_activity,
        train_data_by_activity=train_data_by_activity,
        top_rules_per_activity=top_rules_per_activity,
        min_delta=min_delta,
        tau_max=5,
        min_rules_per_activity=6,
        low_edge_threshold=3,
        top_fallback_source_vars=8,
        top_fallback_target_vars=12,
        min_condition_count=8,
        fallback_min_delta=0.05,
        fallback_pool_size=300,
        core_fraction=0.6,
        tie_breaker_weight=0.25,
        coverage_weight=0.2,
        prior_weight=0.1,
        prototype_weight=0.25,
        min_margin=0.02,
        max_rules_scale=1.6,
        low_edge_budget_ratio=0.3,
    )


def build_rule_library_advanced(
    *,
    settings: Settings,
    var_names: list[str],
    edges_by_activity: dict[int, list[CausalEdge]],
    train_data_by_activity: dict[int, np.ndarray],
    top_rules_per_activity: int,
    min_delta: float,
    tau_max: int,
    min_rules_per_activity: int,
    low_edge_threshold: int,
    top_fallback_source_vars: int,
    top_fallback_target_vars: int,
    min_condition_count: int,
    fallback_min_delta: float,
    fallback_pool_size: int,
    core_fraction: float,
    tie_breaker_weight: float,
    coverage_weight: float,
    prior_weight: float,
    prototype_weight: float,
    min_margin: float,
    max_rules_scale: float,
    low_edge_budget_ratio: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    """Create sign-aware deterministic rules with adaptive budgets and low-edge fallback mining."""
    activity_ids = sorted(edges_by_activity)
    library: dict[int, list[dict[str, Any]]] = {}
    activity_models: dict[int, dict[str, Any]] = {}

    candidates_by_activity: dict[int, list[dict[str, Any]]] = {}
    total_train_points = float(sum(len(train_data_by_activity[activity_id]) for activity_id in activity_ids))
    if total_train_points <= 0:
        total_train_points = 1.0

    for activity_id in activity_ids:
        graph_candidates: list[dict[str, Any]] = []
        for edge in edges_by_activity[activity_id]:
            candidate = build_rule_candidate(
                settings=settings,
                var_names=var_names,
                activity_id=activity_id,
                activity_ids=activity_ids,
                train_data_by_activity=train_data_by_activity,
                src=edge.src,
                dst=edge.dst,
                lag=edge.lag,
                strength=edge.strength,
                sign=edge.sign,
                rule_origin="graph",
                min_condition_count=1,
            )
            if candidate is not None:
                graph_candidates.append(candidate)

        low_edge_mode = len(edges_by_activity[activity_id]) <= low_edge_threshold
        fallback_candidates: list[dict[str, Any]] = []
        if low_edge_mode:
            fallback_candidates = mine_low_edge_rule_candidates(
                settings=settings,
                var_names=var_names,
                activity_id=activity_id,
                activity_ids=activity_ids,
                train_data_by_activity=train_data_by_activity,
                tau_max=tau_max,
                top_source_vars=top_fallback_source_vars,
                top_target_vars=top_fallback_target_vars,
                min_condition_count=min_condition_count,
                min_delta=fallback_min_delta,
                max_candidates=fallback_pool_size,
            )

        candidates_by_activity[activity_id] = graph_candidates + fallback_candidates

    budgets = allocate_activity_rule_budgets(
        candidates_by_activity=candidates_by_activity,
        top_rules_per_activity=top_rules_per_activity,
        min_rules_per_activity=min_rules_per_activity,
        max_rules_scale=max_rules_scale,
    )
    for activity_id in activity_ids:
        if len(edges_by_activity[activity_id]) <= low_edge_threshold:
            low_edge_cap = max(min_rules_per_activity, int(round(top_rules_per_activity * low_edge_budget_ratio)))
            budgets[activity_id] = min(budgets[activity_id], low_edge_cap)

    for activity_id in activity_ids:
        candidates = candidates_by_activity[activity_id]
        has_fallback_candidates = any(rule.get("rule_origin") != "graph" for rule in candidates)
        if has_fallback_candidates:
            selected = select_rules_with_pruning(
                candidates=candidates,
                budget=budgets[activity_id],
                min_delta=min_delta,
                core_fraction=core_fraction,
            )
        else:
            graph_budget = min(len(candidates), top_rules_per_activity)
            selected = select_rules_baseline(
                candidates=candidates,
                budget=graph_budget,
                min_delta=min_delta,
            )
        library[activity_id] = selected

        n_graph_rules = int(sum(1 for rule in selected if rule.get("rule_origin") == "graph"))
        n_fallback_rules = int(sum(1 for rule in selected if rule.get("rule_origin") != "graph"))
        use_prototype = bool(len(selected) == 0 or len(edges_by_activity[activity_id]) <= low_edge_threshold)

        prior = float(len(train_data_by_activity[activity_id]) / total_train_points)
        prototype = np.mean(train_data_by_activity[activity_id], axis=0, dtype=np.float64)
        activity_models[activity_id] = {
            "activity_id": int(activity_id),
            "activity_name": settings.activities[activity_id],
            "rule_budget": int(budgets[activity_id]),
            "n_rules": int(len(selected)),
            "n_graph_rules": int(n_graph_rules),
            "n_fallback_rules": int(n_fallback_rules),
            "core_rule_count": int(sum(1 for rule in selected if bool(rule.get("is_core")))),
            "tie_rule_count": int(sum(1 for rule in selected if not bool(rule.get("is_core")))),
            "edge_count": int(len(edges_by_activity[activity_id])),
            "prior": float(prior),
            "log_prior": float(math.log(max(prior, 1e-12))),
            "score_bias": 0.0,
            "min_margin": float(min_margin),
            "tie_breaker_weight": float(tie_breaker_weight),
            "coverage_weight": float(coverage_weight),
            "prior_weight": float(prior_weight),
            "prototype_weight": float(prototype_weight),
            "use_prototype_fallback": bool(use_prototype),
            "prototype_true_rate": [float(value) for value in prototype.tolist()],
        }

    return library, activity_models


def score_sequence_with_rules(
    sequence: np.ndarray,
    rules: list[dict[str, Any]],
    eps: float = 1e-6,
    missing_rule_score: float = 0.0,
) -> dict[str, float]:
    """Compute hierarchical rule scores with coverage diagnostics."""
    raw_score = 0.0
    raw_used_weight = 0.0
    core_score = 0.0
    core_used_weight = 0.0
    tie_score = 0.0
    tie_used_weight = 0.0
    matched_rules = 0
    matched_core_rules = 0
    matched_tie_rules = 0

    if not rules:
        return {
            "raw_rule_score": float(missing_rule_score),
            "core_score": float(missing_rule_score),
            "tie_score": 0.0,
            "coverage": 0.0,
            "matched_rules": 0.0,
            "matched_core_rules": 0.0,
            "matched_tie_rules": 0.0,
        }

    for rule in rules:
        hits, count = rule_hits_and_count(
            data=sequence,
            src=int(rule["src"]),
            dst=int(rule["dst"]),
            lag=int(rule["lag"]),
            target_state=int(rule["target_dst_state"]),
        )
        if count == 0:
            continue

        p_a = float(np.clip(rule["p_target_train"], eps, 1.0 - eps))
        p_b = float(np.clip(rule["p_target_other"], eps, 1.0 - eps))
        llr = hits * math.log(p_a / p_b) + (count - hits) * math.log((1.0 - p_a) / (1.0 - p_b))
        weight = max(0.0, float(rule.get("normalized_weight", 0.0)))
        raw_score += weight * llr
        raw_used_weight += weight
        matched_rules += 1

        if bool(rule.get("is_core")):
            core_weight = max(0.0, float(rule.get("normalized_core_weight", 0.0)))
            core_score += core_weight * llr
            core_used_weight += core_weight
            matched_core_rules += 1
        else:
            tie_weight = max(0.0, float(rule.get("normalized_tie_weight", 0.0)))
            tie_score += tie_weight * llr
            tie_used_weight += tie_weight
            matched_tie_rules += 1

    coverage = float(min(1.0, raw_used_weight))
    if raw_used_weight <= 0:
        return {
            "raw_rule_score": float(missing_rule_score),
            "core_score": float(missing_rule_score),
            "tie_score": 0.0,
            "coverage": coverage,
            "matched_rules": float(matched_rules),
            "matched_core_rules": float(matched_core_rules),
            "matched_tie_rules": float(matched_tie_rules),
        }

    raw_llr = float(raw_score / raw_used_weight)
    core_llr = float(core_score / core_used_weight) if core_used_weight > 0 else raw_llr
    tie_llr = float(tie_score / tie_used_weight) if tie_used_weight > 0 else 0.0
    return {
        "raw_rule_score": raw_llr,
        "core_score": core_llr,
        "tie_score": tie_llr,
        "coverage": coverage,
        "matched_rules": float(matched_rules),
        "matched_core_rules": float(matched_core_rules),
        "matched_tie_rules": float(matched_tie_rules),
    }


def prototype_similarity_score(sequence: np.ndarray, prototype_true_rate: list[float]) -> float:
    """Simple bounded similarity between observed and activity prototype true-rates."""
    if sequence.size == 0 or not prototype_true_rate:
        return 0.0
    observed = np.mean(sequence, axis=0, dtype=np.float64)
    prototype = np.array(prototype_true_rate, dtype=np.float64)
    if observed.shape != prototype.shape:
        return 0.0
    return float(-np.mean(np.abs(observed - prototype)))


def combine_activity_score(
    *,
    rule_stats: dict[str, float],
    model: dict[str, Any],
    prototype_score: float,
    apply_bias: bool,
) -> float:
    """Combine rule, prior, and fallback components into one deterministic score."""
    score = float(rule_stats["core_score"])
    score += float(model.get("tie_breaker_weight", 0.0)) * float(rule_stats["tie_score"])
    score += float(model.get("coverage_weight", 0.0)) * float(rule_stats["coverage"])
    score += float(model.get("prior_weight", 0.0)) * float(model.get("log_prior", 0.0))

    if bool(model.get("use_prototype_fallback", False)):
        score += float(model.get("prototype_weight", 0.0)) * prototype_score

    if apply_bias:
        score += float(model.get("score_bias", 0.0))
    return float(score)


def score_sequence_for_all_activities(
    *,
    sequence: np.ndarray,
    rule_library: dict[int, list[dict[str, Any]]],
    activity_models: dict[int, dict[str, Any]] | None,
    apply_bias: bool,
) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    """Score one sequence for all activities and return diagnostics."""
    scores: dict[int, float] = {}
    diagnostics: dict[int, dict[str, Any]] = {}

    for activity_id, rules in rule_library.items():
        model = activity_models.get(activity_id, {}) if activity_models is not None else {}
        rule_stats = score_sequence_with_rules(sequence=sequence, rules=rules)
        prototype_score = prototype_similarity_score(
            sequence=sequence,
            prototype_true_rate=model.get("prototype_true_rate", []),
        )
        final_score = combine_activity_score(
            rule_stats=rule_stats,
            model=model,
            prototype_score=prototype_score,
            apply_bias=apply_bias,
        )
        diagnostics[activity_id] = {
            **rule_stats,
            "prototype_score": float(prototype_score),
            "final_score": float(final_score),
        }
        scores[activity_id] = float(final_score)
    return scores, diagnostics


def calibrate_activity_models(
    *,
    rule_library: dict[int, list[dict[str, Any]]],
    activity_models: dict[int, dict[str, Any]],
    train_data_by_activity: dict[int, np.ndarray],
    segment_length: int,
    segment_hop: int,
    max_segments_per_activity: int | None,
    calibration_clip: float,
) -> None:
    """Calibrate class-specific score biases from train segments."""
    segments = build_segments(
        labelled_data=train_data_by_activity,
        segment_length=segment_length,
        segment_hop=segment_hop,
        max_segments_per_activity=max_segments_per_activity,
    )
    if not segments:
        return

    score_buckets: dict[int, dict[str, list[float]]] = {
        activity_id: {"own": [], "other": []}
        for activity_id in rule_library
    }
    for sequence, true_label in segments:
        scores, _ = score_sequence_for_all_activities(
            sequence=sequence,
            rule_library=rule_library,
            activity_models=activity_models,
            apply_bias=False,
        )
        for activity_id, value in scores.items():
            bucket = "own" if activity_id == true_label else "other"
            score_buckets[activity_id][bucket].append(float(value))

    for activity_id, buckets in score_buckets.items():
        own_scores = np.array(buckets["own"], dtype=np.float64)
        other_scores = np.array(buckets["other"], dtype=np.float64)
        if own_scores.size == 0 or other_scores.size == 0:
            continue

        own_median = float(np.median(own_scores))
        other_median = float(np.median(other_scores))
        midpoint = 0.5 * (own_median + other_median)
        separation = own_median - other_median
        score_bias = float(np.clip(-midpoint, -calibration_clip, calibration_clip))

        model = activity_models[activity_id]
        model["score_bias"] = score_bias
        # Increase confidence margin if calibration indicates weak separation.
        if separation < 0.05:
            model["min_margin"] = float(max(float(model.get("min_margin", 0.0)), 0.05))
        model["calibration"] = {
            "own_median": own_median,
            "other_median": other_median,
            "separation": separation,
        }


def classify_sequence_with_rule_library(
    sequence: np.ndarray,
    rule_library: dict[int, list[dict[str, Any]]],
    activity_models: dict[int, dict[str, Any]] | None = None,
    return_details: bool = False,
) -> tuple[int, dict[int, float]] | tuple[int, dict[int, float], dict[int, dict[str, Any]]]:
    """Predict activity for one latent sequence window."""
    scores, diagnostics = score_sequence_for_all_activities(
        sequence=sequence,
        rule_library=rule_library,
        activity_models=activity_models,
        apply_bias=True,
    )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    prediction = ranked[0][0]

    if len(ranked) > 1 and activity_models is not None:
        runner_up = ranked[1][0]
        margin = float(ranked[0][1] - ranked[1][1])
        required_margin = float(
            max(
                activity_models.get(prediction, {}).get("min_margin", 0.0),
                activity_models.get(runner_up, {}).get("min_margin", 0.0),
            ),
        )
        if margin < required_margin:
            # Tie-break with strongest core evidence and coverage.
            tie_candidates = [prediction, runner_up]
            prediction = max(
                tie_candidates,
                key=lambda activity_id: (
                    diagnostics[activity_id]["matched_core_rules"],
                    diagnostics[activity_id]["coverage"],
                    diagnostics[activity_id]["prototype_score"],
                    scores[activity_id],
                ),
            )

    if return_details:
        return prediction, scores, diagnostics
    return prediction, scores


def build_segments(
    *,
    labelled_data: dict[int, np.ndarray],
    segment_length: int,
    segment_hop: int,
    max_segments_per_activity: int | None,
) -> list[tuple[np.ndarray, int]]:
    """Create deterministic overlapping segments for each activity sequence."""
    segments: list[tuple[np.ndarray, int]] = []
    for label, data in labelled_data.items():
        if len(data) < segment_length:
            continue
        starts = list(range(0, len(data) - segment_length + 1, segment_hop))
        if max_segments_per_activity is not None and len(starts) > max_segments_per_activity:
            keep = np.linspace(0, len(starts) - 1, num=max_segments_per_activity, dtype=int)
            starts = [starts[idx] for idx in keep]
        for start in starts:
            segments.append((data[start : start + segment_length], label))
    return segments


def run_rule_based_classifier(
    *,
    settings: Settings,
    rule_library: dict[int, list[dict[str, Any]]],
    activity_models: dict[int, dict[str, Any]],
    test_data_by_activity: dict[int, np.ndarray],
    segment_length: int,
    segment_hop: int,
    max_segments_per_activity: int | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic classifier using extracted symbolic rules."""
    segments = build_segments(
        labelled_data=test_data_by_activity,
        segment_length=segment_length,
        segment_hop=segment_hop,
        max_segments_per_activity=max_segments_per_activity,
    )

    if not segments:
        msg = "No segments available for rule-based classification."
        raise RuntimeError(msg)

    y_true: list[int] = []
    y_pred: list[int] = []
    margins: list[float] = []
    for sequence, label in segments:
        prediction, scores, _ = classify_sequence_with_rule_library(
            sequence=sequence,
            rule_library=rule_library,
            activity_models=activity_models,
            return_details=True,
        )
        ranked = sorted(scores.values(), reverse=True)
        margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0
        margins.append(margin)
        y_true.append(label)
        y_pred.append(prediction)

    labels = sorted(rule_library)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # local import for speed

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[settings.activities[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_segments": len(segments),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "mean_prediction_margin": float(np.mean(margins)) if margins else 0.0,
        "median_prediction_margin": float(np.median(margins)) if margins else 0.0,
    }


def save_rule_library(
    *,
    output_dir: Path,
    rule_library: dict[int, list[dict[str, Any]]],
    activity_models: dict[int, dict[str, Any]],
    settings: Settings,
    metadata: dict[str, Any],
) -> None:
    """Save symbolic rules in JSON and human-readable text."""
    payload = {
        "metadata": metadata,
        "activities": {
            settings.activities[activity_id]: rules
            for activity_id, rules in rule_library.items()
        },
        "activity_models": {
            settings.activities[activity_id]: model
            for activity_id, model in activity_models.items()
        },
    }
    with (output_dir / "classification_rules.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    lines: list[str] = []
    for activity_id in sorted(rule_library):
        activity_name = settings.activities[activity_id]
        lines.append(f"[{activity_name}]")
        for idx, rule in enumerate(rule_library[activity_id], start=1):
            lines.append(f"{idx}. {rule['rule_text']}")
        lines.append("")
    (output_dir / "classification_rules.txt").write_text("\n".join(lines), encoding="utf-8")


def load_raw_graphs(raw_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load manifest and all per-activity raw graph JSON files."""
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        msg = f"Missing manifest: {manifest_path}"
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_text())

    by_activity: dict[int, dict[str, Any]] = {}
    for activity_name, info in manifest["activities"].items():
        raw_path = Path(info["raw_graph_json"])
        if not raw_path.is_absolute() and not raw_path.exists():
            raw_path = raw_dir / raw_path
        payload = json.loads(raw_path.read_text())
        by_activity[int(payload["activity_id"])] = {
            "activity_name": activity_name,
            "n_samples": int(payload["n_samples"]),
            "graph": np.array(payload["graph"], dtype=object),
            "val_matrix": np.array(payload["val_matrix"], dtype=np.float64),
        }
    return manifest, by_activity


def run_path_search_from_raw(
    *,
    settings: Settings,
    raw_dir: Path,
    filter_threshold: float = 0.1,
    path_threshold: float = 0.0,
    max_edges_per_path: int = 4,
    min_edges_per_path: int = 2,
    top_paths_per_activity: int = 150,
    top_rules_per_activity: int = 50,
    rule_min_delta: float = 0.0,
    min_rules_per_activity: int = 6,
    low_edge_threshold: int = 3,
    top_fallback_source_vars: int = 8,
    top_fallback_target_vars: int = 12,
    min_condition_count: int = 8,
    fallback_min_delta: float = 0.05,
    fallback_pool_size: int = 300,
    core_fraction: float = 0.6,
    tie_breaker_weight: float = 0.25,
    coverage_weight: float = 0.2,
    prior_weight: float = 0.1,
    prototype_weight: float = 0.25,
    min_margin: float = 0.02,
    max_rules_scale: float = 1.6,
    low_edge_budget_ratio: float = 0.3,
    calibration_clip: float = 2.0,
    classifier_stride: int = 75,
    classifier_train_ratio: float = 0.7,
    segment_length: int = 64,
    segment_hop: int = 12,
    max_segments_per_activity: int | None = None,
    save_figs: bool = True,
) -> dict[str, Any]:
    """Search discriminative paths from saved raw graphs and run deterministic classification."""
    manifest, raw_graphs = load_raw_graphs(raw_dir=raw_dir)
    tau_min = int(manifest["tau_min"])
    tau_max = int(manifest["tau_max"])
    var_names = list(manifest["var_names"])
    n_vars = len(var_names)
    use_full_onehot = manifest.get("latent_representation", "onehot_flat") == "onehot_flat"
    ind_test_name = str(manifest["independence_test"])

    latents_path = Path(settings.study_path) / "latents" / "latents_hard.npy"
    labels_path = Path(settings.study_path) / "latents" / "labels.npy"
    latents, labels, _, _, n_states = load_latent_hards(
        latent_path=latents_path,
        label_path=labels_path,
        use_full_onehot=use_full_onehot,
    )
    classifier_data = split_by_activity(latents=latents, labels=labels, stride=classifier_stride)
    classifier_train_data, classifier_test_data = split_train_test_by_activity(
        labelled_data=classifier_data,
        train_ratio=classifier_train_ratio,
        min_test_length=segment_length + tau_max,
    )

    out_dir = Path(settings.study_path) / "causal_path_search" / ind_test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_summary: dict[str, Any] = {
        "source_raw_dir": str(raw_dir),
        "independence_test": ind_test_name,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "filter_threshold": filter_threshold,
        "path_threshold": path_threshold,
        "classifier_stride": classifier_stride,
        "classifier_train_ratio": classifier_train_ratio,
        "segment_length": segment_length,
        "segment_hop": segment_hop,
        "top_rules_per_activity": top_rules_per_activity,
        "rule_min_delta": rule_min_delta,
        "min_rules_per_activity": min_rules_per_activity,
        "low_edge_threshold": low_edge_threshold,
        "top_fallback_source_vars": top_fallback_source_vars,
        "top_fallback_target_vars": top_fallback_target_vars,
        "min_condition_count": min_condition_count,
        "fallback_min_delta": fallback_min_delta,
        "fallback_pool_size": fallback_pool_size,
        "core_fraction": core_fraction,
        "tie_breaker_weight": tie_breaker_weight,
        "coverage_weight": coverage_weight,
        "prior_weight": prior_weight,
        "prototype_weight": prototype_weight,
        "min_margin": min_margin,
        "max_rules_scale": max_rules_scale,
        "low_edge_budget_ratio": low_edge_budget_ratio,
        "calibration_clip": calibration_clip,
        "n_states_per_variable": n_states,
        "n_latent_variables": n_vars,
        "activities": {},
    }

    strength_maps: dict[int, dict[tuple[int, int, int], float]] = {}
    candidate_paths: dict[int, list[tuple[tuple[tuple[int, int, int], ...], float]]] = {}
    edges_for_rules: dict[int, list[CausalEdge]] = {}

    for activity_id in sorted(raw_graphs):
        raw = raw_graphs[activity_id]
        activity_name = settings.activities[activity_id]
        raw_graph = raw["graph"]
        raw_val = raw["val_matrix"]

        filtered_graph, filtered_val = filter_graph(
            graph=raw_graph,
            val_matrix=raw_val,
            threshold=filter_threshold,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        path_graph, path_val = filter_graph(
            graph=raw_graph,
            val_matrix=raw_val,
            threshold=path_threshold,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        edges = extract_edges(filtered_graph=filtered_graph, filtered_val=filtered_val)
        path_edges = extract_edges(filtered_graph=path_graph, filtered_val=path_val)
        strengths = edge_strength_map(path_edges)
        candidates = enumerate_best_paths(
            edges=path_edges,
            n_vars=n_vars,
            max_edges=max_edges_per_path,
            min_edges=min_edges_per_path,
            top_k=top_paths_per_activity,
        )
        if not candidates:
            abs_values = np.abs(path_val[:, :, tau_min : tau_max + 1])
            src, dst, lag_offset = np.unravel_index(np.argmax(abs_values), abs_values.shape)
            lag = int(lag_offset + tau_min)
            fallback_edge = (int(src), int(dst), lag)
            fallback_score = float(abs_values[src, dst, lag_offset])
            candidates = [((fallback_edge,), fallback_score)]
            strengths[fallback_edge] = fallback_score

        strength_maps[activity_id] = strengths
        candidate_paths[activity_id] = candidates
        edges_for_rules[activity_id] = path_edges

        activity_dir = out_dir / activity_name.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)
        if save_figs:
            fig, _ = tp.plot_time_series_graph(
                figsize=(6, 6),
                val_matrix=filtered_val,
                graph=filtered_graph,
                var_names=var_names,
                link_colorbar_label="MCI Strength",
            )
            fig.tight_layout()
            fig.savefig(activity_dir / f"{activity_name}.pdf")
            plt.close(fig)

        np.save(activity_dir / "graph.npy", filtered_graph)
        np.save(activity_dir / "val_matrix.npy", filtered_val)
        with (activity_dir / "edges.json").open("w", encoding="utf-8") as file:
            json.dump([asdict(edge) for edge in edges], file, indent=2)
        save_edges_csv(edges=edges, output_path=activity_dir / "edges.csv")

        graph_summary["activities"][activity_name] = {
            "activity_id": activity_id,
            "n_samples_raw": raw["n_samples"],
            "n_edges": len(edges),
            "top_edges": [asdict(edge) for edge in edges[:20]],
            "top_paths": [
                {"path": path_to_strings(path), "score": score}
                for path, score in candidates[:10]
            ],
        }

    rule_library, activity_models = build_rule_library_advanced(
        settings=settings,
        var_names=var_names,
        edges_by_activity=edges_for_rules,
        train_data_by_activity=classifier_train_data,
        top_rules_per_activity=top_rules_per_activity,
        min_delta=rule_min_delta,
        tau_max=tau_max,
        min_rules_per_activity=min_rules_per_activity,
        low_edge_threshold=low_edge_threshold,
        top_fallback_source_vars=top_fallback_source_vars,
        top_fallback_target_vars=top_fallback_target_vars,
        min_condition_count=min_condition_count,
        fallback_min_delta=fallback_min_delta,
        fallback_pool_size=fallback_pool_size,
        core_fraction=core_fraction,
        tie_breaker_weight=tie_breaker_weight,
        coverage_weight=coverage_weight,
        prior_weight=prior_weight,
        prototype_weight=prototype_weight,
        min_margin=min_margin,
        max_rules_scale=max_rules_scale,
        low_edge_budget_ratio=low_edge_budget_ratio,
    )
    calibrate_activity_models(
        rule_library=rule_library,
        activity_models=activity_models,
        train_data_by_activity=classifier_train_data,
        segment_length=segment_length,
        segment_hop=segment_hop,
        max_segments_per_activity=max_segments_per_activity,
        calibration_clip=calibration_clip,
    )
    for activity_id, rules in rule_library.items():
        activity_name = settings.activities[activity_id]
        model = activity_models[activity_id]
        graph_summary["activities"][activity_name]["n_rules"] = len(rules)
        graph_summary["activities"][activity_name]["n_graph_rules"] = model["n_graph_rules"]
        graph_summary["activities"][activity_name]["n_fallback_rules"] = model["n_fallback_rules"]
        graph_summary["activities"][activity_name]["rule_budget"] = model["rule_budget"]
        graph_summary["activities"][activity_name]["core_rule_count"] = model["core_rule_count"]
        graph_summary["activities"][activity_name]["tie_rule_count"] = model["tie_rule_count"]
        graph_summary["activities"][activity_name]["use_prototype_fallback"] = model["use_prototype_fallback"]
        graph_summary["activities"][activity_name]["top_rules"] = [rule["rule_text"] for rule in rules[:10]]

    selected_paths = select_unique_paths(candidate_paths=candidate_paths, strength_maps=strength_maps)
    signatures = build_activity_signatures(
        selected_paths=selected_paths,
        strength_maps=strength_maps,
        labelled_data=classifier_train_data,
        settings=settings,
        n_states=n_states,
    )
    classifier_metrics = run_rule_based_classifier(
        settings=settings,
        rule_library=rule_library,
        activity_models=activity_models,
        test_data_by_activity=classifier_test_data,
        segment_length=segment_length,
        segment_hop=segment_hop,
        max_segments_per_activity=max_segments_per_activity,
    )
    classifier_metrics["train_points_per_activity"] = {
        settings.activities[label]: int(len(data))
        for label, data in classifier_train_data.items()
    }
    classifier_metrics["test_points_per_activity"] = {
        settings.activities[label]: int(len(data))
        for label, data in classifier_test_data.items()
    }

    labels_sorted = classifier_metrics["labels"]
    cm = np.array(classifier_metrics["confusion_matrix"], dtype=int)
    label_names = [settings.activities[idx] for idx in labels_sorted]
    save_confusion_matrix_plot(
        confusion=cm,
        label_names=label_names,
        output_path=out_dir / "deterministic_confusion_matrix.png",
    )
    with (out_dir / "deterministic_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred"] + label_names)
        for idx, row in enumerate(cm):
            writer.writerow([label_names[idx], *row.tolist()])

    path_library = {
        settings.activities[activity_id]: {
            "activity_id": signature.activity_id,
            "own_score": signature.own_score,
            "margin_to_next_activity": signature.margin_to_next_activity,
            "strictly_unique": signature.strictly_unique,
            "path": path_to_strings(signature.path),
            "path_edges": [list(edge) for edge in signature.path],
            "edge_states": [asdict(edge_state) for edge_state in signature.edge_states],
        }
        for activity_id, signature in signatures.items()
    }

    save_rule_library(
        output_dir=out_dir,
        rule_library=rule_library,
        activity_models=activity_models,
        settings=settings,
        metadata={
            "independence_test": ind_test_name,
            "source_raw_dir": str(raw_dir),
            "filter_threshold": filter_threshold,
            "path_threshold": path_threshold,
            "classifier_stride": classifier_stride,
            "classifier_train_ratio": classifier_train_ratio,
            "segment_length": segment_length,
            "segment_hop": segment_hop,
            "n_latent_variables": n_vars,
            "var_names": var_names,
            "min_rules_per_activity": min_rules_per_activity,
            "low_edge_threshold": low_edge_threshold,
            "top_fallback_source_vars": top_fallback_source_vars,
            "top_fallback_target_vars": top_fallback_target_vars,
            "min_condition_count": min_condition_count,
            "fallback_min_delta": fallback_min_delta,
            "fallback_pool_size": fallback_pool_size,
            "core_fraction": core_fraction,
            "tie_breaker_weight": tie_breaker_weight,
            "coverage_weight": coverage_weight,
            "prior_weight": prior_weight,
            "prototype_weight": prototype_weight,
            "min_margin": min_margin,
            "max_rules_scale": max_rules_scale,
            "low_edge_budget_ratio": low_edge_budget_ratio,
            "calibration_clip": calibration_clip,
        },
    )

    with (out_dir / "path_library.json").open("w", encoding="utf-8") as file:
        json.dump(path_library, file, indent=2)
    with (out_dir / "graphs_summary.json").open("w", encoding="utf-8") as file:
        json.dump(graph_summary, file, indent=2)
    with (out_dir / "deterministic_classifier_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(classifier_metrics, file, indent=2)

    return {
        "output_dir": str(out_dir),
        "rule_library": rule_library,
        "path_library": path_library,
        "classifier_metrics": classifier_metrics,
        "graph_summary": graph_summary,
        "activity_models": activity_models,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    settings = Settings()
    parser = argparse.ArgumentParser(description="Search paths/classify from raw LPCMCI JSON graphs.")
    parser.add_argument("--study-path", type=str, default=settings.study_path)
    parser.add_argument("--ind-test", type=str, default="parcorr")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--filter-threshold", type=float, default=0.1)
    parser.add_argument("--path-threshold", type=float, default=0.0)
    parser.add_argument("--max-edges-per-path", type=int, default=4)
    parser.add_argument("--min-edges-per-path", type=int, default=2)
    parser.add_argument("--top-paths-per-activity", type=int, default=150)
    parser.add_argument("--top-rules-per-activity", type=int, default=50)
    parser.add_argument("--rule-min-delta", type=float, default=0.0)
    parser.add_argument("--min-rules-per-activity", type=int, default=6)
    parser.add_argument("--low-edge-threshold", type=int, default=3)
    parser.add_argument("--top-fallback-source-vars", type=int, default=8)
    parser.add_argument("--top-fallback-target-vars", type=int, default=12)
    parser.add_argument("--min-condition-count", type=int, default=8)
    parser.add_argument("--fallback-min-delta", type=float, default=0.05)
    parser.add_argument("--fallback-pool-size", type=int, default=300)
    parser.add_argument("--core-fraction", type=float, default=0.6)
    parser.add_argument("--tie-breaker-weight", type=float, default=0.25)
    parser.add_argument("--coverage-weight", type=float, default=0.2)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--prototype-weight", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--max-rules-scale", type=float, default=1.6)
    parser.add_argument("--low-edge-budget-ratio", type=float, default=0.3)
    parser.add_argument("--calibration-clip", type=float, default=2.0)
    parser.add_argument("--classifier-stride", type=int, default=75)
    parser.add_argument("--classifier-train-ratio", type=float, default=0.7)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--segment-hop", type=int, default=12)
    parser.add_argument("--max-segments-per-activity", type=int, default=None)
    parser.add_argument("--no-save-figs", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    settings = Settings(study_path=args.study_path)
    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir is not None
        else Path(settings.study_path) / "causal_graphs_raw" / args.ind_test
    )
    result = run_path_search_from_raw(
        settings=settings,
        raw_dir=raw_dir,
        filter_threshold=args.filter_threshold,
        path_threshold=args.path_threshold,
        max_edges_per_path=args.max_edges_per_path,
        min_edges_per_path=args.min_edges_per_path,
        top_paths_per_activity=args.top_paths_per_activity,
        top_rules_per_activity=args.top_rules_per_activity,
        rule_min_delta=args.rule_min_delta,
        min_rules_per_activity=args.min_rules_per_activity,
        low_edge_threshold=args.low_edge_threshold,
        top_fallback_source_vars=args.top_fallback_source_vars,
        top_fallback_target_vars=args.top_fallback_target_vars,
        min_condition_count=args.min_condition_count,
        fallback_min_delta=args.fallback_min_delta,
        fallback_pool_size=args.fallback_pool_size,
        core_fraction=args.core_fraction,
        tie_breaker_weight=args.tie_breaker_weight,
        coverage_weight=args.coverage_weight,
        prior_weight=args.prior_weight,
        prototype_weight=args.prototype_weight,
        min_margin=args.min_margin,
        max_rules_scale=args.max_rules_scale,
        low_edge_budget_ratio=args.low_edge_budget_ratio,
        calibration_clip=args.calibration_clip,
        classifier_stride=args.classifier_stride,
        classifier_train_ratio=args.classifier_train_ratio,
        segment_length=args.segment_length,
        segment_hop=args.segment_hop,
        max_segments_per_activity=args.max_segments_per_activity,
        save_figs=not args.no_save_figs,
    )
    print(json.dumps({"output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
