from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from charl_tre.causal.rules import Rule


@dataclass(frozen=True)
class RuleStats:
    """Statistics generated while scoring one sequence against activity rules."""

    raw_rule_score: float
    core_score: float
    tie_score: float
    coverage: float
    matched_rules: int
    matched_core_rules: int
    matched_tie_rules: int


@dataclass(frozen=True)
class ScoringDiagnostics:
    """Detailed breakdown of an activity score."""

    stats: RuleStats
    prototype_score: float
    final_score: float


@dataclass(frozen=True)
class ActivityModel:
    """Optional activity-specific scoring parameters."""

    log_prior: float = 0.0
    prior_weight: float = 0.0
    tie_breaker_weight: float = 0.0
    coverage_weight: float = 0.0
    use_prototype_fallback: bool = False
    prototype_weight: float = 0.0
    score_bias: float = 0.0
    min_margin: float = 0.0
    prototype_mean: tuple[float, ...] = ()


@dataclass(frozen=True)
class BaseMetrics:
    """Classification metrics for one class aggregate."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Performance report for activity classification."""

    overall: BaseMetrics
    weighted: BaseMetrics
    per_activity: dict[int, BaseMetrics]


def _make_segments(data: np.ndarray, segment_length: int) -> dict[int, np.ndarray]:
    """Create stride-one segments using vectorized sliding window views."""
    segments: dict[int, np.ndarray] = {}
    for activity, activity_data in enumerate(data):
        n_windows = activity_data.shape[0]
        if n_windows >= segment_length:
            sw = sliding_window_view(activity_data, window_shape=segment_length, axis=0)
            segments[activity] = np.moveaxis(sw, -1, 1)
        else:
            segments[activity] = np.empty((0, segment_length, *activity_data.shape[1:]), dtype=activity_data.dtype)

    return segments


def _linspace_indices(n: int, count: int) -> np.ndarray:
    """Return integer indices matching np.linspace(..., dtype=int)."""
    if n <= 0 or count <= 0:
        return np.empty(0, dtype=int)
    if count == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n - 1, count, dtype=int)


def _rule_evidence(
    sequence: np.ndarray,
    rule: Rule,
    eps: float = 1e-8,
    min_count: int = 3,
    max_evidence: float = 2.0,
) -> tuple[float, int]:
    """Compute normalized evidence score and sample count for one rule on a target sequence."""
    seq_len = sequence.shape[0]
    if rule.edge.lag >= seq_len:
        return 0.0, 0

    src_seq = sequence[: seq_len - rule.edge.lag, rule.edge.source.mixture, rule.edge.source.component]
    tgt_seq = sequence[rule.edge.lag :, rule.edge.target.mixture, rule.edge.target.component]

    valid_mask = np.isfinite(src_seq) & np.isfinite(tgt_seq) & (src_seq >= rule.source_threshold)
    count = int(valid_mask.sum())
    if count < min_count:
        # Very small match counts are too noisy to provide stable evidence.
        return 0.0, 0

    conditional_mean = float(tgt_seq[valid_mask].mean())
    expected_delta = rule.target_mean_train - rule.target_mean_other
    scale = max(abs(expected_delta), eps)

    observed_delta = rule.target_direction * (conditional_mean - rule.target_mean_other)
    normalized_evidence = float(np.clip(observed_delta / scale, -max_evidence, max_evidence))

    return normalized_evidence, count


def _score_rules(sequence: np.ndarray, rules: list[Rule], eps: float = 1e-8) -> RuleStats:
    """Score a continuous latent sequence against an activity's rules."""
    if not rules:
        return RuleStats(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    raw_score = raw_weight = 0.0
    core_score = core_weight = 0.0
    tie_score = tie_weight = 0.0
    matched = matched_core = matched_tie = 0

    for rule in rules:
        evidence, count = _rule_evidence(sequence, rule, eps)
        if count == 0:
            continue

        weight = max(rule.normalized_weight or 0.0, 0.0)
        raw_score += weight * evidence
        raw_weight += weight
        matched += 1

        if rule.is_core:
            c_weight = max(rule.normalized_core_weight or 0.0, 0.0)
            core_score += c_weight * evidence
            core_weight += c_weight
            matched_core += 1
        else:
            t_weight = max(rule.normalized_tie_weight or 0.0, 0.0)
            tie_score += t_weight * evidence
            tie_weight += t_weight
            matched_tie += 1

    final_raw = raw_score / raw_weight if raw_weight > 0.0 else 0.0
    final_core = core_score / core_weight if core_weight > 0.0 else final_raw
    final_tie = tie_score / tie_weight if tie_weight > 0.0 else 0.0

    return RuleStats(
        raw_rule_score=float(final_raw),
        core_score=float(final_core),
        tie_score=float(final_tie),
        coverage=float(min(raw_weight, 1.0)),
        matched_rules=matched,
        matched_core_rules=matched_core,
        matched_tie_rules=matched_tie,
    )


def _prototype_score(sequence: np.ndarray, prototype_mean: tuple[float, ...] | np.ndarray) -> float:
    """Return the negative mean absolute error relative to an activity prototype."""
    if sequence.size == 0 or len(prototype_mean) == 0:
        return 0.0

    observed = np.nanmean(sequence, axis=0).ravel()
    prototype = np.ravel(np.asarray(prototype_mean, dtype=np.float64))

    if observed.shape != prototype.shape:
        return 0.0

    diff = np.abs(observed - prototype)
    valid = np.isfinite(diff)
    return -float(np.mean(diff[valid])) if valid.any() else 0.0


def _combine(stats: RuleStats, model: ActivityModel, prototype: float, apply_bias: bool = True) -> float:
    """Combine rule score, coverage, priors, prototype fallbacks, and class bias."""
    score = stats.core_score
    score += model.tie_breaker_weight * stats.tie_score
    score += model.coverage_weight * stats.coverage
    score += model.prior_weight * model.log_prior

    if model.use_prototype_fallback:
        score += model.prototype_weight * prototype

    if apply_bias:
        score += model.score_bias

    return float(score)


def _score_sequence(
    sequence: np.ndarray,
    rule_library: dict[int, list[Rule]],
    activity_models: dict[int, ActivityModel] | None = None,
    apply_bias: bool = True,
) -> tuple[dict[int, float], dict[int, ScoringDiagnostics]]:
    """Score a sequence for every activity in the rule library."""
    scores: dict[int, float] = {}
    diagnostics: dict[int, ScoringDiagnostics] = {}

    for activity, rules in rule_library.items():
        model = activity_models.get(activity) if activity_models else None
        stats = _score_rules(sequence, rules)
        proto = _prototype_score(sequence, model.prototype_mean) if model else 0.0
        final_score = _combine(stats, model, proto, apply_bias) if model else stats.core_score

        scores[activity] = final_score
        diagnostics[activity] = ScoringDiagnostics(stats, proto, final_score)

    return scores, diagnostics


def _classify(
    sequence: np.ndarray,
    rule_library: dict[int, list[Rule]],
    activity_models: dict[int, ActivityModel] | None = None,
) -> tuple[int, dict[int, float]]:
    """Classify a sequence, breaking narrow score margins using multi-tiered rule evidence."""
    scores, diagnostics = _score_sequence(sequence, rule_library, activity_models)

    ranked = sorted(scores, key=lambda a: scores[a], reverse=True)
    prediction = ranked[0]

    if len(ranked) > 1 and activity_models:
        runner_up = ranked[1]
        req_margin = max(
            activity_models.get(prediction, ActivityModel()).min_margin,
            activity_models.get(runner_up, ActivityModel()).min_margin,
        )

        if scores[prediction] - scores[runner_up] < req_margin:
            pred_diag = diagnostics[prediction]
            runner_diag = diagnostics[runner_up]

            pred_key = (
                pred_diag.stats.matched_core_rules,
                pred_diag.stats.coverage,
                pred_diag.prototype_score,
                scores[prediction],
            )
            runner_key = (
                runner_diag.stats.matched_core_rules,
                runner_diag.stats.coverage,
                runner_diag.prototype_score,
                scores[runner_up],
            )

            if runner_key > pred_key:
                prediction = runner_up

    return prediction, scores


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> EvaluationMetrics:
    """Compute per-class, macro-averaged, and support-weighted classification metrics."""
    total = len(y_true)
    if total == 0 or not labels:
        empty = BaseMetrics(0.0, 0.0, 0.0, 0.0, 0)
        return EvaluationMetrics(empty, empty, {})

    per_class: dict[int, BaseMetrics] = {}
    overall_acc = float(np.mean(y_pred == y_true))

    for label in labels:
        is_true = y_true == label
        is_pred = y_pred == label

        tp = int(np.sum(is_true & is_pred))
        support = int(is_true.sum())
        pred_count = int(is_pred.sum())

        # In multiclass reports, per-class accuracy is most interpretable as class hit-rate.
        cls_acc = tp / support if support > 0 else 0.0
        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = BaseMetrics(
            accuracy=float(cls_acc),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            support=support,
        )

    class_vals = list(per_class.values())

    macro = BaseMetrics(
        accuracy=overall_acc,
        precision=float(np.mean([m.precision for m in class_vals])),
        recall=float(np.mean([m.recall for m in class_vals])),
        f1=float(np.mean([m.f1 for m in class_vals])),
        support=total,
    )

    weights = np.array([m.support for m in class_vals], dtype=np.float64)
    sum_w = weights.sum()
    weights = weights / sum_w if sum_w > 0 else np.zeros_like(weights)

    weighted = BaseMetrics(
        accuracy=overall_acc,
        precision=float(np.dot(weights, [m.precision for m in class_vals])),
        recall=float(np.dot(weights, [m.recall for m in class_vals])),
        f1=float(np.dot(weights, [m.f1 for m in class_vals])),
        support=total,
    )

    return EvaluationMetrics(macro, weighted, per_class)


def evaluate(
    rule_library: dict[int, list[Rule]],
    test_data: np.ndarray,
    segment_length: int,
    activity_models: dict[int, ActivityModel] | None = None,
) -> EvaluationMetrics:
    """Evaluate activity classification performance across test dataset segments."""
    segments = _make_segments(test_data, segment_length)
    y_true: list[int] = []
    y_pred: list[int] = []

    for activity, activity_segments in segments.items():
        for sequence in activity_segments:
            prediction, _ = _classify(sequence, rule_library, activity_models)
            y_true.append(activity)
            y_pred.append(prediction)

    return _metrics(np.asarray(y_true), np.asarray(y_pred), sorted(rule_library))
