from dataclasses import dataclass

import numpy as np

from charl_tre.causal.rules import Rule, _conditional_target_mean

_LATENT_NDIM = 3


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

    macro_metrics: BaseMetrics
    weighted_metrics: BaseMetrics
    per_class_metrics: dict[int, BaseMetrics]


def make_segments(data: np.ndarray, segment_length: int) -> dict[int, np.ndarray]:
    """Create stride-one segments from activity latent sequences."""
    if segment_length <= 0:
        msg = "segment_length must be positive"
        raise ValueError(msg)

    segments: dict[int, np.ndarray] = {}
    for activity, activity_data in enumerate(data):
        if activity_data.ndim != _LATENT_NDIM:
            msg = "activity latent data must have shape (n_windows, n_mixtures, n_components)"
            raise ValueError(msg)
        n_segments = max(activity_data.shape[0] - segment_length + 1, 0)
        if n_segments:
            segments[activity] = np.stack(
                [activity_data[start : start + segment_length] for start in range(n_segments)],
                axis=0,
            )
        else:
            segments[activity] = np.empty((0, segment_length, *activity_data.shape[1:]), dtype=activity_data.dtype)

    return segments


def linspace_indices(n: int, count: int) -> np.ndarray:
    """Return integer indices matching ``np.linspace(..., dtype=int)``."""
    if n <= 0 or count <= 0:
        return np.empty(0, dtype=int)
    if count == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n - 1, count, dtype=int)


def _rule_evidence(sequence: np.ndarray, rule: Rule, eps: float) -> tuple[float, int]:
    """Score continuous target evidence for one rule in a sequence."""
    conditional_mean, count = _conditional_target_mean(sequence, rule.edge, rule.source_threshold)
    if count == 0:
        return 0.0, 0

    expected_delta = rule.target_mean_train - rule.target_mean_other
    scale = max(abs(expected_delta), eps)
    observed_delta = rule.target_direction * (conditional_mean - rule.target_mean_other)
    return float(observed_delta / scale), count


def score_rules(sequence: np.ndarray, rules: list[Rule], eps: float = 1e-8) -> RuleStats:
    """Score a continuous latent sequence against one activity's rules."""
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
            core_weight_value = max(rule.normalized_core_weight or 0.0, 0.0)
            core_score += core_weight_value * evidence
            core_weight += core_weight_value
            matched_core += 1
        else:
            tie_weight_value = max(rule.normalized_tie_weight or 0.0, 0.0)
            tie_score += tie_weight_value * evidence
            tie_weight += tie_weight_value
            matched_tie += 1

    raw = raw_score / raw_weight if raw_weight else 0.0
    return RuleStats(
        raw_rule_score=raw,
        core_score=core_score / core_weight if core_weight else raw,
        tie_score=tie_score / tie_weight if tie_weight else 0.0,
        coverage=min(raw_weight, 1.0),
        matched_rules=matched,
        matched_core_rules=matched_core,
        matched_tie_rules=matched_tie,
    )


def prototype_score(sequence: np.ndarray, prototype_mean: tuple[float, ...] | np.ndarray) -> float:
    """Return the negative mean absolute distance from an activity prototype."""
    if sequence.size == 0 or len(prototype_mean) == 0:
        return 0.0
    observed = np.nanmean(sequence, axis=0).reshape(-1)
    prototype = np.asarray(prototype_mean, dtype=float).reshape(-1)
    if observed.shape != prototype.shape:
        return 0.0
    return -float(np.nanmean(np.abs(observed - prototype)))


def combine(stats: RuleStats, model: ActivityModel, prototype: float, apply_bias: bool = True) -> float:
    """Combine rule, coverage, prior, prototype, and bias terms."""
    score = stats.core_score
    score += model.tie_breaker_weight * stats.tie_score
    score += model.coverage_weight * stats.coverage
    score += model.prior_weight * model.log_prior
    if model.use_prototype_fallback:
        score += model.prototype_weight * prototype
    if apply_bias:
        score += model.score_bias
    return score


def score_sequence(
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
        stats = score_rules(sequence, rules)
        prototype = prototype_score(sequence, model.prototype_mean) if model else 0.0
        final_score = combine(stats, model, prototype, apply_bias) if model else stats.core_score
        scores[activity] = final_score
        diagnostics[activity] = ScoringDiagnostics(stats, prototype, final_score)
    return scores, diagnostics


def classify(
    sequence: np.ndarray,
    rule_library: dict[int, list[Rule]],
    activity_models: dict[int, ActivityModel] | None = None,
) -> tuple[int, dict[int, float]]:
    """Classify a sequence, using diagnostics to break narrow score margins."""
    scores, diagnostics = score_sequence(sequence, rule_library, activity_models)
    if not scores:
        msg = "rule_library must contain at least one activity"
        raise ValueError(msg)

    ranked = sorted(scores, key=lambda activity: scores[activity], reverse=True)
    prediction = ranked[0]
    if len(ranked) > 1 and activity_models:
        runner_up = ranked[1]
        required_margin = max(
            activity_models.get(prediction, ActivityModel()).min_margin,
            activity_models.get(runner_up, ActivityModel()).min_margin,
        )
        if scores[prediction] - scores[runner_up] < required_margin:
            prediction_key = (
                diagnostics[prediction].stats.matched_core_rules,
                diagnostics[prediction].stats.coverage,
                diagnostics[prediction].prototype_score,
                scores[prediction],
            )
            runner_up_key = (
                diagnostics[runner_up].stats.matched_core_rules,
                diagnostics[runner_up].stats.coverage,
                diagnostics[runner_up].prototype_score,
                scores[runner_up],
            )
            if runner_up_key > prediction_key:
                prediction = runner_up
    return prediction, scores


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> EvaluationMetrics:
    """Compute per-class, macro, and support-weighted classification metrics."""
    per_class: dict[int, BaseMetrics] = {}
    total = len(y_true)
    accuracy = float(np.mean(y_pred == y_true)) if total else 0.0
    for label in labels:
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        support = int(np.sum(y_true == label))
        predicted = int(np.sum(y_pred == label))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = BaseMetrics(accuracy, precision, recall, f1, support)

    if not labels:
        empty = BaseMetrics(0.0, 0.0, 0.0, 0.0, 0)
        return EvaluationMetrics(empty, empty, {})

    class_values = list(per_class.values())
    macro = BaseMetrics(
        accuracy=float(np.mean([metric.accuracy for metric in class_values])),
        precision=float(np.mean([metric.precision for metric in class_values])),
        recall=float(np.mean([metric.recall for metric in class_values])),
        f1=float(np.mean([metric.f1 for metric in class_values])),
        support=total,
    )
    weights = np.array([metric.support for metric in class_values], dtype=float)
    weights = weights / weights.sum() if weights.sum() else np.zeros_like(weights)
    weighted = BaseMetrics(
        accuracy=accuracy,
        precision=float(np.dot(weights, [metric.precision for metric in class_values])),
        recall=float(np.dot(weights, [metric.recall for metric in class_values])),
        f1=float(np.dot(weights, [metric.f1 for metric in class_values])),
        support=total,
    )
    return EvaluationMetrics(macro, weighted, per_class)


def evaluate(
    rule_library: dict[int, list[Rule]],
    test_data: np.ndarray,
    segment_length: int,
    activity_models: dict[int, ActivityModel] | None = None,
) -> EvaluationMetrics:
    """Evaluate activity classification on stride-one test segments."""
    segments = make_segments(test_data, segment_length)
    y_true: list[int] = []
    y_pred: list[int] = []
    for activity, activity_segments in segments.items():
        for sequence in activity_segments:
            prediction, _ = classify(sequence, rule_library, activity_models)
            y_true.append(activity)
            y_pred.append(prediction)

    return _metrics(np.asarray(y_true), np.asarray(y_pred), sorted(rule_library))
