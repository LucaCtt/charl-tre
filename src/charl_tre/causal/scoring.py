"""Rule-based scoring, calibration, and classification."""

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from charl_tre.causal.rules import ActivityModel, Rule, _rule_hits

_MARGIN_EPS = 0.05


@dataclass(frozen=True)
class RuleStats:
    """Statistics generated from scoring a sequence against a set of rules."""

    raw_rule_score: float = 0.0
    core_score: float = 0.0
    tie_score: float = 0.0
    coverage: float = 0.0
    matched_rules: int = 0
    matched_core_rules: int = 0
    matched_tie_rules: int = 0


@dataclass(frozen=True)
class ScoringDiagnostics:
    """Detailed breakdown of how a final activity score was calculated."""

    stats: RuleStats
    prototype_score: float
    final_score: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Performance report for activity classification."""

    accuracy: float
    n_segments: int
    labels: list[int]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    mean_prediction_margin: float
    median_prediction_margin: float


class ActivityScorer:
    """Score sequences against symbolic rules and classify activities."""

    @staticmethod
    def make_segments(
        labelled_data: dict[int, np.ndarray],
        segment_length: int,
        segment_hop: int,
        max_per_activity: int | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Create segments of data for evaluation or calibration.

        Arguments:
            labelled_data: Mapping from activity label to data array (n_samples, n_features).
            segment_length: Length of each segment in samples.
            segment_hop: Step size between segment starts in samples.
            max_per_activity: Optional maximum number of segments to keep per activity.

        Returns:
            List of (segment, label) tuples, where segment is a numpy array of shape (segment_length, n_features).

        """
        segments: list[tuple[np.ndarray, int]] = []

        for label, data in labelled_data.items():
            if len(data) < segment_length:
                continue

            starts = list(range(0, len(data) - segment_length + 1, segment_hop))
            if max_per_activity is not None and len(starts) > max_per_activity:
                keep = np.linspace(0, len(starts) - 1, num=max_per_activity, dtype=int)
                starts = [starts[i] for i in keep]

            segments.extend((data[s : s + segment_length], label) for s in starts)

        return segments

    def score_sequence(
        self,
        sequence: np.ndarray,
        rule_library: dict[int, list[Rule]],
        activity_models: dict[int, ActivityModel] | None,
        apply_bias: bool = True,
    ) -> tuple[dict[int, float], dict[int, ScoringDiagnostics]]:
        """Score a sequence against the rule library and activity models.

        Arguments:
            sequence: Input data array of shape (n_samples, n_features).
            rule_library: Mapping from activity label to list of rules.
            activity_models: Optional mapping from activity label to model parameters.
            apply_bias: Whether to apply score bias from activity models.

        Returns:
            Tuple of (scores, diagnostics), where scores is a mapping from activity label to final score,
            and diagnostics is a mapping from activity label to detailed scoring information.

        """
        scores: dict[int, float] = {}
        diagnostics: dict[int, ScoringDiagnostics] = {}

        for aid, rules in rule_library.items():
            model = (activity_models or {}).get(aid)
            # Handle cases where model might be missing
            proto_rates = model.prototype_true_rate if model else []

            stats = self._score_rules(sequence, rules)
            proto = self._prototype_score(sequence, proto_rates)
            final = self._combine(stats, model, proto, apply_bias) if model else stats.core_score

            diagnostics[aid] = ScoringDiagnostics(stats=stats, prototype_score=proto, final_score=final)
            scores[aid] = final

        return scores, diagnostics

    def classify(
        self,
        sequence: np.ndarray,
        rule_library: dict[int, list[Rule]],
        activity_models: dict[int, ActivityModel] | None = None,
    ) -> tuple[int, dict[int, float]]:
        """Classify a sequence into an activity based on scoring.

        Arguments:
            sequence: Input data array of shape (n_samples, n_features).
            rule_library: Mapping from activity label to list of rules.
            activity_models: Optional mapping from activity label to model parameters.
            return_details: Whether to return detailed diagnostics along with the prediction.

        Returns:
            Tuple of (predicted_activity_id, scores), where predicted_activity_id
            is the label of the predicted activity, and scores is a mapping from activity label to final score.

        """
        scores, diagnostics = self.score_sequence(sequence, rule_library, activity_models, apply_bias=True)

        # Sort by descending final score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        prediction = ranked[0][0]

        # Check margin requirements if multiple candidates exist
        if len(ranked) > 1 and activity_models is not None:
            runner_up = ranked[1][0]
            margin = float(ranked[0][1] - ranked[1][1])

            # Get required margin from participating models
            m1 = activity_models[prediction].min_margin
            m2 = activity_models[runner_up].min_margin
            required_margin = max(m1, m2)

            if margin < required_margin:
                # Tie-breaker hierarchy: core hits -> coverage -> prototype match
                prediction = max(
                    [prediction, runner_up],
                    key=lambda aid: (
                        diagnostics[aid].stats.matched_core_rules,
                        diagnostics[aid].stats.coverage,
                        diagnostics[aid].prototype_score,
                        scores[aid],
                    ),
                )

        return prediction, scores

    def evaluate(
        self,
        rule_library: dict[int, list[Rule]],
        activity_models: dict[int, ActivityModel],
        test_data: dict[int, np.ndarray],
        activity_names: dict[int, str],
        segment_length: int,
        segment_hop: int,
        max_per_activity: int | None = None,
    ) -> EvaluationMetrics:
        """Evaluate classification performance on test data.

        Arguments:
            rule_library: Mapping from activity label to list of rules.
            activity_models: Mapping from activity label to model parameters.
            test_data: Mapping from activity label to data array (n_samples, n_features).
            activity_names: Mapping from activity label to human-readable name.
            segment_length: Length of each segment in samples.
            segment_hop: Step size between segment starts in samples.
            max_per_activity: Optional maximum number of segments to evaluate per activity.

        Returns:
            EvaluationMetrics object containing accuracy, confusion matrix,
            classification report, and margin statistics.

        """
        segments = self.make_segments(test_data, segment_length, segment_hop, max_per_activity)

        y_true: list[int] = []
        y_pred: list[int] = []
        margins: list[float] = []

        for sequence, label in segments:
            prediction, scores = self.classify(sequence, rule_library, activity_models)

            # Calculate prediction margin (difference between top and runner-up score)
            val_list = sorted(scores.values(), reverse=True)
            margin = float(val_list[0] - val_list[1]) if len(val_list) > 1 else 0.0

            y_true.append(label)
            y_pred.append(prediction)
            margins.append(margin)

        labels = sorted(rule_library)

        # Calculate standard metrics using sklearn
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report: dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[activity_names[label] for label in labels],
            output_dict=True,
            zero_division=0,
        )  # pyright: ignore[reportAssignmentType]

        return EvaluationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            n_segments=len(segments),
            labels=labels,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            mean_prediction_margin=float(np.mean(margins)) if margins else 0.0,
            median_prediction_margin=float(np.median(margins)) if margins else 0.0,
        )

    def calibrate(
        self,
        rule_library: dict[int, list[Rule]],
        activity_models: dict[int, ActivityModel],
        train_data: dict[int, np.ndarray],
        segment_length: int,
        segment_hop: int,
        max_per_activity: int | None = None,
        clip: float = 2.0,
    ) -> dict[int, ActivityModel]:
        """Calibrate score biases for each activity based on training data.

        Arguments:
            rule_library: Mapping from activity label to list of rules.
            activity_models: Mapping from activity label to model parameters, which will be updated with score_bias.
            train_data: Mapping from activity label to data array (n_samples, n_features) for calibration.
            segment_length: Length of each segment in samples.
            segment_hop: Step size between segment starts in samples.
            max_per_activity: Optional maximum number of segments to use for calibration per activity.
            clip: Maximum absolute value for score bias to prevent extreme adjustments.

        """
        segments = self.make_segments(train_data, segment_length, segment_hop, max_per_activity)
        if not segments:
            return activity_models

        buckets: dict[int, dict[str, list[float]]] = {aid: {"own": [], "other": []} for aid in rule_library}
        for seq, true_label in segments:
            scores, _ = self.score_sequence(seq, rule_library, activity_models, apply_bias=False)
            for aid, val in scores.items():
                buckets[aid]["own" if aid == true_label else "other"].append(val)

        new_models = {}
        for aid, bkt in buckets.items():
            model = activity_models[aid]
            own, other = np.array(bkt["own"]), np.array(bkt["other"])

            if own.size == 0 or other.size == 0:
                new_models[aid] = model
                continue

            own_med, other_med = np.median(own), np.median(other)
            bias = np.clip(-0.5 * (own_med + other_med), -clip, clip)

            updated_margin = (
                max(model.min_margin, _MARGIN_EPS) if (own_med - other_med) < _MARGIN_EPS else model.min_margin
            )

            new_models[aid] = replace(model, score_bias=float(bias), min_margin=float(updated_margin))

        return new_models

    @staticmethod
    def _score_rules(sequence: np.ndarray, rules: list[Rule], eps: float = 1e-6) -> RuleStats:
        """Score a sequence against a list of rules and compute statistics.

        Arguments:
            sequence: Input data array of shape (n_samples, n_features).
            rules: List of rules to evaluate against the sequence.
            eps: Small value to prevent log(0) issues.

        Returns:
            RuleStats object containing aggregated scores and match counts for the given rules.

        """
        raw_score = raw_weight = core_score = core_weight = tie_score = tie_weight = 0.0
        matched = matched_core = matched_tie = 0

        for rule in rules:
            hits, count = _rule_hits(sequence, rule.src, rule.dst, rule.lag, rule.target_dst_state)
            if count == 0:
                continue

            p_a = np.clip(rule.p_target_train, eps, 1.0 - eps)
            p_b = np.clip(rule.p_target_other, eps, 1.0 - eps)
            llr = hits * math.log(p_a / p_b) + (count - hits) * math.log((1 - p_a) / (1 - p_b))

            w = max(0.0, rule.normalized_weight)
            raw_score += w * llr
            raw_weight += w
            matched += 1

            if rule.is_core:
                cw = max(0.0, rule.normalized_core_weight)
                core_score += cw * llr
                core_weight += cw
                matched_core += 1
            else:
                tw = max(0.0, rule.normalized_tie_weight)
                tie_score += tw * llr
                tie_weight += tw
                matched_tie += 1

        return RuleStats(
            raw_rule_score=raw_score / raw_weight if raw_weight > 0 else 0.0,
            core_score=core_score / core_weight
            if core_weight > 0
            else (raw_score / raw_weight if raw_weight > 0 else 0.0),
            tie_score=tie_score / tie_weight if tie_weight > 0 else 0.0,
            coverage=min(1.0, raw_weight),
            matched_rules=matched,
            matched_core_rules=matched_core,
            matched_tie_rules=matched_tie,
        )

    @staticmethod
    def _prototype_score(sequence: np.ndarray, prototype_true_rate: list[float]) -> float:
        """Compute a prototype score based on the difference between observed feature rates and prototype rates.

        Arguments:
            sequence: Input data array of shape (n_samples, n_features).
            prototype_true_rate: List of expected true rates for each feature based on the activity prototype.

        Returns:
            A score representing how closely the observed feature rates match
            the prototype rates, with higher scores indicating closer matches.
            The score is computed as the negative mean absolute difference between observed and prototype rates.

        """
        if sequence.size == 0 or not prototype_true_rate:
            return 0.0
        observed = np.mean(sequence, axis=0, dtype=np.float64)
        proto = np.array(prototype_true_rate, dtype=np.float64)
        if observed.shape != proto.shape:
            return 0.0
        return float(-np.mean(np.abs(observed - proto)))

    @staticmethod
    def _combine(
        stats: RuleStats,
        model: ActivityModel,
        prototype_score: float,
        apply_bias: bool,
    ) -> float:
        """Combine rule-based scores, prototype score, and model parameters into a final score.

        Arguments:
            stats: Dictionary containing rule-based statistics (core_score, tie_score, coverage).
            model: ActivityModel containing parameters for weighting different score components and bias.
            prototype_score: Score from the prototype matching function.
            apply_bias: Whether to apply the score bias from the model.

        Returns:
            The final combined score for the activity, where higher scores indicate a stronger match to the activity.
            The score is computed as a weighted sum of the core score, tie score, coverage,
            log prior, and prototype score, with an optional bias added.

        """
        # Core score is the base component
        score = stats.core_score

        # Weighted additions from tie-breakers and coverage
        score += model.tie_breaker_weight * stats.tie_score
        score += model.coverage_weight * stats.coverage
        score += model.prior_weight * model.log_prior

        # Optional prototype fallback and calibration bias
        if model.use_prototype_fallback:
            score += model.prototype_weight * prototype_score

        if apply_bias:
            score += model.score_bias

        return float(score)
