import numpy as np

from charl_tre.causal.rules import Rule
from charl_tre.metrics import Metrics, compute_metrics


def _score_activity_rules(activity_data: np.ndarray, rules: list[Rule]) -> float:
    """Score a single activity's test sequences against its rules.

    Arguments:
        activity_data: A 3D numpy array of shape (n_windows, n_mixtures, n_components)
            representing the test sequences for a single activity.
        rules: A list of Rule objects associated with the activity.

    Returns:
        A float representing the average log-likelihood ratio score for the activity based on its rules.

    """
    n_windows = activity_data.shape[0]
    total_llr = 0.0
    matched_rules = 0

    for rule in rules:
        source_seq = activity_data[: n_windows - rule.edge.lag, rule.edge.source.mixture, rule.edge.source.component]
        target_seq = activity_data[rule.edge.lag :, rule.edge.target.mixture, rule.edge.target.component]

        y_active = target_seq[source_seq >= rule.source_threshold]
        if y_active.size > 0:
            total_llr += rule.llr(y_active)
            matched_rules += 1

    return total_llr / float(matched_rules) if matched_rules > 0 else 0.0


def evaluate(
    rules: dict[int, list[Rule]],
    test_data: np.ndarray,
    segment_length: int,
) -> Metrics:
    """Evaluate test data against rules and compute metrics.

    The evaluation is performed by scoring each activity's test data
    against the provided rules and determining the predicted activity based on the highest score.

    The test data is split into segments of the specified length, and each segment is scored against the rules.
    This allows evaluation over multiple segments of the test data for each activity,
    rather than just a single segment per activity

    Arguments:
        rules: A dictionary mapping activity indices to lists of Rule objects.
        test_data: A 3D numpy array of shape (n_activities, n_windows, n_mixtures, n_components)
            representing the test sequences for each activity.
        segment_length: An integer representing the length of each segment in the test data.

    Returns:
        A Metrics object containing classification metrics for the evaluation.

    """
    y_true: list[int] = []
    y_pred: list[int] = []

    for activity in range(test_data.shape[0]):
        n_windows = test_data.shape[1]
        for start in range(0, n_windows - segment_length + 1, segment_length):
            segment = test_data[activity, start : start + segment_length]
            scores = {
                rule_activity: _score_activity_rules(segment, activity_rules)
                for rule_activity, activity_rules in rules.items()
            }
            prediction = max(scores, key=lambda act: scores[act])

            y_true.append(activity)
            y_pred.append(prediction)

    return compute_metrics(y_true, y_pred)
