from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text

ArrayLike = np.ndarray | torch.Tensor | list


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert a numpy array or torch tensor to a numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    if isinstance(array, list):
        return np.array(array)

    return array


@dataclass(frozen=True)
class BaseMetrics:
    """Classification metrics for one class aggregate."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Metrics[LabelT: Hashable]:
    """Classification metrics for multiple classes."""

    multi_class: BaseMetrics
    """Multi-class metrics.

    Accuracy is computed as the total number of correct predictions divided by the total number of instances.
    Precision, recall, and F1 score are computed as support-weighted averages across all classes
    """

    per_class: dict[LabelT, BaseMetrics]
    """Per-class metrics.

    Accuracy, precision, recall, and F1 score are computed for each class individually,
    using a one-vs-rest approach.
    """


def _compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: Hashable) -> BaseMetrics:
    """Compute one-vs-rest classification metrics for a single class label.

    Arguments:
        y_true (np.ndarray): True labels, as a numpy array.
        y_pred (np.ndarray): Predicted labels, as a numpy array.
        label (LabelT): The class label to compute metrics for.

    Returns:
        BaseMetrics: The computed metrics for the specified class label.

    """
    is_true = y_true == label
    is_pred = y_pred == label

    tp = float(np.sum(is_true & is_pred))
    tn = float(np.sum(~is_true & ~is_pred))
    fp = float(np.sum(~is_true & is_pred))
    fn = float(np.sum(is_true & ~is_pred))

    class_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    support = int(np.sum(is_true))

    return BaseMetrics(
        accuracy=float(class_acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        support=support,
    )


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Metrics[Hashable]:
    """Compute per-class and support-weighted overall classification metrics.

    Class labels are inferred from the union of values present in `y_true` and `y_pred`.

    Arguments:
        y_true (ArrayLike): True labels, as a numpy array, torch tensor, or list.
        y_pred (ArrayLike): Predicted labels, as a numpy array, torch tensor, or list.

    Returns:
        Metrics[Hashable]: The computed metrics for each class label and overall metrics.

    Raises:
        ValueError: If `y_true` and `y_pred` do not have the same shape.

    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.shape != y_pred.shape:
        msg = "y_true and y_pred must have the same shape"
        raise ValueError(msg)

    total = len(y_true)
    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    per_class = {label: _compute_class_metrics(y_true, y_pred, label) for label in labels}

    overall_support = sum(m.support for m in per_class.values())
    if overall_support > 0:
        weighted_precision = sum(m.precision * m.support for m in per_class.values()) / overall_support
        weighted_recall = sum(m.recall * m.support for m in per_class.values()) / overall_support
        weighted_f1 = sum(m.f1 * m.support for m in per_class.values()) / overall_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    accuracy = float(np.sum(y_true == y_pred)) / total if total > 0 else 0.0

    overall = BaseMetrics(
        accuracy=accuracy,
        precision=weighted_precision,
        recall=weighted_recall,
        f1=weighted_f1,
        support=overall_support,
    )

    return Metrics(multi_class=overall, per_class=per_class)


def metrics_summary(metrics: Metrics, labels: dict[Hashable, str] | list[str] | None = None) -> Text:
    """Create a table summarizing the evaluation metrics.

    Arguments:
        metrics (EvaluationMetrics): Evaluation metrics containing per-activity and overall metrics.
        labels (dict[Hashable, str] | list[str] | None): Optional mapping of activity indices to human-readable labels.
            If a dictionary is provided, it should map activity indices to their corresponding labels.
            If a list is provided, it should contain labels in the order of activity indices.
            If None, activity indices will be used as labels.

    Returns:
        Text: A rich Text object containing the formatted metrics table.

    """
    table = Table()

    table.add_column("Metric", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("F1", justify="right", style="green")
    table.add_column("Precision", justify="right", style="green")
    table.add_column("Recall", justify="right", style="green")

    def format_metric(metric: BaseMetrics) -> list[str]:
        return [
            f"{metric.accuracy:.4f}",
            f"{metric.f1:.4f}",
            f"{metric.precision:.4f}",
            f"{metric.recall:.4f}",
        ]

    # Per-activity metrics
    for activity, activity_metrics in metrics.per_class.items():
        if isinstance(labels, list):
            label = labels[activity] if 0 <= activity < len(labels) else str(activity)
        else:
            label = labels.get(activity, str(activity)) if labels else str(activity)

        table.add_row(
            label,
            *format_metric(activity_metrics),
            end_section=activity == list(metrics.per_class.keys())[-1],
        )

    # Overall metrics
    table.add_row("Overall", *format_metric(metrics.multi_class), style="bold magenta")

    console = Console()
    with console.capture() as capture:
        console.print(table)

    return Text.from_ansi(capture.get())
