import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tigramite import plotting as tp

from charl_tre.causal.rules import ActivityModel, Rule
from charl_tre.causal.scoring import EvaluationMetrics

from .types import ActivityGraphResult, CausalEdge


def path_to_strings(path: tuple[tuple[int, int, int], ...]) -> list[str]:
    """Convert a path of (src, dst, lag) tuples into human-readable strings."""
    return [f"Z{src} -[{lag}]-> Z{dst}" for src, dst, lag in path]


class ResultsWriter:
    """Handles serialization of causal discovery results, rule libraries, and evaluation metrics."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the writer and ensure the output directory exists.

        Arguments:
            output_dir: The base directory where all results will be stored.

        """
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)

    def write_activity(
        self,
        result: ActivityGraphResult,
        var_names: list[str],
        save_fig: bool,
    ) -> Path:
        """Write activity-specific graph data, edges, and optional visualizations.

        Arguments:
            result: The graph discovery result containing edges and adjacency matrices.
            var_names: List of human-readable variable names for labeling.
            save_fig: If True, renders and saves a PDF visualization of the graph.

        Returns:
            The Path to the created activity directory.

        """
        activity_dir = self.out / result.activity_name.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)

        # Save raw matrices for downstream numerical analysis
        np.save(activity_dir / "graph.npy", result.filtered_graph)
        np.save(activity_dir / "val_matrix.npy", result.filtered_val)

        # Serialize edges to both JSON and CSV formats
        with (activity_dir / "edges.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in result.edges], f, indent=2)

        self.write_edges_csv(result.edges, activity_dir / "edges.csv")

        if save_fig:
            self._save_graph_figure(result, var_names, activity_dir)

        return activity_dir

    def write_rule_library(
        self,
        rule_library: dict[int, list[Rule]],
        activity_models: dict[int, ActivityModel],
        metadata: dict[str, Any],
    ) -> None:
        """Serialize the trained rule library and activity models to JSON and text.

        Arguments:
            rule_library: Mapping of activity IDs to their selected Rule objects.
            activity_models: Mapping of activity IDs to ActivityModel metadata.
            metadata: General pipeline metadata (e.g., timestamps, versioning).

        """
        # Convert dataclasses to dicts for JSON serialization
        payload = {
            "metadata": metadata,
            "activities": {
                model.activity_name: [asdict(r) for r in rule_library[aid]] for aid, model in activity_models.items()
            },
            "activity_models": {model.activity_name: asdict(model) for model in activity_models.values()},
        }

        self.write_json("classification_rules.json", payload)

        # Generate a human-readable text summary of the rules
        lines = []
        for aid in sorted(rule_library):
            model = activity_models[aid]
            lines.append(f"[{model.activity_name}]")
            for idx, rule in enumerate(rule_library[aid], start=1):
                lines.append(f"{idx}. {rule.rule_text}")
            lines.append("")

        (self.out / "classification_rules.txt").write_text("\n".join(lines), encoding="utf-8")

    def write_evaluation_report(self, metrics: EvaluationMetrics) -> None:
        """Write the final performance metrics and confusion matrix.

        Arguments:
            metrics: Structured evaluation results from the Scorer.

        """
        # Save metrics as JSON
        self.write_json("evaluation_metrics.json", asdict(metrics))

        # Generate the visual confusion matrix
        # (Assuming label_names are derived from the report or passed in)
        label_names = [k for k in metrics.classification_report if k not in ("accuracy", "macro avg", "weighted avg")]
        self.write_confusion_matrix(np.array(metrics.confusion_matrix), label_names)

    def write_confusion_matrix(
        self,
        confusion: np.ndarray,
        label_names: list[str],
        filename_base: str = "deterministic_confusion_matrix",
    ) -> None:
        """Generate and save a visual heatmap and a CSV version of the confusion matrix.

        Arguments:
            confusion: A square 2D array of shape (n_classes, n_classes).
            label_names: List of activity names corresponding to the matrix indices.
            filename_base: Base name for the output files (png and csv).

        """
        if confusion.size == 0:
            return

        # 1. Visual Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(len(label_names)),
            yticks=np.arange(len(label_names)),
            xticklabels=label_names,
            yticklabels=label_names,
            ylabel="True activity",
            xlabel="Predicted activity",
            title="Activity Classifier Confusion Matrix",
        )

        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cell counts with contrast-aware coloring
        threshold = confusion.max() / 2
        for i, j in np.ndindex(confusion.shape):
            ax.text(
                j,
                i,
                str(int(confusion[i, j])),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > threshold else "black",
            )

        fig.tight_layout()
        fig.savefig(self.out / f"{filename_base}.png")
        plt.close(fig)

        # 2. CSV Serialization
        csv_path = self.out / f"{filename_base}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred", *label_names])
            for idx, row in enumerate(confusion):
                writer.writerow([label_names[idx], *row.tolist()])

    def write_json(self, filename: str, data: Any) -> None:  # noqa: ANN401
        """Write arbitrary data to a JSON file in the output directory."""
        with (self.out / filename).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def write_raw_activity(
        self,
        activity_id: int,
        activity_name: str,
        n_samples: int,
        graph: np.ndarray,
        val_matrix: np.ndarray,
    ) -> None:
        """Save the raw graph and value matrices for a given activity."""
        activity_dir = self.out / activity_name.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "activity_id": activity_id,
            "activity_name": activity_name,
            "n_samples": n_samples,
            "graph": graph.tolist(),
            "val_matrix": val_matrix.tolist(),
        }
        with (activity_dir / "raw_graph.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def write_edges_csv(edges: list[CausalEdge], path: Path) -> None:
        """Save a list of causal edges to a CSV file.

        Arguments:
            edges: List of edge objects to serialize.
            path: Target file path.

        """
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["src", "dst", "lag", "strength", "sign", "mark"])
            writer.writeheader()
            for edge in edges:
                writer.writerow(asdict(edge))

    def _save_graph_figure(self, result: ActivityGraphResult, var_names: list[str], directory: Path) -> None:
        """Render the Tigramite time-series graph."""
        fig, _ = tp.plot_time_series_graph(
            figsize=(6, 6),
            val_matrix=result.filtered_val,
            graph=result.filtered_graph,
            var_names=var_names,
            link_colorbar_label="MCI Strength",
        )
        fig.tight_layout()
        fig.savefig(directory / f"{result.activity_name}.pdf")
        plt.close(fig)
