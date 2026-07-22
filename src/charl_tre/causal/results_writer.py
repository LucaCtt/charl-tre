import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tigramite import plotting as tp

from charl_tre.causal.rules import ActivityModel, Rule
from charl_tre.causal.types import ActivityGraphResult, CausalEdge


class ResultsWriter:
    """Handles serialization of causal discovery results, rule libraries, and evaluation metrics."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the writer and ensure the output directory exists.

        Arguments:
            output_dir: The base directory where all results will be stored.
                Will be created if it does not exist.

        """
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_activity(
        self,
        result: ActivityGraphResult,
        var_names: list[str],
        save_fig: bool = True,
    ) -> Path:
        """Write activity-specific graph data, edges, and optional visualizations.

        Arguments:
            result: The graph discovery result containing edges and adjacency matrices.
            var_names: List of human-readable variable names for labeling.
            save_fig: If True, renders and saves a PDF visualization of the graph.

        Returns:
            The Path to the created activity directory.

        """
        activity_dir = self._output_dir / result.activity_name.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)

        # Save raw matrices for downstream numerical analysis
        np.save(activity_dir / "graph.npy", result.filtered_graph)
        np.save(activity_dir / "val_matrix.npy", result.filtered_val)

        # Serialize edges to both JSON and CSV formats
        with (activity_dir / "edges.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in result.edges], f, indent=2)

        self._write_edges_json(result.edges, activity_dir / "edges.json")

        if save_fig:
            self._save_graph_figure(result, var_names, activity_dir)

        return activity_dir

    def write_rule_library(
        self,
        rule_library: dict[int, list[Rule]],
        activity_models: dict[int, ActivityModel],
    ) -> None:
        """Serialize the trained rule library and activity models to JSON and text.

        Arguments:
            rule_library: Mapping of activity IDs to their selected Rule objects.
            activity_models: Mapping of activity IDs to ActivityModel metadata.

        """
        # Convert dataclasses to dicts for JSON serialization
        payload = {
            "activities": {
                model.activity_name: [asdict(r) for r in rule_library[aid]] for aid, model in activity_models.items()
            },
            "activity_models": {model.activity_name: asdict(model) for model in activity_models.values()},
        }

        with (self._output_dir / "rules.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Generate a human-readable text summary of the rules
        lines = []
        for aid in sorted(rule_library):
            model = activity_models[aid]
            lines.append(f"[{model.activity_name}]")
            for idx, rule in enumerate(rule_library[aid], start=1):
                lines.append(f"{idx}. {rule.rule_text}")
            lines.append("")

        (self._output_dir / "rules.txt").write_text("\n".join(lines), encoding="utf-8")

    def write_raw_activity(
        self,
        activity_id: int,
        activity_name: str,
        n_samples: int,
        graph: np.ndarray,
        val_matrix: np.ndarray,
    ) -> None:
        """Save the raw graph and value matrices for a given activity."""
        activity_dir = self._output_dir / activity_name.replace(" ", "_")
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


    def _write_edges_json(self, edges: list[CausalEdge], path: Path) -> None:
        """Save a list of causal edges to a JSON file.

        Arguments:
            edges (list[CausalEdge]): List of edge objects to serialize.
            path (Path): Target file path.

        """
        with path.open("w", newline="", encoding="utf-8") as f:
            json.dump([asdict(edge) for edge in edges], f)

    def _save_graph_figure(self, result: ActivityGraphResult, var_names: list[str], directory: Path) -> None:
        """Render the Tigramite time-series graph.

        Arguments:
            result (ActivityGraphResult): The activity graph result containing the filtered graph and value matrix.
            var_names (list[str]): List of variable names for labeling the graph.
            directory (Path): Directory to save the figure.

        """
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
