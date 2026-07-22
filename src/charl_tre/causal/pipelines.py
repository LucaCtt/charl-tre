import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from charl_tre.causal.graph import GraphProcessor
from charl_tre.causal.latents_loader import LatentsLoader
from charl_tre.causal.lcpcmci import LPCMCIRunner
from charl_tre.causal.paths import PathSearcher
from charl_tre.causal.results_writer import ResultsWriter
from charl_tre.causal.rules import RuleBuilder
from charl_tre.causal.scoring import ActivityScorer
from charl_tre.causal.types import ActivityGraphResult


def path_to_strings(path: tuple[tuple[int, int, int], ...]) -> list[str]:
    """Convert a path of (src, dst, lag) tuples into human-readable strings."""
    return [f"Z{src} -[{lag}]-> Z{dst}" for src, dst, lag in path]


def _make_var_names(n_mixtures: int, n_components: int) -> list[str]:
    return [f"m{i}_alpha{j}" for i in range(n_mixtures) for j in range(n_components)]


class RawDiscoveryPipeline:
    """Run LPCMCI for all activities and persist raw graph tensors as JSON."""

    def __init__(
        self,
        study_path: Path,
        activities: list[str],
        tau_min: int,
        tau_max: int,
        pc_alpha: float,
        max_workers: int | None = None,
    ) -> None:
        """Initialize pipeline with settings and tuning parameters."""
        self._study_path = study_path
        self._activities = activities
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._pc_alpha = pc_alpha
        self._max_workers = max_workers

    def run(self) -> None:
        """Execute the pipeline and return a summary of outputs."""
        train_data, _ = LatentsLoader(self._study_path).load()

        _, n_mixtures, n_components = next(iter(train_data.values())).shape
        var_names = _make_var_names(n_mixtures, n_components)

        runner = LPCMCIRunner(
            tau_min=self._tau_min,
            tau_max=self._tau_max,
        )
        lpcmci_results = runner.run(train_data, var_names, self._max_workers)

        out_dir = Path(self._study_path) / "causal_graphs_raw"
        writer = ResultsWriter(out_dir)

        for activity_id in sorted(lpcmci_results):
            val_matrix, graph = lpcmci_results[activity_id]
            writer.write_raw_activity(
                activity_id=activity_id,
                activity_name=self._activities[activity_id],
                n_samples=len(train_data[activity_id]),
                graph=graph,
                val_matrix=val_matrix,
            )


class PathSearchPipeline:
    """Load raw graphs, mine rules, calibrate, evaluate, and write all outputs."""

    def __init__(
        self,
        raw_dir: Path,
        filter_threshold: float = 0.1,
        path_threshold: float = 0.0,
        max_edges_per_path: int = 4,
        min_edges_per_path: int = 2,
        top_paths_per_activity: int = 150,
        # Classifier
        classifier_train_ratio: float = 0.7,
        segment_length: int = 64,
        segment_hop: int = 12,
        max_segments_per_activity: int | None = None,
    ) -> None:
        """Initialize pipeline with settings and tuning parameters."""
        self.raw_dir = raw_dir
        self.filter_threshold = filter_threshold
        self.path_threshold = path_threshold
        self.max_edges_per_path = max_edges_per_path
        self.min_edges_per_path = min_edges_per_path
        self.top_paths_per_activity = top_paths_per_activity
        self.classifier_train_ratio = classifier_train_ratio
        self.segment_length = segment_length
        self.segment_hop = segment_hop
        self.max_segments = max_segments_per_activity

    def run(self) -> None:
        """Execute the pipeline and return a summary of outputs."""
        raw_graphs = self._load_raw(self.raw_dir)

        loader = LatentsLoader(s.study_path)
        train_data, test_data = loader.load()

        activity_names: dict[int, str] = {aid: s.activities[aid] for aid in sorted(raw_graphs)}
        processor = GraphProcessor(tau_min=tau_min, tau_max=tau_max)
        out_dir = Path(s.study_path) / "causal_path_search" / ind_test_name
        writer = ResultsWriter(out_dir)

        graph_results: dict[int, ActivityGraphResult] = {}
        for activity_id in sorted(raw_graphs):
            raw = raw_graphs[activity_id]
            result = processor.process(
                activity_id=activity_id,
                activity_name=s.activities[activity_id],
                n_samples=raw["n_samples"],
                raw_graph=raw["graph"],
                raw_val=raw["val_matrix"],
                filter_threshold=self.filter_threshold,
                path_threshold=self.path_threshold,
                max_edges_per_path=self.max_edges_per_path,
                min_edges_per_path=self.min_edges_per_path,
                top_paths_per_activity=self.top_paths_per_activity,
            )
            graph_results[activity_id] = result
            writer.write_activity(result, var_names)

        candidate_paths = {aid: r.candidate_paths for aid, r in graph_results.items()}
        strength_maps = {aid: r.strength_map for aid, r in graph_results.items()}
        edges_for_rules = {aid: r.path_edges for aid, r in graph_results.items()}

        rule_builder = RuleBuilder(
            activity_names=activity_names,
            var_names=var_names,
        )
        rule_library, activity_models = rule_builder.build(edges_for_rules, train_data)

        scorer = ActivityScorer()
        scorer.calibrate(
            rule_library=rule_library,
            activity_models=activity_models,
            train_data=train_data,
            segment_length=self.segment_length,
            segment_hop=self.segment_hop,
            max_per_activity=self.max_segments,
            clip=self.selection_cfg.calibration_clip,
        )

        searcher = PathSearcher()
        selected_paths = searcher.select_unique(candidate_paths, strength_maps)
        signatures = searcher.build_signatures(
            selected_paths=selected_paths,
            strength_maps=strength_maps,
            labelled_data=train_data,
            activity_names=activity_names,
            n_states=n_states,
        )

        path_library = {
            activity_names[aid]: {
                "activity_id": sig.activity_id,
                "own_score": sig.own_score,
                "margin_to_next_activity": sig.margin_to_next_activity,
                "strictly_unique": sig.strictly_unique,
                "path": path_to_strings(sig.path),
                "path_edges": [list(e) for e in sig.path],
                "edge_states": [asdict(es) for es in sig.edge_states],
            }
            for aid, sig in signatures.items()
        }

        writer.write_rule_library(rule_library, activity_models)
        writer._write_json("path_library.json", path_library)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw(raw_dir: Path) -> ict[int, dict]:
        raw_graphs: dict[int, dict] = {}

        for info in manifest.activities.values():
            raw_path = Path(info["raw_graph_json"])
            if not raw_path.is_absolute() and not raw_path.exists():
                raw_path = raw_dir / raw_path
            payload = json.loads(raw_path.read_text())
            raw_graphs[int(payload["activity_id"])] = {
                "activity_name": payload["activity_name"],
                "n_samples": int(payload["n_samples"]),
                "graph": np.array(payload["graph"], dtype=object),
                "val_matrix": np.array(payload["val_matrix"], dtype=np.float64),
            }
        return manifest, raw_graphs

    def _graph_summary(
        self,
        graph_results: dict[int, ActivityGraphResult],
        activity_names: dict[int, str],
        manifest: dict,
        var_names: list[str],
    ) -> dict:
        summary: dict = {
            "source_raw_dir": str(self.raw_dir),
            "independence_test": manifest["independence_test"],
            "tau_min": manifest["tau_min"],
            "tau_max": manifest["tau_max"],
            "filter_threshold": self.filter_threshold,
            "path_threshold": self.path_threshold,
            "n_latent_variables": len(var_names),
            "activities": {},
        }

        for aid, result in sorted(graph_results.items()):
            summary["activities"][activity_names[aid]] = {
                "activity_id": aid,
                "n_samples_raw": result.n_samples,
                "n_edges": len(result.edges),
                "top_edges": [e.__dict__ for e in result.edges[:20]],
                "top_paths": [
                    {"path": path_to_strings(path), "score": score} for path, score in result.candidate_paths[:10]
                ],
            }
        return summary
