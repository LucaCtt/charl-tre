"""High-level pipeline orchestrators.

Two pipelines are provided:

* ``RawDiscoveryPipeline`` — runs LPCMCI and saves raw ``(graph, val_matrix)``
  tensors per activity as JSON.  Equivalent to the old ``run_raw_causal_discovery``.

* ``PathSearchPipeline`` — loads previously saved raw graphs, extracts edges and
  paths, builds symbolic rules, calibrates models, evaluates a classifier, and
  writes all outputs.  Equivalent to the old ``run_path_search_from_raw``.

Both pipelines accept a ``Settings`` object and keyword-only tuning parameters.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from charl_tre.causal.data import LatentDataLoader
from charl_tre.causal.graph import GraphProcessor
from charl_tre.causal.io import ResultsWriter, path_to_strings
from charl_tre.causal.lcpcmci import LPCMCIRunner
from charl_tre.causal.paths import PathSearcher
from charl_tre.causal.rules import FallbackMiningConfig, RuleBudgetConfig, RuleBuilder, SelectionConfig
from charl_tre.causal.scoring import ActivityScorer
from charl_tre.causal.types import ActivityGraphResult
from charl_tre.settings import Settings


def _make_var_names(latent_dim: int, n_categories: int, use_full_onehot: bool) -> list[str]:
    if use_full_onehot:
        return [f"Z{i}_C{j}" for i in range(latent_dim) for j in range(n_categories)]
    return [f"Z{i}" for i in range(latent_dim)]


def _effective_tau_max(settings: Settings, tau_max: int | None) -> int:
    return tau_max if tau_max is not None else settings.classifier_window_size // settings.vae_window_size - 1


def _effective_stride(settings: Settings, stride: int | None) -> int:
    return stride if stride is not None else settings.vae_window_size


def _validate_tau(tau_min: int, tau_max: int) -> None:
    if tau_min < 1 or tau_max < tau_min:
        msg = f"Invalid tau range: tau_min={tau_min}, tau_max={tau_max}"
        raise ValueError(msg)


class RawDiscoveryPipeline:
    """Run LPCMCI for all activities and persist raw graph tensors as JSON."""

    def __init__(
        self,
        settings: Settings,
        tau_min: int = 1,
        tau_max: int | None = None,
        pc_alpha: float = 1e-3,
        stride: int | None = None,
        use_full_onehot: bool = True,
        max_workers: int | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize pipeline with settings and tuning parameters."""
        self.settings = settings
        self.tau_min = tau_min
        self.tau_max_arg = tau_max
        self.pc_alpha = pc_alpha
        self.stride_arg = stride
        self.use_full_onehot = use_full_onehot
        self.max_workers = max_workers
        self.debug = debug

    def run(self) -> dict[str, Any]:
        """Execute the pipeline and return a summary of outputs."""
        s = self.settings
        tau_max = _effective_tau_max(s, self.tau_max_arg)
        stride = _effective_stride(s, self.stride_arg)
        _validate_tau(self.tau_min, tau_max)

        loader = LatentDataLoader(s.study_path)
        latents, labels, latent_dim, n_categories, _ = loader.load(use_full_onehot=self.use_full_onehot)
        var_names = _make_var_names(latent_dim, n_categories, self.use_full_onehot)
        labelled_data = loader.split_by_activity(latents, labels, stride)

        runner = LPCMCIRunner(
            tau_min=self.tau_min,
            tau_max=tau_max,
        )
        lpcmci_results = runner.run_parallel(labelled_data, var_names, self.max_workers)

        out_dir = Path(s.study_path) / "causal_graphs_raw"
        writer = ResultsWriter(out_dir)

        for activity_id in sorted(lpcmci_results):
            val_matrix, graph = lpcmci_results[activity_id]
            writer.write_raw_activity(
                activity_id=activity_id,
                activity_name=s.activities[activity_id],
                n_samples=len(labelled_data[activity_id]),
                graph=graph,
                val_matrix=val_matrix,
            )

        manifest = {
            "study_path": s.study_path,
            "tau_min": self.tau_min,
            "tau_max": tau_max,
            "pc_alpha": self.pc_alpha,
            "stride": stride,
            "latent_representation": "onehot_flat" if self.use_full_onehot else "argmax_index",
            "latent_dim": latent_dim,
            "n_categories": n_categories,
            "n_latent_variables": len(var_names),
            "var_names": var_names,
            "activities": {
                s.activities[aid]: {
                    "activity_id": aid,
                    "raw_graph_json": f"{s.activities[aid].replace(' ', '_')}/raw_graph.json",
                }
                for aid in sorted(lpcmci_results)
            },
        }
        writer.write_json("manifest.json", manifest)
        return {"output_dir": str(out_dir), "manifest": manifest}


class PathSearchPipeline:
    """Load raw graphs, mine rules, calibrate, evaluate, and write all outputs."""

    def __init__(
        self,
        settings: Settings,
        raw_dir: Path,
        filter_threshold: float = 0.1,
        path_threshold: float = 0.0,
        max_edges_per_path: int = 4,
        min_edges_per_path: int = 2,
        top_paths_per_activity: int = 150,
        # Rule builder
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
        # Classifier
        classifier_stride: int = 75,
        classifier_train_ratio: float = 0.7,
        segment_length: int = 64,
        segment_hop: int = 12,
        max_segments_per_activity: int | None = None,
        save_figs: bool = True,
    ) -> None:
        """Initialize pipeline with settings and tuning parameters."""
        self.settings = settings
        self.raw_dir = raw_dir
        self.filter_threshold = filter_threshold
        self.path_threshold = path_threshold
        self.max_edges_per_path = max_edges_per_path
        self.min_edges_per_path = min_edges_per_path
        self.top_paths_per_activity = top_paths_per_activity
        self.budget_cfg = RuleBudgetConfig(
            top_rules_per_activity=top_rules_per_activity,
            min_rules_per_activity=min_rules_per_activity,
            max_rules_scale=max_rules_scale,
            low_edge_threshold=low_edge_threshold,
            low_edge_budget_ratio=low_edge_budget_ratio,
        )
        self.fallback_cfg = FallbackMiningConfig(
            top_source_vars=top_fallback_source_vars,
            top_target_vars=top_fallback_target_vars,
            min_condition_count=min_condition_count,
            min_delta=fallback_min_delta,
            pool_size=fallback_pool_size,
        )
        self.selection_cfg = SelectionConfig(
            core_fraction=core_fraction,
            min_delta=rule_min_delta,
            tie_breaker_weight=tie_breaker_weight,
            coverage_weight=coverage_weight,
            prior_weight=prior_weight,
            prototype_weight=prototype_weight,
            min_margin=min_margin,
            calibration_clip=calibration_clip,
        )
        self.classifier_stride = classifier_stride
        self.classifier_train_ratio = classifier_train_ratio
        self.segment_length = segment_length
        self.segment_hop = segment_hop
        self.max_segments = max_segments_per_activity
        self.save_figs = save_figs

    def run(self) -> dict[str, Any]:
        """Execute the pipeline and return a summary of outputs."""
        s = self.settings
        manifest, raw_graphs = self._load_raw(self.raw_dir)
        tau_min = int(manifest["tau_min"])
        tau_max = int(manifest["tau_max"])
        var_names = list(manifest["var_names"])
        ind_test_name = str(manifest["independence_test"])
        use_full_onehot = manifest.get("latent_representation", "onehot_flat") == "onehot_flat"

        loader = LatentDataLoader(s.study_path)
        latents, labels, _, _, n_states = loader.load(use_full_onehot=use_full_onehot)
        classifier_data = loader.split_by_activity(latents, labels, self.classifier_stride)
        train_data, test_data = loader.train_test_split(
            classifier_data,
            self.classifier_train_ratio,
            min_test_length=self.segment_length + tau_max,
        )

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
            writer.write_activity(result, var_names, self.save_figs)

        candidate_paths = {aid: r.candidate_paths for aid, r in graph_results.items()}
        strength_maps = {aid: r.strength_map for aid, r in graph_results.items()}
        edges_for_rules = {aid: r.path_edges for aid, r in graph_results.items()}

        rule_builder = RuleBuilder(
            activity_names=activity_names,
            var_names=var_names,
            budget_cfg=self.budget_cfg,
            fallback_cfg=self.fallback_cfg,
            selection_cfg=self.selection_cfg,
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

        classifier_metrics = scorer.evaluate(
            rule_library=rule_library,
            activity_models=activity_models,
            test_data=test_data,
            activity_names=activity_names,
            segment_length=self.segment_length,
            segment_hop=self.segment_hop,
            max_per_activity=self.max_segments,
        )

        label_names = [activity_names[i] for i in classifier_metrics.labels]
        cm = np.array(classifier_metrics.confusion_matrix, dtype=int)
        writer.write_confusion_matrix(cm, label_names)

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

        metadata = self._metadata(manifest)
        writer.write_rule_library(rule_library, activity_models, metadata)
        writer.write_json("path_library.json", path_library)
        writer.write_json(
            "graphs_summary.json",
            self._graph_summary(graph_results, activity_names, manifest, var_names),
        )
        writer.write_json("deterministic_classifier_metrics.json", classifier_metrics)

        return {
            "output_dir": str(out_dir),
            "rule_library": rule_library,
            "path_library": path_library,
            "classifier_metrics": classifier_metrics,
            "activity_models": activity_models,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw(raw_dir: Path) -> tuple[dict, dict[int, dict]]:
        manifest_path = raw_dir / "manifest.json"
        if not manifest_path.exists():
            msg = f"Missing manifest: {manifest_path}"
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text())
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

    def _metadata(self, manifest: dict) -> dict:
        cfg = self.selection_cfg
        fbk = self.fallback_cfg
        bgt = self.budget_cfg
        return {
            "independence_test": manifest["independence_test"],
            "source_raw_dir": str(self.raw_dir),
            "filter_threshold": self.filter_threshold,
            "path_threshold": self.path_threshold,
            "classifier_stride": self.classifier_stride,
            "classifier_train_ratio": self.classifier_train_ratio,
            "segment_length": self.segment_length,
            "segment_hop": self.segment_hop,
            "n_latent_variables": len(manifest["var_names"]),
            "var_names": manifest["var_names"],
            "min_rules_per_activity": bgt.min_rules_per_activity,
            "low_edge_threshold": bgt.low_edge_threshold,
            "top_fallback_source_vars": fbk.top_source_vars,
            "top_fallback_target_vars": fbk.top_target_vars,
            "min_condition_count": fbk.min_condition_count,
            "fallback_min_delta": fbk.min_delta,
            "fallback_pool_size": fbk.pool_size,
            "core_fraction": cfg.core_fraction,
            "tie_breaker_weight": cfg.tie_breaker_weight,
            "coverage_weight": cfg.coverage_weight,
            "prior_weight": cfg.prior_weight,
            "prototype_weight": cfg.prototype_weight,
            "min_margin": cfg.min_margin,
            "max_rules_scale": bgt.max_rules_scale,
            "low_edge_budget_ratio": bgt.low_edge_budget_ratio,
            "calibration_clip": cfg.calibration_clip,
        }
