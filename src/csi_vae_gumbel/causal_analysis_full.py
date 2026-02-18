from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.lpcmci import LPCMCI

from csi_vae_gumbel.settings import Settings


@dataclass(frozen=True)
class CausalEdge:
    src: int
    dst: int
    lag: int
    strength: float
    sign: int
    mark: str


@dataclass(frozen=True)
class PathEdgeState:
    src: int
    dst: int
    lag: int
    strength: float
    src_state: int
    dst_state: int
    support: float
    info_gain: float
    joint_probs: tuple[float, ...]
    src_probs: tuple[float, ...]
    dst_probs: tuple[float, ...]


@dataclass(frozen=True)
class ActivityPathSignature:
    activity_id: int
    activity_name: str
    path: tuple[tuple[int, int, int], ...]
    own_score: float
    margin_to_next_activity: float
    strictly_unique: bool
    edge_states: tuple[PathEdgeState, ...]


@dataclass(frozen=True)
class ActivityAnalysisResult:
    activity_id: int
    activity_name: str
    n_samples: int
    filtered_graph: np.ndarray
    filtered_val: np.ndarray
    raw_graph: np.ndarray
    raw_val: np.ndarray
    edges: list[CausalEdge]
    path_edges: list[CausalEdge]
    candidate_paths: list[tuple[tuple[tuple[int, int, int], ...], float]]
    strength_map: dict[tuple[int, int, int], float]


def get_independence_test(ind_test_name: str) -> tuple[Any, bool]:
    """Build the Tigramite conditional-independence test."""
    normalized = ind_test_name.lower()
    is_categorical = normalized in {"cmiknn", "gsquared", "cmisymb"}

    match normalized:
        case "parcorr":
            return ParCorr(significance="analytic"), is_categorical
        case "robustparcorr":
            return RobustParCorr(significance="analytic"), is_categorical
        case "gpdc":
            return GPDCtorch(), is_categorical
        case "gsquared":
            return Gsquared(), is_categorical
        case "cmisymb":
            return CMIsymb(), is_categorical
        case "cmiknn":
            return CMIknn(), is_categorical
        case _:
            msg = f"Unknown independence test: {ind_test_name}"
            raise ValueError(msg)


def compute_gsquared_signs(data: np.ndarray, val_matrix: np.ndarray, tau_min: int, tau_max: int) -> None:
    """Infer a sign for each G-squared link via Spearman correlation."""
    n_vars = val_matrix.shape[0]
    for src in range(n_vars):
        for dst in range(n_vars):
            for lag in range(tau_min, tau_max + 1):
                strength = val_matrix[src, dst, lag]
                if strength <= 0:
                    continue
                y_series = data[lag:, dst]
                x_series = data[:-lag, src]
                res = stats.spearmanr(x_series, y_series)
                rho = float(res.statistic)  # pyright: ignore[reportAttributeAccessIssue]
                val_matrix[src, dst, lag] *= np.sign(rho)


def load_latent_hards(
    latent_path: Path,
    label_path: Path,
    use_full_onehot: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Load hard latents as either full one-hot space or argmax-compressed categories."""
    latents_hard = np.load(latent_path)
    labels = np.load(label_path).astype(int)

    if latents_hard.ndim != 3:
        msg = f"Expected (T, latent_dim, n_categories) for latents_hard, got {latents_hard.shape}"
        raise ValueError(msg)
    if labels.ndim != 1:
        msg = f"Expected (T,) for labels, got {labels.shape}"
        raise ValueError(msg)
    if latents_hard.shape[0] != labels.shape[0]:
        msg = "latents_hard and labels have mismatched first dimension."
        raise ValueError(msg)

    latent_dim = int(latents_hard.shape[1])
    n_categories = int(latents_hard.shape[2])

    if use_full_onehot:
        latents = latents_hard.reshape(latents_hard.shape[0], latent_dim * n_categories).astype(np.int8)
        n_states = 2
    else:
        latents = np.argmax(latents_hard, axis=2).astype(np.int16)
        n_states = n_categories
    return latents, labels, latent_dim, n_categories, n_states


def split_by_activity(latents: np.ndarray, labels: np.ndarray, stride: int) -> dict[int, np.ndarray]:
    """Create one latent time-series per activity label."""
    if stride < 1:
        msg = "stride must be >= 1."
        raise ValueError(msg)

    grouped: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        seq = latents[labels == label]
        grouped[int(label)] = seq[::stride]
    return grouped


def filter_graph(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    threshold: float,
    tau_min: int,
    tau_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop weak, ambiguous, and contemporaneous links."""
    filtered_graph = graph.copy()
    filtered_val = val_matrix.copy()

    weak_links = np.abs(filtered_val) < threshold
    filtered_val[weak_links] = 0
    filtered_graph[weak_links] = ""

    for src in range(filtered_graph.shape[0]):
        for dst in range(filtered_graph.shape[1]):
            for lag in range(filtered_graph.shape[2]):
                mark = filtered_graph[src, dst, lag]
                if not mark:
                    continue
                if lag < tau_min or lag > tau_max:
                    filtered_graph[src, dst, lag] = ""
                    filtered_val[src, dst, lag] = 0
                    continue
                # Keep lagged links that point into the target variable.
                if mark not in {"-->", "o->"}:
                    filtered_graph[src, dst, lag] = ""
                    filtered_val[src, dst, lag] = 0
    return filtered_graph, filtered_val


def extract_edges(filtered_graph: np.ndarray, filtered_val: np.ndarray) -> list[CausalEdge]:
    """Convert Tigramite matrices to a list of directed weighted edges."""
    edges: list[CausalEdge] = []
    for src in range(filtered_graph.shape[0]):
        for dst in range(filtered_graph.shape[1]):
            for lag in range(filtered_graph.shape[2]):
                mark = filtered_graph[src, dst, lag]
                if mark not in {"-->", "o->"}:
                    continue
                value = float(filtered_val[src, dst, lag])
                strength = abs(value)
                if strength == 0:
                    continue
                edges.append(
                    CausalEdge(
                        src=src,
                        dst=dst,
                        lag=lag,
                        strength=strength,
                        sign=int(np.sign(value)),
                        mark=mark,
                    ),
                )
    return sorted(edges, key=lambda edge: edge.strength, reverse=True)


def edge_strength_map(edges: list[CausalEdge]) -> dict[tuple[int, int, int], float]:
    """Create a lookup map for edge strengths."""
    return {(edge.src, edge.dst, edge.lag): edge.strength for edge in edges}


def enumerate_best_paths(
    edges: list[CausalEdge],
    n_vars: int,
    max_edges: int,
    min_edges: int,
    top_k: int,
) -> list[tuple[tuple[tuple[int, int, int], ...], float]]:
    """Enumerate high-strength simple paths in the activity graph."""
    adjacency: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.src].append((edge.dst, edge.lag, edge.strength))

    for src in adjacency:
        adjacency[src].sort(key=lambda item: item[2], reverse=True)

    scored_paths: dict[tuple[tuple[int, int, int], ...], float] = {}

    def dfs(
        current: int,
        visited: set[int],
        path: list[tuple[int, int, int]],
        score: float,
    ) -> None:
        if len(path) >= min_edges:
            signature = tuple(path)
            prev = scored_paths.get(signature, float("-inf"))
            if score > prev:
                scored_paths[signature] = score

        if len(path) >= max_edges:
            return

        for dst, lag, strength in adjacency.get(current, []):
            if dst in visited:
                continue
            visited.add(dst)
            path.append((current, dst, lag))
            dfs(dst, visited, path, score + strength)
            path.pop()
            visited.remove(dst)

    for start in range(n_vars):
        dfs(current=start, visited={start}, path=[], score=0.0)

    # Always keep single-edge paths available for uniqueness assignment.
    for edge in edges:
        signature = ((edge.src, edge.dst, edge.lag),)
        prev = scored_paths.get(signature, float("-inf"))
        if edge.strength > prev:
            scored_paths[signature] = edge.strength

    ranked = sorted(scored_paths.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def score_path(path: tuple[tuple[int, int, int], ...], strengths: dict[tuple[int, int, int], float]) -> float:
    """Compute total strength of a path in one activity graph."""
    return float(sum(strengths.get(edge, 0.0) for edge in path))


def select_unique_paths(
    candidate_paths: dict[int, list[tuple[tuple[tuple[int, int, int], ...], float]]],
    strength_maps: dict[int, dict[tuple[int, int, int], float]],
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Assign one unique path per activity with max discriminative margin."""
    activity_ids = sorted(candidate_paths)
    all_paths: list[tuple[tuple[int, int, int], ...]] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()

    for activity_id in activity_ids:
        for path, _ in candidate_paths[activity_id]:
            if path in seen:
                continue
            seen.add(path)
            all_paths.append(path)

    if len(all_paths) < len(activity_ids):
        # Fallback: keep the best discriminative path per activity even if strict uniqueness is impossible.
        fallback: dict[int, tuple[tuple[int, int, int], ...]] = {}
        for activity_id in activity_ids:
            own_map = strength_maps[activity_id]
            best_path: tuple[tuple[int, int, int], ...] | None = None
            best_margin = float("-inf")
            for path, _ in candidate_paths[activity_id]:
                own_score = score_path(path, own_map)
                other_scores = [
                    score_path(path, strength_maps[other_id])
                    for other_id in activity_ids
                    if other_id != activity_id
                ]
                margin = own_score - max(other_scores, default=0.0)
                if margin > best_margin:
                    best_margin = margin
                    best_path = path
            if best_path is None:
                msg = f"No candidate paths available for activity {activity_id}."
                raise RuntimeError(msg)
            fallback[activity_id] = best_path
        return fallback

    # Score matrix: discriminative margin for assigning path_j to activity_i.
    matrix = np.full((len(activity_ids), len(all_paths)), fill_value=-1e9, dtype=np.float64)
    for row, activity_id in enumerate(activity_ids):
        own_candidates = {path for path, _ in candidate_paths[activity_id]}
        own_map = strength_maps[activity_id]
        for col, path in enumerate(all_paths):
            if path not in own_candidates:
                continue
            own_score = score_path(path, own_map)
            other_scores = [
                score_path(path, strength_maps[other_id])
                for other_id in activity_ids
                if other_id != activity_id
            ]
            matrix[row, col] = own_score - max(other_scores, default=0.0)

    row_ind, col_ind = linear_sum_assignment(-matrix)
    assigned: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for row, col in zip(row_ind, col_ind, strict=True):
        assigned[activity_ids[row]] = all_paths[col]
    return assigned


def edge_joint_distribution(data: np.ndarray, src: int, dst: int, lag: int, n_states: int) -> np.ndarray:
    """Compute normalized joint distribution of lagged edge states."""
    n_pairs = n_states * n_states
    if lag <= 0 or len(data) <= lag:
        return np.full(n_pairs, fill_value=1.0 / n_pairs, dtype=np.float64)

    src_values = data[:-lag, src].astype(int)
    dst_values = data[lag:, dst].astype(int)
    encoded = src_values * n_states + dst_values
    counts = np.bincount(encoded, minlength=n_pairs).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return np.full(n_pairs, fill_value=1.0 / n_pairs, dtype=np.float64)
    return counts / total


def informative_state_pair(
    joint_probs: np.ndarray,
    n_states: int,
) -> tuple[int, int, float, float, np.ndarray, np.ndarray]:
    """Pick the most informative state pair using pointwise MI contribution."""
    joint = joint_probs.reshape(n_states, n_states)
    src_probs = np.sum(joint, axis=1)
    dst_probs = np.sum(joint, axis=0)
    expected = np.outer(src_probs, dst_probs)

    eps = 1e-12
    pmi_contrib = joint * np.log((joint + eps) / (expected + eps))
    best_idx = np.unravel_index(int(np.argmax(pmi_contrib)), pmi_contrib.shape)
    info_gain = float(max(pmi_contrib[best_idx], 0.0))
    support = float(joint[best_idx])

    if info_gain <= 0:
        best_idx = np.unravel_index(int(np.argmax(joint)), joint.shape)
        support = float(joint[best_idx])
        info_gain = 0.0

    return int(best_idx[0]), int(best_idx[1]), support, info_gain, src_probs, dst_probs


def build_activity_signatures(
    selected_paths: dict[int, tuple[tuple[int, int, int], ...]],
    strength_maps: dict[int, dict[tuple[int, int, int], float]],
    labelled_data: dict[int, np.ndarray],
    settings: Settings,
    n_states: int,
) -> dict[int, ActivityPathSignature]:
    """Add state templates and discriminative margins to each selected path."""
    signatures: dict[int, ActivityPathSignature] = {}
    for activity_id, path in selected_paths.items():
        own_score = score_path(path, strength_maps[activity_id])
        other_scores = [
            score_path(path, strength_maps[other_id])
            for other_id in selected_paths
            if other_id != activity_id
        ]
        margin = own_score - max(other_scores, default=0.0)
        edge_states: list[PathEdgeState] = []
        for src, dst, lag in path:
            joint_probs = edge_joint_distribution(
                data=labelled_data[activity_id],
                src=src,
                dst=dst,
                lag=lag,
                n_states=n_states,
            )
            src_state, dst_state, support, info_gain, src_probs, dst_probs = informative_state_pair(
                joint_probs=joint_probs,
                n_states=n_states,
            )
            edge_states.append(
                PathEdgeState(
                    src=src,
                    dst=dst,
                    lag=lag,
                    strength=float(strength_maps[activity_id].get((src, dst, lag), 0.0)),
                    src_state=src_state,
                    dst_state=dst_state,
                    support=support,
                    info_gain=info_gain,
                    joint_probs=tuple(float(x) for x in joint_probs.tolist()),
                    src_probs=tuple(float(x) for x in src_probs.tolist()),
                    dst_probs=tuple(float(x) for x in dst_probs.tolist()),
                ),
            )

        signatures[activity_id] = ActivityPathSignature(
            activity_id=activity_id,
            activity_name=settings.activities[activity_id],
            path=path,
            own_score=own_score,
            margin_to_next_activity=margin,
            strictly_unique=margin > 0,
            edge_states=tuple(edge_states),
        )
    return signatures


def segment_sequences(
    labelled_data: dict[int, np.ndarray],
    segment_length: int,
    segment_hop: int,
    max_segments_per_activity: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Split activity sequences into fixed-size chunks for deterministic evaluation."""
    if segment_length < 2:
        msg = "segment_length must be >= 2."
        raise ValueError(msg)
    if segment_hop < 1:
        msg = "segment_hop must be >= 1."
        raise ValueError(msg)

    segments: list[tuple[np.ndarray, int]] = []
    for label, data in labelled_data.items():
        if len(data) < segment_length:
            continue

        starts = list(range(0, len(data) - segment_length + 1, segment_hop))
        if max_segments_per_activity is not None and len(starts) > max_segments_per_activity:
            # Deterministic subsampling over the full sequence.
            keep = np.linspace(0, len(starts) - 1, num=max_segments_per_activity, dtype=int)
            starts = [starts[idx] for idx in keep]

        for start in starts:
            end = start + segment_length
            segments.append((data[start:end], label))
    return segments


def split_train_test_by_activity(
    labelled_data: dict[int, np.ndarray],
    train_ratio: float,
    min_test_length: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Create deterministic train/test splits per activity without shuffling."""
    if not (0.1 <= train_ratio <= 0.9):
        msg = "train_ratio must be in [0.1, 0.9]."
        raise ValueError(msg)

    train: dict[int, np.ndarray] = {}
    test: dict[int, np.ndarray] = {}
    for label, data in labelled_data.items():
        n = len(data)
        if n <= 2:
            train[label] = data
            test[label] = data
            continue

        split_idx = int(n * train_ratio)
        split_idx = max(1, min(split_idx, n - 1))
        if n - split_idx < min_test_length:
            split_idx = max(1, n - min_test_length)
        train[label] = data[:split_idx]
        test[label] = data[split_idx:]
    return train, test


def edge_pair_frequency(
    sequence: np.ndarray,
    src: int,
    dst: int,
    lag: int,
    src_state: int,
    dst_state: int,
) -> float:
    """Compute how often one lagged state-pair appears in a sequence."""
    if lag <= 0 or len(sequence) <= lag:
        return 0.0
    src_values = sequence[:-lag, src]
    dst_values = sequence[lag:, dst]
    hits = (src_values == src_state) & (dst_values == dst_state)
    return float(np.mean(hits))


def edge_distribution_similarity(
    sequence: np.ndarray,
    edge: PathEdgeState,
    n_states: int,
) -> float:
    """Compare edge-state distributions using Jensen-Shannon similarity."""
    observed = edge_joint_distribution(
        data=sequence,
        src=edge.src,
        dst=edge.dst,
        lag=edge.lag,
        n_states=n_states,
    )
    template = np.asarray(edge.joint_probs, dtype=np.float64)
    dist = float(jensenshannon(template, observed, base=2.0))
    if np.isnan(dist):
        return 0.0
    return max(0.0, 1.0 - dist)


def score_sequence_for_activity(
    sequence: np.ndarray,
    signature: ActivityPathSignature,
    n_states: int,
) -> float:
    """Score one sequence against an activity path signature."""
    if not signature.edge_states:
        return 0.0

    weights = np.array([edge.strength * (1.0 + edge.info_gain) for edge in signature.edge_states], dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0

    edge_scores = np.array(
        [
            0.7
            * edge_distribution_similarity(
                sequence=sequence,
                edge=edge,
                n_states=n_states,
            )
            + 0.3
            * (
                1.0
                - abs(
                    edge_pair_frequency(
                        sequence=sequence,
                        src=edge.src,
                        dst=edge.dst,
                        lag=edge.lag,
                        src_state=edge.src_state,
                        dst_state=edge.dst_state,
                    )
                    - edge.support
                )
            )
            for edge in signature.edge_states
        ],
        dtype=np.float64,
    )
    return float(np.dot(weights, edge_scores) / weight_sum)


def run_deterministic_classifier(
    signatures: dict[int, ActivityPathSignature],
    labelled_data: dict[int, np.ndarray],
    settings: Settings,
    segment_length: int,
    segment_hop: int,
    n_states: int,
    max_segments_per_activity: int | None = None,
) -> dict[str, Any]:
    """Classify sequences by scoring activity-specific causal paths."""
    segments = segment_sequences(
        labelled_data=labelled_data,
        segment_length=segment_length,
        segment_hop=segment_hop,
        max_segments_per_activity=max_segments_per_activity,
    )
    if not segments:
        msg = "No segments available for deterministic classification."
        raise RuntimeError(msg)
    y_true: list[int] = []
    y_pred: list[int] = []

    for sequence, label in segments:
        scores = {
            activity_id: score_sequence_for_activity(
                sequence=sequence,
                signature=signature,
                n_states=n_states,
            )
            for activity_id, signature in signatures.items()
        }
        prediction = max(scores, key=scores.get)
        y_true.append(label)
        y_pred.append(prediction)

    labels = sorted(signatures)
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
    }


def save_confusion_matrix_plot(
    confusion: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> None:
    """Save confusion matrix as a figure."""
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
        title="Deterministic path-based classifier confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion.max() / 2 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                str(int(confusion[i, j])),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_edges_csv(edges: list[CausalEdge], output_path: Path) -> None:
    """Store one activity edge list as CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["src", "dst", "lag", "strength", "sign", "mark"],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow(asdict(edge))


def path_to_strings(path: tuple[tuple[int, int, int], ...]) -> list[str]:
    """Render path edges for concise human-readable output."""
    return [f"Z{src} -[{lag}]-> Z{dst}" for src, dst, lag in path]


def analyze_activity(
    activity_id: int,
    activity_name: str,
    data: np.ndarray,
    var_names: list[str],
    ind_test_name: str,
    is_categorical_test: bool,
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    threshold: float,
    debug: bool,
    max_edges_per_path: int,
    min_edges_per_path: int,
    top_paths_per_activity: int,
) -> ActivityAnalysisResult:
    """Run LPCMCI and candidate-path extraction for one activity."""
    activity_data = data if is_categorical_test else data.astype(np.float64, copy=False)
    data_type = np.ones_like(data, dtype=np.int16) if is_categorical_test else np.zeros_like(data, dtype=np.int16)
    dataframe = pp.DataFrame(activity_data, var_names=var_names, data_type=data_type)

    ind_test, _ = get_independence_test(ind_test_name)
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=ind_test, verbosity=int(debug))
    results = lpcmci.run_lpcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)

    raw_val = results["val_matrix"].copy()
    raw_graph = results["graph"].copy()
    if ind_test_name.lower() == "gsquared":
        compute_gsquared_signs(data=activity_data, val_matrix=raw_val, tau_min=tau_min, tau_max=tau_max)

    filtered_graph, filtered_val = filter_graph(
        graph=raw_graph,
        val_matrix=raw_val,
        threshold=threshold,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    path_graph, path_val = filter_graph(
        graph=raw_graph,
        val_matrix=raw_val,
        threshold=0.0,
        tau_min=tau_min,
        tau_max=tau_max,
    )

    edges = extract_edges(filtered_graph=filtered_graph, filtered_val=filtered_val)
    path_edges = extract_edges(filtered_graph=path_graph, filtered_val=path_val)
    strengths = edge_strength_map(path_edges)
    candidate_paths = enumerate_best_paths(
        edges=path_edges,
        n_vars=len(var_names),
        max_edges=max_edges_per_path,
        min_edges=min_edges_per_path,
        top_k=top_paths_per_activity,
    )

    if not candidate_paths:
        abs_values = np.abs(raw_val[:, :, tau_min : tau_max + 1])
        src, dst, lag_offset = np.unravel_index(np.argmax(abs_values), abs_values.shape)
        lag = int(lag_offset + tau_min)
        fallback_edge = (int(src), int(dst), lag)
        fallback_score = float(abs_values[src, dst, lag_offset])
        candidate_paths = [((fallback_edge,), fallback_score)]
        strengths[fallback_edge] = fallback_score

    return ActivityAnalysisResult(
        activity_id=activity_id,
        activity_name=activity_name,
        n_samples=len(data),
        filtered_graph=filtered_graph,
        filtered_val=filtered_val,
        raw_graph=raw_graph,
        raw_val=raw_val,
        edges=edges,
        path_edges=path_edges,
        candidate_paths=candidate_paths,
        strength_map=strengths,
    )


def run_full_causal_analysis(
    settings: Settings,
    ind_test_name: str = "parcorr",
    threshold: float = 0.1,
    tau_min: int = 1,
    tau_max: int | None = None,
    pc_alpha: float = 1e-3,
    stride: int | None = None,
    max_edges_per_path: int = 4,
    min_edges_per_path: int = 2,
    top_paths_per_activity: int = 150,
    segment_length: int = 128,
    segment_hop: int = 32,
    classifier_stride: int = 1,
    classifier_train_ratio: float = 0.7,
    max_segments_per_activity: int | None = None,
    max_workers: int | None = None,
    use_full_onehot: bool = True,
    save_figs: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """Run end-to-end causal analysis with graph export and deterministic classification."""
    tau_max_eff = tau_max if tau_max is not None else settings.test_window_size // settings.train_window_size - 1
    stride_eff = stride if stride is not None else settings.train_window_size

    if tau_min < 1 or tau_max_eff < tau_min:
        msg = "Invalid tau range."
        raise ValueError(msg)

    _, is_categorical_test = get_independence_test(ind_test_name)

    latents_path = Path(settings.study_path) / "latents" / "latents_hard.npy"
    labels_path = Path(settings.study_path) / "latents" / "labels.npy"
    latents, labels, latent_dim, n_categories, n_states = load_latent_hards(
        latent_path=latents_path,
        label_path=labels_path,
        use_full_onehot=use_full_onehot,
    )
    labelled_data = split_by_activity(latents=latents, labels=labels, stride=stride_eff)
    classifier_data = split_by_activity(latents=latents, labels=labels, stride=classifier_stride)
    classifier_train_data, classifier_test_data = split_train_test_by_activity(
        labelled_data=classifier_data,
        train_ratio=classifier_train_ratio,
        min_test_length=segment_length + tau_max_eff,
    )

    n_vars = latents.shape[1]
    if use_full_onehot:
        var_names = [f"Z{latent_idx}_C{cat_idx}" for latent_idx in range(latent_dim) for cat_idx in range(n_categories)]
    else:
        var_names = [f"Z{idx}" for idx in range(n_vars)]
    output_dir = Path(settings.study_path) / "causal_analysis_full" / ind_test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_summary: dict[str, Any] = {
        "independence_test": ind_test_name,
        "threshold": threshold,
        "tau_min": tau_min,
        "tau_max": tau_max_eff,
        "stride": stride_eff,
        "classifier_stride": classifier_stride,
        "classifier_train_ratio": classifier_train_ratio,
        "segment_length": segment_length,
        "segment_hop": segment_hop,
        "latent_representation": "onehot_flat" if use_full_onehot else "argmax_index",
        "latent_dim": latent_dim,
        "n_categories": n_categories,
        "n_states_per_variable": n_states,
        "n_latent_variables": n_vars,
        "activities": {},
    }

    edge_lists: dict[int, list[CausalEdge]] = {}
    strength_maps: dict[int, dict[tuple[int, int, int], float]] = {}
    candidate_paths: dict[int, list[tuple[tuple[tuple[int, int, int], ...], float]]] = {}
    activity_items = sorted(labelled_data.items())
    max_workers_eff = max_workers if max_workers is not None else len(activity_items)
    cpu_count = os.cpu_count() or len(activity_items)
    max_workers_eff = max(1, min(max_workers_eff, len(activity_items), cpu_count))

    analysis_results: dict[int, ActivityAnalysisResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers_eff) as executor:
        futures = {
            executor.submit(
                analyze_activity,
                activity_id=activity_id,
                activity_name=settings.activities[activity_id],
                data=data,
                var_names=var_names,
                ind_test_name=ind_test_name,
                is_categorical_test=is_categorical_test,
                tau_min=tau_min,
                tau_max=tau_max_eff,
                pc_alpha=pc_alpha,
                threshold=threshold,
                debug=debug,
                max_edges_per_path=max_edges_per_path,
                min_edges_per_path=min_edges_per_path,
                top_paths_per_activity=top_paths_per_activity,
            ): activity_id
            for activity_id, data in activity_items
        }
        for future in as_completed(futures):
            result = future.result()
            analysis_results[result.activity_id] = result

    for activity_id in sorted(analysis_results):
        result = analysis_results[activity_id]
        activity_name = result.activity_name

        edge_lists[activity_id] = result.edges
        strength_maps[activity_id] = result.strength_map
        candidate_paths[activity_id] = result.candidate_paths

        activity_dir = output_dir / activity_name.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)

        if save_figs:
            fig, _ = tp.plot_time_series_graph(
                figsize=(6, 6),
                val_matrix=result.filtered_val,
                graph=result.filtered_graph,
                var_names=var_names,
                link_colorbar_label="MCI Strength",
            )
            fig.tight_layout()
            fig.savefig(activity_dir / f"{activity_name}.pdf")
            plt.close(fig)

        np.save(activity_dir / "graph.npy", result.filtered_graph)
        np.save(activity_dir / "val_matrix.npy", result.filtered_val)
        with (activity_dir / "edges.json").open("w", encoding="utf-8") as file:
            json.dump([asdict(edge) for edge in result.edges], file, indent=2)
        save_edges_csv(edges=result.edges, output_path=activity_dir / "edges.csv")

        graph_summary["activities"][activity_name] = {
            "activity_id": activity_id,
            "n_samples": int(result.n_samples),
            "n_edges": len(result.edges),
            "top_edges": [asdict(edge) for edge in result.edges[:20]],
            "top_paths": [
                {"path": path_to_strings(path), "score": score}
                for path, score in result.candidate_paths[:10]
            ],
        }

    selected_paths = select_unique_paths(candidate_paths=candidate_paths, strength_maps=strength_maps)
    signatures = build_activity_signatures(
        selected_paths=selected_paths,
        strength_maps=strength_maps,
        labelled_data=classifier_train_data,
        settings=settings,
        n_states=n_states,
    )

    classifier_metrics = run_deterministic_classifier(
        signatures=signatures,
        labelled_data=classifier_test_data,
        settings=settings,
        segment_length=segment_length,
        segment_hop=segment_hop,
        n_states=n_states,
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
        output_path=output_dir / "deterministic_confusion_matrix.png",
    )

    with (output_dir / "deterministic_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
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

    with (output_dir / "path_library.json").open("w", encoding="utf-8") as file:
        json.dump(path_library, file, indent=2)
    with (output_dir / "graphs_summary.json").open("w", encoding="utf-8") as file:
        json.dump(graph_summary, file, indent=2)
    with (output_dir / "deterministic_classifier_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(classifier_metrics, file, indent=2)

    return {
        "output_dir": str(output_dir),
        "path_library": path_library,
        "classifier_metrics": classifier_metrics,
        "graph_summary": graph_summary,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full causal analysis."""
    settings = Settings()
    parser = argparse.ArgumentParser(description="Run full-latent causal analysis on latent_hard codes.")
    parser.add_argument("--study-path", type=str, default=settings.study_path)
    parser.add_argument("--ind-test", type=str, default="parcorr")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--tau-min", type=int, default=1)
    parser.add_argument("--tau-max", type=int, default=None)
    parser.add_argument("--pc-alpha", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=settings.train_window_size)
    parser.add_argument("--max-edges-per-path", type=int, default=4)
    parser.add_argument("--min-edges-per-path", type=int, default=2)
    parser.add_argument("--top-paths-per-activity", type=int, default=150)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--segment-hop", type=int, default=32)
    parser.add_argument("--classifier-stride", type=int, default=1)
    parser.add_argument("--classifier-train-ratio", type=float, default=0.7)
    parser.add_argument("--max-segments-per-activity", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--argmax-latents", action="store_true")
    parser.add_argument("--no-save-figs", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    settings = Settings(study_path=args.study_path)

    result = run_full_causal_analysis(
        settings=settings,
        ind_test_name=args.ind_test,
        threshold=args.threshold,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        stride=args.stride,
        max_edges_per_path=args.max_edges_per_path,
        min_edges_per_path=args.min_edges_per_path,
        top_paths_per_activity=args.top_paths_per_activity,
        segment_length=args.segment_length,
        segment_hop=args.segment_hop,
        classifier_stride=args.classifier_stride,
        classifier_train_ratio=args.classifier_train_ratio,
        max_segments_per_activity=args.max_segments_per_activity,
        max_workers=args.max_workers,
        use_full_onehot=not args.argmax_latents,
        save_figs=not args.no_save_figs,
        debug=args.debug,
    )
    print(json.dumps({"output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
