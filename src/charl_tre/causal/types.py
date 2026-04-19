"""Shared dataclasses for the causal discovery pipeline."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalEdge:
    """Represents a causal edge from src to dst with a certain lag, strength, sign, and mark."""

    src: int
    dst: int
    lag: int
    strength: float
    sign: int
    mark: str


@dataclass(frozen=True)
class PathEdgeState:
    """Represents the state of an edge in a causal path, including probabilities and information gain."""

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
    """Represents a causal path signature for an activity, including the path and its score."""

    activity_id: int
    activity_name: str
    path: tuple[tuple[int, int, int], ...]
    own_score: float
    margin_to_next_activity: float
    strictly_unique: bool
    edge_states: tuple[PathEdgeState, ...]


@dataclass
class ActivityGraphResult:
    """Raw + filtered graph tensors and derived edges for one activity."""

    activity_id: int
    activity_name: str
    n_samples: int
    raw_graph: np.ndarray
    raw_val: np.ndarray
    filtered_graph: np.ndarray
    filtered_val: np.ndarray
    edges: list[CausalEdge]
    path_edges: list[CausalEdge]
    candidate_paths: list[tuple[tuple[tuple[int, int, int], ...], float]]
    strength_map: dict[tuple[int, int, int], float]
