"""Graph filtering, edge extraction, and path enumeration."""

from collections import defaultdict

import numpy as np

from charl_tre.causal.types import ActivityGraphResult, CausalEdge

_DIRECTED_MARKS = frozenset({"-->", "o->"})
"""Edge marks that represent a lagged directed link into the target variable."""


def _edge_strength_map(edges: list[CausalEdge]) -> dict[tuple[int, int, int], float]:
    """Create a mapping from (src, dst, lag) to edge strength for quick lookup.

    Arguments:
        edges: A list of CausalEdge objects.

    Returns:
        A dictionary mapping (src, dst, lag) tuples to their corresponding edge strength.

    """
    return {(e.src, e.dst, e.lag): e.strength for e in edges}


def _extract_edges(graph: np.ndarray, val_matrix: np.ndarray) -> list[CausalEdge]:
    """Extract a list of CausalEdge objects from the filtered graph and value matrices.

    Arguments:
        graph: 3D array of edge marks after filtering.
        val_matrix: 3D array of edge strength values after filtering.

    Returns:
        A list of CausalEdge objects representing the directed edges in the graph,
        sorted by strength in descending order.

    """
    edges: list[CausalEdge] = []
    for src in range(graph.shape[0]):
        for dst in range(graph.shape[1]):
            for lag in range(graph.shape[2]):
                mark = graph[src, dst, lag]
                if mark not in _DIRECTED_MARKS:
                    continue
                value = float(val_matrix[src, dst, lag])
                if value == 0:
                    continue
                edges.append(
                    CausalEdge(
                        src=src,
                        dst=dst,
                        lag=lag,
                        strength=abs(value),
                        sign=int(np.sign(value)),
                        mark=mark,
                    ),
                )
    return sorted(edges, key=lambda e: e.strength, reverse=True)


def _enumerate_paths(
    edges: list[CausalEdge],
    n_vars: int,
    max_edges: int,
    min_edges: int,
    top_k: int,
) -> list[tuple[tuple[tuple[int, int, int], ...], float]]:
    """Enumerate and score candidate paths based on the edges.

    Arguments:
        edges: A list of CausalEdge objects to use for path enumeration.
        n_vars: Total number of variables in the graph.
        max_edges: Maximum number of edges allowed in a candidate path.
        min_edges: Minimum number of edges required in a candidate path.
        top_k: Number of top-ranked paths to return.

    Returns:
        A list of tuples where each tuple contains a candidate path (as a tuple of edges)
        and its corresponding score, sorted by score in descending order.

    """
    adjacency = _build_adjacency(edges)
    scored_paths: dict[tuple[tuple[int, int, int], ...], float] = {}

    for start in range(n_vars):
        _score_paths_from_start(
            start=start,
            adjacency=adjacency,
            min_edges=min_edges,
            max_edges=max_edges,
            scored_paths=scored_paths,
        )

    _record_single_edge_paths(edges, scored_paths)
    return sorted(scored_paths.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _build_adjacency(edges: list[CausalEdge]) -> dict[int, list[tuple[int, int, float]]]:
    """Build an adjacency list from the list of edges for efficient path enumeration.

    Arguments:
        edges: A list of CausalEdge objects.

    Returns:
        A dictionary mapping each source variable to a list of tuples containing (dst, lag, strength),
        sorted by strength in descending order.

    """
    adjacency: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.src].append((edge.dst, edge.lag, edge.strength))

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: item[2], reverse=True)

    return adjacency


def _score_paths_from_start(
    start: int,
    adjacency: dict[int, list[tuple[int, int, float]]],
    min_edges: int,
    max_edges: int,
    scored_paths: dict[tuple[tuple[int, int, int], ...], float],
) -> None:
    """Perform a depth-first search to enumerate paths starting from a given variable and score them.

    Arguments:
        start: The starting variable index for path enumeration.
        adjacency: The adjacency list representing the graph structure.
        min_edges: Minimum number of edges required in a candidate path.
        max_edges: Maximum number of edges allowed in a candidate path.
        scored_paths: A dictionary to store the best score for each unique path encountered.

    """

    def dfs(current: int, visited: set[int], path: list[tuple[int, int, int]], score: float) -> None:
        if len(path) >= min_edges:
            key = tuple(path)
            if score > scored_paths.get(key, float("-inf")):
                scored_paths[key] = score
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

    dfs(start, {start}, [], 0.0)


def _record_single_edge_paths(
    edges: list[CausalEdge],
    scored_paths: dict[tuple[tuple[int, int, int], ...], float],
) -> None:
    """Ensure that single-edge paths are included in the scored paths with their strength as the score.

    Arguments:
        edges: A list of CausalEdge objects to consider for single-edge paths.
        scored_paths: The dictionary of scored paths to update with single-edge paths.

    """
    for edge in edges:
        key = ((edge.src, edge.dst, edge.lag),)
        if edge.strength > scored_paths.get(key, float("-inf")):
            scored_paths[key] = edge.strength


class GraphProcessor:
    """Convert raw LPCMCI graph tensors into edges and ranked candidate paths."""

    def __init__(self, tau_min: int, tau_max: int) -> None:
        """Initialize with lag range for valid causal links."""
        self.__tau_min = tau_min
        self.__tau_max = tau_max

    def process(
        self,
        activity_id: int,
        activity_name: str,
        n_samples: int,
        raw_graph: np.ndarray,
        raw_val: np.ndarray,
        filter_threshold: float,
        path_threshold: float,
        max_edges_per_path: int,
        min_edges_per_path: int,
        top_paths_per_activity: int,
    ) -> ActivityGraphResult:
        """Process raw graph tensors to extract edges and candidate paths.

        Arguments:
            activity_id: Unique identifier for the activity.
            activity_name: Human-readable name of the activity.
            n_samples: Number of samples used to infer the graph.
            raw_graph: 3D array of edge marks from LPCMCI.
            raw_val: 3D array of edge strength values from LPCMCI.
            filter_threshold: Minimum absolute strength to keep an edge.
            path_threshold: Minimum absolute strength to consider an edge for paths.
            max_edges_per_path: Maximum number of edges in a candidate path.
            min_edges_per_path: Minimum number of edges in a candidate path.
            top_paths_per_activity: Number of top-ranked paths to return.

        Returns:
            An ActivityGraphResult containing all processed information.

        """
        filtered_graph, filtered_val = self.__filter(raw_graph, raw_val, filter_threshold)
        path_graph, path_val = self.__filter(raw_graph, raw_val, path_threshold)

        edges = _extract_edges(filtered_graph, filtered_val)
        path_edges = _extract_edges(path_graph, path_val)
        strength_map = _edge_strength_map(path_edges)

        n_vars = raw_graph.shape[0]
        candidates = _enumerate_paths(
            edges=path_edges,
            n_vars=n_vars,
            max_edges=max_edges_per_path,
            min_edges=min_edges_per_path,
            top_k=top_paths_per_activity,
        )
        if not candidates:
            candidates, strength_map = self.__fallback_path(path_val, strength_map)

        return ActivityGraphResult(
            activity_id=activity_id,
            activity_name=activity_name,
            n_samples=n_samples,
            raw_graph=raw_graph,
            raw_val=raw_val,
            filtered_graph=filtered_graph,
            filtered_val=filtered_val,
            edges=edges,
            path_edges=path_edges,
            candidate_paths=candidates,
            strength_map=strength_map,
        )

    def __filter(
        self,
        graph: np.ndarray,
        val_matrix: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter the graph and value matrices based on strength threshold and valid lag range.

        Arguments:
            graph: 3D array of edge marks from LPCMCI.
            val_matrix: 3D array of edge strength values from LPCMCI.
            threshold: Minimum absolute strength to keep an edge.

        Returns:
            A tuple of (filtered_graph, filtered_val) where edges below the threshold
            or outside valid lag range are removed.

        """
        filtered_graph = graph.copy()
        filtered_val = val_matrix.copy()

        weak = np.abs(filtered_val) < threshold
        filtered_val[weak] = 0
        filtered_graph[weak] = ""

        for src in range(filtered_graph.shape[0]):
            for dst in range(filtered_graph.shape[1]):
                for lag in range(filtered_graph.shape[2]):
                    mark = filtered_graph[src, dst, lag]
                    if not mark:
                        continue
                    if lag < self.__tau_min or lag > self.__tau_max or mark not in _DIRECTED_MARKS:
                        filtered_graph[src, dst, lag] = ""
                        filtered_val[src, dst, lag] = 0.0

        return filtered_graph, filtered_val

    def __fallback_path(
        self,
        path_val: np.ndarray,
        strength_map: dict,
    ) -> tuple[list, dict]:
        """Fallback to the single strongest edge if no multi-edge paths are found.

        Arguments:
            path_val: 3D array of edge strength values used for path scoring.
            strength_map: Current mapping of edges to their strengths.

        Returns:
            A tuple containing a list with the single strongest edge as a candidate path and an updated
            strength map including this edge.

        """
        abs_values = np.abs(path_val[:, :, self.__tau_min : self.__tau_max + 1])
        src, dst, lag_offset = np.unravel_index(np.argmax(abs_values), abs_values.shape)
        lag = int(lag_offset) + self.__tau_min
        edge = (int(src), int(dst), lag)
        score = float(abs_values[src, dst, lag_offset])
        strength_map = dict(strength_map)
        strength_map[edge] = score

        return [((edge,), score)], strength_map
