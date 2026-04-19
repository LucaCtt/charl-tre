"""Unique path selection and activity signature building."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from charl_tre.causal.types import ActivityPathSignature, PathEdgeState

# Type aliases for readability.
_Path = tuple[tuple[int, int, int], ...]
_CandidateList = list[tuple[_Path, float]]
_StrengthMap = dict[tuple[int, int, int], float]


def _score_path(path: _Path, strengths: _StrengthMap) -> float:
    """Compute the total strength of a path based on its edges.

    Arguments:
        path: A sequence of edges, where each edge is a tuple (src, dst, lag).
        strengths: A mapping from (src, dst, lag) to edge strength.

    Returns:
        The total strength of the path, calculated as the sum of the strengths of its edges.

    """
    return float(sum(strengths.get(e, 0.0) for e in path))


class PathSearcher:
    """Select one unique discriminative path per activity and enrich it with state statistics."""

    def select_unique(
        self,
        candidate_paths: dict[int, _CandidateList],
        strength_maps: dict[int, _StrengthMap],
    ) -> dict[int, _Path]:
        """Assign one unique path per activity maximising discriminative margin."""
        activity_ids = sorted(candidate_paths)
        all_paths, seen = [], set()
        for activity_id in activity_ids:
            for path, _ in candidate_paths[activity_id]:
                if path not in seen:
                    seen.add(path)
                    all_paths.append(path)

        if len(all_paths) < len(activity_ids):
            return self._greedy_fallback(activity_ids, candidate_paths, strength_maps)

        # Hungarian assignment on discriminative-margin matrix.
        matrix = np.full((len(activity_ids), len(all_paths)), fill_value=-1e9, dtype=np.float64)
        own_candidates_by_activity = {aid: {p for p, _ in candidate_paths[aid]} for aid in activity_ids}
        for row, activity_id in enumerate(activity_ids):
            own_map = strength_maps[activity_id]
            for col, path in enumerate(all_paths):
                if path not in own_candidates_by_activity[activity_id]:
                    continue
                own_score = _score_path(path, own_map)
                other_max = max(
                    (_score_path(path, strength_maps[oid]) for oid in activity_ids if oid != activity_id),
                    default=0.0,
                )
                matrix[row, col] = own_score - other_max

        row_ind, col_ind = linear_sum_assignment(-matrix)
        return {activity_ids[r]: all_paths[c] for r, c in zip(row_ind, col_ind, strict=True)}

    def build_signatures(
        self,
        selected_paths: dict[int, _Path],
        strength_maps: dict[int, _StrengthMap],
        labelled_data: dict[int, np.ndarray],
        activity_names: dict[int, str],
        n_states: int,
    ) -> dict[int, ActivityPathSignature]:
        """Compute state templates and discriminative margins for each path."""
        signatures: dict[int, ActivityPathSignature] = {}
        for activity_id, path in selected_paths.items():
            own_score = _score_path(path, strength_maps[activity_id])
            other_max = max(
                (_score_path(path, strength_maps[oid]) for oid in selected_paths if oid != activity_id),
                default=0.0,
            )
            margin = own_score - other_max
            edge_states = [
                self._build_edge_state(
                    src,
                    dst,
                    lag,
                    data=labelled_data[activity_id],
                    strength=strength_maps[activity_id].get((src, dst, lag), 0.0),
                    n_states=n_states,
                )
                for src, dst, lag in path
            ]
            signatures[activity_id] = ActivityPathSignature(
                activity_id=activity_id,
                activity_name=activity_names[activity_id],
                path=path,
                own_score=own_score,
                margin_to_next_activity=margin,
                strictly_unique=margin > 0,
                edge_states=tuple(edge_states),
            )
        return signatures

    @staticmethod
    def _greedy_fallback(
        activity_ids: list[int],
        candidate_paths: dict[int, _CandidateList],
        strength_maps: dict[int, _StrengthMap],
    ) -> dict[int, _Path]:
        result: dict[int, _Path] = {}
        for activity_id in activity_ids:
            own_map = strength_maps[activity_id]
            best_path, best_margin = None, float("-inf")
            for path, _ in candidate_paths[activity_id]:
                own_score = _score_path(path, own_map)
                other_max = max(
                    (_score_path(path, strength_maps[oid]) for oid in activity_ids if oid != activity_id),
                    default=0.0,
                )
                margin = own_score - other_max
                if margin > best_margin:
                    best_margin = margin
                    best_path = path
            if best_path is None:
                msg = f"No candidate paths for activity {activity_id}."
                raise RuntimeError(msg)
            result[activity_id] = best_path
        return result

    @staticmethod
    def _edge_joint_distribution(data: np.ndarray, src: int, dst: int, lag: int, n_states: int) -> np.ndarray:
        n_pairs = n_states * n_states
        if lag <= 0 or len(data) <= lag:
            return np.full(n_pairs, 1.0 / n_pairs)
        encoded = data[:-lag, src].astype(int) * n_states + data[lag:, dst].astype(int)
        counts = np.bincount(encoded, minlength=n_pairs).astype(np.float64)
        total = counts.sum()
        return counts / total if total > 0 else np.full(n_pairs, 1.0 / n_pairs)

    @classmethod
    def _build_edge_state(
        cls,
        src: int,
        dst: int,
        lag: int,
        data: np.ndarray,
        strength: float,
        n_states: int,
    ) -> PathEdgeState:
        joint_probs = cls._edge_joint_distribution(data, src, dst, lag, n_states)
        joint = joint_probs.reshape(n_states, n_states)
        src_probs = joint.sum(axis=1)
        dst_probs = joint.sum(axis=0)
        eps = 1e-12
        pmi = joint * np.log((joint + eps) / (np.outer(src_probs, dst_probs) + eps))
        best = np.unravel_index(int(np.argmax(pmi)), pmi.shape)
        info_gain = float(max(pmi[best], 0.0))
        support = float(joint[best])
        if info_gain <= 0:
            best = np.unravel_index(int(np.argmax(joint)), joint.shape)
            support = float(joint[best])
        return PathEdgeState(
            src=src,
            dst=dst,
            lag=lag,
            strength=strength,
            src_state=int(best[0]),
            dst_state=int(best[1]),
            support=support,
            info_gain=info_gain,
            joint_probs=tuple(float(x) for x in joint_probs),
            src_probs=tuple(float(x) for x in src_probs),
            dst_probs=tuple(float(x) for x in dst_probs),
        )
