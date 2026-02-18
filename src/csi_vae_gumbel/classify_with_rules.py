from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from csi_vae_gumbel.path_search_from_raw import classify_sequence_with_rule_library


def load_rule_library(
    rules_path: Path,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, str], dict[str, Any], dict[int, dict[str, Any]]]:
    """Load rule library exported by path_search_from_raw."""
    payload = json.loads(rules_path.read_text())
    activity_rules = payload["activities"]
    metadata = payload.get("metadata", {})
    activity_models_by_name = payload.get("activity_models", {})

    library: dict[int, list[dict[str, Any]]] = {}
    id_to_name: dict[int, str] = {}
    activity_models: dict[int, dict[str, Any]] = {}
    for activity_name, rules in activity_rules.items():
        model = activity_models_by_name.get(activity_name)
        activity_id: int | None = None
        if rules:
            activity_id = int(rules[0]["activity_id"])
        elif model is not None and "activity_id" in model:
            activity_id = int(model["activity_id"])
        if activity_id is None:
            continue

        library[activity_id] = rules
        id_to_name[activity_id] = activity_name
        if model is not None:
            activity_models[activity_id] = model

    # Include any model-only activities that may not have explicit rules.
    for activity_name, model in activity_models_by_name.items():
        if "activity_id" not in model:
            continue
        activity_id = int(model["activity_id"])
        if activity_id not in library:
            library[activity_id] = []
            id_to_name[activity_id] = activity_name
            activity_models[activity_id] = model

    return library, id_to_name, metadata, activity_models


def load_latent_binary_sequence(latents_path: Path, expected_vars: int | None = None) -> np.ndarray:
    """Load latent sequence and convert to binary variable matrix (T, n_vars)."""
    array = np.load(latents_path)
    if array.ndim == 3:
        # (T, latent_dim, n_categories) one-hot hard latents
        sequence = array.reshape(array.shape[0], array.shape[1] * array.shape[2]).astype(np.int8)
    elif array.ndim == 2:
        sequence = array.astype(np.int8)
    else:
        msg = f"Unsupported latent shape {array.shape}. Expected 2D or 3D array."
        raise ValueError(msg)

    if expected_vars is not None and sequence.shape[1] != expected_vars:
        msg = f"Variable mismatch: rules expect {expected_vars}, input has {sequence.shape[1]}."
        raise ValueError(msg)
    return sequence


def segment_sequence(sequence: np.ndarray, segment_length: int, segment_hop: int) -> list[np.ndarray]:
    """Split sequence into overlapping windows."""
    if len(sequence) < segment_length:
        return []
    return [
        sequence[start : start + segment_length]
        for start in range(0, len(sequence) - segment_length + 1, segment_hop)
    ]


def classify_latent_sequence(
    *,
    rule_library: dict[int, list[dict[str, Any]]],
    activity_models: dict[int, dict[str, Any]],
    id_to_name: dict[int, str],
    sequence: np.ndarray,
    segment_length: int,
    segment_hop: int,
) -> dict[str, Any]:
    """Classify a latent sequence with deterministic rules."""
    windows = segment_sequence(sequence=sequence, segment_length=segment_length, segment_hop=segment_hop)
    if not windows:
        msg = "Sequence is shorter than segment_length."
        raise RuntimeError(msg)

    segment_predictions: list[dict[str, Any]] = []
    vote_counts: dict[int, int] = {activity_id: 0 for activity_id in rule_library}

    for idx, window in enumerate(windows):
        pred_id, scores = classify_sequence_with_rule_library(
            sequence=window,
            rule_library=rule_library,
            activity_models=activity_models if activity_models else None,
        )
        vote_counts[pred_id] += 1
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        margin = float(top_scores[0][1] - top_scores[1][1]) if len(top_scores) > 1 else 0.0
        segment_predictions.append(
            {
                "segment_index": idx,
                "predicted_activity_id": pred_id,
                "predicted_activity": id_to_name.get(pred_id, str(pred_id)),
                "scores": {id_to_name.get(activity_id, str(activity_id)): score for activity_id, score in scores.items()},
                "margin": margin,
            },
        )

    final_id = max(vote_counts, key=vote_counts.get)
    return {
        "predicted_activity_id": final_id,
        "predicted_activity": id_to_name.get(final_id, str(final_id)),
        "n_segments": len(windows),
        "votes": {id_to_name.get(activity_id, str(activity_id)): count for activity_id, count in vote_counts.items()},
        "segment_predictions": segment_predictions,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Classify a latent sequence using exported symbolic rules.")
    parser.add_argument("--rules", type=str, required=True, help="Path to classification_rules.json")
    parser.add_argument("--latents", type=str, required=True, help="Path to latent npy (2D or 3D)")
    parser.add_argument("--segment-length", type=int, default=None)
    parser.add_argument("--segment-hop", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    rules_path = Path(args.rules)
    latents_path = Path(args.latents)

    rule_library, id_to_name, metadata, activity_models = load_rule_library(rules_path=rules_path)
    if not rule_library:
        msg = "Rule library is empty."
        raise RuntimeError(msg)

    segment_length = args.segment_length if args.segment_length is not None else int(metadata.get("segment_length", 128))
    segment_hop = args.segment_hop if args.segment_hop is not None else int(metadata.get("segment_hop", 32))
    expected_vars = int(metadata["n_latent_variables"]) if "n_latent_variables" in metadata else None

    sequence = load_latent_binary_sequence(latents_path=latents_path, expected_vars=expected_vars)
    result = classify_latent_sequence(
        rule_library=rule_library,
        activity_models=activity_models,
        id_to_name=id_to_name,
        sequence=sequence,
        segment_length=segment_length,
        segment_hop=segment_hop,
    )

    output_path = Path(args.output) if args.output else rules_path.parent / "new_sequence_classification.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"predicted_activity": result["predicted_activity"], "output_json": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
