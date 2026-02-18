from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tigramite import data_processing as pp
from tigramite.lpcmci import LPCMCI

from csi_vae_gumbel.causal_analysis_full import (
    compute_gsquared_signs,
    get_independence_test,
    load_latent_hards,
    split_by_activity,
)
from csi_vae_gumbel.settings import Settings


def run_activity_lpcmci_raw(
    *,
    activity_id: int,
    activity_name: str,
    data: np.ndarray,
    var_names: list[str],
    ind_test_name: str,
    is_categorical_test: bool,
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    debug: bool,
) -> dict[str, Any]:
    """Run LPCMCI for one activity and return raw graph tensors."""
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

    return {
        "activity_id": activity_id,
        "activity_name": activity_name,
        "n_samples": int(len(data)),
        "graph": raw_graph,
        "val_matrix": raw_val,
    }


def run_raw_causal_discovery(
    *,
    settings: Settings,
    ind_test_name: str = "parcorr",
    tau_min: int = 1,
    tau_max: int | None = None,
    pc_alpha: float = 1e-3,
    stride: int | None = None,
    use_full_onehot: bool = True,
    max_workers: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run LPCMCI only and save raw graph tensors as JSON."""
    tau_max_eff = tau_max if tau_max is not None else settings.test_window_size // settings.train_window_size - 1
    stride_eff = stride if stride is not None else settings.train_window_size
    if tau_min < 1 or tau_max_eff < tau_min:
        msg = "Invalid tau range."
        raise ValueError(msg)

    _, is_categorical_test = get_independence_test(ind_test_name)
    latents_path = Path(settings.study_path) / "latents" / "latents_hard.npy"
    labels_path = Path(settings.study_path) / "latents" / "labels.npy"
    latents, labels, latent_dim, n_categories, _ = load_latent_hards(
        latent_path=latents_path,
        label_path=labels_path,
        use_full_onehot=use_full_onehot,
    )
    labelled_data = split_by_activity(latents=latents, labels=labels, stride=stride_eff)

    if use_full_onehot:
        var_names = [f"Z{latent_idx}_C{cat_idx}" for latent_idx in range(latent_dim) for cat_idx in range(n_categories)]
    else:
        var_names = [f"Z{idx}" for idx in range(latents.shape[1])]

    out_dir = Path(settings.study_path) / "causal_graphs_raw" / ind_test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    activity_items = sorted(labelled_data.items())
    max_workers_eff = max_workers if max_workers is not None else len(activity_items)
    cpu_count = os.cpu_count() or len(activity_items)
    max_workers_eff = max(1, min(max_workers_eff, len(activity_items), cpu_count))

    activity_results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers_eff) as executor:
        futures = {
            executor.submit(
                run_activity_lpcmci_raw,
                activity_id=activity_id,
                activity_name=settings.activities[activity_id],
                data=data,
                var_names=var_names,
                ind_test_name=ind_test_name,
                is_categorical_test=is_categorical_test,
                tau_min=tau_min,
                tau_max=tau_max_eff,
                pc_alpha=pc_alpha,
                debug=debug,
            ): activity_id
            for activity_id, data in activity_items
        }
        for future in as_completed(futures):
            result = future.result()
            activity_results[result["activity_id"]] = result

    for activity_id in sorted(activity_results):
        result = activity_results[activity_id]
        activity_dir = out_dir / result["activity_name"].replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "activity_id": result["activity_id"],
            "activity_name": result["activity_name"],
            "n_samples": result["n_samples"],
            "graph": result["graph"].tolist(),
            "val_matrix": result["val_matrix"].tolist(),
        }
        with (activity_dir / "raw_graph.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file)

    manifest = {
        "study_path": settings.study_path,
        "independence_test": ind_test_name,
        "tau_min": tau_min,
        "tau_max": tau_max_eff,
        "pc_alpha": pc_alpha,
        "stride": stride_eff,
        "latent_representation": "onehot_flat" if use_full_onehot else "argmax_index",
        "latent_dim": latent_dim,
        "n_categories": n_categories,
        "n_latent_variables": len(var_names),
        "var_names": var_names,
        "activities": {
            settings.activities[activity_id]: {
                "activity_id": activity_id,
                "raw_graph_json": f"{settings.activities[activity_id].replace(' ', '_')}/raw_graph.json",
            }
            for activity_id in sorted(activity_results)
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {"output_dir": str(out_dir), "manifest": manifest}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    settings = Settings()
    parser = argparse.ArgumentParser(description="Run raw LPCMCI and save raw graphs as JSON.")
    parser.add_argument("--study-path", type=str, default=settings.study_path)
    parser.add_argument("--ind-test", type=str, default="parcorr")
    parser.add_argument("--tau-min", type=int, default=1)
    parser.add_argument("--tau-max", type=int, default=None)
    parser.add_argument("--pc-alpha", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=settings.train_window_size)
    parser.add_argument("--argmax-latents", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    settings = Settings(study_path=args.study_path)
    result = run_raw_causal_discovery(
        settings=settings,
        ind_test_name=args.ind_test,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        stride=args.stride,
        use_full_onehot=not args.argmax_latents,
        max_workers=args.max_workers,
        debug=args.debug,
    )
    print(json.dumps({"output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
