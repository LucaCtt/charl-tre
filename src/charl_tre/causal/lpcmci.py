import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from causalts.ci_tests import ParCorrGPU, SplitKCIGPU
from causalts.tigramite_discovery import run_lpcmci

_DEVICE: str | None = None  # worker-process global, set once by _init_worker


@dataclass(frozen=True)
class LPCMCIVariable:
    """Represents a variable in the LPCMCI discovery process, identified by its mixture and component indices.

    Any ordering of variables should depend exclusively on the mixture and component indices, in this specific order.
    """

    mixture: int
    component: int

    def __str__(self) -> str:
        return f"Z_{self.mixture}_{self.component}"


@dataclass(frozen=True)
class LPCMCIParams:
    """Parameters for LPCMCI discovery."""

    variables: list[LPCMCIVariable] = field(default_factory=list)
    tau_min: int = 1
    tau_max: int = 5
    pc_alpha: float = 0.05
    max_p_global: int = 5
    n_preliminary_iterations: int = 1
    ci_test: Literal["parcorr", "split_kci"] = "parcorr"


class LPCMCIActivityError(Exception):
    """Exception raised when an activity fails during LPCMCI discovery."""

    def __init__(self, activity_id: int, cause: BaseException) -> None:
        """Initialize the exception with the activity ID and the cause of the failure."""
        super().__init__(f"activity {activity_id} failed: {cause!r}")
        self.activity_id = activity_id
        self.__cause__ = cause


def _init_worker(device_queue: "mp.Queue[str]") -> None:
    """Initialize a worker process for LPCMCI discovery.

    This is needed to set each worker's device to a fixed value for its lifetime.
    Other solutions, such as setting the device depending on the activity ID,
    would lead to recycled workers being assigned to different devices,
    causing memory leaks on the GPUs.
    """
    global _DEVICE  # noqa: PLW0603
    _DEVICE = device_queue.get()


def _run_one_activity(
    activity_id: int,
    activity_latents: np.ndarray,
    params: LPCMCIParams,
    cache_dir: str | None = None,
) -> tuple[int, np.ndarray, dict]:
    """Run LPCMCI for one activity.

    See the [Causal-TS documentation](https://causal-ts.readthedocs.io/en/latest/api/autoapi/causalts/tigramite_discovery/index.html#causalts.tigramite_discovery.run_pcmci)
    for more information about returned values.

    No sorting of the variables is done here, the ordering of the variables in the adjacency matrix
    corresponds to the order of `params.variables`.

    Arguments:
        activity_id (int): ID of the activity.
        activity_latents (np.ndarray): Latents of shape (n_windows, n_mixtures, n_components).
        params (LPCMCIParams): LPCMCIParams object.
        cache_dir (str | None): Directory to store temporary files for the tests.

    Returns:
        tuple[int, np.ndarray, dict]:
            - activity_id (int): ID of the activity.
            - graph (np.ndarray): Estimated graph of shape (n_vars, n_vars, tau_max + 1).
            - info (dict): Additional information from LPCMCI

    """
    n_windows, n_mixtures, n_components = activity_latents.shape

    var_names = [str(v) for v in params.variables]

    # Build LPCMCI input DataFrame
    flat = activity_latents.reshape(n_windows, n_mixtures * n_components)
    df = pd.DataFrame(flat, columns=var_names)

    match params.ci_test:
        case "split_kci":
            ci_test = SplitKCIGPU
        case _:
            ci_test = ParCorrGPU

    graph, info = run_lpcmci(
        df,
        tau_min=params.tau_min,
        tau_max=params.tau_max,
        pc_alpha=params.pc_alpha,
        max_p_global=params.max_p_global,
        n_preliminary_iterations=params.n_preliminary_iterations,
        ci_test=ci_test(
            data=flat,
            device=_DEVICE,
            cache_dir=cache_dir,
            data_hash=hash(flat.tobytes()),
        ),
    )
    return activity_id, graph, info


def run_lpcmci_batch(
    latents: np.ndarray,
    params: LPCMCIParams,
    max_workers: int | None = None,
    cache_dir: str | None = None,
) -> np.ndarray:
    """Run LPCMCI for every activity in `latents`, in parallel.

    No sorting of the variables is done here, the ordering of the variables in the adjacency matrix
    corresponds to the order of `params.variables`.

    Arguments:
        latents (np.ndarray): Latents of shape (n_activities, n_windows, n_mixtures, n_components).
        params (LPCMCIParams): LPCMCIParams object.
        max_workers (int | None): Maximum number of workers to use for parallel processing.
            Default is None, which means no maximum limit (i.e., use all available cores).
        cache_dir (str | None): Directory to store temporary files for the tests.

    Returns:
        np.ndarray: Adjacency matrix of shape
            (n_activities, len(params.variables), len(params.variables), params.tau_max + 1),
            containing tuples of (bool, float) for each pair of variables and lag.
            The first element of the tuple indicates whether a causal link exists,
            and the second element is the corresponding value from the LPCMCI output.

    """
    n_activities, _, n_mixtures, n_components = latents.shape
    if len(params.variables) != n_mixtures * n_components:
        msg = (
            f"Number of variables in params ({len(params.variables)}) does not match "
            f"the number of mixtures * components in latents ({n_mixtures * n_components})."
        )
        raise ValueError(msg)

    adjacency_matrix = np.empty(
        (n_activities, len(params.variables), len(params.variables), params.tau_max + 1),
        dtype=np.dtype([("mark", np.bool_), ("value", np.float32)]),
    )

    ctx = mp.get_context("spawn")
    device_queue: "mp.Queue[str]" = ctx.Queue()
    for i in range(max_workers or os.cpu_count() or 1):
        device_queue.put(f"cuda:{i % torch.cuda.device_count()}")
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(device_queue,),
    )

    try:
        futures = [pool.submit(_run_one_activity, i, latents[i], params, cache_dir) for i in range(n_activities)]

        for future in as_completed(futures):
            try:
                activity_id, _, info = future.result()
            except Exception as e:
                raise LPCMCIActivityError(futures.index(future), e) from e

            has_link = info["graph_raw"] == "-->"
            adjacency_matrix[activity_id]["mark"] = has_link
            adjacency_matrix[activity_id]["value"] = np.where(has_link, info["val_matrix"], 0.0)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return adjacency_matrix
