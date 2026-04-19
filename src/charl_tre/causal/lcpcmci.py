"""Tigramite LPCMCI wrapper with independence test construction."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI


class LPCMCIRunner:
    """Run LPCMCI with ParCorr for one or many activities, optionally in parallel."""

    def __init__(
        self,
        tau_min: int = 1,
        tau_max: int = 5,
        pc_alpha: float = 1e-3,
    ) -> None:
        """Initialize the runner with LPCMCI parameters."""
        self.__tau_min = tau_min
        self.__tau_max = tau_max
        self.__pc_alpha = pc_alpha

    def run_one(
        self,
        data: np.ndarray,
        var_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run LPCMCI for one activity and return the val_matrix and graph.

        Arguments:
            data: 2D array of shape (n_samples, n_variables) containing the time series data for one activity.
            var_names: List of variable names corresponding to the columns in `data`.

        Returns:
            A tuple containing:
            - val_matrix: 2D array of shape (n_variables, n_variables) containing
                the p-values of the independence tests for each pair of variables and time lags.
            - graph: 2D array of shape (n_variables, n_variables) containing the inferred causal graph structure,
                where graph[i, j] = 1 indicates a causal link from variable i to variable j
                at some time lag, and graph[i, j] = 0 indicates no causal link.

        """
        dataframe = pp.DataFrame(
            data.astype(np.float64, copy=False),
            var_names=var_names,
            data_type=np.zeros_like(data, dtype=np.int16),
        )
        lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"))
        results = lpcmci.run_lpcmci(tau_min=self.__tau_min, tau_max=self.__tau_max, pc_alpha=self.__pc_alpha)

        val_matrix = results["val_matrix"].copy()
        graph = results["graph"].copy()
        return val_matrix, graph

    def run_parallel(
        self,
        labelled_data: dict[int, np.ndarray],
        var_names: list[str],
        max_workers: int | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Run LPCMCI for multiple activities in parallel and return a mapping from activity ID to results.

        Arguments:
            labelled_data: A dictionary mapping activity IDs to their corresponding time series data arrays.
            var_names: List of variable names corresponding to the columns in the data arrays.
            max_workers: Maximum number of worker threads to use for parallel execution.
                If None, it will use as many workers as there are activities.

        Returns:
            A dictionary mapping each activity ID to a tuple containing:
            - val_matrix: 2D array of shape (n_variables, n_variables) containing the p-values
                of the independence tests for each pair of variables and time lags.
            - graph: 2D array of shape (n_variables, n_variables) containing the inferred causal graph structure,
                where graph[i, j] = 1 indicates a causal link from variable i to variable j
                at some time lag, and graph[i, j] = 0 indicates no causal link.

        """
        items = sorted(labelled_data.items())
        n_workers = self._clamp_workers(max_workers, len(items))
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self.run_one, data=data, var_names=var_names): activity_id
                for activity_id, data in items
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    @staticmethod
    def _clamp_workers(max_workers: int | None, n_items: int) -> int:
        cpu_count = os.cpu_count() or n_items
        effective = max_workers if max_workers is not None else n_items
        return max(1, min(effective, n_items, cpu_count))
