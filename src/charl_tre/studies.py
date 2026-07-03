from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna


@dataclass
class StudyResult:
    """Data class to store the results of a single Optuna studyl."""

    trial_number: int
    trial_value: float
    params: dict[str, Any]


def make_study(study_name: str, storage_dir: str | None, seed: int) -> optuna.Study:
    """Create (or load) an Optuna study backed by a journal file.

    Arguments:
        study_name: The name of the study to create or load.
        storage_dir: The directory to use for storage. If None, the study will be created without persistent storage.
        seed: The seed to use for the random number generator.

    Returns:
        An Optuna Study object.

    """
    if storage_dir:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        journal_path = f"{storage_dir}/{study_name}.sqlite"
    else:
        journal_path = ":memory:"

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{journal_path}",
        heartbeat_interval=60,
        grace_period=120,
        heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=3),
    )

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed),
        direction="maximize",
        load_if_exists=True,
    )


def read_study(launch_dir: str) -> optuna.Study:
    """Read an Optuna study from a journal file in the specified launch directory.

    Arguments:
        launch_dir (Path): The directory where the Optuna study SQLite file is located.

    Returns:
        optuna.Study: An Optuna Study object loaded from the SQLite file in the launch directory.

    """
    study_files = sorted([f.name for f in Path(launch_dir).iterdir() if f.is_file() and f.suffix == ".sqlite"])
    if len(study_files) == 0:
        msg = f"No study files found in {launch_dir}."
        raise FileNotFoundError(msg)

    study = study_files[-1]
    return optuna.load_study(study_name=study.split(".")[0], storage=f"sqlite:///{Path(launch_dir) / study}")


def get_best_model(study: optuna.Study) -> StudyResult:
    """Return the best model across all studies based on the highest seed accuracy.

    Arguments:
        study: An Optuna Study object containing the trials to evaluate.

    Returns:
        StudyResult: A dataclass containing the details of the best model found across all studies.

    """
    study_df = study.trials_dataframe()
    completed = study_df[study_df["state"] == "COMPLETE"].copy()
    if completed.empty:
        msg = "No completed trials found in the study."
        raise ValueError(msg)

    study_best = StudyResult(
        trial_number=0,
        trial_value=0.0,
        params={},
    )

    for _, trial in completed.iterrows():
        if trial["value"] > study_best.trial_value:
            study_best = StudyResult(
                trial_number=trial["number"],
                trial_value=trial["value"],
                params=trial.filter(like="params_").rename(lambda x: x.replace("params_", "")).to_dict(),
            )

    return study_best
