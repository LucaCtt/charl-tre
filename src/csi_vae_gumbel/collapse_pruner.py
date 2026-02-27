import numpy as np
import optuna
from optuna.pruners import BasePruner


class CollapsePruner(BasePruner):
    """Prune trials on irrevesible categorical posterior collapse."""

    def __init__(
        self,
        n_categories: int,
        patience: int = 10,
        warmup_steps: int = 50,
        min_entropy_frac: float = 0.05,
    ) -> None:
        """Initialize the CollapsePruner.

        Arguments:
            n_categories: Number of categorical states (K).
            patience: Number of consecutive epochs to inspect.
            warmup_steps: Epochs before pruning is allowed.
            min_entropy_frac: Fraction of log(K) below which entropy is 'collapsed'.

        """
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.min_entropy_frac = min_entropy_frac
        self.max_entropy = np.log(n_categories)

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:  # noqa: ARG002
        """Decide whether to prune the trial based on entropy history.

        Arguments:
            study: The Optuna study object (not used here).
            trial: The Optuna trial object containing intermediate values and user attributes.

        Returns:
            True if the trial should be pruned, False otherwise.

        """
        steps = list(trial.intermediate_values.keys())

        if not steps or max(steps) < self.warmup_steps:
            return False

        if len(steps) < self.patience:
            return False

        entropy_hist = trial.user_attrs.get("entropy_history", [])
        entropy = np.array(entropy_hist[-self.patience :])

        # If mean entropy is below the threshold for the last 'patience' epochs, prune the trial
        return bool(entropy.mean() < self.min_entropy_frac * self.max_entropy)
