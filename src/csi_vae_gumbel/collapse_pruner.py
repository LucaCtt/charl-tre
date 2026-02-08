import numpy as np
import optuna
from optuna.pruners import BasePruner


class CollapsePruner(BasePruner):
    """Prune trials only on irreversible categorical posterior collapse."""

    def __init__(
        self,
        n_categories: int,
        patience: int = 10,
        warmup_steps: int = 50,
        min_entropy_frac: float = 0.02,
        max_entropy_slope: float = 1e-4,
    ) -> None:
        """Initialize the KLCollapsePruner.

        Arguments:
            n_categories: Number of categorical states (K).
            patience: Number of consecutive epochs to inspect.
            warmup_steps: Epochs before pruning is allowed.
            min_entropy_frac: Fraction of log(K) below which entropy is 'collapsed'.
            max_entropy_slope: Max allowed entropy increase to consider it dead.

        """
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.min_entropy_frac = min_entropy_frac
        self.max_entropy_slope = max_entropy_slope
        self.max_entropy = float(np.log(n_categories))

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

        entropy_hist = trial.user_attrs.get("entropy_history", [])

        if len(entropy_hist) < self.patience:
            return False

        ent = np.array(entropy_hist[-self.patience :])

        # Entropy is very low
        if np.mean(ent) > self.min_entropy_frac * self.max_entropy:
            return False

        # Entropy is not recovering
        slope = np.polyfit(np.arange(len(ent)), ent, 1)[0]
        return slope <= self.max_entropy_slope
