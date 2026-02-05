import optuna
from optuna.pruners import BasePruner


class KLCollapsePruner(BasePruner):
    """Stateless pruner that avoids private API access."""

    def __init__(self, patience: int = 3, warmup_steps: int = 5, eps: float = 1e-4) -> None:
        """Initialize the pruner with patience, epsilon threshold, and warmup steps.

        Arguments:
            patience: Number of consecutive steps with low KL divergence to trigger pruning.
            warmup_steps: Minimum number of steps before pruning is considered.
            eps: Threshold below which KL divergence is considered collapsed.

        """
        self.patience = patience
        self.eps = eps
        self.warmup_steps = warmup_steps

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:  # noqa: ARG002
        """Decide whether to prune the trial based on KL divergence collapse."""
        # Use intermediate_values as a proxy for the current epoch count
        steps = list(trial.intermediate_values.keys())
        if not steps or max(steps) < self.warmup_steps:
            return False

        #  Get the history of KL losses from user_attrs
        kl_history = trial.user_attrs.get("kl_history", [])

        if not kl_history:
            return False

        # Check the last 'patience' entries
        recent_kls = kl_history[-self.patience :]

        if len(recent_kls) < self.patience:
            return False

        # Prune if all recent KL values are below epsilon
        if all(kl < self.eps for kl in recent_kls):
            return True

        # Prune if all recent KL values are identical (within a small tolerance)
        return bool(all(abs(recent_kls[i] - recent_kls[i - 1]) < self.eps for i in range(1, len(recent_kls))))
