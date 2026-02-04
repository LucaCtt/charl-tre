import optuna
from optuna.pruners import BasePruner
from optuna.trial import FrozenTrial


class KLCollapsePruner(BasePruner):
    """Early stopping utility to halt training when KL divergence collapses."""

    def __init__(self, patience: int = 3, eps: float = 1e-6) -> None:
        """Initialize EarlyStopping with a patience parameter.

        Arguments:
            patience: Number of epochs to wait for improvement before stopping.
            eps: Threshold to consider KL divergence as near-zero.

        """
        self.__patience = patience
        self.__eps = eps
        self.__current_trial = 0
        self.__kl_counter = 0

    def prune(self, study: optuna.Study, trial: FrozenTrial) -> bool:  # noqa: ARG002
        """Determine whether to prune the current trial.

        Arguments:
            study: The Optuna study object.
            trial: The current Optuna trial object.

        Returns:
            True if the trial should be pruned, False otherwise.

        """
        if self.__current_trial != trial.number:
            self.__current_trial = trial.number
            self.__kl_counter = 0

        epoch_kl_loss = trial.user_attrs.get("epoch_kl_loss", 0)

        if epoch_kl_loss < self.__eps:
            self.__kl_counter += 1
        else:
            self.__kl_counter = 0

        return self.__kl_counter >= self.__patience
