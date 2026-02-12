class GumbelTemperatureAnnealer:
    """Annealer for the Gumbel-Softmax temperature parameter."""

    def __init__(
        self,
        start_tau: float = 1.5,
        min_tau: float = 0.4,
        n_epochs: int = 100,
        start_epoch: int = 0,
    ) -> None:
        """Initialize the Gumbel temperature annealer.

        Arguments:
            start_tau: initial temperature value.
            min_tau: minimum temperature value after annealing.
            n_epochs: total number of epochs for annealing.
            start_epoch: epoch index at which annealing begins (default 0).

        """
        self.__start_tau = start_tau
        self.__min_tau = min_tau
        self.__n_epochs = n_epochs
        self.__start_epoch = start_epoch

    def step(self, epoch: int) -> float:
        """Compute temperature at given epoch.

        Arguments:
            epoch: current epoch index.

        Returns:
            Current temperature value.

        """
        if epoch < self.__start_epoch:
            return self.__start_tau

        progress = (epoch - self.__start_epoch) / (self.__n_epochs - self.__start_epoch)
        tau = self.__start_tau - progress * (self.__start_tau - self.__min_tau)
        return max(self.__min_tau, tau)
