class KLScheduler:
    """Handles the warming up of the KL divergence weight (beta)."""

    def __init__(
        self,
        start_epoch: int = 0,
        ramp_epochs: int = 10,
        max_weight: float = 9e-4,
    ) -> None:
        """Set up the KL weight scheduler.

        Arguments:
            start_epoch: Epoch to start increasing the KL weight.
            ramp_epochs: Number of epochs over which to increase the KL weight.
            max_weight: Maximum KL weight to reach.

        """
        self.__start_epoch = start_epoch
        self.__ramp_epochs = ramp_epochs
        self.__max_weight = max_weight

    def get_weight(self, epoch: int) -> float:
        """Get the KL weight for the given epoch.

        Arguments:
            epoch: Current epoch number.

        Returns:
            The KL weight for the current epoch.

        """
        if epoch < self.__start_epoch:
            return 0.0

        progress = (epoch - self.__start_epoch) / self.__ramp_epochs
        return self.__max_weight * min(progress, 1.0)
