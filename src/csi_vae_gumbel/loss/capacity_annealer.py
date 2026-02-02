class CapacityAnnealer:
    """Linear capacity annealer.

    Ct = min(C_max, (t / t_ramp) * C_max)

    Where t is the number of steps (or epochs) since start_step. If t_ramp <= 0,
    capacity immediately equals C_max.

    Usage:
        annealer = CapacityAnnealer(c_max=25.0, t_ramp=10000)
        c = annealer(step)  # returns capacity at given step

    """

    def __init__(self, max_capacity: float = 2, ramp_epochs: int = 30, start_epoch: int = 10) -> None:
        """Initialize the capacity annealer.

        Arguments:
            max_capacity: maximum capacity (C_max).
            ramp_epochs: number of steps (or epochs) to linearly ramp to C_max.
            start_epoch: epoch index at which annealing begins (default 0).

        """
        self.__c_max = max_capacity
        self.__t_ramp = ramp_epochs
        self.__start_epoch = start_epoch

    def step(self, epoch: int) -> float:
        """Compute capacity at given epoch.

        Arguments:
            epoch: current epoch index.

        Returns:
            Current capacity value.

        """
        epoch += 1

        t = max(0, epoch - self.__start_epoch)
        if self.__t_ramp == 0:
            return self.__c_max

        frac = t / float(self.__t_ramp)
        return min(self.__c_max, frac * self.__c_max)
