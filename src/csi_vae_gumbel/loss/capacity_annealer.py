class CapacityAnnealer:
    """Linear capacity annealer.

    Ct = min(C_max, (t / t_ramp) * C_max)

    Where t is the number of steps (or epochs) since start_step. If t_ramp <= 0,
    capacity immediately equals C_max.

    Usage:
        annealer = CapacityAnnealer(c_max=25.0, t_ramp=10000)
        c = annealer(step)  # returns capacity at given step

    """

    def __init__(self, c_max: float = 2, t_ramp: int = 30, start_epoch: int = 10) -> None:
        """Initialize the capacity annealer.

        Arguments:
            c_max: maximum capacity (C_max).
            t_ramp: number of steps (or epochs) to linearly ramp to C_max.
            start_epoch: epoch index at which annealing begins (default 0).

        """
        self.c_max = c_max
        self.t_ramp = t_ramp
        self.start_epoch = start_epoch

    def step(self, epoch: int) -> float:
        """Compute capacity at given epoch.

        Arguments:
            epoch: current integer epoch.

        Returns:
            Current capacity value.

        """
        t = max(0, epoch - self.start_epoch)

        if self.t_ramp == 0:
            return self.c_max

        frac = t / float(self.t_ramp)
        return min(self.c_max, frac * self.c_max)
