class CapacityScheduler:
    """Linear capacity scheduler.

    Ct = min(C_max, (t / t_ramp) * C_max)

    Where t is the number of steps (or epochs) since start_step. If t_ramp <= 0,
    capacity immediately equals C_max.

    Usage:
        annealer = CapacityAnnealer(c_max=25.0, t_ramp=10000)
        c = annealer(step)  # returns capacity at given step

    """

    def __init__(self, c_max: float = 1, t_ramp: int = 10, start_step: int = 0) -> None:
        """Initialize the capacity annealer.

        Arguments:
            c_max: maximum capacity (C_max).
            t_ramp: number of steps (or epochs) to linearly ramp to C_max.
            start_step: step index at which annealing begins (default 0).

        """
        if c_max < 0:
            msg = "c_max must be non-negative"
            raise ValueError(msg)
        if t_ramp is None:
            t_ramp = 0
        if t_ramp < 0:
            msg = "t_ramp must be non-negative or zero"
            raise ValueError(msg)
        self.c_max = float(c_max)
        self.t_ramp = int(t_ramp)
        self.start_step = int(start_step)

    def step(self, step: int) -> float:
        """Compute capacity at given global step (or epoch).

        Arguments:
            step: current integer step (>= 0)

        Returns:
            Current capacity value.

        """
        step = int(step)
        t = max(0, step - self.start_step)
        if self.t_ramp == 0:
            return self.c_max
        frac = t / float(self.t_ramp)
        return min(self.c_max, frac * self.c_max)
