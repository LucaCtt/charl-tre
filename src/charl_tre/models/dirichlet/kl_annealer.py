class KLAnnealer:
    """Cyclical KL weight annealer.

    Each cycle linearly ramps the weight from 0 to kl_final over ramp_epochs,
    then holds at kl_final for the remaining epochs in the cycle.

    Implements https://aclanthology.org/N19-1021/
    """

    def __init__(
        self,
        total_epochs: int,
        n_cycles: int = 4,
        kl_final: float = 4.0,
        ramp_fraction: float = 0.5,
    ) -> None:
        """Initialize the KL annealer.

        Arguments:
            total_epochs (int): Total number of epochs for the entire training run.
            n_cycles (int): Number of cycles to divide the epochs into.
            kl_final (float): Final KL weight to reach at the end of each ramp.
            ramp_fraction (float): Fraction of each cycle spent ramping up the KL weight (between 0 and 1).

        """
        self._epoch = 0
        self._schedule = self._build_schedule(total_epochs, n_cycles, kl_final, ramp_fraction)
        self._weight = self._schedule[0]

    @staticmethod
    def _build_schedule(
        total_epochs: int,
        n_cycles: int,
        kl_final: float,
        ramp_fraction: float,
    ) -> list[float]:
        cycle_len = total_epochs // n_cycles
        ramp_len = max(1, int(cycle_len * ramp_fraction))

        schedule = []
        for _ in range(n_cycles):
            schedule += [kl_final * i / ramp_len for i in range(ramp_len)]
            schedule += [kl_final] * (cycle_len - ramp_len)

        # Pad any remaining epochs (due to integer division) at kl_final
        schedule += [kl_final] * (total_epochs - len(schedule))

        return schedule

    def step(self) -> None:
        """Advance to the next epoch and update the KL weight accordingly."""
        if self._epoch < len(self._schedule):
            self._weight = self._schedule[self._epoch]
            self._epoch += 1

    @property
    def weight(self) -> float:
        """Get the current KL weight."""
        return self._weight
