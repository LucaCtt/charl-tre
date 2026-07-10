class GumbelTemperatureAnnealer:
    """Annealer for the Gumbel-Softmax temperature parameter."""

    def __init__(
        self,
        total_epochs: int,
        start_tau: float = 2,
        final_tau: float = 0.5,
    ) -> None:
        """Initialize the Gumbel temperature annealer.

        Arguments:
            total_epochs: total number of epochs for annealing.
            start_tau: initial temperature value.
            final_tau: minimum temperature value after annealing.

        """
        self._epoch = 0
        self._schedule = self._build_schedule(start_tau, final_tau, total_epochs)
        self._value = self._schedule[0]

    @staticmethod
    def _build_schedule(start_tau: float, final_tau: float, total_epochs: int) -> list[float]:
        """Build the temperature schedule for the entire training run.

        Arguments:
            start_tau: initial temperature value.
            final_tau: minimum temperature value after annealing.
            total_epochs: total number of epochs for annealing.

        Returns:
            List of temperature values for each epoch.

        """
        return [
            max(final_tau, start_tau - (start_tau - final_tau) * epoch / (total_epochs - 1))
            for epoch in range(total_epochs)
        ]

    def step(self) -> None:
        """Advance one epoch in the schedule and return the current temperature value."""
        if self._epoch < len(self._schedule):
            self._value = self._schedule[self._epoch]
            self._epoch += 1

    @property
    def value(self) -> float:
        """Current temperature value."""
        return self._value
