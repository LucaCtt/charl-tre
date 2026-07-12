class GumbelTemperature:
    """Annealer for the Gumbel-Softmax temperature parameter."""

    def __init__(
        self,
        total_epochs: int,
        temperature_start: float = 2,
        temperature_final: float = 0.5,
    ) -> None:
        """Initialize the Gumbel temperature annealer.

        Arguments:
            total_epochs: total number of epochs for annealing.
            temperature_start: initial temperature value.
            temperature_final: minimum temperature value after annealing.

        """
        self._epoch = 0
        self._schedule = self._build_schedule(temperature_start, temperature_final, total_epochs)
        self._value = self._schedule[0]

    @staticmethod
    def _build_schedule(temperature_start: float, temperature_final: float, total_epochs: int) -> list[float]:
        """Build the temperature schedule for the entire training run.

        Arguments:
            temperature_start: initial temperature value.
            temperature_final: minimum temperature value after annealing.
            total_epochs: total number of epochs for annealing.

        Returns:
            List of temperature values for each epoch.

        """
        return [
            max(
                temperature_final,
                temperature_start - (temperature_start - temperature_final) * epoch / (total_epochs - 1),
            )
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
