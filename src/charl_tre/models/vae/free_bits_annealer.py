class FreeBitsAnnealer:
    """Linear free-bits annealer.

    Free bits start at ``start_value`` and decay linearly to ``end_value``
    over ``total_epochs`` scheduler steps.
    """

    def __init__(
        self,
        total_epochs: int,
        start_value: float,
        end_value: float = 0.0,
    ) -> None:
        """Initialize a linear-decay free-bits schedule.

        Arguments:
            total_epochs: Number of epochs across which to apply the schedule.
            start_value: Initial free-bits value at epoch 0.
            end_value: Final free-bits value at the end of the schedule.

        """
        if total_epochs <= 0:
            msg = "total_epochs must be positive."
            raise ValueError(msg)

        self._epoch = 0
        self._schedule = self._build_schedule(total_epochs, start_value, end_value)
        self._value = self._schedule[0]

    @staticmethod
    def _build_schedule(total_epochs: int, start_value: float, end_value: float) -> list[float]:
        if total_epochs == 1:
            return [end_value]

        delta = (start_value - end_value) / (total_epochs - 1)
        return [max(end_value, start_value - delta * idx) for idx in range(total_epochs)]

    def step(self) -> None:
        """Advance one epoch in the schedule."""
        if self._epoch < len(self._schedule):
            self._value = self._schedule[self._epoch]
            self._epoch += 1

    @property
    def value(self) -> float:
        """Current free-bits value."""
        return self._value
