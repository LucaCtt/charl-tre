import numpy as np


class GumbelAnnealer:
    """Handles the exponential decay of the Gumbel-Softmax temperature (tau)."""

    def __init__(
        self,
        start_tau: float = 1.0,
        min_tau: float = 0.1,
        decay_rate: float = 0.013,  # Controls how fast it drops
        step_interval: int = 1,  # Usually decay every epoch
    ) -> None:
        """Initialize the Gumbel-Softmax temperature annealer.

        Arguments:
            start_tau: Initial temperature (usually 1.0 or higher).
            min_tau: The floor temperature (don't go below this, or gradients vanish).
            decay_rate: The rate of decay. Lower means slower cooling.
            step_interval: How often to update the temperature.

        """
        self.__tau = start_tau
        self.__min_tau = min_tau
        self.__decay_rate = decay_rate
        self.__step_interval = step_interval

    def step(self, epoch: int) -> float:
        """Calculate the temperature for the current epoch.

        Formula: tau = max(min_tau, start_tau * exp(-decay_rate * epoch))

        Arguments:
            epoch: Current epoch number.

        Returns:
            The updated temperature for the Gumbel-Softmax distribution.

        """
        if epoch % self.__step_interval == 0:
            new_tau = self.__tau * np.exp(-self.__decay_rate * epoch)
            self.__tau = max(self.__min_tau, new_tau)

        return self.__tau
