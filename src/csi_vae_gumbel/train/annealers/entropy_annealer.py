from typing import Literal


class EntropyAnnealer:
    """Entropy annealing with mode switching.

    Entropy term in VAE loss:
      - Early: entropy bonus  (encourage exploration)
      - Late:  entropy penalty (encourage commitment)

    λ_t ramps up over time, and its sign flips at switch_epoch.
    """

    def __init__(
        self,
        final_weight: float = 1e-5,
        n_epochs: int = 100,
        switch_epoch: int = 60,
    ) -> None:
        """Initialize the entropy annealer.

        Arguments:
            final_weight: Final |λ| value.
            n_epochs: Total number of epochs.
            switch_epoch: Epoch at which bonus → penalty.
            schedule: Annealing schedule for |λ|.

        """
        self.__final_weight = final_weight
        self.__n_epochs = n_epochs
        self.__switch_epoch = switch_epoch

    def step(self, epoch: int) -> tuple[float, Literal["bonus", "penalty"]]:
        """Get entropy weight and mode for the given epoch.

        Arguments:
            epoch: Current epoch number.

        Returns:
            Tuple of (entropy weight, entropy mode).

        """
        progress = min(epoch / self.__n_epochs, 1.0)
        weight = self.__final_weight * progress

        if epoch < self.__switch_epoch:
            # Entropy bonus: -λ H(q)
            return weight, "bonus"

        # Entropy penalty: +λ H(q)
        return weight, "penalty"
