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
        max_weight: float = 0.05,
        n_epochs: int = 100,
        switch_epoch: int = 60,
    ) -> None:
        """Initialize the entropy annealer.

        Arguments:
            max_weight: Final |λ| value.
            n_epochs: Total number of epochs.
            switch_epoch: Epoch at which bonus → penalty.
            schedule: Annealing schedule for |λ|.

        """
        self.max_weight = max_weight
        self.n_epochs = n_epochs
        self.switch_epoch = switch_epoch

    def step(self, epoch: int) -> tuple[float, Literal["bonus", "penalty"]]:
        """Get entropy weight and mode for the given epoch.

        Arguments:
            epoch: Current epoch number.

        Returns:
            Tuple of (entropy weight, entropy mode).

        """
        progress = min(epoch / self.n_epochs, 1.0)
        weight = self.max_weight * progress

        if epoch < self.switch_epoch:
            # Entropy bonus: -λ H(q)
            return weight, "bonus"

        # Entropy penalty: +λ H(q)
        return weight, "penalty"
