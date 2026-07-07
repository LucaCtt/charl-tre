class DeadLossError(Exception):
    """Raised when the loss becomes NaN or infinite during training."""

    def __init__(self) -> None:
        """Initialize the error with the problematic loss value."""
        super().__init__("Loss became NaN or infinite.")


class PosteriorCollapseError(Exception):
    """Raised when the VAE posterior collapses during training."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Posterior collapse detected.")
