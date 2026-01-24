class EarlyStopping:
    """Early stopping utility to halt training when validation loss stops improving."""

    def __init__(self, patience: int) -> None:
        """Initialize EarlyStopping with a patience parameter.

        Arguments:
            patience: Number of epochs to wait for improvement before stopping.

        """
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        """Check if training should stop early based on loss improvement."""
        # If loss improved, reset counter and continue training
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False

        # If loss did not improve, increment counter
        self.counter += 1
        return self.counter >= self.patience
