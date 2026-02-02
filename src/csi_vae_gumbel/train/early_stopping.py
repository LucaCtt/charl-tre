class EarlyStopping:
    """Early stopping utility to halt training when validation loss stops improving or KL divergence collapses."""

    def __init__(self, patience: int = 10) -> None:
        """Initialize EarlyStopping with a patience parameter.

        Arguments:
            patience: Number of epochs to wait for improvement before stopping.

        """
        self.__patience = patience
        self.__best_total_loss = float("inf")

        self.__loss_counter = 0
        self.__kl_counter = 0

    def step(self, total_loss: float, kl_loss: float, eps: float = 1e-6) -> bool:
        """Check if training should stop early based on loss improvement.

        Arguments:
            total_loss: Current total loss value.
            kl_loss: Current KL divergence loss value.
            eps: Threshold to consider KL divergence as near-zero.

        Returns:
            True if training should stop early, False otherwise.

        """
        # If loss improved, reset counter and continue training
        if total_loss < self.__best_total_loss:
            self.__best_total_loss = total_loss
            self.__loss_counter = 0
        else:
            self.__loss_counter += 1

        if kl_loss < eps:
            self.__kl_counter += 1
        else:
            self.__kl_counter = 0

        return self.__loss_counter >= self.__patience or self.__kl_counter >= self.__patience
