import torch
from torch import nn


class BasicNNClassifier(nn.Module):
    """Basic neural network classifier.

    Arguments:
        input_dim (int): Dimension of the input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Dimension of the hidden layer.

    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """Initialize the classifier model."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (N, num_classes)."""
        return self.net(x)
