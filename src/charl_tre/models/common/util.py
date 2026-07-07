import math

import torch
from torch import nn


def is_dead(tensor: torch.Tensor) -> bool:
    """Check if a tensor contains NaN or infinite values.

    Arguments:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor contains NaN or infinite values, False otherwise.

    """
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


def build_fc(in_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """Build FC block with n_layers, keeping hidden dims as multiples of 8.

    Arguments:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        n_layers (int): Number of layers in the FC block.
        dropout (float): Dropout rate for the FC block.

    Returns:
        nn.Sequential: A sequential model containing the FC block.

    """
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    # Geometrically interpolate hidden dims, only internal ones must be multiples of 8
    dims = (
        [in_dim]
        + [_next_multiple_of_8(int(in_dim * ((out_dim / in_dim) ** (i / (n_layers - 1))))) for i in range(1, n_layers)]
        + [out_dim]
    )

    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < n_layers - 1:
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


def _next_multiple_of_8(n: int) -> int:
    """Round n up to the next multiple of 8.

    Arguments:
        n (int): Integer to round up.

    Returns:
        int: The next multiple of 8 greater than or equal to n.

    """
    return math.ceil(n / 8) * 8
