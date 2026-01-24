"""Gumbel-Softmax sampling with numerical stability improvements."""

import torch
from torch import nn


@torch.no_grad()
def _sample_gumbel_like(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Draw Gumbel(0,1) noise of the same shape as `tensor` with robust clamping of uniforms.

    Arguments:
        tensor (torch.Tensor): Tensor whose shape to match.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Sampled Gumbel noise.

    """
    # Clamp away from {0,1} to avoid -log(-log(U)) overflow/NaN
    u = torch.rand_like(tensor).clamp_(eps, 1.0 - eps)
    return -torch.log(-torch.log(u))


def gumbel_softmax_stable(
    logits: torch.Tensor,
    tau: float,
    dim: int = -1,
    eps_u: float = 1e-6,
    center: bool = True,
) -> torch.Tensor:
    """Numerically stable (soft) Gumbel-Softmax.

    Clamps uniforms, enforces a minimum temperature,
    optional centering before softmax (shift-invariant, avoids overflow).

    Arguments:
        logits (torch.Tensor): Logits of the categorical distribution.
        tau (float): Temperature parameter.
        dim (int): Dimension along which to apply softmax.
        eps_u (float): Small constant for numerical stability of uniform sampling.
        center (bool): Whether to center logits before softmax.

    Returns:
        torch.Tensor: Sampled tensor from the Gumbel-Softmax distribution.

    """
    tau_eff = max(float(tau), 1e-3)
    g = _sample_gumbel_like(logits, eps=eps_u)
    y = logits + g
    if center:
        y = y - y.max(dim=dim, keepdim=True).values
    y = y / tau_eff
    return nn.functional.softmax(y, dim=dim)


def gumbel_softmax_straight_through_stable(
    logits: torch.Tensor,
    tau: float,
    dim: int = -1,
    eps_u: float = 1e-6,
    center: bool = True,
) -> torch.Tensor:
    """Straight-Through Gumbel-Softmax.

    In forward pass we get a hard one-hot sample, in backward gradients flow as if soft (y_soft).

    Arguments:
        logits (torch.Tensor): Logits of the categorical distribution.
        tau (float): Temperature parameter.
        dim (int): Dimension along which to apply softmax.
        eps_u (float): Small constant for numerical stability of uniform sampling.
        center (bool): Whether to center logits before softmax.

    Returns:
        torch.Tensor: Sampled tensor from the Gumbel-Softmax distribution with straight-through estimator.

    """
    y_soft = gumbel_softmax_stable(logits, tau=tau, dim=dim, eps_u=eps_u, center=center)

    # Hard one-hot in forward
    _, k = y_soft.max(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim, k, 1.0)

    # Straight-through trick: forward = y_hard, backward = y_soft
    return (y_hard - y_soft).detach() + y_soft
