"""Compensation d'ombres (shadow removal) sur les B-scans OCT."""

from __future__ import annotations

import numpy as np
import torch


def _cumtrapz(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Integrale cumulative trapezoidale (pas unitaire), 0 en tete."""
    n = x.size(dim)
    if n < 2:
        return torch.zeros_like(x)
    avg = 0.5 * (x.narrow(dim, 0, n - 1) + x.narrow(dim, 1, n - 1))
    c = torch.cumsum(avg, dim=dim)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    zero = torch.zeros(pad_shape, dtype=c.dtype, device=c.device)
    return torch.cat([zero, c], dim=dim)


def _cumtrapz_from_bottom(x: torch.Tensor, dim: int) -> torch.Tensor:
    """``flipud(cumtrapz(flipud(x)))`` : integrale du bas de la colonne jusqu'au pixel."""
    xf = torch.flip(x, dims=[dim])
    c = _cumtrapz(xf, dim)
    return torch.flip(c, dims=[dim])


def correct_shadow(
    image,
    n: float = 4.0,
    a: float = 1.0,
    axial_dim: int | None = None,
    return_all: bool = False,
):
    """Compensation d'ombres OCT (portage de ``correctShadow.m``)."""
    is_numpy = isinstance(image, np.ndarray)
    x = torch.as_tensor(image, dtype=torch.float32) if is_numpy else image.float()

    if axial_dim is None:
        axial_dim = x.ndim - 2

    # L : compense sur I**n (la sortie utilisee pour la RPE).
    x_n = x.pow(n)
    denom_L = a * _cumtrapz_from_bottom(x, axial_dim).pow(n)
    L = torch.where(denom_L != 0, x_n / denom_L, torch.zeros_like(x_n))

    if not return_all:
        return L.numpy() if is_numpy else L

    # J : compense sur I
    denom_J = a * _cumtrapz_from_bottom(x, axial_dim)
    J = torch.where(denom_J != 0, x / denom_J, torch.zeros_like(x))
    K = J.pow(n)
    if is_numpy:
        return J.numpy(), K.numpy(), L.numpy()
    return J, K, L
