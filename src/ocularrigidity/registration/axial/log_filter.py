"""Filtre Laplacien d'une Gaussienne (LoG) sur les B-scans OCT.

Portage Python de ``laplacianOfAGaussian.m`` (+ l'usage RPE de
``EYYstrain_analysis.m`` : ``kernel = fspecial('log', 9, 3); -conv2(I, kernel, 'same')``).

Le noyau ``fspecial('log', hsize, sigma)`` de MATLAB est reproduit a l'identique
par ``fspecial_log`` ci-dessous, puis convolue (avec negation, comme le MATLAB)
pour donner une reponse positive sur les couches claires fines telles que la RPE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fspecial_log(hsize: int, sigma: float) -> np.ndarray:
    """Reproduction exacte de ``fspecial('log', hsize, sigma)`` (MATLAB).

    Renvoie un noyau carre ``(hsize, hsize)`` de somme nulle (float64).
    """
    # Coordonnees centrees : equivalent de meshgrid(-siz:siz) pour hsize impair,
    # et gere aussi hsize pair (coordonnees demi-entieres), comme MATLAB.
    coords = np.arange(hsize, dtype=np.float64) - (hsize - 1) / 2.0
    x, y = np.meshgrid(coords, coords)

    std2 = float(sigma) ** 2
    arg = -(coords*coords) / (2.0 * std2)
    h = np.exp(arg)
    # MATLAB : annule les valeurs negligeables puis normalise la Gaussienne.
    h[h < np.finfo(float).eps * h.max()] = 0.0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh

    # Laplacien de la Gaussienne, puis recentrage a somme nulle.
    h1 = h * (coords*coords - 2.0 * std2) / (std2 ** 2)
    h = h1 - h1.sum() / (hsize)
    return h


def laplacian_of_gaussian(
    image,
    kernel_size: int,
    sigma: float,
):
    """LoG negatif ``-conv2(image, fspecial('log', kernel_size, sigma), 'same')``.

    Parameters
    ----------
    image : np.ndarray | torch.Tensor
        B-scan ``(H, W)`` ou pile ``(T, H, W)``.
    kernel_size : int
        Taille (cote) du noyau LoG carre.
    sigma : float
        Ecart-type de la Gaussienne (« taille du lissage »).

    Returns
    -------
    np.ndarray | torch.Tensor
        Image filtree (meme type/forme que l'entree). Bords en zero-padding
        (comme ``conv2(..., 'same')``).
    """
    is_numpy = isinstance(image, np.ndarray)
    x = torch.as_tensor(image, dtype=torch.float32) if is_numpy else image.float()

    squeeze_batch = x.ndim == 2
    if squeeze_batch:
        x = x.unsqueeze(0)  # (1, H, W)
    x4 = x.unsqueeze(1)  # (N, 1, H, W)

    kernel = torch.as_tensor(
        fspecial_log(int(kernel_size), float(sigma)),
        dtype=torch.float32,
        device=x.device,
    ).view(1, 1, kernel_size, 1)

    # Le noyau LoG est symetrique : conv2 (convolution) == conv2d (correlation).
    # 'same' pour un noyau impair == zero-padding de (kernel_size // 2).
    pad = kernel_size // 2
    out = -F.conv2d(x4, kernel, padding=(pad,0))
    out = out.squeeze(1)
    if squeeze_batch:
        out = out.squeeze(0)
    return out.numpy() if is_numpy else out
