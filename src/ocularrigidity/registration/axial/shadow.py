"""Compensation d'ombres (shadow removal) sur les B-scans OCT.

Portage Python de ``correctShadow.m`` (projet MATLAB OneCycleStrainNASA).

Algorithme d'apres :
    Girard, M. J., Strouthidis, N. G., Ethier, C. R., & Mari, J. M. (2011).
    Shadow removal and contrast enhancement in optical coherence tomography
    images of the human optic nerve head. Investigative ophthalmology &
    visual science, 52(10), 7738-7748.
    https://iovs.arvojournals.org/article.aspx?articleid=2165830

Principe : le long de chaque A-scan (colonne = axe axial/profondeur), on divise
l'intensite d'un pixel par l'energie situee PLUS EN PROFONDEUR (l'integrale de
l'intensite du bas de la colonne jusqu'au pixel). Les vaisseaux/ombres, qui
attenuent tout ce qui est en dessous d'eux, sont ainsi compenses et le contraste
des couches profondes (dont la RPE) est rehausse.

Le MATLAB expose trois sorties ``[J, K, L]`` ; c'est la 3e (``L``, calculee sur
``I**n``) qui sert a rehausser la RPE dans le pipeline
(``[~,~,M_comp] = correctShadow(M, 4, 0.8)``). ``correct_shadow`` renvoie donc
``L`` par defaut ; passer ``return_all=True`` pour recuperer ``(J, K, L)``.
"""

from __future__ import annotations

import numpy as np
import torch


def _cumtrapz(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Integrale cumulative trapezoidale (pas unitaire), 0 en tete.

    Equivalent de ``cumtrapz`` MATLAB : meme taille que ``x``, premier element
    nul le long de ``dim`` (``out[k] = out[k-1] + (x[k-1] + x[k]) / 2``).
    """
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
    """``flipud(cumtrapz(flipud(x)))`` : integrale du bas de la colonne jusqu'au pixel.

    Le resultat vaut 0 sur la derniere ligne (fond de l'A-scan) et l'integrale
    totale de la colonne sur la premiere ligne.
    """
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
    """Compensation d'ombres OCT (portage de ``correctShadow.m``).

    Parameters
    ----------
    image : np.ndarray | torch.Tensor
        B-scan ``(H, W)`` ou pile de B-scans ``(T, H, W)`` (ou ``(B, C, H, W)``).
        L'axe axial (profondeur, le long duquel courent les A-scans) est
        l'avant-dernier par defaut (``H``).
    n : float, default 4.0
        Exposant applique a l'intensite (``I**n``) avant compensation. Rehausse
        le contraste ; ``4`` pour la RPE (cf. pipeline MATLAB).
    a : float, default 1.0
        Facteur d'echelle du denominateur (energie sous le pixel).
    axial_dim : int, optional
        Axe axial explicite. Par defaut ``image.ndim - 2`` (l'axe ``H``).
    return_all : bool, default False
        Si vrai, renvoie ``(J, K, L)`` comme le MATLAB au lieu de seulement ``L``.

    Returns
    -------
    np.ndarray | torch.Tensor
        Image compensee ``L`` (meme type/forme que l'entree). Avec
        ``return_all`` : le triplet ``(J, K, L)``.
    """
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

    # J : compense sur I ; K = J**n. Fournis pour fidelite au MATLAB.
    denom_J = a * _cumtrapz_from_bottom(x, axial_dim)
    J = torch.where(denom_J != 0, x / denom_J, torch.zeros_like(x))
    K = J.pow(n)
    if is_numpy:
        return J.numpy(), K.numpy(), L.numpy()
    return J, K, L
