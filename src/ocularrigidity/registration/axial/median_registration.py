"""Second recalage : alignement axial de chaque A-scan sur la mediane du volume.

Apres le recalage de la video entiere (``register_masks_by_displacement``), on
raffine l'alignement de la RPE ainsi :

  1. mediane temporelle du volume recale -> image de reference (``temporal_median``) ;
  2. compensation d'ombres (``correct_shadow``) de la mediane ET de chaque image ;
  3. filtrage Laplacien-d'une-Gaussienne (``laplacian_of_gaussian``) des resultats,
     ce qui isole les couches claires fines (dont la RPE) ;
  4. pour chaque A-scan (colonne) de chaque image, recalage vertical sur l'A-scan
     correspondant de la mediane par correlation de phase spectrale (meme principe
     que ``estimate_lateral_shift_fullframe``, ici 1D le long de l'axe axial),
     borne par un deplacement vertical maximal ;
  5. application du deplacement par colonne aux images BRUTES recalees (et masques)
     via ``F.grid_sample``.

Le pretraitement (etapes 2-3) ne sert qu'a ESTIMER les deplacements ; ils sont
appliques aux pixels d'origine, sans jamais renvoyer d'image compensee/filtree.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ocularrigidity.registration.rigid import temporal_median
from ocularrigidity.registration.axial.shadow import correct_shadow
from ocularrigidity.registration.axial.log_filter import laplacian_of_gaussian


def _preprocess(
    x,
    use_shadow: bool,
    shadow_n: float,
    shadow_a: float,
    use_log: bool,
    log_kernel_size: int,
    log_sigma: float,
):
    """Pretraitement RPE decouple : compensation d'ombres et/ou LoG, ou aucun.

    ``use_shadow`` et ``use_log`` sont independants : on peut appliquer l'un,
    l'autre, les deux, ou aucun (dans ce cas ``x`` est renvoye tel quel, et la
    correlation de phase opere sur les pixels bruts).
    """
    if use_shadow:
        x = correct_shadow(x, shadow_n, shadow_a)
    if use_log:
        x = laplacian_of_gaussian(x, log_kernel_size, log_sigma)
    return x


def _axial_window_band(
    H: int, bandpass: tuple[float, float], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fenetre de Hann axiale (H,) + masque passe-bande spectral (H//2+1,).

    La fenetre de Hann est INDISPENSABLE ici : ``correct_shadow`` divise par
    l'energie cumulee depuis le bas de l'A-scan, ce qui fait exploser les
    dernieres lignes (denominateur ~ 0) en pics fixes qui, sans fenetrage,
    verrouillent la correlation de phase sur le decalage nul. Le passe-bande
    retire en plus la composante continue et le bruit haute frequence. Meme
    strategie que ``estimate_lateral_shift_fullframe``.
    """
    win = torch.hann_window(H, periodic=False, device=device)
    fy = torch.fft.rfftfreq(H, device=device)
    f_lo, f_hi = bandpass
    band = (1.0 - torch.exp(-((fy / f_lo) ** 2))) * torch.exp(-((fy / f_hi) ** 2))
    return win, band


def _phase_corr_vshift(
    x_pre: torch.Tensor,
    Fm_conj: torch.Tensor,
    win: torch.Tensor,
    band: torch.Tensor,
    H: int,
    center: int,
    y_lo: int,
    y_hi: int,
    subpixel: bool,
    eps: float,
) -> torch.Tensor:
    """Deplacement vertical par colonne d'un lot pretraite ``x_pre`` (b, H, W).

    ``Fm_conj`` est ``conj(rfft(win * median_pre, axe axial))`` (1, H//2+1, W).
    ``win`` (H,) et ``band`` (H//2+1,) proviennent de ``_axial_window_band`` et
    doivent etre les MEMES que ceux utilises pour ``Fm_conj``. Renvoie ``dy``
    (b, W) : le decalage a appliquer (``sample_y = grid_y + dy``) pour aligner
    chaque colonne sur la mediane.
    """
    x = x_pre - x_pre.mean(dim=1, keepdim=True)  # retire la composante continue
    x = x * win.view(1, H, 1)  # fenetrage de Hann axial
    Fx = torch.fft.rfft(x, dim=1)  # (b, H//2+1, W)
    cps = Fx * Fm_conj
    cps = cps / (cps.abs() + eps)  # correlation de phase (phase-only)
    cps = cps * band.view(1, -1, 1)  # passe-bande spectral
    corr = torch.fft.irfft(cps, n=H, dim=1)  # (b, H, W)
    corr = torch.fft.fftshift(corr, dim=1)  # decalage nul au centre

    # Restreint la recherche du pic a +-max_vshift autour du centre.
    window = torch.full((H,), float("-inf"), device=corr.device)
    window[y_lo:y_hi] = 0.0
    corr = corr + window.view(1, H, 1)

    peak = corr.argmax(dim=1).clamp(1, H - 2)  # (b, W)
    if subpixel:
        ym1 = torch.gather(corr, 1, (peak - 1).unsqueeze(1)).squeeze(1)
        y0 = torch.gather(corr, 1, peak.unsqueeze(1)).squeeze(1)
        yp1 = torch.gather(corr, 1, (peak + 1).unsqueeze(1)).squeeze(1)
        denom = ym1 - 2 * y0 + yp1
        offset = torch.where(
            torch.isfinite(denom) & (denom != 0),
            0.5 * (ym1 - yp1) / denom,
            torch.zeros_like(y0),
        ).clamp(-1.0, 1.0)
        peak_sub = peak.float() + offset
    else:
        peak_sub = peak.float()
    return peak_sub - center


@torch.inference_mode()
def estimate_ascan_vshift_to_median(
    frames_pre,
    median_pre,
    max_vshift: int = 30,
    subpixel: bool = True,
    batch_size: int = 64,
    device: str = "cuda",
    bandpass: tuple[float, float] = (0.02, 0.5),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Deplacement vertical par A-scan alignant ``frames_pre`` sur ``median_pre``.

    Entrees DEJA pretraitees (compensation + LoG). Correlation de phase spectrale
    le long de l'axe axial (H), colonne par colonne, avec fenetrage de Hann et
    passe-bande (cf. ``_axial_window_band``).

    Parameters
    ----------
    frames_pre : (T, H, W) ; median_pre : (H, W)
    max_vshift : int
        Deplacement vertical maximal (px) teste de chaque cote.
    bandpass : (float, float)
        Bornes basse/haute du passe-bande spectral (fraction de la freq. de Nyquist).

    Returns
    -------
    torch.Tensor
        ``dy`` (T, W) sur ``device`` : convention ``sample_y = grid_y + dy``.
    """
    if isinstance(frames_pre, np.ndarray):
        frames_pre = torch.from_numpy(frames_pre)
    if isinstance(median_pre, np.ndarray):
        median_pre = torch.from_numpy(median_pre)
    T, H, W = frames_pre.shape
    center = H // 2

    win, band = _axial_window_band(H, bandpass, device)
    m = median_pre.to(device).float()
    m = (m - m.mean(dim=0, keepdim=True)) * win.view(H, 1)
    Fm_conj = torch.fft.rfft(m, dim=0).conj().unsqueeze(0)  # (1, H//2+1, W)

    yw = int(max_vshift)
    y_lo, y_hi = max(0, center - yw), min(H, center + yw + 1)

    dy = torch.empty((T, W), dtype=torch.float32, device=device)
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        x = frames_pre[start:end].to(device).float()
        dy[start:end] = _phase_corr_vshift(
            x, Fm_conj, win, band, H, center, y_lo, y_hi, subpixel, eps
        )
    return dy


@torch.inference_mode()
def register_ascans_to_median(
    frames,
    masks=None,
    *,
    max_vshift: int = 30,
    use_shadow: bool = True,
    shadow_n: float = 4.0,
    shadow_a: float = 0.8,
    use_log: bool = True,
    log_kernel_size: int = 9,
    log_sigma: float = 3.0,
    subpixel: bool = True,
    batch_size: int = 64,
    device: str = "cuda",
    bandpass: tuple[float, float] = (0.02, 0.5),
    ignore_zeros_median: bool = True,
    verbose: bool = True,
    return_median: bool = False,
):
    """Recale chaque A-scan du volume sur la mediane (identification de la RPE).

    Enchaine mediane -> compensation d'ombres -> LoG -> correlation de phase par
    A-scan -> application par ``grid_sample`` aux pixels bruts. Le pretraitement
    est fait par lots pour borner la memoire.

    Parameters
    ----------
    frames : (T, H, W)
        Volume DEJA recale (sortie de ``register_masks_by_displacement``).
    masks : (T, H, W), optional
        Masques a transporter avec les images (meme deplacement par colonne).
    max_vshift : int
        Deplacement vertical maximal (px) — parametre d'entree demande.
    use_shadow, use_log : bool
        Activent independamment la compensation d'ombres et le filtre LoG. Les
        deux, l'un, l'autre, ou aucun (correlation sur les pixels bruts).
    shadow_n, shadow_a : float
        Parametres de ``correct_shadow`` (compensation d'ombres).
    log_kernel_size, log_sigma : int, float
        Taille du noyau et sigma (lissage) du LoG.

    Returns
    -------
    (registered_frames, registered_masks, dy)
        ``registered_frames`` (T, H, W) — MEME dtype que l'entree — CPU,
        ``registered_masks`` (T, H, W) bool CPU (ou ``None``), ``dy`` (T, W) float32
        CPU. Avec ``return_median``, renvoie en plus le template median (H, W) numpy.
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    if masks is not None and isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    T, H, W = frames.shape
    center = H // 2
    frame_dtype = frames.dtype  # on preserve le dtype d'entree en sortie

    # 1) mediane temporelle (volume en memoire) -> reference.
    median = temporal_median(
        frames, ignore_zeros=ignore_zeros_median, device=device
    )  # (H, W) sur device

    # 2-3) compensation et/ou LoG de la mediane (decouples), puis rfft de reference.
    median_pre = _preprocess(
        median, use_shadow, shadow_n, shadow_a, use_log, log_kernel_size, log_sigma
    )
    win, band = _axial_window_band(H, bandpass, device)
    m = (median_pre - median_pre.mean(dim=0, keepdim=True)) * win.view(H, 1)
    Fm_conj = torch.fft.rfft(m, dim=0).conj().unsqueeze(0)

    yw = int(max_vshift)
    y_lo, y_hi = max(0, center - yw), min(H, center + yw + 1)

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    norm_x_full = (xs.view(1, 1, W).expand(1, H, W)) / (W - 1) * 2 - 1

    dy_out = torch.empty((T, W), dtype=torch.float32, device="cpu")
    frames_out = torch.empty((T, H, W), dtype=frame_dtype, device="cpu")
    masks_out = (
        torch.empty((T, H, W), dtype=torch.bool, device="cpu")
        if masks is not None
        else None
    )

    for start in tqdm(
        range(0, T, batch_size),
        desc="A-scan -> median",
        disable=not verbose,
        leave=False,
    ):
        end = min(start + batch_size, T)
        t = end - start
        raw = frames[start:end].to(device).float()  # (t, H, W)

        # 4) pretraitement du lot (decouple) puis correlation de phase par colonne.
        pre = _preprocess(
            raw, use_shadow, shadow_n, shadow_a, use_log, log_kernel_size, log_sigma
        )
        dy = _phase_corr_vshift(
            pre, Fm_conj, win, band, H, center, y_lo, y_hi, subpixel, eps=1e-8
        )  # (t, W)
        if not subpixel:
            dy = dy.round()

        # 5) application du deplacement par colonne aux pixels bruts (+ masque).
        if masks is not None:
            mk = masks[start:end].to(device).float()
            data = torch.stack([mk, raw], dim=1)  # (t, 2, H, W)
        else:
            data = raw.unsqueeze(1)  # (t, 1, H, W)

        grid_y = ys.view(1, H, 1).expand(t, H, W)
        sample_y = grid_y + dy.view(t, 1, W)  # deplacement PAR COLONNE
        norm_y = sample_y / (H - 1) * 2 - 1
        norm_x = norm_x_full.expand(t, H, W)
        grid = torch.stack([norm_x, norm_y], dim=-1)

        reg = F.grid_sample(
            data, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        if masks is not None:
            masks_out[start:end] = (reg[:, 0] > 0.5).cpu()
            frames_out[start:end] = reg[:, 1].to(frame_dtype).cpu()
        else:
            frames_out[start:end] = reg[:, 0].to(frame_dtype).cpu()
        dy_out[start:end] = dy.cpu()

        del raw, pre, data, reg
        if device != "cpu":
            torch.cuda.empty_cache()

    if return_median:
        return frames_out, masks_out, dy_out, median.cpu().numpy()
    return frames_out, masks_out, dy_out
