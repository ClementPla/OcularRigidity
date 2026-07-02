import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ocularrigidity.registration.axial.utils import temporal_median


def _axial_window_band(
    H: int, bandpass: tuple[float, float], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
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
    subpixel: bool = True,
    batch_size: int = 64,
    device: str = "cuda",
    bandpass: tuple[float, float] = (0.02, 0.5),
    ignore_zeros_median: bool = True,
    verbose: bool = True,
    return_median: bool = False,
):
    """Recale chaque A-scan du volume sur la mediane (identification de la RPE).


    Parameters
    ----------
    frames : (T, H, W)
        Volume DEJA recale (sortie de ``register_masks_by_displacement``).
    masks : (T, H, W), optional
        Masques a transporter avec les images (meme deplacement par colonne).
    max_vshift : int
        Deplacement vertical maximal (px) — parametre d'entree demande.

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
    frame_dtype = frames.dtype  # on preserve le dtype d'entree en sortie

    # 1) mediane temporelle (volume en memoire) -> reference.
    median = temporal_median(
        frames, ignore_zeros=ignore_zeros_median, device=device
    )  # (H, W) sur device

    y0, y1 = 0, H
    Hc = y1 - y0
    center = Hc // 2

    win, band = _axial_window_band(Hc, bandpass, device)
    m = (median - median.mean(dim=0, keepdim=True)) * win.view(Hc, 1)
    Fm_conj = torch.fft.rfft(m, dim=0).conj().unsqueeze(0)

    yw = int(max_vshift)
    y_lo, y_hi = max(0, center - yw), min(Hc, center + yw + 1)

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

        dy = _phase_corr_vshift(
            raw[:, y0:y1, :],
            Fm_conj,
            win,
            band,
            Hc,
            center,
            y_lo,
            y_hi,
            subpixel,
            eps=1e-8,
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

        del raw, data, reg
        if device != "cpu":
            torch.cuda.empty_cache()

    if return_median:
        return frames_out, masks_out, dy_out, median.cpu().numpy()
    return frames_out, masks_out, dy_out
