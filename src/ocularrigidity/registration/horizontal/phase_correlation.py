import torch
import numpy as np
import torch.nn.functional as F


def estimate_lateral_shift_xcorr_subpixel(
    curve: torch.Tensor,
    ref_curve: torch.Tensor,
    max_shift: int = None,
    drop_edges: int = 75,
    subpixel: bool = True,
) -> torch.Tensor:
    """
    curve: T x W, ref_curve: W
    Returns dx: T, the sub-pixel lateral shift that best aligns
    curve[t] to ref_curve via 1D cross-correlation and parabolic interpolation.
    When ``subpixel`` is False the parabolic fit is skipped and the integer-pixel
    cross-correlation peak is returned directly.
    """
    T, W = curve.shape
    if max_shift is None:
        max_shift = W // 4

    curve = curve[:, drop_edges : W - drop_edges]
    ref_curve = ref_curve[drop_edges : W - drop_edges]
    T, W = curve.shape
    c = curve - curve.mean(dim=1, keepdim=True)
    r = ref_curve - ref_curve.mean()

    shifts = torch.arange(-max_shift, max_shift + 1, device=curve.device)
    num_shifts = len(shifts)
    scores = torch.zeros(T, num_shifts, device=curve.device)

    # 1. Compute integer cross-correlation scores
    for i, s in enumerate(shifts):
        s = int(s.item())
        if s >= 0:
            c_seg = c[:, : W - s] if s > 0 else c
            r_seg = r[s:] if s > 0 else r
        else:
            c_seg = c[:, -s:]
            r_seg = r[: W + s]
        scores[:, i] = (c_seg * r_seg).sum(dim=1)

    # 2. Find the best integer shift
    best_idx = scores.argmax(dim=1)
    dx = shifts[best_idx].float()

    # 3. Sub-pixel Interpolation (Parabolic Fit)
    # Ensure we only interpolate if the peak isn't touching the boundary edges
    valid_mask = (best_idx > 0) & (best_idx < num_shifts - 1)

    if subpixel and valid_mask.any():
        idx_valid = best_idx[valid_mask]

        # Get scores at the peak (y0) and its neighbors (y_minus_1, y_plus_1)
        y_minus_1 = scores[valid_mask, idx_valid - 1]
        y_0 = scores[valid_mask, idx_valid]
        y_plus_1 = scores[valid_mask, idx_valid + 1]

        # Parabolic interpolation formula
        denominator = 2 * (y_minus_1 - 2 * y_0 + y_plus_1)

        # Protect against division by zero (e.g., flat regions)
        nonzero_mask = denominator != 0

        offset = torch.zeros_like(y_0)
        offset[nonzero_mask] = (
            y_minus_1[nonzero_mask] - y_plus_1[nonzero_mask]
        ) / denominator[nonzero_mask]

        # Add the fractional offset to the integer shift
        dx[valid_mask] += offset

    return dx


@torch.inference_mode()
def estimate_lateral_shift_fullframe(
    frames: torch.Tensor,
    ref: torch.Tensor = None,
    downsample_to: tuple[int, int] = (512, 512),
    max_shift: int = 16,
    max_vshift: int = 512,
    batch_size: int = 256,
    device: str = "cuda",
    bandpass: tuple[float, float] = (0.02, 0.5),
    return_confidence: bool = False,
    subpixel: bool = True,
) -> torch.Tensor:
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    T, H, W = frames.shape
    h, w = downsample_to
    center = w // 2

    # crop to the 3/4 central region of the frame for lateral shift estimation, to avoid edge artifacts
    crop_h = int(H * 3 / 4)
    crop_y_start = (H - crop_h) // 2
    crop_w = int(W * 3 / 4)
    crop_x_start = (W - crop_w) // 2

    frames = frames[
        :, crop_y_start : crop_y_start + crop_h, crop_x_start : crop_x_start + crop_w
    ]
    ref = (
        ref[crop_y_start : crop_y_start + crop_h, crop_x_start : crop_x_start + crop_w]
        if ref is not None
        else None
    )

    # Hann window kills FFT wrap-around edge artifacts.
    win = (
        torch.hann_window(h, periodic=False, device=device)[:, None]
        * torch.hann_window(w, periodic=False, device=device)[None, :]
    )

    # Reference: temporal median template (computed on a subsample for speed).
    if ref is None:
        idx = torch.linspace(0, T - 1, min(T, 64)).round().long()
        ref = frames[idx].to(device).float().median(dim=0).values
    else:
        ref = ref.to(device).float()
    r = F.interpolate(ref[None, None], size=(h, w), mode="area")[0, 0]
    r = (r - r.mean()) * win
    Fr_conj = torch.fft.rfft2(r)[None].conj()  # (1, h, w//2+1)

    # Restrict the peak search to +-max_shift (converted to downsampled pixels).
    max_shift_down = max(1, int(round(max_shift * w / crop_w)))
    positions = torch.arange(w, device=device) - center  # zero shift at `center`
    in_window = positions.abs() <= max_shift_down

    f_lo, f_hi = bandpass
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    fr = torch.sqrt(fy**2 + fx**2)
    band = ((1.0 - torch.exp(-((fr / f_lo) ** 2))) * torch.exp(-((fr / f_hi) ** 2)))[
        None
    ]
    y_center = h // 2
    yw = max(1, int(round(max_vshift * h / crop_h)))
    y_lo, y_hi = max(0, y_center - yw), min(h, y_center + yw + 1)

    dx = torch.empty(T, device=device, dtype=torch.float32)
    conf = torch.empty(T, device=device, dtype=torch.float32)
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        x = frames[start:end].to(device).float()
        x = F.interpolate(x.unsqueeze(1), size=(h, w), mode="area").squeeze(1)
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) * win

        cps = torch.fft.rfft2(x) * Fr_conj
        cps = cps / (cps.abs() + 1e-8)  # phase only
        cps = cps * band
        corr = torch.fft.irfft2(cps, s=(h, w))
        corr = torch.fft.fftshift(corr, dim=(-2, -1))  # zero shift at center

        corr_x = corr[:, y_lo:y_hi, :].sum(dim=1)  # (b, w)
        # Light 1D smoothing suppresses isolated speckle spikes.
        corr_x = F.avg_pool1d(corr_x.unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
        corr_x = corr_x.masked_fill(~in_window[None], -float("inf"))
        peak = corr_x.argmax(dim=1).clamp(1, w - 2)  # (b,)

        ym1 = corr_x.gather(1, (peak - 1)[:, None]).squeeze(1)
        y0 = corr_x.gather(1, peak[:, None]).squeeze(1)
        yp1 = corr_x.gather(1, (peak + 1)[:, None]).squeeze(1)
        if subpixel:
            denom = ym1 - 2 * y0 + yp1
            offset = torch.where(
                torch.isfinite(denom) & (denom != 0),
                0.5 * (ym1 - yp1) / denom,
                torch.zeros_like(y0),
            ).clamp(-1.0, 1.0)
            peak_sub = peak.float() + offset
        else:
            peak_sub = peak.float()

        win_vals = corr_x[:, in_window]
        conf[start:end] = (y0 - win_vals.mean(dim=1)) / (win_vals.std(dim=1) + 1e-8)

        dx[start:end] = (center - peak_sub) * (crop_w / w)

    # Pixel-precise request: snap to whole ORIGINAL-resolution pixels. The peak is
    # already integer on the downsampled grid, but the back-projection
    # (× crop_w / w) turns it into a fraction; rounding here makes the returned
    # shift an integer number of pixels in the original image (not the
    # downsampled one). With ``subpixel`` the fractional peak offset is kept.
    if not subpixel:
        dx = dx.round()

    if return_confidence:
        return dx, conf

    return dx
