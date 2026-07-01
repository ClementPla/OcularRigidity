import torch
import torch.nn.functional as F
import numpy as np
from ocularrigidity.registration.horizontal.phase_correlation import (
    estimate_lateral_shift_fullframe,
    estimate_lateral_shift_xcorr_subpixel,
)
from ocularrigidity.registration.horizontal.utils import (
    _interp1d,
    _median_filter_1d,
    smooth_translations,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)
from tqdm.auto import tqdm


def robust_temporal_dx(
    dx: torch.Tensor,
    conf: torch.Tensor = None,
    conf_z: float = 1.0,
    k: float = 2.5,
    win: int = 55,
    max_velocity: float = 4.0,
) -> torch.Tensor:
    """Reject temporally inconsistent lateral shifts and re-interpolate them.
    A frame is kept only if it satisfies every enabled criterion:

      - ``conf >= conf_z`` (when ``conf`` is given): drops frames whose
        correlation peak was too weak to mean anything. Note confidence can only
        *remove* a frame, never rescue an inconsistent one -- a sharp
        (high-confidence) peak that locks onto a different lateral feature than
        its neighbours is exactly the jitter we want to discard, so the temporal
        tests below take precedence.
      - ``|dx - rolling_median| <= k * MAD``: the primary consistency test,
        scaled by the robust median-absolute-deviation. ``k`` is tight by default
        because real motion does not jump frame-to-frame.
      - ``|dx - rolling_median| <= max_velocity``: a hard cap (in pixels) on how
        far a single frame may stray from the local trend. We measure deviation
        from the rolling median rather than the raw consecutive difference
        ``|dx[t] - dx[t-1]|`` on purpose: an isolated outlier inflates *two*
        consecutive differences and would wrongly condemn its innocent
        neighbour, whereas the median trend stays put.

    Rejected frames are linearly interpolated from the surviving ones.
    """
    dx = dx.float()
    good = torch.ones_like(dx, dtype=torch.bool)
    if conf is not None:
        good &= conf >= conf_z
    med = _median_filter_1d(dx, win)
    resid = (dx - med).abs()
    # Primary consistency test: deviation from the local median, robust-scaled.
    good &= resid <= k * (torch.median(resid) + 1e-6)
    # Hard velocity cap: reject anything that strays too far from the trend.
    if max_velocity is not None:
        good &= resid <= max_velocity
    if good.all() or good.sum() < 2:
        return dx
    idx = torch.arange(dx.numel(), device=dx.device, dtype=torch.float32)
    interp = _interp1d(idx, idx[good], dx[good])
    return torch.where(good, dx, interp)


@torch.inference_mode()
@torch.no_grad()
def register_masks_by_displacement(
    raw_masks: torch.Tensor,
    raw_frames: torch.Tensor,
    correct_dx: bool = True,
    flatten_rpe: bool = False,
    batch_size: int = 256,
    device: str = "cuda",
    verbose: bool = True,
    return_params: bool = False,
    lateral_method: str = "xcorr",
    subpixel: bool = True,
    max_shift: int = 16,
):
    """Register frames/masks by lateral shift + vertical boundary displacement.

    Returns ``(registered_masks, registered_frames)``. When ``return_params`` is
    True, additionally returns a ``params`` dict with the transform that was
    applied: ``{"dx": (T,) lateral shift, "dy": (T, W) vertical displacement}``
    (both CPU tensors). ``dx`` is all-zeros when ``correct_dx`` is False.

    ``lateral_method`` selects how the lateral shift is estimated:
      - ``"xcorr"``: 1D cross-correlation of vertical-mean profiles (default).
      - ``"fullframe"``: 2D phase correlation of the full frames
    """
    if isinstance(raw_masks, np.ndarray):
        raw_masks = torch.from_numpy(raw_masks)
    if isinstance(raw_frames, np.ndarray):
        raw_frames = torch.from_numpy(raw_frames)
    T, H, W = raw_masks.shape
    mask_dtype = raw_masks.dtype
    frame_dtype = raw_frames.dtype
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    # Vertical-mean profiles are only needed by the "xcorr" lateral method.
    need_profiles = correct_dx and lateral_method == "xcorr"
    bms_list = []
    csi_list = []
    profile_chunks = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        masks_chunk = raw_masks[start:end].to(torch.float32)
        bm, csi = extract_boundaries_fast(masks_chunk.cpu().numpy())
        bm, csi = clean_boundaries(bm, csi)
        bms_list.append(torch.from_numpy(bm).to(device))
        csi_list.append(torch.from_numpy(csi).to(device))

        if need_profiles:
            frames_chunk = raw_frames[start:end].to(torch.float32).to(device)
            padded_frames = F.pad(
                frames_chunk.unsqueeze(1), (2, 2, 2, 2), mode="replicate"
            )
            blurred_frames = F.avg_pool2d(
                padded_frames, kernel_size=5, stride=1
            )  # Fast blur alternative
            profile_chunks.append(blurred_frames.squeeze(1).mean(dim=1))

    all_bms = torch.cat(bms_list, dim=0).to(device)
    # Find the reference indices: count the number of masks pixels per frame, take the median, and pick the frame closest to that median as the reference.
    mask_counts = raw_masks.sum(dim=(1, 2))
    median_count = mask_counts.median()
    ref_idx = (mask_counts - median_count).abs().argmin()
    ref_bm = all_bms[ref_idx]

    if correct_dx:
        if lateral_method == "fullframe":
            global_dx, conf = estimate_lateral_shift_fullframe(
                raw_frames,
                ref=raw_frames[
                    ref_idx
                ],  # same anchor as the vertical reference (frame 0)
                batch_size=batch_size,
                device=device,
                return_confidence=True,
                subpixel=subpixel,
                max_shift=max_shift,
            )
            global_dx = robust_temporal_dx(global_dx, conf=conf)
        elif lateral_method == "xcorr":
            profiles = torch.cat(profile_chunks, dim=0).to(device)
            global_dx = estimate_lateral_shift_xcorr_subpixel(
                profiles,
                profiles[ref_idx],
                101,
                drop_edges=100,
                subpixel=subpixel,
            ).to(device)  # T
            global_dx = robust_temporal_dx(global_dx)
        else:
            raise ValueError(f"Unknown lateral_method: {lateral_method!r}")
        # Real lateral eye motion is low-frequency; a wider Gaussian absorbs the
        # residual frame-to-frame jitter that survives outlier rejection.
        global_dx = smooth_translations(global_dx, sigma=4.0)

    registered_masks_chunks = []
    registered_frames_chunks = []
    displacement_chunks = []

    for start in tqdm(
        range(0, T, batch_size),
        desc="Registering frames",
        disable=not verbose,
        leave=False,
    ):
        end = min(start + batch_size, T)
        t = end - start

        masks_chunk = raw_masks[start:end].to(torch.float32).to(device).unsqueeze(1)
        frames_chunk = raw_frames[start:end].to(torch.float32).to(device).unsqueeze(1)
        data_chunk = torch.cat([masks_chunk, frames_chunk], dim=1)  # t x 2 x H x W

        grid_y = ys.view(1, H, 1).expand(t, H, W)
        grid_x = xs.view(1, 1, W).expand(t, H, W)

        bm_chunk = all_bms[start:end].to(device)  # t x W

        if correct_dx:
            # Grab the pre-computed, correctly smoothed shifts for this chunk
            dx_chunk = global_dx[start:end]

            sample_x = grid_x - dx_chunk.view(t, 1, 1)
            norm_x = sample_x / (W - 1) * 2 - 1
            norm_y = grid_y / (H - 1) * 2 - 1
            grid_x_only = torch.stack([norm_x, norm_y], dim=-1)
            data_x_aligned = F.grid_sample(
                data_chunk,
                grid_x_only,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            bm_aligned, csi_aligned = extract_boundaries_fast(
                data_x_aligned[:, 0].cpu().numpy()
            )
            bm_aligned, csi_aligned = clean_boundaries(bm_aligned, csi_aligned)
            bm_aligned = torch.from_numpy(bm_aligned).to(device)
            bm_aligned = bm_aligned.to(device)
        else:
            data_x_aligned = data_chunk
            bm_aligned = bm_chunk
        displacement = bm_aligned - ref_bm.unsqueeze(0)  # t x W

        if flatten_rpe:
            displacement = bm_aligned - ref_bm.mean()  # t x W

        # Replace NaN to zeros in the displacement
        displacement = torch.nan_to_num(displacement, nan=0.0)
        if return_params:
            displacement_chunks.append(displacement.detach().cpu())
        sample_y = grid_y + displacement.view(t, 1, -1)
        norm_y2 = sample_y / (H - 1) * 2 - 1
        norm_x2 = grid_x / (W - 1) * 2 - 1
        grid_y_only = torch.stack([norm_x2, norm_y2], dim=-1)

        reg_data_chunk = F.grid_sample(
            data_x_aligned,
            grid_y_only,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        registered_masks_chunks.append(reg_data_chunk[:, 0].to(mask_dtype).cpu())
        registered_frames_chunks.append(reg_data_chunk[:, 1].to(frame_dtype).cpu())

        del masks_chunk, frames_chunk, data_x_aligned
        del bm_chunk, bm_aligned, displacement
        if correct_dx:
            del dx_chunk
        torch.cuda.empty_cache()

    registered_masks_out = torch.cat(registered_masks_chunks, dim=0)
    registered_frames_out = torch.cat(registered_frames_chunks, dim=0)

    if return_params:
        if correct_dx:
            dx_out = global_dx.detach().cpu().to(torch.float32)
        else:
            dx_out = torch.zeros(T, dtype=torch.float32)
        dy_out = torch.cat(displacement_chunks, dim=0)  # T x W
        params = {"dx": dx_out, "dy": dy_out}
        return registered_masks_out, registered_frames_out, params

    return registered_masks_out, registered_frames_out
