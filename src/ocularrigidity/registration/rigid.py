import torch
import torch.nn.functional as F
import numpy as np
from ocularrigidity.segmentation.postprocess.smoothing import extract_boundaries_gpu
from tqdm.auto import tqdm


def smooth_translations(dx: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Applies a 1D Gaussian filter to smooth out high-frequency jitter in dx."""
    if sigma <= 0:
        return dx

    radius = int(4 * sigma + 0.5)

    # Create 1D Gaussian kernel
    x = torch.arange(-radius, radius + 1, device=dx.device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(
        1, 1, -1
    )  # Formatted for conv1d: (out_channels, in_channels, width)

    # Pad dx to handle edges cleanly
    dx_padded = F.pad(dx.view(1, 1, -1), (radius, radius), mode="replicate")

    # Convolve
    dx_smoothed = F.conv1d(dx_padded, kernel)
    return dx_smoothed.view(-1)


def estimate_lateral_shift_xcorr_subpixel(
    curve: torch.Tensor,
    ref_curve: torch.Tensor,
    max_shift: int = None,
    drop_edges: int = 75,
) -> torch.Tensor:
    """
    curve: T x W, ref_curve: W
    Returns dx: T, the sub-pixel lateral shift that best aligns
    curve[t] to ref_curve via 1D cross-correlation and parabolic interpolation.
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

    if valid_mask.any():
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


def estimate_lateral_shift_phase_correlation(
    curve: torch.Tensor, ref_curve: torch.Tensor, max_shift: int = 75
) -> torch.Tensor:
    """
    Computes sub-pixel lateral shift using 1D Phase Correlation via FFT.
    curve: T x W, ref_curve: W
    """
    T, W = curve.shape

    # 1. Compute FFTs
    F_curve = torch.fft.fft(curve, dim=-1)
    F_ref = torch.fft.fft(ref_curve, dim=-1).unsqueeze(0)  # 1 x W

    # 2. Compute Cross-Power Spectrum
    cross_power = F_curve * torch.conj(F_ref)
    normalized_cross_power = cross_power / (torch.abs(cross_power) + 1e-8)

    # 3. Inverse FFT to get the phase correlation peak
    peak_profile = torch.fft.ifft(normalized_cross_power, dim=-1).real

    # Shift the zero-frequency component to the center (so center is 0 shift)
    peak_profile = torch.fft.fftshift(peak_profile, dim=-1)

    # The center of the shifted array corresponds to 0 displacement
    center_idx = W // 2

    # Search for peak within the allowed max_shift window
    search_start = center_idx - max_shift
    search_end = center_idx + max_shift + 1
    windowed_peaks = peak_profile[:, search_start:search_end]

    # Find integer peak location
    best_idx_window = windowed_peaks.argmax(dim=1)
    best_idx_global = search_start + best_idx_window

    # Convert back to real shifts (-max_shift to +max_shift)
    dx = (best_idx_global - center_idx).float()

    # 4. Optional: Sub-pixel refinement via Centroid/Center-of-Mass around peak
    # (Much more stable in phase correlation than parabolic fitting)
    for t in range(T):
        idx = best_idx_global[t]
        if 0 < idx < W - 1:
            # Take a 3-pixel neighborhood
            weights = peak_profile[t, idx - 1 : idx + 2]
            weights = torch.clamp(weights, min=0)  # Ensure positive weights
            if weights.sum() > 0:
                local_centroid = (
                    weights[0] * -1 + weights[1] * 0 + weights[2] * 1
                ) / weights.sum()
                dx[t] += local_centroid

    return dx


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
):
    if isinstance(raw_masks, np.ndarray):
        raw_masks = torch.from_numpy(raw_masks)
    if isinstance(raw_frames, np.ndarray):
        raw_frames = torch.from_numpy(raw_frames)
    T, H, W = raw_masks.shape
    mask_dtype = raw_masks.dtype
    frame_dtype = raw_frames.dtype

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    bms_list = []
    csi_list = []
    profile_chunks = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        masks_chunk = raw_masks[start:end].to(torch.float32)
        frames_chunk = raw_frames[start:end].to(torch.float32).cuda()
        bm, csi = extract_boundaries_gpu(masks_chunk, to_numpy=False)

        padded_frames = F.pad(frames_chunk.unsqueeze(1), (2, 2, 2, 2), mode="replicate")
        blurred_frames = F.avg_pool2d(
            padded_frames, kernel_size=5, stride=1
        )  # Fast blur alternative

        profile_chunk = blurred_frames.squeeze(1).mean(dim=1)
        bms_list.append(bm)
        csi_list.append(csi)
        profile_chunks.append(profile_chunk)

    profiles = torch.cat(profile_chunks, dim=0).cuda()
    all_bms = torch.cat(bms_list, dim=0).cuda()
    ref_bm = all_bms[0]
    if correct_dx:
        global_dx = estimate_lateral_shift_xcorr_subpixel(
            profiles,
            profiles[0],
            101,
            drop_edges=100,
        ).to(device)  # T
        global_dx = smooth_translations(global_dx, sigma=2.0)

    registered_masks_chunks = []
    registered_frames_chunks = []

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

            bm_aligned, _ = extract_boundaries_gpu(
                data_x_aligned[:, 0], to_numpy=False
            )  # t x W
            bm_aligned = bm_aligned.to(device)
        else:
            data_x_aligned = data_chunk
            bm_aligned = bm_chunk
        displacement = bm_aligned - ref_bm.unsqueeze(0)  # t x W

        displacement = bm_aligned - ref_bm.unsqueeze(0)

        if flatten_rpe:
            displacement = bm_aligned - ref_bm.mean()  # t x W
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

        # Cleanup chunk memory
        del masks_chunk, frames_chunk, data_x_aligned
        del bm_chunk, bm_aligned, displacement
        if correct_dx:
            del dx_chunk
        torch.cuda.empty_cache()

    return torch.cat(registered_masks_chunks, dim=0), torch.cat(
        registered_frames_chunks, dim=0
    )
