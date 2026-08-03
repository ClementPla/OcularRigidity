import torch
import torch.nn.functional as F
from ocularrigidity.registration.lateral.correlation import (
    frame_correlation_dx,
    profile_correlation_dx,
)
from ocularrigidity.registration.layout import restore_layout, to_bchw, to_gray
from ocularrigidity.registration.lateral.utils import (
    robust_temporal_dx,
    smooth_translations,
)
from ocularrigidity.segmentation.fovea.from_ilm import (
    estimate_fovea,
)
from tqdm.auto import tqdm


def _vertical_mean_profiles(raw_frames, *, batch_size, device):
    """Blurred vertical-mean intensity profile ``(T, W)`` per frame (xcorr input)."""
    profiles = []
    for start in range(0, len(raw_frames), batch_size):
        chunk = raw_frames[start : start + batch_size].to(device, torch.float32)
        blurred = F.avg_pool2d(
            F.pad(chunk.unsqueeze(1), (2, 2, 2, 2), mode="replicate"),
            kernel_size=5,
            stride=1,
        )
        profiles.append(blurred.squeeze(1).mean(dim=1))
    return torch.cat(profiles, dim=0)


def estimate_lateral_dx(
    raw_frames,
    ref_idx,
    lateral_method,
    smooth_transversal,
    smooth_transversal_sigma,
    subpixel,
    max_shift,
    batch_size,
    device,
    scale_factor=1.0,
    crop_factor=0.66,
    bandpass=(0.02, 0.5),
):
    """Per-frame lateral shift ``dx`` (T,) aligning each frame onto ``ref_idx``.

    ``"fullframe"`` uses 2D phase correlation; ``"xcorr"`` cross-correlates the
    vertical-mean profiles. Temporal outliers are rejected and the trace smoothed;
    it is rounded to whole pixels unless ``subpixel``.

    Colour frames are reduced to luminance: a lateral shift is a property of the
    scene, so estimating it per channel would only add noise.
    """
    raw_frames = to_gray(raw_frames)
    if lateral_method == "both":
        dx_fullframe = estimate_lateral_dx(
            raw_frames,
            ref_idx,
            "fullframe",
            subpixel=subpixel,
            max_shift=max_shift,
            batch_size=batch_size,
            device=device,
            smooth_transversal=smooth_transversal,
            smooth_transversal_sigma=smooth_transversal_sigma,
            scale_factor=scale_factor,
            crop_factor=crop_factor,
            bandpass=bandpass,
        )
        dx_xcorr = estimate_lateral_dx(
            raw_frames,
            ref_idx,
            "xcorr",
            subpixel=subpixel,
            max_shift=max_shift,
            batch_size=batch_size,
            device=device,
            smooth_transversal=smooth_transversal,
            smooth_transversal_sigma=smooth_transversal_sigma,
            scale_factor=scale_factor,
            crop_factor=crop_factor,
            bandpass=bandpass,
        )
        dx = (dx_fullframe + dx_xcorr) / 2
    elif lateral_method == "fullframe":
        dx, conf = frame_correlation_dx(
            raw_frames,
            ref=raw_frames[ref_idx],
            batch_size=batch_size,
            device=device,
            return_confidence=True,
            subpixel=subpixel,
            max_shift=max_shift,
            scale_factor=scale_factor,
            crop_factor=crop_factor,
            bandpass=bandpass,
        )
        dx = robust_temporal_dx(dx, conf=conf)
    elif lateral_method == "xcorr":
        profiles = _vertical_mean_profiles(
            raw_frames, batch_size=batch_size, device=device
        )
        dx = profile_correlation_dx(
            profiles, profiles[ref_idx], max_shift, drop_edges=100, subpixel=subpixel
        ).to(device)
        dx = robust_temporal_dx(dx)
    else:
        raise ValueError(f"Unknown lateral_method: {lateral_method!r}")

    if smooth_transversal:
        dx = smooth_translations(dx, sigma=smooth_transversal_sigma)
    return dx if subpixel else dx.round()


def fovea_correction(raw_frames, raw_masks, ref_idx, batch_size, device, verbose):
    """Estimate the fovea location from the ILM and shift the frames/masks accordingly.

    ``raw_frames`` may be gray or colour, in any layout; the fovea is located on
    the luminance, and the frames come back in the layout they arrived in.
    """
    raw_frames, layout = to_bchw(torch.as_tensor(raw_frames))
    fovea_locations = estimate_fovea(to_gray(raw_frames), raw_masks)
    ref_fovea_x = fovea_locations[ref_idx][None, 0]
    ref_fovea_y = fovea_locations[ref_idx][None, 1]
    dx_fovea = ref_fovea_x - fovea_locations[:, 0]
    dx_fovea = torch.as_tensor(dx_fovea, device=device, dtype=torch.float32)
    dy_fovea = torch.as_tensor(
        ref_fovea_y - fovea_locations[:, 1], device=device, dtype=torch.float32
    )
    T, H, W = raw_masks.shape
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    registered_masks_chunks = []
    registered_frames_chunks = []
    raw_masks = torch.as_tensor(raw_masks)
    mask_dtype, frame_dtype = raw_masks.dtype, raw_frames.dtype

    for start in tqdm(
        range(0, T, batch_size),
        desc="Aligning foveas",
        disable=not verbose,
        leave=False,
    ):
        end = min(start + batch_size, T)
        t = end - start

        masks_chunk = raw_masks[start:end].to(device, torch.float32).unsqueeze(1)
        frames_chunk = raw_frames[start:end].to(device, torch.float32)
        # Mask as channel 0, so it takes the same warp: t x (1 + C) x H x W.
        data = torch.cat([masks_chunk, frames_chunk], dim=1)

        grid_y = ys.view(1, H, 1).expand(t, H, W)
        grid_x = xs.view(1, 1, W).expand(t, H, W)

        # Lateral shift, then re-read the BM from the x-aligned mask so the
        # vertical displacement is measured in the already-shifted frame.
        dx = dx_fovea[start:end].view(t, 1, 1)
        dy = dy_fovea[start:end].view(t, 1, 1)
        norm_x = (grid_x - dx) / (W - 1) * 2 - 1
        norm_y = (grid_y - dy) / (H - 1) * 2 - 1
        data = F.grid_sample(
            data,
            torch.stack([norm_x, norm_y], dim=-1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        registered_masks_chunks.append(data[:, 0].to(mask_dtype).cpu())
        registered_frames_chunks.append(data[:, 1:].to(frame_dtype).cpu())
    return (
        torch.cat(registered_masks_chunks, dim=0),
        restore_layout(torch.cat(registered_frames_chunks, dim=0), layout),
    )
