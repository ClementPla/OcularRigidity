import torch
import torch.nn.functional as F
from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.lateral.dx import estimate_lateral_dx, fovea_correction

from ocularrigidity.registration.axial.median_registration import (
    register_ascans_to_median,
)

from ocularrigidity.registration.layout import restore_layout, to_bchw, to_gray
from ocularrigidity.registration.postprocess import filter_bad_ascans_per_bms
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)
from tqdm.auto import tqdm


def _all_bm_boundaries(raw_masks, *, batch_size, device):
    """Cleaned BM boundary ``(T, W)`` of every frame, on ``device``."""
    bms = []
    for start in range(0, len(raw_masks), batch_size):
        chunk = raw_masks[start : start + batch_size].to(torch.float32).cpu().numpy()
        bm, _ = clean_boundaries(*extract_boundaries_fast(chunk))
        bms.append(torch.from_numpy(bm).to(device))
    return torch.cat(bms, dim=0)


def register_videos(
    raw_masks: torch.Tensor,
    raw_frames: torch.Tensor,
    config: RegistrationConfig | None = None,
    *,
    device: str = "cuda",
    verbose: bool = True,
    return_params: bool = False,
):
    """Register a video by lateral (x) shift, then per-column vertical (y) BM alignment.

    ``raw_frames`` is either gray ``(T, H, W)`` or colour ``(T, H, W, 3)``, and
    comes back in the layout it went in. Colour channels are warped together (the
    shifts are estimated once, on the luminance), so they stay in register.

    Returns ``(registered_masks, registered_frames)``, plus a ``params`` dict
    ``{"dx": (T,), "dy": (T, W), "bad_columns": (W,)}`` when ``return_params``.

    - ``lateral_method``: ``"xcorr"`` (vertical-mean profiles) or ``"fullframe``
      (2D phase correlation); ``dx`` is all-zeros when ``correct_transversal`` is False.
    - ``flatten_rpe``: align the BM to a constant row rather than to the reference
      frame's BM curve.
    - ``axial_refinement``: optional second axial pass aligning each A-scan on
      the volume's temporal median (adds ``"dy_median"`` (T, W) to ``params``).

    Columns whose BM is unreliable (``filter_bad_ascans_per_bms``) are zeroed in
    both frames and masks, across all frames.
    """
    with torch.inference_mode(), torch.no_grad():
        if config is None:
            config = RegistrationConfig()
        # Unpack once here so the body (and the call sites) stay config-driven.
        correct_transversal = config.correct_transversal
        correct_axial = config.correct_axial
        flatten_rpe = config.flatten_rpe
        axial_refinement = config.axial_refinement
        fovea_correction_enabled = config.fovea_correction_enabled
        transversal_bandpass = config.transversal_bandpass
        axial_bandpass = config.axial_bandpass
        lateral_method = config.lateral_method
        max_lateral_shift = config.max_lateral_shift
        smooth_transversal = config.smooth_transversal
        smooth_transversal_sigma = config.smooth_transversal_sigma
        max_axial_shift = config.max_axial_shift
        subpixel = config.subpixel
        scale_factor = config.scale_factor
        crop_factor = config.crop_factor
        batch_size = config.batch_size

        raw_masks = torch.as_tensor(raw_masks)
        # Work channels-first throughout (a gray video is just C = 1) and restore
        # the caller's layout on the way out.
        raw_frames, layout = to_bchw(torch.as_tensor(raw_frames))

        T, H, W = raw_masks.shape
        mask_dtype, frame_dtype = raw_masks.dtype, raw_frames.dtype

        # Reference frame: the one whose mask area is closest to the temporal median.
        mask_counts = raw_masks.sum(dim=(1, 2))
        ref_idx = (mask_counts - mask_counts.median()).abs().argmin()
        if fovea_correction_enabled:
            raw_masks, raw_frames = fovea_correction(
                raw_frames,
                raw_masks,
                ref_idx=ref_idx,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
        if (not correct_axial) and (not correct_transversal):
            params = {
                "dx": torch.zeros(T, dtype=torch.float32),
                "dy": torch.zeros(T, W, dtype=torch.float32),
                "bad_columns": torch.zeros(W, dtype=torch.bool),
            }
            untouched = restore_layout(raw_frames, layout)
            if return_params:
                return raw_masks, untouched, params
            return raw_masks, untouched
        all_bms = _all_bm_boundaries(raw_masks, batch_size=batch_size, device=device)
        ref_bm = all_bms[ref_idx]

        # --- Lateral (x) registration: estimated once, decoupled from the y warp ---
        global_dx = (
            estimate_lateral_dx(
                to_gray(raw_frames),  # the shift is measured on the luminance
                ref_idx,
                lateral_method,
                subpixel=subpixel,
                max_shift=max_lateral_shift,
                batch_size=batch_size,
                device=device,
                smooth_transversal=smooth_transversal,
                smooth_transversal_sigma=smooth_transversal_sigma,
                scale_factor=scale_factor,
                crop_factor=crop_factor,
                bandpass=transversal_bandpass,
            )
            if correct_transversal
            else None
        )

        # --- Vertical (y) registration onto the reference BM ----------------------
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        # BM level every column is warped onto: a scalar (flatten) or the ref curve.
        target_bm = torch.nanmean(ref_bm) if flatten_rpe else ref_bm.unsqueeze(0)

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

            masks_chunk = raw_masks[start:end].to(device, torch.float32).unsqueeze(1)
            frames_chunk = raw_frames[start:end].to(device, torch.float32)
            # Mask rides along as channel 0, so it takes the exact same warp:
            # t x (1 + C) x H x W — 2 channels for gray, 4 for colour.
            data = torch.cat([masks_chunk, frames_chunk], dim=1)

            grid_y = ys.view(1, H, 1).expand(t, H, W)
            grid_x = xs.view(1, 1, W).expand(t, H, W)

            # Lateral (x) shift: warp both channels when correct_transversal.
            if correct_transversal:
                dx = global_dx[start:end].view(t, 1, 1)
                norm_x = (grid_x - dx) / (W - 1) * 2 - 1
                norm_y = grid_y / (H - 1) * 2 - 1
                data = F.grid_sample(
                    data,
                    torch.stack([norm_x, norm_y], dim=-1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )

            # Per-column vertical displacement onto the target BM level, when correct_axial.
            if correct_axial:
                # Re-read the BM from the (possibly x-aligned) mask so the vertical
                # displacement is measured in the already-shifted frame.
                if correct_transversal:
                    bm, _ = clean_boundaries(
                        *extract_boundaries_fast(data[:, 0].cpu().numpy())
                    )
                    bm_aligned = torch.from_numpy(bm).to(device)
                else:
                    bm_aligned = all_bms[start:end]
                displacement = torch.nan_to_num(bm_aligned - target_bm, nan=0.0)
                if not subpixel:
                    displacement = displacement.round()
            else:
                displacement = torch.zeros(t, W, device=device)  # x-only: no y warp
            if return_params:
                displacement_chunks.append(displacement.cpu())

            norm_x = grid_x / (W - 1) * 2 - 1
            norm_y = (grid_y + displacement.view(t, 1, W)) / (H - 1) * 2 - 1
            reg = F.grid_sample(
                data,
                torch.stack([norm_x, norm_y], dim=-1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            registered_masks_chunks.append(reg[:, 0].to(mask_dtype).cpu())
            registered_frames_chunks.append(
                reg[:, 1:].to(frame_dtype).cpu()
            )  # (t, C, H, W)

        registered_masks = torch.cat(registered_masks_chunks, dim=0)
        registered_frames = torch.cat(registered_frames_chunks, dim=0)

        params = None
        if return_params:
            dx_out = (
                global_dx.detach().cpu().float()
                if correct_transversal
                else torch.zeros(T, dtype=torch.float32)
            )
            params = {"dx": dx_out, "dy": torch.cat(displacement_chunks, dim=0)}

        # Optional second pass: axial A-scan alignment on the temporal median (RPE).
        if axial_refinement:
            registered_frames, registered_masks, dy_median = register_ascans_to_median(
                registered_frames,
                registered_masks,
                max_vshift=max_axial_shift,
                subpixel=subpixel,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
                bandpass=axial_bandpass,
            )
            if params is not None:
                params["dy_median"] = dy_median

        # Blank A-scan columns whose BM is unreliable, in both frames and masks.
        # Columns are the last axis of both (frames are still channels-first), so
        # the same indexing covers gray and colour.
        bad_cols = filter_bad_ascans_per_bms(registered_masks)
        if bool(bad_cols.any()):
            registered_frames[..., bad_cols] = 0
            registered_masks[..., bad_cols] = 0
        if params is not None:
            params["bad_columns"] = bad_cols

        registered_frames = restore_layout(registered_frames, layout)
        if return_params:
            return registered_masks, registered_frames, params
        return registered_masks, registered_frames
