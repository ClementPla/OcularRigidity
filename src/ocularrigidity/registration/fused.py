"""Segment and register a volume in a single pass over the GPU.

Run as separate stages, segmentation and registration each encode every frame,
and the encoder dominates the per-frame cost. One encode feeding both heads is
therefore close to a halving, and is why this module exists.
``model.encoder(x)[2:]`` is bit-identical to ``encoder.forward_features(x)``
and ``segmentation_head(decoder(encoder(x)))`` reproduces ``model(x)``, so the
split costs no accuracy — both are asserted by the tests.

Choosing the *reference frame* is the part that does not fall out for free: it
must be fixed before the pass begins, by a rule needing mask areas the pass has
not produced yet, and holding every frame's pyramid instead would be tens of GB
per volume. A short probe pass over ``config.probe_frames`` frames supplies
them; the chosen frame's percentile against the *full* area distribution is
reported afterwards, so an under-sized probe is visible rather than silent.

Median area alone is not enough — a frame at the median area can sit at an
extreme of the eye's axial drift, and registering onto it pushes far-drift
frames off the canvas as all-zero warps. Area only picks a *pivot*; the
reference is the probe frame at the median of the axial trajectory measured
from it (``reference_selection="motion_medoid"``), which minimises the largest
warp in the volume.

    probe   N frames  ->  encode -> mask -> area        -> reference frame
    main    all frames ->  encode -> mask                       (segmentation)
                                  \\-> regressor(ref, .) -> dx, dy  (registration)
    warp    all frames ->  grid_sample(mask+frame by dx, dy)

Only the warp revisits the frames, and it is a grid_sample, not an encode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.deep_learning.models.losses import warp
from ocularrigidity.registration.postprocess import filter_bad_ascans_per_bms
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component_gpu,
)

__all__ = ["FusedResult", "segment_and_register", "register"]

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Everything one video's fused pass produced.

    ``raw_masks`` is the segmentation before any warp — the artifact the
    standalone segmentation stage used to write, kept because it is what lets a
    later run resume at the registration without segmenting again.
    """

    registered_frames: np.ndarray  # (T, H, W) uint8
    transform: dict  # {"dx": (T,), "dy": (T, W)} float32
    ref_idx: int
    timings: dict
    raw_masks: np.ndarray | None = None  # (T, H, W) bool, unregistered
    registered_masks: np.ndarray | None = None  # (T, H, W) bool
    ref_percentile: float | None = None  # of the reference's area in the full distribution


def _normalise(frames_u8: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` uint8 -> ``(B, 1, H, W)`` float, the encoder's scaling.

    The same ``(x/255 - 0.5) / 0.5`` that ``prepare_data.py`` wrote the training
    crops with and that ``segmentation.inference.infer`` applies; the regressor
    was trained on features from exactly this input, so it is not negotiable.
    """
    return frames_u8.unsqueeze(1).float().div_(255.0).sub_(0.5).div_(0.5)


@torch.inference_mode()
def _encode_segment(
    seg_model,
    batch_u8: torch.Tensor,
    amp_dtype: torch.dtype,
    keep_largest_cc: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One encode, two outputs: the mask and the pyramid the regressor wants.

    Returns ``(mask (B, H, W) bool, pyramid list fine->coarse)``. The mask is
    cleaned on the device — cc3d would be a CPU round-trip per batch, and it
    holds the GIL, so it would stall the pass instead of overlapping with it.
    """
    with torch.autocast("cuda", dtype=amp_dtype):
        feats = seg_model.model.encoder(_normalise(batch_u8))
        logits = seg_model.model.segmentation_head(seg_model.model.decoder(feats))
    if keep_largest_cc:
        mask = keep_largest_connected_component_gpu(logits.squeeze(1) > 0.0)
    else:
        mask = logits.squeeze(1) > 0.0
    # feats[0] and feats[1] are the stride-1 input and an empty stride-2 stage;
    # the regressor's pyramid is the four real scales, i.e. forward_features().
    return mask, list(feats[2:])


@torch.inference_mode()
def _encode(seg_model, batch_u8: torch.Tensor, amp_dtype: torch.dtype) -> list[torch.Tensor]:
    with torch.autocast("cuda", dtype=amp_dtype):
        return list(seg_model.model.encoder(_normalise(batch_u8))[2:])


def _trajectory_medoid(
    reg_model, pyramid: list[torch.Tensor], pivot: int, batch: int, amp_dtype: torch.dtype
) -> int:
    n = pyramid[0].shape[0]
    if n <= 2:
        return pivot
    pivot_feats = [f[pivot : pivot + 1] for f in pyramid]
    traj = torch.empty(n, dtype=torch.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        moving = [f[s:e] for f in pyramid]
        fixed = [f.expand(e - s, -1, -1, -1) for f in pivot_feats]
        with torch.autocast("cuda", dtype=amp_dtype):
            _, b_dy = reg_model(fixed, moving)
        traj[s:e] = b_dy.float().mean(dim=1).cpu()
    return int((traj - traj.median()).abs().argmin().item())


def _resolve_ref_idx(ref_idx: int | None, config: RegistrationConfig, T: int) -> int | None:
    if ref_idx is None and isinstance(config.reference_selection, int):
        ref_idx = config.reference_selection
    return None if ref_idx is None else range(T)[ref_idx]


@torch.inference_mode()
def segment_and_register(
    frames: np.ndarray | torch.Tensor,
    seg_model,
    reg_model,
    config: RegistrationConfig,
    *,
    ref_idx: int | None = None,
    device: str = "cuda",
    verbose: bool = True,
) -> FusedResult:
    """Segment and register one already-trimmed volume in a single encode pass.

    ``frames`` is ``(T, H, W)`` uint8, with ``skip_first_n_frames`` /
    ``drop_last_n_frames`` already applied by the caller — the transform is
    indexed on the trimmed volume, exactly as the classical cache is.

    ``(dx, dy)`` come from the trained regressor alone. No temporal filtering
    is applied to them: the model is the estimator, and smoothing its output
    with the classical ``robust_temporal_dx`` would mix two estimators whose
    failure modes are unrelated.

    ``ref_idx`` (or an int ``config.reference_selection``) fixes the reference
    frame and skips the probe; negative indices count from the end.
    """
    import time

    timings: dict = {}

    def _tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(np.ascontiguousarray(frames))
    T, H, W = frames.shape
    batch = config.fused_batch_size
    amp_dtype = torch.bfloat16

    t0 = _tick()
    d_frames = frames.to(device, non_blocking=True)
    d_masks = torch.empty((T, H, W), dtype=torch.bool, device=device)
    timings["upload"] = _tick() - t0
    ref_idx = _resolve_ref_idx(ref_idx, config, T)
    if ref_idx is not None:
        ref_feats = _encode(seg_model, d_frames[ref_idx : ref_idx + 1], amp_dtype)
    else:
        # --- probe: choose the reference frame ---------------------------------
        t0 = _tick()
        n_probe = int(min(max(config.probe_frames, 1), T))
        probe_idx = np.linspace(0, T - 1, n_probe).round().astype(int)
        probe_area = torch.empty(n_probe, dtype=torch.float64, device=device)
        probe_feats: list = []
        for s in range(0, n_probe, batch):
            sel = probe_idx[s : s + batch]
            mask, feats = _encode_segment(seg_model, d_frames[sel], amp_dtype)
            probe_area[s : s + len(sel)] = mask.sum(dim=(1, 2)).double()
            probe_feats.append(feats)
            # The probe's *masks* are recomputed in the main pass rather than kept:
            # N is a percent or two of T, and stitching them in would complicate
            # the main loop for no measurable gain.
        # Concatenate into one pyramid over all probe frames: (N, C, h, w) per scale.
        probe_pyramid = [
            torch.cat([f[i] for f in probe_feats], dim=0)
            for i in range(len(probe_feats[0]))
        ]
        del probe_feats

        # Pivot: the area rule, applied to the subsample.
        pivot_local = int((probe_area - probe_area.median()).abs().argmin().item())
        ref_local = pivot_local

        if config.reference_selection == "motion_medoid":
            # Area alone is not enough on a subsample: a frame can sit at the median
            # area and still be at the extreme of the eye's axial drift, which makes
            # every warp in the volume larger than it needs to be. So measure dy
            # from the pivot to every probe frame and take the frame at the *median
            # of that trajectory* — the centre of the motion, not of the areas.
            ref_local = _trajectory_medoid(
                reg_model, probe_pyramid, pivot_local, batch, amp_dtype
            )

        ref_idx = int(probe_idx[ref_local])
        ref_feats = [f[ref_local : ref_local + 1].clone() for f in probe_pyramid]
        probe_pyramid = None  # ~1.5 GB; the reference was cloned out of it
        timings["probe"] = _tick() - t0

    # --- main pass: one encode per frame, feeding both heads ---------------
    t0 = _tick()
    dx = torch.empty(T, dtype=torch.float32)
    dy = torch.empty(T, W, dtype=torch.float32)
    # Areas are accumulated a batch at a time, never reduced over the whole
    # volume: `d_masks.sum(dim=(1, 2))` on a (T, 1536, 1024) bool tensor
    # promotes to int64 and asks for a 35 GB intermediate, which survives on an
    # idle card and OOMs the moment a second shard shares it.
    areas = torch.empty(T, dtype=torch.float64)
    for s in range(0, T, batch):
        e = min(s + batch, T)
        mask, feats = _encode_segment(
            seg_model, d_frames[s:e], amp_dtype, keep_largest_cc=config.keep_largest_cc
        )
        d_masks[s:e] = mask
        areas[s:e] = mask.sum(dim=(1, 2)).double().cpu()
        fixed = [f.expand(e - s, -1, -1, -1) for f in ref_feats]
        with torch.autocast("cuda", dtype=amp_dtype):
            b_dx, b_dy = reg_model(fixed, feats)  # already in pixels
        dx[s:e], dy[s:e] = b_dx.float().cpu(), b_dy.float().cpu()
    timings["encode+segment+register"] = _tick() - t0

    # The reference is picked from a subsample, so say where it actually landed
    # in the distribution it was estimating. 50 is the ideal; a value far from
    # it means `probe_frames` is too small for this volume.
    ref_percentile = float((areas < areas[ref_idx]).double().mean().item() * 100.0)

    # --- warp: mask rides as channel 0, so it takes the exact same transform -
    t0 = _tick()
    raw_masks = np.empty((T, H, W), dtype=bool)
    registered_masks = np.empty((T, H, W), dtype=bool)
    registered_frames = np.empty((T, H, W), dtype=np.uint8)
    for s in range(0, T, batch):
        e = min(s + batch, T)
        raw_masks[s:e] = d_masks[s:e].cpu().numpy()
        data = torch.stack([d_masks[s:e].float(), d_frames[s:e].float()], dim=1)
        out, _ = warp(data, dx[s:e].to(device), dy[s:e].to(device), (H, W))
        registered_masks[s:e] = (out[:, 0] > 0.5).cpu().numpy()
        registered_frames[s:e] = out[:, 1].clamp(0, 255).to(torch.uint8).cpu().numpy()
    timings["warp"] = _tick() - t0

    del d_frames, d_masks
    torch.cuda.empty_cache()

    # Blank the columns whose BM is unreliable, in frames and masks alike —
    # the same postprocess `register_videos` ends on, so the artifacts this
    # writes mean the same thing to every downstream stage.
    bad_cols = None
    if config.filter_bad_columns:
        bad_cols = filter_bad_ascans_per_bms(registered_masks)
        if bool(bad_cols.any()):
            cols = bad_cols.cpu().numpy()
            registered_frames[..., cols] = 0
            registered_masks[..., cols] = 0

    transform = {
        "dx": dx.numpy(),
        "dy": dy.numpy(),
        "bad_columns": (
            bad_cols.cpu().numpy() if bad_cols is not None else np.zeros(W, dtype=bool)
        ),
    }

    if verbose:
        stages = " | ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        logger.info(f"fused: T={T} ref={ref_idx} (p{ref_percentile:.0f}) | {stages}")

    return FusedResult(
        raw_masks=raw_masks,
        registered_masks=registered_masks,
        registered_frames=registered_frames,
        transform=transform,
        ref_idx=ref_idx,
        ref_percentile=ref_percentile,
        timings=timings,
    )


@torch.inference_mode()
def register(
    frames: np.ndarray | torch.Tensor,
    seg_model,
    reg_model,
    config: RegistrationConfig,
    *,
    ref_idx: int | None = None,
    device: str = "cuda",
    verbose: bool = True,
) -> FusedResult:
    """Register one already-trimmed volume without segmenting it.

    Without ``ref_idx`` the reference is the probe frame at the median of the
    axial trajectory, pivoted on the central probe frame (no masks, no area).
    """
    import time

    timings: dict = {}

    def _tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(np.ascontiguousarray(frames))
    T, H, W = frames.shape
    batch = config.fused_batch_size
    amp_dtype = torch.bfloat16

    t0 = _tick()
    d_frames = frames.to(device, non_blocking=True)
    timings["upload"] = _tick() - t0

    t0 = _tick()
    ref_idx = _resolve_ref_idx(ref_idx, config, T)
    if ref_idx is not None:
        ref_feats = _encode(seg_model, d_frames[ref_idx : ref_idx + 1], amp_dtype)
    else:
        n_probe = int(min(max(config.probe_frames, 1), T))
        probe_idx = np.linspace(0, T - 1, n_probe).round().astype(int)
        probe_pyramid = [
            torch.cat(scale, dim=0)
            for scale in zip(
                *(
                    _encode(seg_model, d_frames[probe_idx[s : s + batch]], amp_dtype)
                    for s in range(0, n_probe, batch)
                )
            )
        ]
        ref_local = _trajectory_medoid(
            reg_model, probe_pyramid, n_probe // 2, batch, amp_dtype
        )
        ref_idx = int(probe_idx[ref_local])
        ref_feats = [f[ref_local : ref_local + 1].clone() for f in probe_pyramid]
        del probe_pyramid
    timings["reference"] = _tick() - t0

    t0 = _tick()
    dx = torch.empty(T, dtype=torch.float32)
    dy = torch.empty(T, W, dtype=torch.float32)
    registered_frames = np.empty((T, H, W), dtype=np.uint8)
    for s in range(0, T, batch):
        e = min(s + batch, T)
        feats = _encode(seg_model, d_frames[s:e], amp_dtype)
        fixed = [f.expand(e - s, -1, -1, -1) for f in ref_feats]
        with torch.autocast("cuda", dtype=amp_dtype):
            b_dx, b_dy = reg_model(fixed, feats)
        b_dx, b_dy = b_dx.float(), b_dy.float()
        out, _ = warp(d_frames[s:e, None].float(), b_dx, b_dy, (H, W))
        registered_frames[s:e] = out[:, 0].clamp(0, 255).to(torch.uint8).cpu().numpy()
        dx[s:e], dy[s:e] = b_dx.cpu(), b_dy.cpu()
    timings["encode+register+warp"] = _tick() - t0

    del d_frames
    torch.cuda.empty_cache()

    transform = {
        "dx": dx.numpy(),
        "dy": dy.numpy(),
        "bad_columns": np.zeros(W, dtype=bool),
    }

    if verbose:
        stages = " | ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        logger.info(f"register: T={T} ref={ref_idx} | {stages}")

    return FusedResult(
        registered_frames=registered_frames,
        transform=transform,
        ref_idx=ref_idx,
        timings=timings,
    )
