from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

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
    """Everything one video's fused pass produced."""

    registered_frames: np.ndarray  # (T, H, W) uint8
    transform: dict  # {"dx": (T,), "dy": (T, W)} float32
    ref_idx: int
    timings: dict
    raw_masks: np.ndarray | None = None  # (T, H, W) bool, unregistered
    registered_masks: np.ndarray | None = None  # (T, H, W) bool
    ref_percentile: float | None = (
        None  # of the reference's area in the full distribution
    )


def _normalise(frames_u8: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` uint8 -> ``(B, 1, H, W)`` float, the encoder's scaling."""
    return frames_u8.unsqueeze(1).float().div_(255.0).sub_(0.5).div_(0.5)


@torch.inference_mode()
def _encode_segment(
    seg_model,
    batch_u8: torch.Tensor,
    amp_dtype: torch.dtype,
    keep_largest_cc: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One encode, two outputs: the mask and the pyramid the regressor wants."""
    with torch.autocast("cuda", dtype=amp_dtype):
        feats = seg_model.model.encoder(_normalise(batch_u8))
        logits = seg_model.model.segmentation_head(seg_model.model.decoder(feats))
    if keep_largest_cc:
        mask = keep_largest_connected_component_gpu(logits.squeeze(1) > 0.0)
    else:
        mask = logits.squeeze(1) > 0.0
    # feats[0] and feats[1] are the stride-1 input and an empty stride-2 stage
    return mask, list(feats[2:])


@torch.inference_mode()
def _encode(
    seg_model, batch_u8: torch.Tensor, amp_dtype: torch.dtype
) -> list[torch.Tensor]:
    with torch.autocast("cuda", dtype=amp_dtype):
        return list(seg_model.model.encoder(_normalise(batch_u8))[2:])


def _trajectory_medoid(
    reg_model,
    pyramid: list[torch.Tensor],
    pivot: int,
    batch: int,
    amp_dtype: torch.dtype,
    img_shape: tuple[int, int],
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
            _, b_dy = reg_model(fixed, moving, img_shape=img_shape)
        traj[s:e] = b_dy.float().mean(dim=1).cpu()
    return int((traj - traj.median()).abs().argmin().item())


def _upper_interface(masks: torch.Tensor) -> torch.Tensor:
    """Row of the first True of each column of ``(B, H, W)`` masks, NaN if none."""
    top = torch.argmax(masks.to(torch.uint8), dim=1).float()
    return torch.where(masks.any(dim=1), top, torch.full_like(top, float("nan")))


def _smooth_along_x(r: torch.Tensor, sigma: float) -> torch.Tensor:
    """NaN-aware median then Gaussian of ``r (B, W)`` along x"""
    if sigma <= 0:
        return r
    k = int(sigma)
    if k > 0:
        padded = F.pad(r[:, None], (k, k), value=float("nan"))[:, 0]
        r = padded.unfold(1, 2 * k + 1, 1).nanmedian(dim=-1).values
    half = int(3 * sigma) + 1
    x = torch.arange(-half, half + 1, device=r.device, dtype=r.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)[None, None]
    valid = torch.isfinite(r)
    num = F.conv1d(torch.where(valid, r, torch.zeros_like(r))[:, None], kernel, padding=half)
    den = F.conv1d(valid.to(r.dtype)[:, None], kernel, padding=half)
    return torch.where(den > 1e-3, num / den.clamp_min(1e-3), torch.full_like(num, float("nan")))[:, 0]


def _bm_alignment(
    masks: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    ref_idx: int,
    shape: tuple[int, int],
    batch: int,
    sigma: float,
) -> torch.Tensor:
    """What to add to ``dy (T, W)`` so every frame's BM lands on the reference's."""
    device = masks.device

    def registered_bm(s: int, e: int) -> torch.Tensor:
        out, _ = warp(masks[s:e, None].float(), dx[s:e].to(device), dy[s:e].to(device), shape)
        return _upper_interface(out[:, 0] > 0.5)

    reference = registered_bm(ref_idx, ref_idx + 1)
    correction = torch.empty_like(dy)
    for s in range(0, masks.shape[0], batch):
        e = min(s + batch, masks.shape[0])
        r = _smooth_along_x(registered_bm(s, e) - reference, sigma)
        fill = torch.nan_to_num(r.nanmedian(dim=1, keepdim=True).values, nan=0.0)
        correction[s:e] = torch.where(torch.isfinite(r), r, fill).cpu()
    return correction


def _resolve_ref_idx(
    ref_idx: int | None, config: RegistrationConfig, T: int
) -> int | None:
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
    """Segment and register one already-trimmed volume in a single encode pass."""
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
            ref_local = _trajectory_medoid(
                reg_model, probe_pyramid, pivot_local, batch, amp_dtype, (H, W)
            )

        ref_idx = int(probe_idx[ref_local])
        ref_feats = [f[ref_local : ref_local + 1].clone() for f in probe_pyramid]
        probe_pyramid = None
        timings["probe"] = _tick() - t0

    # --- main pass: one encode per frame, feeding both heads ---------------
    t0 = _tick()
    dx = torch.empty(T, dtype=torch.float32)
    dy = torch.empty(T, W, dtype=torch.float32)
    areas = torch.empty(T, dtype=torch.float64)
    for s in tqdm(
        range(0, T, batch),
        desc="segmenting + registering",
        disable=not verbose,
        leave=False,
    ):
        e = min(s + batch, T)
        mask, feats = _encode_segment(
            seg_model, d_frames[s:e], amp_dtype, keep_largest_cc=config.keep_largest_cc
        )
        d_masks[s:e] = mask
        areas[s:e] = mask.sum(dim=(1, 2)).double().cpu()
        fixed = [f.expand(e - s, -1, -1, -1) for f in ref_feats]
        with torch.autocast("cuda", dtype=amp_dtype):
            b_dx, b_dy = reg_model(  # in pixels
                fixed, feats, img_shape=(H, W), bulk_only=config.dy_bulk_only
            )
        dx[s:e], dy[s:e] = b_dx.float().cpu(), b_dy.float().cpu()
    timings["encode+segment+register"] = _tick() - t0

    if config.dy_align_bm:
        t0 = _tick()
        dy = dy + _bm_alignment(
            d_masks, dx, dy, ref_idx, (H, W), batch, config.dy_align_bm_sigma
        )
        timings["align-bm"] = _tick() - t0

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

    # Blank the columns whose BM is unreliable, in frames and masks alike
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
    """Register one already-trimmed volume without segmenting it."""
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
            reg_model, probe_pyramid, n_probe // 2, batch, amp_dtype, (H, W)
        )
        ref_idx = int(probe_idx[ref_local])
        ref_feats = [f[ref_local : ref_local + 1].clone() for f in probe_pyramid]
        del probe_pyramid
    timings["reference"] = _tick() - t0

    t0 = _tick()
    dx = torch.empty(T, dtype=torch.float32)
    dy = torch.empty(T, W, dtype=torch.float32)
    registered_frames = np.empty((T, H, W), dtype=np.uint8)
    for s in tqdm(range(0, T, batch), desc="registering frames", disable=not verbose, leave=False):
        e = min(s + batch, T)
        feats = _encode(seg_model, d_frames[s:e], amp_dtype)
        fixed = [f.expand(e - s, -1, -1, -1) for f in ref_feats]
        with torch.autocast("cuda", dtype=amp_dtype):
            b_dx, b_dy = reg_model(
                fixed, feats, img_shape=(H, W), bulk_only=config.dy_bulk_only
            )
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
