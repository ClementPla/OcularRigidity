from warnings import warn


import torch
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm
from typing import Optional, Tuple

from ocularrigidity.registration.utils import _interp1d_batch_at

warn(
    "The apply module is deprecated and will likely be removed in a future release. ",
    category=DeprecationWarning,
)


@torch.inference_mode()
def apply_registration_torch(
    image: torch.Tensor,
    params: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    flatten_bms: Optional[torch.Tensor] = None,
    ref_idx_for_flatten: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.as_tensor(image)
    if isinstance(params, np.ndarray):
        params = torch.as_tensor(params)
    return _apply_registration_torch_streaming(
        image,
        params,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        flatten_bms=flatten_bms,
        ref_idx_for_flatten=ref_idx_for_flatten,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )


def _apply_registration_batch_torch(
    image_4d: torch.Tensor,
    dx: torch.Tensor,
    sx: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    flatten_curves: Optional[torch.Tensor] = None,
    y_target: Optional[float] = None,
) -> torch.Tensor:
    """Apply registration to one mini-batch already on the compute device.

    Matches apply_y_transform_cv2 semantics, including flatten mode and
    optional disabling of dx/sx in the horizontal inverse map.
    """
    B, _, H, W = image_4d.shape
    if (
        dx.shape[0] != B
        or sx.shape[0] != B
        or a.shape[0] != B
        or b.shape[0] != B
        or c.shape[0] != B
    ):
        raise ValueError("All parameter vectors must have batch size B")

    cx = (W - 1) * 0.5
    dx = dx.view(B, 1, 1)
    sx = sx.view(B, 1, 1)
    a = a.view(B, 1, 1)
    b = b.view(B, 1, 1)
    c = c.view(B, 1, 1)

    grid_dtype = a.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image_4d.device, dtype=grid_dtype),
        torch.arange(W, device=image_4d.device, dtype=grid_dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    x_s = (xx - cx - dx) / sx + cx

    if flatten_curves is None:
        y_s = (yy - b * x_s - c) / a
    else:
        if flatten_curves.shape != (B, W):
            raise ValueError(
                f"flatten_curves must have shape ({B}, {W}); got {tuple(flatten_curves.shape)}"
            )
        if y_target is None:
            y_target_t = (
                (
                    a.squeeze(-1).squeeze(-1) * flatten_curves
                    + b.squeeze(-1).squeeze(-1)
                    * torch.arange(
                        W, device=image_4d.device, dtype=grid_dtype
                    ).unsqueeze(0)
                    + c.squeeze(-1).squeeze(-1)
                )
                .mean(dim=1, keepdim=True)
                .view(B, 1, 1)
            )
        else:
            y_target_t = torch.tensor(
                y_target,
                device=image_4d.device,
                dtype=grid_dtype,
            ).view(1, 1, 1)
        fc_shifted = _interp1d_batch_at(flatten_curves, x_s)
        y_s = (yy - y_target_t) / a + fc_shifted

    if align_corners:
        x_norm = 2.0 * x_s / max(W - 1, 1) - 1.0
        y_norm = 2.0 * y_s / max(H - 1, 1) - 1.0
    else:
        x_norm = (2.0 * x_s + 1.0) / W - 1.0
        y_norm = (2.0 * y_s + 1.0) / H - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(
        image_4d,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def _apply_registration_torch_streaming(
    image: torch.Tensor,
    params: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    flatten_bms: Optional[torch.Tensor] = None,
    ref_idx_for_flatten: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Resample each frame so a source pixel (y, x) ends up at
    (a*y + b*x + c, sx*(x - cx) + cx + dx) in the output.

    Args:
        image: (T, H, W) or (T, C, H, W).
        params: (T, 5) as (dx, sx, a, b, c), or legacy (T, 4) as
            (dx, a, b, c), or (T, 3) as (a, b, c).
        flatten_bms: Optional (T, W) baseline curves for flatten mode.
        ref_idx_for_flatten: Reference index used to compute y_target.
        use_dx, use_sx: Toggle horizontal shift / scale in warp mapping.

    Returns:
        Resampled stack with the same shape as the input.
    """
    src_device = image.device
    if device is None:
        device = (
            image.device
            if image.is_cuda
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

    if image.dim() == 3:
        image_4d = image.unsqueeze(1)
        squeeze_back = True
    elif image.dim() == 4:
        image_4d = image
        squeeze_back = False
    else:
        raise ValueError(f"image must be 3D or 4D; got {image.dim()}D")

    input_dtype = image_4d.dtype
    compute_dtype = (
        image_4d.dtype if torch.is_floating_point(image_4d) else torch.float32
    )

    T, _, H, W = image_4d.shape
    dx_all, sx_all, a_all, b_all, c_all = _split_params_torch(
        params,
        T,
        device=src_device,
    )

    if (flatten_bms is None) != (ref_idx_for_flatten is None):
        raise ValueError(
            "Must provide both flatten_bms and ref_idx_for_flatten or neither"
        )
    flatten_all = None
    y_target = None
    if flatten_bms is not None:
        flatten_all = torch.as_tensor(
            flatten_bms,
            device=src_device,
            dtype=compute_dtype,
        )
        if flatten_all.shape != (T, W):
            raise ValueError(
                f"flatten_bms must have shape ({T}, {W}); got {tuple(flatten_all.shape)}"
            )
        if not (0 <= ref_idx_for_flatten < T):
            raise IndexError(
                f"ref_idx_for_flatten {ref_idx_for_flatten} out of range for T={T}"
            )
        y_target = float(flatten_all[ref_idx_for_flatten].mean().item())

    if batch_size is None:
        batch_size = T
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")

    out = torch.empty(
        (T, image_4d.shape[1], H, W), device=src_device, dtype=compute_dtype
    )
    for start in tqdm(
        range(0, T, batch_size),
        desc="Registering frames",
        disable=not verbose,
        leave=False,
    ):
        end = min(start + batch_size, T)
        image_batch = image_4d[start:end].to(device=device, dtype=compute_dtype)
        dx_batch = dx_all[start:end].to(device=image_batch.device, dtype=compute_dtype)
        sx_batch = sx_all[start:end].to(device=image_batch.device, dtype=compute_dtype)
        a_batch = a_all[start:end].to(device=image_batch.device, dtype=compute_dtype)
        b_batch = b_all[start:end].to(device=image_batch.device, dtype=compute_dtype)
        c_batch = c_all[start:end].to(device=image_batch.device, dtype=compute_dtype)
        flatten_batch = None
        if flatten_all is not None:
            flatten_batch = flatten_all[start:end].to(
                device=image_batch.device,
                dtype=compute_dtype,
            )
        out_batch = _apply_registration_batch_torch(
            image_batch,
            dx_batch,
            sx_batch,
            a_batch,
            b_batch,
            c_batch,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            flatten_curves=flatten_batch,
            y_target=y_target,
        )

        # Keep the full output on the same device as the input image.
        out[start:end] = out_batch.to(src_device)

    if input_dtype == torch.bool:
        out = out > 0.5
    elif out.dtype != input_dtype:
        out = out.to(input_dtype)
    return out.squeeze(1) if squeeze_back else out


def apply_registration_lines_torch(
    lines: torch.Tensor,
    params: torch.Tensor,
    *,
    flatten_bms: Optional[torch.Tensor] = None,
    ref_idx_for_flatten: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Register line y-coordinates instead of full images.

    Mirrors the warp used by ``apply_registration_torch`` but operates
    directly on curves. For each integer output column ``xx``, returns the
    y-value of the registered line.

    Args:
        lines: (T, W) for one line per frame, or (T, N, W) for N lines.
        params: (T, 3/4/5) as in ``apply_registration_torch``.
        flatten_bms: Optional (T, W) baselines enabling flatten mode.
        ref_idx_for_flatten: Reference frame for ``y_target`` in flatten mode.

    Returns:
        Registered lines with the same shape as ``lines``.
    """
    if isinstance(lines, np.ndarray):
        lines = torch.as_tensor(lines)
    if isinstance(params, np.ndarray):
        params = torch.as_tensor(params)

    squeeze_back = lines.dim() == 2
    if squeeze_back:
        lines = lines.unsqueeze(1)
    if lines.dim() != 3:
        raise ValueError(f"lines must be 2D (T, W) or 3D (T, N, W); got {lines.dim()}D")

    T, N, W = lines.shape
    src_device = lines.device
    if device is None:
        device = (
            src_device
            if src_device.type == "cuda"
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

    dtype = torch.float32
    lines_d = lines.to(device=device, dtype=dtype)
    dx, sx, a, b, c = _split_params_torch(params, T, device=device)

    if (flatten_bms is None) != (ref_idx_for_flatten is None):
        raise ValueError(
            "Must provide both flatten_bms and ref_idx_for_flatten or neither"
        )
    flatten_d = None
    y_target = None
    if flatten_bms is not None:
        flatten_d = torch.as_tensor(flatten_bms, device=device, dtype=dtype)
        if flatten_d.shape != (T, W):
            raise ValueError(
                f"flatten_bms must have shape ({T}, {W}); got {tuple(flatten_d.shape)}"
            )
        if not (0 <= ref_idx_for_flatten < T):
            raise IndexError(
                f"ref_idx_for_flatten {ref_idx_for_flatten} out of range for T={T}"
            )
        y_target = float(flatten_d[ref_idx_for_flatten].mean().item())

    cx = (W - 1) * 0.5
    xx = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)  # (1, W)
    x_s = (xx - cx - dx.view(T, 1)) / sx.view(T, 1) + cx  # (T, W)

    a_ = a.view(T, 1)
    out = torch.empty_like(lines_d)

    if flatten_d is None:
        affine = b.view(T, 1) * x_s + c.view(T, 1)  # (T, W)
        for n in range(N):
            line_at_xs = _interp1d_batch_at(lines_d[:, n, :], x_s)  # (T, W)
            out[:, n, :] = a_ * line_at_xs + affine
    else:
        flatten_at_xs = _interp1d_batch_at(flatten_d, x_s)  # (T, W)
        for n in range(N):
            line_at_xs = _interp1d_batch_at(lines_d[:, n, :], x_s)
            out[:, n, :] = a_ * (line_at_xs - flatten_at_xs) + y_target

    return (out.to(src_device)).squeeze(1) if squeeze_back else out.to(src_device)


def _split_params_torch(
    params: torch.Tensor,
    T: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize params to (dx, sx, a, b, c), supporting 3/4/5-column inputs."""
    if params.ndim != 2:
        raise ValueError(f"params must be 2D, got shape {tuple(params.shape)}")
    if params.shape[0] != T:
        raise ValueError(f"params first dim {params.shape[0]} != image T={T}")
    dtype = torch.float32
    p = params.to(device=device, dtype=dtype)
    if p.shape[1] == 5:
        dx, sx, a, b, c = [p[:, i] for i in range(5)]
    elif p.shape[1] == 4:
        dx = p[:, 0]
        sx = torch.ones(T, device=device, dtype=dtype)
        a, b, c = p[:, 1], p[:, 2], p[:, 3]
    elif p.shape[1] == 3:
        dx = torch.zeros(T, device=device, dtype=dtype)
        sx = torch.ones(T, device=device, dtype=dtype)
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
    else:
        raise ValueError(
            f"params must have 3, 4, or 5 columns; got shape {tuple(p.shape)}"
        )
    return dx, sx, a, b, c
