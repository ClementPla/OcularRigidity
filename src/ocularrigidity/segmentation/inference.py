from typing import Optional, Tuple

import torch
import numpy as np
from tqdm.auto import tqdm

from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.postprocess.graphcut_gpu import (
    graphcut_masks_from_probs_batch_torch,
)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def infer(
    module: ChoroidSegmentationModule,
    data: torch.Tensor | np.ndarray,
    scale_factor: float | tuple[float, float] = 1.0,
    resize_to: tuple[int, int] | None = None,
    batch_size: int = 8,
    return_logit: bool = False,
    use_graphcut: bool = True,
    graphcut_kwargs: dict | None = None,
    device: str = "cuda",
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    verbose: bool = False,
) -> np.ndarray:
    # 1. Convert to tensor without copying if possible; keep uint8 in CPU RAM
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if data.ndim == 3:
        data = data.unsqueeze(1)

    n, _, org_h, org_w = data.shape
    gc_kwargs = graphcut_kwargs or {}

    # Pin memory so data transfers to GPU are truly asynchronous
    if not data.is_pinned() and device == "cuda":
        data = data.pin_memory()

    # Pre-allocate GPU tensor for batch outputs (avoids inner-loop CPU syncs)
    if return_logit:
        gpu_predictions = torch.empty(
            (n, org_h, org_w), dtype=torch.float32, device=device
        )
    else:
        gpu_predictions = torch.empty(
            (n, org_h, org_w), dtype=torch.bool, device=device
        )

    for start in tqdm(range(0, n, batch_size), desc="Inference", disable=not verbose):
        end = min(start + batch_size, n)

        # Fast async transfer of uint8 data
        chunk = data[start:end].to(device, non_blocking=True)

        # 2. Normalize ON GPU (fast FP16 vectorized operations)
        if chunk.dtype == torch.uint8:
            chunk = chunk.to(dtype=amp_dtype).mul_(1.0 / 255.0).sub_(0.5).div_(0.5)
        else:
            chunk = chunk.to(dtype=amp_dtype)

        # Resizing / Padding on GPU
        if resize_to is not None:
            chunk = F.interpolate(
                chunk, size=resize_to, mode="bilinear", align_corners=False
            )
        elif scale_factor != 1.0:
            chunk = F.interpolate(
                chunk, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )

        h, w = chunk.shape[2], chunk.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            chunk = F.pad(chunk, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = module(chunk)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        if (scale_factor != 1.0) or (resize_to is not None):
            out = F.interpolate(
                out, size=(org_h, org_w), mode="bilinear", align_corners=True
            )

        # 3. Store results directly in GPU memory without syncing to CPU every batch
        if return_logit:
            gpu_predictions[start:end] = out.squeeze(1)
        elif use_graphcut:
            probs = torch.sigmoid(out.float() * 0.5).squeeze(1)
            masks = graphcut_masks_from_probs_batch_torch(probs, **gc_kwargs)
            gpu_predictions[start:end] = masks
        else:
            gpu_predictions[start:end] = out.squeeze(1) > 0.0

    # Single transfer back to CPU host memory
    predictions = gpu_predictions.cpu().numpy()

    if return_logit:
        return predictions

    # Post-processing (CPU bottleneck)
    predictions = keep_largest_connected_component(predictions)
    return predictions
