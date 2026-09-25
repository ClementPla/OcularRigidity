
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
import torch.nn.functional as F



@torch.inference_mode()
def infer(
    module: ChoroidSegmentationModule,
    data: torch.Tensor | np.ndarray,
    scale_factor: float | tuple[float, float] = 1.0,
    resize_to: tuple[int, int] | None = None,
    batch_size: int = 8,
    return_logit: bool = False,
    use_graphcut: bool = False,
    graphcut_kwargs: dict | None = None,
    device: str = "cuda",
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    post_process: bool = True,
    pin_memory: bool = True,
    accumulate_on_cpu: bool = False,
    pad_last_batch: bool = False,
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
    if pin_memory and not data.is_pinned() and device == "cuda":
        data = data.pin_memory()

    # Pre-allocate the whole-volume output (avoids inner-loop CPU syncs when it
    # lives on the device; see ``accumulate_on_cpu`` for when it should not)
    buffer_device = "cpu" if accumulate_on_cpu else device
    predictions_buf = torch.empty(
        (n, org_h, org_w),
        dtype=torch.float32 if return_logit else torch.bool,
        device=buffer_device,
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

        # Fill the tail batch out to batch_size with copies of its last frame,
        # so the module only ever sees one input shape. The copies are sliced
        # back off below and never reach the output.
        n_real = end - start
        batch_padding = batch_size - n_real if pad_last_batch else 0
        if batch_padding > 0:
            chunk = torch.cat([chunk, chunk[-1:].expand(batch_padding, -1, -1, -1)])

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = module(chunk)

        if batch_padding > 0:
            out = out[:n_real]

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        if (scale_factor != 1.0) or (resize_to is not None):
            out = F.interpolate(
                out, size=(org_h, org_w), mode="bilinear", align_corners=True
            )

        # 3. Store the batch. Assigning into a host buffer is a cross-device
        # copy_, so this is the same line either way.
        if return_logit:
            predictions_buf[start:end] = out.squeeze(1)
        elif use_graphcut:
            probs = torch.sigmoid(out.float() * 0.5).squeeze(1)
            masks = graphcut_masks_from_probs_batch_torch(probs, **gc_kwargs)
            predictions_buf[start:end] = masks
        else:
            predictions_buf[start:end] = out.squeeze(1) > 0.0

    # No-op when the buffer is already in host memory.
    predictions = predictions_buf.cpu().numpy()

    if return_logit:
        return predictions

    if not post_process:
        return predictions

    predictions = keep_largest_connected_component(predictions)
    return predictions
