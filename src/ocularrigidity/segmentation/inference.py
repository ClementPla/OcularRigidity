from typing import Tuple

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


@torch.inference_mode()
def infer(
    module: ChoroidSegmentationModule,
    data: torch.Tensor | np.ndarray,
    scale_factor: float | Tuple[float, float] = 1.0,
    batch_size: int = 8,
    return_logit: bool = False,
    use_graphcut: bool = True,
    graphcut_kwargs: dict | None = None,
    device: str = "cuda",
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    verbose=False,
) -> np.ndarray:
    """
    Provide a convenient wrapper around the segmentation model for inference on a full video cube, with optional resizing and postprocessing.

    Args:
        module (ChoroidSegmentationModule):
        data (torch.Tensor | np.ndarray): Input video cube, expected shape (T, C, H, W) or (T, H, W). If uint8, will be normalized to [0,1] and standardized.
        scale_factor (float | Tuple[float, float], optional): Scale factor for resizing the input. Defaults to 1.0.
        batch_size (int, optional): Batch size for inference. Defaults to 8.
        return_logit (bool, optional): Whether to return logits instead of binary masks. Defaults to False.
        use_graphcut (bool, optional): Whether to use graph cut for postprocessing. Defaults to True.
        graphcut_kwargs (dict | None, optional): Keyword arguments for graph cut. Defaults to None.
        device (str, optional): Device to run inference on. Defaults to "cuda".
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        amp_dtype (torch.dtype, optional): Data type for mixed precision. Defaults to torch.float16.
        verbose (bool, optional): Whether to show a progress bar. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    if data.ndim == 3:
        data = data.unsqueeze(1)
    if data.max() > 1:
        data = ((data / 255.0) - 0.5) / 0.5

    module = module.to(device).eval()
    n = data.shape[0]
    org_h, org_w = data.shape[2], data.shape[3]

    if return_logit:
        predictions = np.empty((n, org_h, org_w), dtype=np.float32)
    else:
        predictions = np.empty((n, org_h, org_w), dtype=bool)

    gc_kwargs = graphcut_kwargs or {}

    for start in tqdm(range(0, n, batch_size), desc="Inference", disable=not verbose):
        end = min(start + batch_size, n)
        chunk = data[start:end].to(device, non_blocking=True)

        if scale_factor != 1.0:
            chunk = torch.nn.functional.interpolate(
                chunk,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = module(chunk)

        if scale_factor != 1.0:
            out = torch.nn.functional.interpolate(
                out,
                size=(org_h, org_w),
                mode="bilinear",
                align_corners=False,
            )

        if return_logit:
            predictions[start:end] = out.squeeze(1).cpu().numpy()
        elif use_graphcut:
            probs = torch.sigmoid(out.float() * 0.5).squeeze(
                1
            )  # (B, H, W) stays on GPU
            masks = graphcut_masks_from_probs_batch_torch(probs, **gc_kwargs)
            predictions[start:end] = masks.cpu().numpy()
        else:
            predictions[start:end] = (out > 0.0).squeeze(1).cpu().numpy()

    if return_logit:
        return predictions

    predictions = keep_largest_connected_component(predictions)
    return predictions
