import torch
import numpy as np


@torch.inference_mode()
def temporal_median(
    frames: torch.Tensor,
    ignore_zeros: bool = True,
    device: str = "cuda",
    row_chunk: int = 64,
) -> torch.Tensor:
    """Mediane temporelle d'un volume recale ``(T, H, W)`` -> template ``(H, W)``.

    Returns
    -------
    torch.Tensor
        Template median ``(H, W)`` (float32, sur ``device``).
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    T, H, W = frames.shape
    out = torch.empty((H, W), dtype=torch.float32, device=device)
    for r0 in range(0, H, row_chunk):
        r1 = min(r0 + row_chunk, H)
        block = frames[:, r0:r1, :].to(device, torch.float32)  # (T, h, W)
        if ignore_zeros:
            block = block.masked_fill(block == 0, float("nan"))
            med = torch.nanmedian(block, dim=0).values
            med = torch.nan_to_num(med, nan=0.0)
        else:
            med = block.median(dim=0).values
        out[r0:r1] = med
        del block
    return out
