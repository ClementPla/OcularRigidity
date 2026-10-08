import cc3d
import torch
import numpy as np
from tqdm.auto import tqdm


def keep_largest_connected_component(masks):
    """Per-frame largest CC."""
    masks = np.ascontiguousarray(masks)
    out = np.zeros_like(masks, dtype=bool)
    u8 = masks.view(np.uint8) if masks.dtype == np.bool_ else masks.astype(np.uint8)

    rows = masks.any(axis=2)

    for i in tqdm(
        range(masks.shape[0]), desc="Processing Frames", position=1, leave=False
    ):
        row = rows[i]
        if not row.any():
            continue
        lo = int(np.argmax(row))
        hi = len(row) - int(np.argmax(row[::-1]))
        out[i, lo:hi] = cc3d.largest_k(u8[i, lo:hi], k=1, connectivity=4) > 0
    return out


def keep_largest_connected_component_gpu(masks: "torch.Tensor") -> "torch.Tensor":
    """Per-frame largest 4-connected component, without leaving the GPU."""
    import cupy as cp
    from cucim.skimage.measure import label as cu_label
    from torch.utils.dlpack import from_dlpack, to_dlpack

    B, H, W = masks.shape
    stride = W + 1  # the +1 is the blank separator column
    montage = torch.zeros((H, B * stride), dtype=torch.uint8, device=masks.device)
    montage.view(H, B, stride)[:, :, :W] = masks.permute(1, 0, 2).to(torch.uint8)

    labelled = cu_label(cp.from_dlpack(to_dlpack(montage)), connectivity=1, background=0)
    labels = from_dlpack(labelled.astype(cp.int32).toDlpack()).long()

    n = int(labels.max().item())
    if n == 0:
        return torch.zeros_like(masks)

    counts = torch.bincount(labels.reshape(-1), minlength=n + 1)
    counts[0] = 0

    # Which frame each label belongs to.
    frame_of = torch.zeros(n + 1, dtype=torch.long, device=labels.device)
    columns = (torch.arange(B * stride, device=labels.device) // stride).expand(H, -1)
    frame_of.scatter_(0, labels.reshape(-1), columns.reshape(-1))

    packed = counts.long() * (n + 1) + torch.arange(n + 1, device=labels.device)
    best = torch.full((B,), -1, dtype=torch.long, device=labels.device)
    best.scatter_reduce_(0, frame_of[1:], packed[1:], reduce="amax", include_self=True)

    keep = torch.zeros(n + 1, dtype=torch.bool, device=labels.device)
    keep[best[best >= 0] % (n + 1)] = True
    keep[0] = False

    return keep[labels].view(H, B, stride)[:, :, :W].permute(1, 0, 2).contiguous()
