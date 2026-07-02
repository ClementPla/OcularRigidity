import kornia as K
import torch
import numpy as np


def median_blur_video(
    video: torch.Tensor | np.ndarray,
    kernel_size: int = 5,
    batch_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Apply a median blur to each frame of a video (T, H, W)."""
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    video = video
    blurred_video = []
    for start in range(0, len(video), batch_size):
        chunk = (
            video[start : start + batch_size].unsqueeze(1).float().to(device)
        )  # (B, 1, H, W)
        blurred_chunk = K.filters.median_blur(chunk, kernel_size=kernel_size)
        blurred_video.append(blurred_chunk.squeeze(1).cpu())  # (B, H, W)
    return torch.cat(blurred_video, dim=0)
