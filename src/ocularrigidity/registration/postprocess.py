import numpy as np
import torch
import torch.nn.functional as F
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)


_BM_JUMP_THRESHOLD_PX = 20.0  # saut horizontal de la BM (px) juge discontinu
_BM_BAD_FRAME_FRACTION = (
    0.5  # colonne noircie si mauvaise dans >= cette fraction des frames
)
_BM_MARGIN_COLUMNS = 15  # marge (colonnes) noircie de chaque cote d'une discontinuite


def filter_bad_ascans_per_bms(
    registered_masks: torch.Tensor | np.ndarray,
    jump_threshold: float = _BM_JUMP_THRESHOLD_PX,
    frame_fraction: float = _BM_BAD_FRAME_FRACTION,
    margin: int = _BM_MARGIN_COLUMNS,
) -> torch.Tensor:
    if isinstance(registered_masks, torch.Tensor):
        registered_masks = registered_masks.cpu().numpy()

    if registered_masks.ndim == 4:
        # Squeeze channel 1
        registered_masks = np.squeeze(registered_masks, axis=1)
    bm, csi = extract_boundaries_fast(registered_masks)
    bm, _ = clean_boundaries(bm, csi)
    bm = torch.from_numpy(bm)  # (T, W)
    nan_bad = torch.isnan(bm)  # (T, W)

    step = (bm[:, 1:] - bm[:, :-1]).abs() > jump_threshold  # (T, W-1)
    jump_bad = torch.zeros_like(nan_bad)
    jump_bad[:, 1:] |= step  # marque la colonne de droite du saut
    jump_bad[:, :-1] |= step  # ... et celle de gauche
    bad = nan_bad | jump_bad  # (T, W)
    bad_cols = bad.float().mean(dim=0) >= frame_fraction  # (W,)
    if margin > 0:
        # Dilatation 1D (max-pool) : elargit chaque colonne signalee de +-margin.
        k = 2 * int(margin) + 1
        bad_cols = (
            F.max_pool1d(
                bad_cols.float().view(1, 1, -1), kernel_size=k, stride=1, padding=margin
            )
            .view(-1)
            .bool()
        )
    return bad_cols
