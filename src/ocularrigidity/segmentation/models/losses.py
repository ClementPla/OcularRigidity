import torch
import torch.nn as nn
import torch.nn.functional as F


def column_thickness(pred: torch.Tensor) -> torch.Tensor:
    """
    Compute soft thickness per column.
    pred: (B, 1, H, W) float in [0, 1]
    Returns: (B, W) thickness per column
    """
    return pred.sum(dim=2).squeeze(1)  # sum over H


def thickness_smoothness_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalize sharp changes in predicted thickness across adjacent columns,
    but only MORE than the target's own variation.

    This lets naturally varying thickness through, but penalizes extra jitter
    and discontinuities the network added.
    """
    thick_pred = column_thickness(pred)  # (B, W)
    thick_target = column_thickness(target)  # (B, W)

    grad_pred = torch.abs(thick_pred[:, 1:] - thick_pred[:, :-1])
    grad_target = torch.abs(thick_target[:, 1:] - thick_target[:, :-1])

    # Only penalize predicted gradient that exceeds target's gradient.
    # (prevents penalizing naturally sharp transitions at the choroid edge)
    excess = F.relu(grad_pred - grad_target - 1.0)  # 1-pixel tolerance
    return excess.mean()


def gap_detection_loss(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Penalize empty columns surrounded by non-empty columns.
    pred: (B, 1, H, W) in [0, 1]
    """
    thick = column_thickness(pred)  # (B, W)
    col_has = torch.sigmoid(10 * (thick - threshold))  # soft "has mask" per column

    # What would the neighbors suggest this column should be?
    # avg of left+right neighbors — if both neighbors have mask, this column "should" too
    expected = F.avg_pool1d(
        col_has.unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)

    # Penalize columns where neighbors say "has mask" but this column doesn't
    gap = F.relu(expected - col_has)
    return gap.mean()
