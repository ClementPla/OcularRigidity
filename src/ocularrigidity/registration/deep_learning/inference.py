from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.deep_learning.models.regressor import (
    RegistrationRegressor,
)

__all__ = ["load_registration_regressor"]


def load_registration_regressor(
    checkpoint: str | Path, device: str = "cuda"
) -> Tuple[RegistrationRegressor, dict]:
    """Rebuild the regressor from ``checkpoint`` and load its weights."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Registration checkpoint not found: {checkpoint}")

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ck.get("args", {}) or {}

    model = RegistrationRegressor(
        in_channels=ck["in_channels"],
        embed=ck["embed"],
        img_shape=ck["img_shape"],
        cascade=ck["cascade"],
        scales=ck["scales"],
        use_correlation=ck.get("use_correlation", True),
        corr_dy_radius=args.get("corr_dy_radius", 6),
        corr_dx_radius=args.get("corr_dx_radius", 2),
        dx_step=args.get("dx_step", 4.0),
    )
    model.load_state_dict(ck["model"], strict=True)

    meta = {k: v for k, v in ck.items() if k != "model"}
    return model.to(device).eval(), meta


