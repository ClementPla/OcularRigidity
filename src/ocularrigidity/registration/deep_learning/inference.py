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
    """Rebuild the regressor from ``checkpoint`` and load its weights.

    Returns ``(model, meta)``; ``meta`` is the checkpoint dict minus the
    weights, so a caller can log which epoch and objective it is running and
    stamp them into the registration cache.

    The state dict is loaded strictly: a silent shape mismatch here would show
    up as a plausible-looking but wrong transform on every video of the cohort.
    """
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
        # Not top-level keys. reg_cascade_v1/v2 both recorded (6, 2), which is
        # also the constructor default, but the trainer's own argparse default
        # is 10 — so reading them from the checkpoint rather than relying on
        # either default is what keeps a future run loadable.
        corr_dy_radius=args.get("corr_dy_radius", 6),
        corr_dx_radius=args.get("corr_dx_radius", 2),
        # Also under `args`, and the one that fails *silently*: dx is
        # ``sum(ddx_i) * dx_step``, so loading a model trained at 8.0 with the
        # constructor's 4.0 halves every lateral displacement it predicts, with
        # nothing to signal it. Checkpoints written before --dx-step existed
        # carry no value and correctly fall back to the 4.0 they trained with.
        dx_step=args.get("dx_step", 4.0),
    )
    model.load_state_dict(ck["model"], strict=True)

    meta = {k: v for k, v in ck.items() if k != "model"}
    return model.to(device).eval(), meta


