"""SiNC-rPPG architectures (PhysNet, RPNet) from https://github.com/CVRL/SiNC-rPPG.

Input ``(B, C, T, H, W)``, output ``(B, T)``. The spatial head (4 max-pools then a
4x4 avg-pool with stride 2) reduces to 1x1 only for 64x64 frames: other sizes give
``(B, T * h * w)`` after the final flatten.
"""

from .PhysNet import PhysNet
from .PhysNetRPPGEncoder import PhysNetRPPGEncoder
from .RPNet import RPNet

# physnet_rppg_enc : encodeur du PhysNet de rPPG-Toolbox (PhysNetRPPG.py), poids preentraines
# optionnels (``pretrained=<.pth>``), sans reduction temporelle, sortie (B, T).
MODELS = {"physnet": PhysNet, "rpnet": RPNet, "physnet_rppg_enc": PhysNetRPPGEncoder}


def select_model(model_type: str, input_channels: int = 1, drop_p: float = 0.5, **kwargs):
    try:
        cls = MODELS[model_type.lower()]
    except KeyError:
        raise ValueError(f"unknown model {model_type!r}, expected one of {sorted(MODELS)}")
    return cls(input_channels=input_channels, drop_p=drop_p, **kwargs)
