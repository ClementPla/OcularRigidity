from functools import lru_cache
from pathlib import Path

import numpy as np

from ocularrigidity.consts import REGISTRATOR_REVISION, SEGMENTATION_REVISION
from ocularrigidity.data.compression import cube_to_mp4_fastest
from ocularrigidity.data.io import save_mask_atomic
from ocularrigidity.registration.deep_learning.models.regressor import (
    RegistrationRegressor,
)
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule


def get_choroid_segmentation_model():
    """Automatically fetch the choroid segmentation model's weights from Hugging Face."""
    return ChoroidSegmentationModule.from_pretrained(
        "ClementP/ChoroidSegmentationModule", revision=SEGMENTATION_REVISION
    ).eval()


def get_registration_model(revision: str = REGISTRATOR_REVISION):
    """Automatically fetch the registration model's weights from Hugging Face."""
    return RegistrationRegressor.from_pretrained(
        "ClementP/OCTVideoRegistration", revision=revision
    ).eval()


@lru_cache(maxsize=1)
def get_model(device: str = "cuda"):
    """The frozen choroid segmentation model, loaded straight onto ``device``."""
    model = get_choroid_segmentation_model().eval().to(device)
    if float(next(model.parameters()).abs().max()) == 0.0:
        raise RuntimeError(
            f"segmentation weights are all zero after loading onto {device!r} -- "
            "the checkpoint did not load; refusing to run with a dead encoder"
        )
    return model


def write_raw_mask(raw_mask: np.ndarray, out_path: Path) -> None:
    """Largest connected component, then ``mask.npz``."""
    save_mask_atomic(keep_largest_connected_component(raw_mask), out_path)


def write_cycle_mask(mask: np.ndarray, out_path: Path) -> None:
    """``segmented_cycles.npz`` and an mp4 preview next to it."""
    save_mask_atomic(mask, out_path)
    cube_to_mp4_fastest(
        (mask * 255).astype(np.uint8),
        Path(out_path).with_suffix(".mp4"),
        fps=30,
        cq=20,
    )
