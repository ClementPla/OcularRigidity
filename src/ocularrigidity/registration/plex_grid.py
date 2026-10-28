"""Resampling between the Spectralis grid and the PLEX grid the networks were trained on.

The choroid segmentation model and the cascade_v9 registration regressor were
trained on PLEX Elite B-scans: ``AXIAL_PIXEL_SIZE_MM`` x ``TRANVERSAL_PIXEL_SIZE_MM``
per pixel (1.95 um x 5.9 um), 1024 columns wide. A Spectralis B-scan (496 x 768,
~3.9 um x ~11.5 um) is brought to that pixel size by ``to_plex_grid``, then
center-cropped to ``width`` columns -- on SANS data that keeps the central
~6 mm of a ~8.8 mm scan. ``width=None`` keeps the whole resampled width (rounded
down to a multiple of 32): every column, at a width the regressor never saw.

``PlexGeometry`` records what the forward resampling did, so that results
computed on the PLEX grid (masks, probabilities, transforms) can be brought back
to the native grid: ``plex_to_native`` and ``transform_to_native``. Columns of the
native grid outside the crop get nothing, and ``native_coverage`` says which.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ocularrigidity.consts import (
    AXIAL_PIXEL_SIZE_MM,
    TRANVERSAL_PIXEL_SIZE_MM,
)

PLEX_WIDTH = 1024


@dataclass(frozen=True)
class PlexGeometry:
    native_h: int
    native_w: int
    axial_mm: float  # native pixel size (ScaleY)
    lateral_mm: float  # native pixel size (ScaleX)
    new_h: int  # resampled size, before the crop
    new_w: int
    x0: int  # first kept column of the resampled frame
    width: int  # kept columns

    @property
    def plex_axial_mm(self) -> float:
        return self.native_h * self.axial_mm / self.new_h

    @property
    def plex_lateral_mm(self) -> float:
        return self.native_w * self.lateral_mm / self.new_w

    @property
    def shape(self) -> tuple[int, int]:
        return self.new_h, self.width

    def as_dict(self) -> dict:
        return asdict(self) | {"plex_axial_um": 1e3 * self.plex_axial_mm,
                               "plex_lateral_um": 1e3 * self.plex_lateral_mm,
                               "covered_mm": self.width * self.plex_lateral_mm,
                               "native_mm": self.native_w * self.lateral_mm}


def plex_geometry(H, W, axial_mm, lateral_mm, width: int | None = PLEX_WIDTH) -> PlexGeometry:
    new_h = int(round(H * axial_mm / AXIAL_PIXEL_SIZE_MM / 32)) * 32
    new_w = int(round(W * lateral_mm / TRANVERSAL_PIXEL_SIZE_MM))
    full = new_w // 32 * 32
    width = full if width is None else min(width, full)
    x0 = (new_w - width) // 2
    return PlexGeometry(H, W, float(axial_mm), float(lateral_mm), new_h, new_w, x0, width)


def to_plex_grid(frames, axial_mm, lateral_mm, width: int | None = PLEX_WIDTH):
    """Resample B-scans to the PLEX pixel size the networks were trained on.

    Returns ``(frames (T, new_h, width) uint8, PlexGeometry)``.
    """
    T, H, W = frames.shape
    g = plex_geometry(H, W, axial_mm, lateral_mm, width)
    out = np.stack(
        [cv2.resize(f, (g.new_w, g.new_h), interpolation=cv2.INTER_CUBIC) for f in frames]
    )
    out = np.ascontiguousarray(out[:, :, g.x0 : g.x0 + g.width])
    return out, g


# --------------------------------------------------------------------------- #
# Back to the native grid
# --------------------------------------------------------------------------- #
def native_coverage(g: PlexGeometry) -> np.ndarray:
    """(W,) float in [0, 1]: fraction of each native column inside the crop."""
    canvas = np.zeros((1, g.new_w), np.float32)
    canvas[:, g.x0 : g.x0 + g.width] = 1.0
    return cv2.resize(canvas, (g.native_w, 1), interpolation=cv2.INTER_AREA)[0]


def plex_to_native(arr: np.ndarray, g: PlexGeometry, inter_flag=cv2.INTER_AREA) -> np.ndarray:
    """(T, new_h, width) -> (T, H, W) float32, area-averaged.

    Exact inverse geometry of ``to_plex_grid``: the crop is put back on a
    ``new_w``-wide zero canvas, which is then area-resized to the native size.
    Columns outside the crop come out 0, edge columns partially covered come out
    attenuated -- use ``native_coverage`` to tell them apart.
    """
    T = arr.shape[0]
    out = np.empty((T, g.native_h, g.native_w), np.float32)
    canvas = np.zeros((g.new_h, g.new_w), np.float32)
    for t in range(T):
        canvas[:, g.x0 : g.x0 + g.width] = arr[t]
        out[t] = cv2.resize(canvas, (g.native_w, g.native_h), interpolation=inter_flag)
    return out


def transform_to_native(dx: np.ndarray, dy: np.ndarray, g: PlexGeometry):
    """PLEX-grid ``dx (T,)``, ``dy (T, width)`` -> native ``dx (T,)``, ``dy (T, W)``, in native pixels.

    Native column ``xn`` sits at resampled column ``(xn + 0.5) new_w / W - 0.5``
    (pixel-center convention of ``cv2.resize``), i.e. at crop column ``... - x0``;
    ``dy`` is linearly interpolated there and scaled by the axial pixel ratio.
    Columns outside the crop get the nearest edge value: they are to be blanked
    by the caller (``native_coverage``).
    """
    xn = np.arange(g.native_w)
    xp = (xn + 0.5) * g.new_w / g.native_w - 0.5 - g.x0
    cols = np.arange(g.width)
    dy_n = np.stack([np.interp(xp, cols, row) for row in np.asarray(dy, np.float64)])
    dy_n *= g.native_h / g.new_h
    dx_n = np.asarray(dx, np.float64) * g.native_w / g.new_w
    return dx_n.astype(np.float32), dy_n.astype(np.float32)
