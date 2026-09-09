"""Heidelberg HEYEX DICOM exports: hierarchy, scan geometry, layer annotations.

    from ocularrigidity.data.heyex import build_tree, load_acquisition

    tree = build_tree(root)                    # patient / visit / eye / scan type
    acq = load_acquisition(path)               # pixels + layers, aligned
    acq.layers["ILM"]                          # (n_bscans, width), NaN where unset

See :mod:`ocularrigidity.data.heyex.dicom` for why the B-scans come from the
private E2E blob rather than from DICOM ``PixelData``.
"""

from .dicom import (
    Acquisition,
    build_tree,
    classify_scan,
    estimate_frame_offsets,
    extract_pdfs,
    frame_geometry,
    index_export,
    load_acquisition,
    read_bscans_from_blob,
    read_layers,
)

__all__ = [
    "Acquisition",
    "build_tree",
    "classify_scan",
    "estimate_frame_offsets",
    "extract_pdfs",
    "frame_geometry",
    "index_export",
    "load_acquisition",
    "plot_acquisition",
    "plot_bscan_layers",
    "read_bscans_from_blob",
    "read_layers",
]


def __getattr__(name):
    # Defer the plotting module so importing the readers does not build the
    # figure helpers. (This does not save the matplotlib import itself: eyepy
    # pulls matplotlib in at import time regardless.)
    if name in ("plot_acquisition", "plot_bscan_layers"):
        from . import plotting

        globals()["plot_acquisition"] = plotting.plot_acquisition
        globals()["plot_bscan_layers"] = plotting.plot_bscan_layers
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
