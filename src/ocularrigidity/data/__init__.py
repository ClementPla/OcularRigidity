from ..consts import OIMHS_ROOT

__all__ = ["OIMHS_ROOT", "ChoroidSegmentationDataModule", "Database"]


def __getattr__(name):
    # Lazily import the torch / PyTorch-Lightning-heavy datamodule, so that
    # lightweight submodules (e.g. ocularrigidity.data.spectralis) can be
    # imported without dragging in the whole deep-learning stack.
    if name in ("ChoroidSegmentationDataModule", "Database"):
        from .datamodule import ChoroidSegmentationDataModule, Database

        globals()["ChoroidSegmentationDataModule"] = ChoroidSegmentationDataModule
        globals()["Database"] = Database
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
