from ..consts import OIMHS_ROOT

__all__ = ["OIMHS_ROOT", "ChoroidSegmentationDataModule", "Database"]


def __getattr__(name):
    if name in ("ChoroidSegmentationDataModule", "Database"):
        from .datamodule import ChoroidSegmentationDataModule, Database

        globals()["ChoroidSegmentationDataModule"] = ChoroidSegmentationDataModule
        globals()["Database"] = Database
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
