from ..consts import OIMHS_ROOT

__all__ = ["OIMHS_ROOT", "ChoroidSegmentationDataModule", "Database"]


def __getattr__(name):
    """Lazily expose the datamodule symbols (PEP 562).

    ``.datamodule`` pulls in the training stack (pytorch_lightning →
    torchmetrics → torchvision / torchaudio / transformers). Importing it
    eagerly here forced every lightweight consumer — e.g.
    ``ocularrigidity.data.compression`` / ``.io`` / ``.measurements`` used by the
    viewers — to load that whole stack, which also makes them fragile to
    torch/torchvision/torchaudio ABI mismatches. Defer it until actually used so
    ``from ocularrigidity.data import ChoroidSegmentationDataModule`` still works
    but ``import ocularrigidity.data.compression`` stays torch-free.
    """
    if name in ("ChoroidSegmentationDataModule", "Database"):
        from .datamodule import ChoroidSegmentationDataModule, Database

        return {
            "ChoroidSegmentationDataModule": ChoroidSegmentationDataModule,
            "Database": Database,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
