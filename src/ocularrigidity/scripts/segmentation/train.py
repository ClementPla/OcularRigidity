from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule
from ocularrigidity.data import ChoroidSegmentationDataModule, Database, OIMHS_ROOT
from pytorch_lightning import seed_everything

seed_everything(42)


def train():
    datamodule = ChoroidSegmentationDataModule(
        root=OIMHS_ROOT,
        database=Database.OIMHS,
        batch_size=16,
        num_workers=4,
    )
    model = ChoroidSegmentationModule()
    checkpoint_callback = ModelCheckpoint(
        monitor="dice",
        dirpath="checkpoints",
        filename="choroid-segmentation-{epoch:02d}-{dice:.2f}",
        save_top_k=3,
        mode="max",
    )
    logger = WandbLogger(project="ocular-rigidity", name="choroid-segmentation")
    trainer = Trainer(
        max_epochs=150,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="dice", mode="max", patience=10),
        ],
        logger=logger,
        accelerator="auto",
        precision="bf16-mixed",
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
