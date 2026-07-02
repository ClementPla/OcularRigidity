from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ocularrigidity.segmentation.fovea import (
    FoveaKeypointDataModule,
    FoveaKeypointModule,
)

seed_everything(42)

# Path to the annotation CSV (columns: image, x, y, patient). Adjust to your data.
ANNOTATIONS_CSV = "/media/clement/HD/Santiago/OcularRigidity/annotations/fovea.csv"


def train():
    datamodule = FoveaKeypointDataModule(
        annotations_csv=ANNOTATIONS_CSV,
        img_size=(256, 512),
        batch_size=32,
        num_workers=4,
        max_x_translate=0.35,
    )
    model = FoveaKeypointModule(encoder_name="resnet34")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_x_mae_px",
        dirpath="checkpoints",
        filename="fovea-keypoint-{epoch:02d}-{val_x_mae_px:.2f}",
        save_top_k=3,
        mode="min",
    )
    logger = WandbLogger(project="ocular-rigidity", name="fovea-keypoint")
    trainer = Trainer(
        max_epochs=150,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_x_mae_px", mode="min", patience=15),
        ],
        logger=logger,
        accelerator="auto",
        precision="bf16-mixed",
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
