import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from monai.losses.dice import DiceCELoss
from torch import optim
from torchmetrics.segmentation import DiceScore
from torchmetrics import MetricCollection
from ocularrigidity.segmentation.models.losses import (
    gap_detection_loss,
    thickness_smoothness_loss,
)
import kornia as K
from huggingface_hub import PyTorchModelHubMixin


class ChoroidSegmentationModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        arch="unet",
        encoder_name="se_resnet50",
        w_smooth=0.05,
        w_gap=0.05,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=1,
            classes=1,
            *args,
            **kwargs,
        )
        self.loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
        self.metrics = MetricCollection(
            {
                "dice": DiceScore(num_classes=1, average="micro"),
            }
        )
        self.w_smooth = w_smooth
        self.w_gap = w_gap

    def forward(self, x):
        x = self.model(x)
        return x

    def get_loss(self, outputs, masks):
        loss = self.loss_fn(outputs, masks)
        proba = outputs.sigmoid()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        smooth_loss = thickness_smoothness_loss(proba, masks)
        gap_loss = gap_detection_loss(proba)
        return loss + self.w_smooth * smooth_loss + self.w_gap * gap_loss

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.unsqueeze(1)
        outputs = self(images)
        loss = self.get_loss(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.unsqueeze(1)
        outputs = self(images)
        loss = self.get_loss(outputs, masks)
        self.metrics.update(outputs > 0, masks > 0)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.compute(), prog_bar=True, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        # We return the optimizer and the scheduler in a dictionary format, as required by PyTorch Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
