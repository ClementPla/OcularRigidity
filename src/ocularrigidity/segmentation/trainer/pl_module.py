import pytorch_lightning as pl
import torch
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

from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM

_UM_PER_PX = AXIAL_PIXEL_SIZE_MM * 1e3


def _lower_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Row of the last positive pixel per column of ``(B, H, W)`` bool, -1 if none."""
    H = mask.shape[1]
    last = H - 1 - mask.flip(1).int().argmax(1)
    return torch.where(mask.any(1), last, torch.full_like(last, -1))


class ChoroidSegmentationModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        arch="unet",
        encoder_name="se_resnet50",
        w_smooth=0.05,
        w_gap=0.05,
        scheduler: str = "restarts",
        max_epochs: int | None = None,
        warmup_epochs: int = 0,
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
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self._reset_boundary_errors()

    def _reset_boundary_errors(self):
        # sums (pixels) and column counts over the validation epoch
        self._thick_err = self._thick_n = 0.0
        self._csi_err = self._csi_n = 0.0

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
        self._update_boundary_errors(outputs[:, 0] > 0, masks[:, 0] > 0)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    @torch.no_grad()
    def _update_boundary_errors(self, pred: torch.Tensor, gt: torch.Tensor):
        """Per-column errors on the columns the ground truth covers: choroid
        thickness (what deltaCT is made of) and the lower boundary (the
        choroid-sclera interface, the hard one). Dice hardly moves for a few
        pixels of boundary shift; these do."""
        has_gt = gt.any(1)
        t_err = (pred.sum(1) - gt.sum(1)).abs().float()
        self._thick_err += float(t_err[has_gt].sum())
        self._thick_n += float(has_gt.sum())
        both = has_gt & pred.any(1)
        c_err = (_lower_boundary(pred) - _lower_boundary(gt)).abs().float()
        self._csi_err += float(c_err[both].sum())
        self._csi_n += float(both.sum())

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.compute(), prog_bar=True, sync_dist=True)
        self.metrics.reset()
        if self._thick_n:
            self.log_dict(
                {
                    "thickness_mae_um": self._thick_err / self._thick_n * _UM_PER_PX,
                    "csi_mae_um": self._csi_err / max(self._csi_n, 1.0) * _UM_PER_PX,
                },
                prog_bar=True,
                sync_dist=True,
            )
        self._reset_boundary_errors()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.scheduler == "restarts":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif self.scheduler == "cosine":
            # One cosine over the run, after an optional linear warm-up
            if not self.max_epochs:
                raise ValueError("scheduler='cosine' needs max_epochs")
            warm = max(0, int(self.warmup_epochs))
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.max_epochs - warm), eta_min=1e-6
            )
            if warm:
                lr_scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    [
                        optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=0.01, total_iters=warm
                        ),
                        lr_scheduler,
                    ],
                    milestones=[warm],
                )
        else:
            raise ValueError(f"unknown scheduler {self.scheduler!r}")
        # We return the optimizer and the scheduler in a dictionary format, as
        # required by PyTorch Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
