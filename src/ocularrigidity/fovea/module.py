"""Lightning module that localizes the foveal point in an OCT B-scan.

Reuses the same backbone family as ``ChoroidSegmentationModule`` (smp), but
outputs a single-channel heatmap decoded to a sub-pixel coordinate via DSNT.
Only the x-coordinate is consumed by the lateral registration, but the model is
supervised on the full (x, y) point (the annotated foveal pit center).
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import optim

from ocularrigidity.fovea.dsnt import (
    dsnt,
    euclidean_loss,
    flat_softmax,
    js_reg_loss,
    normalized_to_pixel,
)


class FoveaKeypointModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        arch: str = "unet",
        encoder_name: str = "resnet34",
        heatmap_sigma: float = 1.0,
        w_reg: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=1,
            classes=1,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (coords, heatmap): coords (B, 1, 2) normalized; heatmap (B, 1, H, W)."""
        heatmap = flat_softmax(self.model(x))
        coords = dsnt(heatmap)
        return coords, heatmap

    def _step(self, batch, stage: str) -> torch.Tensor:
        images, target_xy = batch  # images (B,1,H,W); target_xy (B,1,2) normalized
        coords, heatmap = self(images)

        eucl = euclidean_loss(coords, target_xy).mean()
        reg = js_reg_loss(heatmap, target_xy, self.hparams.heatmap_sigma).mean()
        loss = eucl + self.hparams.w_reg * reg

        # Report lateral error in heatmap pixels (the quantity registration cares about).
        w = heatmap.shape[-1]
        x_mae_px = (
            (
                normalized_to_pixel(coords[..., 0], w)
                - normalized_to_pixel(target_xy[..., 0], w)
            )
            .abs()
            .mean()
        )
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), sync_dist=True)
        self.log(f"{stage}_eucl", eucl, sync_dist=True)
        self.log(f"{stage}_x_mae_px", x_mae_px, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    @torch.inference_mode()
    def predict_coords_px(self, x: torch.Tensor) -> torch.Tensor:
        """Predict (x, y) in pixels of the *input* tensor. x: (B,1,H,W) -> (B,2)."""
        coords, heatmap = self(x)
        h, w = heatmap.shape[-2:]
        px = normalized_to_pixel(coords[:, 0, 0], w)
        py = normalized_to_pixel(coords[:, 0, 1], h)
        return torch.stack([px, py], dim=-1)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
