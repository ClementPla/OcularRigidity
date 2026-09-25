"""Lightning wrapper: augmentation, SiNC loss, whole-video validation.

Speed augmentation resamples the raw window on the GPU and keeps the native
frame spacing for the output times, so a clip played at ``speed`` carries its
pulse at ``speed`` × the true rate. That spreads the batch's frequencies, which
is what the variance loss needs; merely rescaling timestamps would not, since
the network would still see identical frames.

Validation has two parts:

* ``validation_step`` — the same losses on fixed validation clips;
* ``on_validation_epoch_end`` — the network run over every held-out video with
  a measured HR, the in-band spectral peak taken as its rate, and the error
  against HR logged (MAE, share within 10 %, share at a harmonic).
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch

from ocularrigidity.motion.pulsation.sinc.losses import (
    SiNCLoss,
    SiNCLossConfig,
    peak_frequency,
    rate_prior,
)
from ocularrigidity.motion.pulsation.sinc.model import PulseNet


@dataclass
class SiNCTrainConfig:
    clip_len: int = 870  # ~10 s at 87 Hz
    min_speed: float = 0.7
    max_speed: float = 1.4
    flip_columns: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-2
    width: int = 32
    head_width: int = 64
    # Variance-loss prior: "cohort" (training-population rates × speed
    # augmentation, see losses.rate_prior) or "uniform" (the paper's).
    prior: str = "cohort"
    loss: SiNCLossConfig = field(default_factory=SiNCLossConfig)


def resample_clips(
    w: torch.Tensor,
    dt: torch.Tensor,
    clip_len: int,
    speed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear resampling of raw windows ``(B, L_raw, 2, W)`` at ``speed``.

    Returns ``x (B, clip_len, 2, W)``, frame mask ``(B, clip_len)`` and times
    ``(B, clip_len)`` on the native spacing. A resampled frame touching a gap is
    a gap (NaN propagates through the interpolation).
    """
    B, L_raw = w.shape[:2]
    span = (clip_len - 1) * speed
    start = torch.rand(B, device=w.device) * (L_raw - 2 - span).clamp_min(0)
    k = torch.arange(clip_len, device=w.device, dtype=w.dtype)
    pos = start[:, None] + k[None] * speed[:, None]
    i0 = pos.floor().long().clamp(max=L_raw - 2)
    a = (pos - i0)[..., None, None]
    b = torch.arange(B, device=w.device)[:, None]
    x = w[b, i0] * (1 - a) + w[b, i0 + 1] * a
    mask = ~torch.isnan(x).all(dim=(2, 3))
    t = k[None] * dt[:, None]
    return x, mask, t


class SiNCModule(pl.LightningModule):
    def __init__(self, config: SiNCTrainConfig | None = None, val_videos=None):
        """``val_videos``: list of ``(x, dt, hr)`` for whole-video validation."""
        super().__init__()
        if isinstance(config, dict):  # reloaded from a checkpoint's hparams
            config = SiNCTrainConfig(
                **{**config, "loss": SiNCLossConfig(**config["loss"])}
            )
        self.config = cfg = config or SiNCTrainConfig()
        if cfg.prior not in ("cohort", "uniform"):
            raise ValueError(f"unknown prior {cfg.prior!r}")
        self.save_hyperparameters({"config": asdict(cfg)})
        self.model = PulseNet(cfg.width, cfg.head_width)
        self.loss = SiNCLoss(cfg.loss)
        self.val_videos = val_videos or []

    def set_cohort_prior(self, rates_bpm: np.ndarray) -> None:
        """Variance-loss prior from population rates (no-op for ``prior="uniform"``).

        Call once before training; the prior is a buffer, so checkpoints keep it.
        """
        cfg = self.config
        if cfg.prior != "cohort":
            return
        self.loss.set_prior(
            rate_prior(
                self.loss.band_freqs.cpu().numpy(),
                rates_bpm,
                speed_range=(cfg.min_speed, cfg.max_speed),
            )
        )

    def on_load_checkpoint(self, checkpoint):
        # Checkpoints from before the prior was a buffer trained with uniform.
        checkpoint["state_dict"].setdefault("loss.prior", self.loss.prior.clone())

    def forward(self, x, mask):
        return self.model(x, mask)

    def _step(self, batch, augment: bool):
        w, dt = batch
        cfg = self.config
        B = w.shape[0]
        if augment:
            speed = torch.empty(B, device=w.device).uniform_(
                cfg.min_speed, cfg.max_speed
            )
        else:
            speed = torch.ones(B, device=w.device)
        x, mask, t = resample_clips(w, dt, cfg.clip_len, speed)
        if augment and cfg.flip_columns:
            flip = torch.rand(B, device=w.device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        y = self(x, mask)
        return self.loss(y, t, mask)

    def training_step(self, batch, batch_idx):
        out = self._step(batch, augment=True)
        for k in ("loss", "bandwidth", "sparsity", "variance"):
            self.log(f"train/{k}", out[k], prog_bar=k == "loss")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step(batch, augment=False)
        for k in ("loss", "bandwidth", "sparsity", "variance"):
            self.log(f"val/{k}", out[k], prog_bar=k == "loss")

    @torch.no_grad()
    def predict_video(self, x: np.ndarray, dt: float):
        """Whole-video waveform ``(T,)``, frame mask and times, as tensors."""
        xt = torch.from_numpy(x.astype(np.float32))[None].to(self.device)
        mask = ~torch.isnan(xt).all(dim=(2, 3))
        t = torch.arange(xt.shape[1], device=self.device, dtype=torch.float32) * dt
        y = self(xt, mask)
        return y[0], mask[0], t

    @torch.no_grad()
    def estimate_bpm(self, x: np.ndarray, dt: float) -> float:
        y, mask, t = self.predict_video(x, dt)
        cfg = self.config.loss
        f, _, _ = peak_frequency(y[None], t[None], mask[None], cfg.f_lo, cfg.f_hi)
        return float(f[0]) * 60.0

    def on_validation_epoch_end(self):
        if not self.val_videos or self.trainer.sanity_checking:
            return
        est = np.array([self.estimate_bpm(x, dt) for x, dt, _ in self.val_videos])
        hr = np.array([h for _, _, h in self.val_videos])
        for k, v in rate_metrics(est, hr).items():
            self.log(f"val/{k}", v, prog_bar=k in ("mae_bpm", "spearman"))

    def configure_optimizers(self):
        cfg = self.config
        opt = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.trainer.max_epochs, 1)
        )
        return [opt], [sched]


def rate_metrics(est_bpm: np.ndarray, hr: np.ndarray) -> dict[str, float]:
    """Agreement of estimated rates with measured HR.

    The measured HR is not necessarily taken during the scan, so absolute error
    mixes estimation error with physiological drift; the correlations (Pearson,
    and Spearman, which a few harmonic outliers cannot dominate) say whether
    the estimate tracks the between-subject variation.
    """
    from scipy.stats import pearsonr, spearmanr

    rel = est_bpm / hr
    return dict(
        pearson=float(pearsonr(est_bpm, hr)[0]),
        spearman=float(spearmanr(est_bpm, hr)[0]),
        mae_bpm=float(np.mean(np.abs(est_bpm - hr))),
        within_10pct=float(np.mean(np.abs(rel - 1) <= 0.10)),
        harmonic=float(
            np.mean((np.abs(rel - 2) <= 0.15) | (np.abs(rel - 0.5) <= 0.075))
        ),
    )
