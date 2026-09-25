"""Train the Stage-0 SiNC rate model and compare it with the current chain.

    python -m ocularrigidity.scripts.pulsation.train_sinc cache
    python -m ocularrigidity.scripts.pulsation.train_sinc train --epochs 60
    python -m ocularrigidity.scripts.pulsation.train_sinc evaluate --ckpt <path>

``cache`` reads every measure.pkl of the run once (~1 h from the HDD).
``evaluate`` writes a per-video table of measured HR, the stored (HR-anchored)
rate, the chain without expected BPM, and SiNC, plus a metric summary.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from ocularrigidity.consts import RUN_ROOT
from ocularrigidity.motion.pulsation.sinc import (
    SiNCModule,
    SiNCTrainConfig,
    rate_metrics,
)
from ocularrigidity.motion.pulsation.sinc.data import (
    SiNCClipDataset,
    build_cache,
    load_index,
    load_video,
    worker_init_fn,
)
from ocularrigidity.motion.pulsation.sinc.evaluate import open_band_table, sinc_table

SINC_ROOT = RUN_ROOT / "sinc"
MEASURES_ROOT = RUN_ROOT / "measures"


def cmd_cache(args):
    index = build_cache(
        MEASURES_ROOT, args.cache_dir, workers=args.workers, limit=args.limit
    )
    print(index["split"].value_counts().to_string())
    print(f"HR available: {index['HR'].notna().sum()} / {len(index)}")


def cmd_train(args):
    seed_everything(args.seed)
    index = load_index(args.cache_dir)
    cfg = SiNCTrainConfig(
        max_speed=args.max_speed, lr=args.lr, width=args.width, prior=args.prior
    )

    train_idx = index[index["split"] == "train"]
    val_idx = index[index["split"] == "val"]
    train_ds = SiNCClipDataset(
        args.cache_dir,
        train_idx,
        clip_len=cfg.clip_len,
        max_speed=cfg.max_speed,
        samples_per_epoch=args.samples_per_epoch,
    )
    val_ds = SiNCClipDataset(
        args.cache_dir,
        val_idx,
        clip_len=cfg.clip_len,
        max_speed=cfg.max_speed,
        samples_per_epoch=512,
        seed=0,
    )
    # Fixed validation clips: draw them once so every epoch sees the same ones.
    val_clips = [val_ds[i] for i in range(len(val_ds))]

    with_hr = val_idx[val_idx["HR"].notna()]
    val_videos = [
        (load_video(args.cache_dir, r.video), r.dt, r.HR) for r in with_hr.itertuples()
    ]
    print(
        f"train videos {len(train_ds.videos)} | val videos {len(val_ds.videos)} "
        f"({len(val_videos)} with HR)"
    )

    # The prior sees the training patients' HR distribution only in aggregate.
    module = SiNCModule(cfg, val_videos=val_videos)
    module.set_cohort_prior(train_idx["HR"].to_numpy())
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.gpu],
        precision="bf16-mixed",
        logger=CSVLogger(SINC_ROOT / "runs", name=args.name),
        callbacks=[
            ModelCheckpoint(
                # val/loss kept falling while the rate got worse (harmonic
                # lock), so the default selects on the rate itself.
                monitor=args.monitor,
                mode="min",
                save_top_k=2,
                save_last=True,
                filename="{epoch:03d}-{" + args.monitor + ":.4f}",
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(),
        ],
        limit_train_batches=args.limit_batches,
        log_every_n_steps=10,
    )
    trainer.fit(
        module,
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=args.workers > 0,
        ),
        DataLoader(val_clips, batch_size=args.batch_size),
    )


def cmd_evaluate(args):
    index = load_index(args.cache_dir)
    module = SiNCModule.load_from_checkpoint(args.ckpt, map_location=f"cuda:{args.gpu}")
    df = index[index["split"] == args.split][
        ["video", "HR", "pipeline_bpm", "pipeline_confidence", "gap_frac"]
    ]
    df = df.merge(sinc_table(module, args.cache_dir, args.split), on="video")
    if not args.skip_open_band:
        open_band = open_band_table(
            args.cache_dir,
            MEASURES_ROOT,
            args.split,
            out=Path(args.cache_dir) / f"open_band_{args.split}.csv",
        )
        df = df.merge(open_band, on="video", how="left")

    out = Path(args.ckpt).with_name(f"eval_{Path(args.ckpt).stem}_{args.split}.csv")
    df.to_csv(out, index=False)

    hr = df[df["HR"].notna()]
    rows = {}
    for col in ("pipeline_bpm", "open_band_bpm", "sinc_bpm"):
        if col in hr:
            ok = hr[col].notna()
            rows[col] = rate_metrics(
                hr.loc[ok, col].to_numpy(), hr.loc[ok, "HR"].to_numpy()
            )
    print(f"{len(hr)} {args.split} videos with measured HR")
    print(
        "(pipeline_bpm is HR-anchored: its band is ±30 % of HR, so not a fair baseline)"
    )
    print(pd.DataFrame(rows).T.round(3).to_string())
    if "open_band_bpm" in df:
        agree = np.abs(df["sinc_bpm"] / df["open_band_bpm"] - 1) <= 0.10
        print(
            f"SiNC within 10 % of open-band chain (all {args.split} videos): {agree.mean():.2%}"
        )
    print(f"per-video table: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["cache", "train", "evaluate"])
    p.add_argument("--cache-dir", type=Path, default=SINC_ROOT / "cache")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit", type=int, help="cache only the first N videos (smoke tests)"
    )
    # train
    p.add_argument("--name", default="stage0")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--samples-per-epoch", type=int, default=4096)
    p.add_argument("--limit-batches", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--max-speed", type=float, default=1.4)
    p.add_argument("--prior", choices=["cohort", "uniform"], default="cohort")
    p.add_argument("--monitor", default="val/mae_bpm")
    # evaluate
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--skip-open-band", action="store_true")
    args = p.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "evaluate": cmd_evaluate}[args.command](
        args
    )


if __name__ == "__main__":
    main()
