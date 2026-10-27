"""Train the choroid segmentation model on the annotated set + OIMHS, at the PLEX pixel size.

    # The current reference (SegFormer MiT-B1, warm-restart schedule, early stopping)
    python -m ocularrigidity.scripts.segmentation.train

    # A named model (segmentation/trainer/presets.py) or any smp arch / encoder
    python -m ocularrigidity.scripts.segmentation.train --model upernet_convnext_t \\
        --scheduler cosine --epochs 60 --patience 0

Started by ``wandb agent`` (see ``scripts/segmentation/sweep.py``), the run's
sweep parameters override the command line; their names are the argparse dest
names (``batch_size``, ``w_smooth``, ...). The best checkpoint (on
``--monitor``) and a ``results.json`` go to ``--out``, under
``checkpoints/segmentation_sweep/<sweep id>/`` for sweep runs; the validation
metrics of the best checkpoint go to the W&B summary as ``best/<metric>``.
"""

import argparse
import json
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ocularrigidity.data import ChoroidSegmentationDataModule, Database
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule
from ocularrigidity.segmentation.trainer.presets import MODELS, registration_ok

# metric -> mode
MONITORS = {"dice": "max", "thickness_mae_um": "min", "csi_mae_um": "min"}
SWEEP_ROOT = Path("checkpoints/segmentation_sweep")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(MODELS), default=None, help="named preset")
    ap.add_argument("--arch", default="segformer")
    ap.add_argument("--encoder", default="mit_b1")
    ap.add_argument("--encoder-weights", default="imagenet", help="'none' for random init")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--w-smooth", type=float, default=0.1)
    ap.add_argument("--w-gap", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--crop", type=int, nargs=2, default=(1024, 768), metavar=("H", "W"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--scheduler", choices=("restarts", "cosine"), default="restarts")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="cosine only")
    ap.add_argument("--patience", type=int, default=10, help="early stopping; 0 disables it")
    ap.add_argument("--monitor", choices=list(MONITORS), default="dice")
    ap.add_argument(
        "--databases",
        nargs="+",
        default=[Database.ANNOTATED, Database.OIMHS],
        choices=[Database.ANNOTATED, Database.OIMHS],
    )
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="default: checkpoints/<run name>")
    ap.add_argument("--wandb-project", default="ocular-rigidity")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-tags", nargs="*", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap


def apply_preset(args, ap, explicit: set[str]):
    """Fill the options of ``args.model``, except those set explicitly (CLI or sweep)."""
    if args.model is None:
        return
    for key, value in MODELS[args.model].items():
        if key in ("arch", "encoder") or (
            key not in explicit and getattr(args, key) == ap.get_default(key)
        ):
            setattr(args, key, value)


def run_name(args) -> str:
    return args.model or f"{args.arch}_{args.encoder}"


def _plain(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def init_wandb(args, ap):
    """Open the W&B run; under a sweep agent, take its parameters into ``args``."""
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
    )
    sweep_keys = set(run.config.keys()) if run.sweep_id else set()
    for key in sweep_keys:
        if not hasattr(args, key):
            raise ValueError(f"sweep parameter {key!r} is not a train.py option")
        value = run.config[key]
        setattr(args, key, tuple(value) if key == "crop" else value)
    apply_preset(args, ap, explicit=sweep_keys)

    if run.sweep_id:
        run.name = f"{run_name(args)}-{run.id}"
        if args.out is None:
            args.out = SWEEP_ROOT / run.sweep_id / run.name
    run.tags = tuple(
        set(run.tags or ())
        | {args.arch, args.encoder}
        | ({"registration_ok"} if registration_ok(args.encoder) else set())
    )
    run.config.update(
        {k: v for k, v in _plain(args).items() if k not in sweep_keys},
        allow_val_change=True,
    )
    for metric, mode in MONITORS.items():
        run.define_metric(metric, summary=mode)
    return run


def train(args, ap):
    run = None
    if args.no_wandb:
        apply_preset(args, ap, explicit=set())
    else:
        run = init_wandb(args, ap)
    if args.out is None:
        args.out = Path("checkpoints") / run_name(args)
    args.out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, workers=True)

    datamodule = ChoroidSegmentationDataModule(
        databases=args.databases,
        crop_size=tuple(args.crop),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = ChoroidSegmentationModule(
        arch=args.arch,
        encoder_name=args.encoder,
        encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_smooth=args.w_smooth,
        w_gap=args.w_gap,
        scheduler=args.scheduler,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    mode = MONITORS[args.monitor]
    checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        dirpath=args.out,
        filename=f"{{epoch:02d}}-{{{args.monitor}:.4f}}",
        save_top_k=1,
        mode=mode,
        save_last=True,
    )
    # Dice and the thickness error peak at different epochs: keep both
    thickness_checkpoint = ModelCheckpoint(
        monitor="thickness_mae_um",
        dirpath=args.out,
        filename="thickness-{epoch:02d}-{thickness_mae_um:.2f}",
        save_top_k=1,
        mode="min",
    )
    callbacks = [checkpoint, thickness_checkpoint]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor=args.monitor, mode=mode, patience=args.patience))
    logger = False
    if run is not None:
        logger = WandbLogger(experiment=run)
        callbacks.append(LearningRateMonitor("epoch"))

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate,
    )
    trainer.fit(model, datamodule=datamodule)

    def validate(ckpt_path):
        out = trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)[0]
        return {k: float(v) for k, v in out.items()}

    # Final weights first: validating a checkpoint loads it into ``model``
    last = validate(None)
    thickness_best = thickness_checkpoint.best_model_path
    by_thickness = validate(thickness_best) if thickness_best else {}
    # Best checkpoint on --monitor: what the sweep ranks on
    best = checkpoint.best_model_path
    val = validate(best or None)
    results = {
        "arch": args.arch,
        "encoder": args.encoder,
        "registration_ok": registration_ok(args.encoder),
        "best_checkpoint": best,
        "epochs_run": trainer.current_epoch,
        "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "val": val,
        "val_last": last,
        "val_best_thickness": by_thickness,
        "best_thickness_checkpoint": thickness_best,
        "args": _plain(args),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    if run is not None:
        run.summary.update({f"best/{k}": v for k, v in val.items()})
        run.summary.update({f"last/{k}": v for k, v in last.items()})
        run.summary.update({f"best_thickness/{k}": v for k, v in by_thickness.items()})
        run.summary.update(
            {
                "n_params_M": results["n_params_M"],
                "registration_ok": results["registration_ok"],
                "best_checkpoint": best,
            }
        )
        run.finish()
    print(json.dumps(val, indent=2))


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args(), parser)
