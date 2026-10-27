"""W&B sweeps for the choroid segmentation model.

The sweeps are defined in ``scripts/segmentation/sweeps/*.yaml`` (stage 1:
``architecture``, every model preset with one recipe; stage 2: ``hparams``, a
Bayesian search on the winner's recipe). Each run is
``scripts/segmentation/train.py`` started by ``wandb agent``, which reads its
parameters from the run config.

    # Create the sweep on W&B, prints its id
    python -m ocularrigidity.scripts.segmentation.sweep create architecture

    # One agent per GPU; for a grid, agents stop when the grid is done
    python -m ocularrigidity.scripts.segmentation.sweep agent <sweep id> --gpus 0 1

    # Results table, from the W&B API (also written to summary.csv)
    python -m ocularrigidity.scripts.segmentation.sweep summary <sweep id>

The plain ``wandb sweep <yaml>`` / ``wandb agent <id>`` commands work too, run
from the repository root.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from ocularrigidity.scripts.registration.ablation import check_gpus
from ocularrigidity.scripts.segmentation.train import SWEEP_ROOT

SWEEPS_DIR = Path(__file__).resolve().parent / "sweeps"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROJECT = "ocular-rigidity-seg-sweep"
VRAM_GB_PER_RUN = 24  # rough upper bound at batch 8, 1024x768, bf16


def load_config(name: str) -> dict:
    path = Path(name)
    if not path.exists():
        path = SWEEPS_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_entity(args) -> str:
    """``--entity``, else the account's default (``wandb agent`` refuses no entity)."""
    if not args.entity:
        import wandb

        args.entity = wandb.Api().default_entity
        if not args.entity:
            raise SystemExit("no default W&B entity: pass --entity")
    return args.entity


def create(args) -> None:
    import wandb

    config = load_config(args.config)
    resolve_entity(args)
    sweep_id = wandb.sweep(config, project=args.project, entity=args.entity)
    print(f"\nsweep id: {sweep_id}")
    print(f"next: python -m ocularrigidity.scripts.segmentation.sweep agent {sweep_id} --gpus 0 1")


def agent(args) -> None:
    problems = check_gpus(args.gpus, VRAM_GB_PER_RUN)
    if problems and not args.skip_vram_check:
        print("not enough free VRAM:\n  " + "\n  ".join(problems))
        print("pass --gpus with the free cards only, or --skip-vram-check.")
        raise SystemExit(1)

    cmd = [
        sys.executable, "-m", "wandb", "agent",
        "--project", args.project,
        "--entity", resolve_entity(args),
    ]
    if args.count:
        cmd += ["--count", str(args.count)]
    cmd.append(args.sweep_id)

    log_dir = SWEEP_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu in args.gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log = open(log_dir / f"agent_{args.sweep_id.split('/')[-1]}_gpu{gpu}.log", "a")
        print(f"[gpu {gpu}] agent started (log: {log.name})")
        # From the repo root: the sweep command and the checkpoint paths are relative to it
        procs.append(
            (gpu, log, subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT))
        )
    for gpu, log, proc in procs:
        proc.wait()
        log.close()
        print(f"[gpu {gpu}] agent exited with code {proc.returncode}")


def summary(args) -> None:
    import pandas as pd
    import wandb

    api = wandb.Api()
    path = args.sweep_id
    if path.count("/") == 0:
        path = f"{resolve_entity(args)}/{args.project}/{path}"
    sweep = api.sweep(path)
    rows = []
    for run in sweep.runs:
        cfg, s = run.config, run.summary
        rows.append(
            {
                "run": run.name,
                "state": run.state,
                "model": cfg.get("model"),
                "params_M": s.get("n_params_M"),
                "dice": s.get("best/dice"),
                "thickness_mae_um": s.get("best/thickness_mae_um"),
                "csi_mae_um": s.get("best/csi_mae_um"),
                "registration_ok": s.get("registration_ok"),
            }
        )
    if not rows:
        print("no runs yet")
        return
    metric = sweep.config.get("metric", {})
    key = metric.get("name", "best/dice").removeprefix("best/")
    df = pd.DataFrame(rows).sort_values(key, ascending=metric.get("goal") == "minimize")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = SWEEP_ROOT / sweep.id / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwritten to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1],
    )
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--entity", default=None)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create a sweep from a yaml")
    p.add_argument("config", help=f"a yaml path or a name in {SWEEPS_DIR.name}/")
    p.set_defaults(func=create)

    p = sub.add_parser("agent", help="run one agent per gpu")
    p.add_argument("sweep_id")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--count", type=int, default=None, help="runs per agent (default: until done)")
    p.add_argument("--skip-vram-check", action="store_true")
    p.set_defaults(func=agent)

    p = sub.add_parser("summary", help="results table of a sweep")
    p.add_argument("sweep_id")
    p.set_defaults(func=summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
