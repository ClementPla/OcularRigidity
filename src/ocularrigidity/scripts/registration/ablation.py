"""Ablation sweep over the registration regressor's architecture and objective."""

import argparse
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import torch

from ocularrigidity.registration.deep_learning.train_registration import OBJECTIVE_KEYS

REFERENCE: Dict[str, object] = {
    "epochs": 10,
    "batch-size": 4,
    "lr": 5e-4,
    "pairs-per-video": 120,
    "triplets-per-video": 40,
    "shifts-per-video": 40,
    "shift-max": 8,
    "dx-step": 8.0,
    "bm-scale": 0.5,
    "num-workers": 8,
    "videos-in-flight": 2,
    "seed": 0,
    "w-feature": 1.0,
    "w-photometric": 1.0,
    "w-lateral": 1.0,
    "w-bm": 0.1,
    "w-smooth": 0.05,
    "w-curvature": 0.02,
    "w-coverage": 0.5,
    "w-cycle": 0.3,
    "w-shift": 1.0,
}

EVAL_OBJECTIVE = {
    k: float(REFERENCE[k.replace("_", "-")]) for k in OBJECTIVE_KEYS
}

SEC_PER_TRAIN_STEP = 3.8
SEC_PER_VAL_STEP = 1.9
VRAM_GB_PER_ARM = 37
SEC_STARTUP_PER_ARM = 265

N_TRAIN_VIDEOS = 68
N_VAL_VIDEOS = 12

# : What the sweep actually runs with, as opposed to REFERENCE's record of what :
# reg_cascade_v9 used.
SWEEP_PAIRS_PER_VIDEO = 40


def hours_per_arm(pairs_per_video: int, batch: int, epochs: int) -> float:
    """Projected wall-clock of one arm."""
    train_steps = N_TRAIN_VIDEOS * pairs_per_video / batch
    val_steps = N_VAL_VIDEOS * pairs_per_video / batch
    per_epoch = train_steps * SEC_PER_TRAIN_STEP + val_steps * SEC_PER_VAL_STEP
    return (SEC_STARTUP_PER_ARM + epochs * per_epoch) / 3600.0


# : ``arm name -> flag overrides``.
ARMS: Dict[str, Dict[str, object]] = {
    "reference": {},
    # --- architecture ---
    "no_cascade": {"no-cascade": True},
    "no_cascade_no_corr": {"no-cascade": True, "no-correlation": True},
    # --- objective, one term at a time ---
    "no_feature": {"w-feature": 0.0},
    "no_photometric": {"w-photometric": 0.0},
    "no_lateral": {"w-lateral": 0.0},
    "no_bm": {"w-bm": 0.0},
    "no_smooth": {"w-smooth": 0.0},
    "no_curvature": {"w-curvature": 0.0},
    "no_coverage": {"w-coverage": 0.0},
    "no_cycle": {"w-cycle": 0.0},
    "no_shift": {"w-shift": 0.0},
}


def tags_for(arm: str) -> List[str]:
    """W&B tags for one arm: its name, plus what kind of ablation it is, so a sweep can be filtered down to just the objective or just the architecture."""
    if arm == "reference":
        return ["reference"]
    kind = "architecture" if "cascade" in arm or "corr" in arm else "objective"
    return [arm, kind]


def build_command(
    arm: str, overrides: Dict[str, object], args: argparse.Namespace
) -> List[str]:
    """The full ``train_registration`` invocation for one arm."""
    flags = dict(REFERENCE)
    flags.update(overrides)
    flags["epochs"] = args.epochs
    flags["pairs-per-video"] = args.pairs_per_video

    cmd = [
        args.python,
        "-u",
        "-m",
        "ocularrigidity.registration.deep_learning.train_registration",
        "--root",
        str(args.root),
        "--out",
        str(args.out / arm),
        # Score every arm on the reference objective, not its own.
        "--eval-objective",
        json.dumps(EVAL_OBJECTIVE),
        "--eval-baseline",
    ]
    for key, value in flags.items():
        if isinstance(value, bool):
            if value:  # store_true flags carry no argument
                cmd.append(f"--{key}")
        else:
            cmd += [f"--{key}", str(value)]

    if args.wandb_project:
        cmd += [
            "--wandb-project",
            args.wandb_project,
            "--wandb-name",
            arm,
            "--wandb-group",
            args.group,
            "--wandb-tags",
            *tags_for(arm),
        ]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
        if args.wandb_checkpoint:
            cmd.append("--wandb-checkpoint")
    return cmd


def check_gpus(gpus: List[str], required_gb: int) -> List[str]:
    """Free VRAM per requested gpu, as a list of human-readable problems."""
    problems = []
    for gpu in gpus:
        try:
            idx = int(gpu)
            free_b, total_b = torch.cuda.mem_get_info(idx)
        except Exception as exc:
            problems.append(f"gpu {gpu}: cannot be queried ({exc})")
            continue
        free_gb = free_b / 1024**3
        if free_gb < required_gb:
            problems.append(
                f"gpu {gpu}: {free_gb:.1f} GB free of {total_b / 1024**3:.1f} GB, "
                f"needs ~{required_gb} GB"
            )
    return problems


def run_arm(
    arm: str, cmd: List[str], gpu: str, out_dir: Path, log_dir: Path
) -> Dict[str, object]:
    """Run one arm to completion on ``gpu``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    started = time.time()
    print(f"[gpu {gpu}] {arm}: start  (log: {log_path})", flush=True)
    with open(log_path, "w") as fh:
        fh.write(shlex.join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    ok = proc.returncode == 0
    print(
        f"[gpu {gpu}] {arm}: {'done' if ok else f'FAILED rc={proc.returncode}'} "
        f"in {timedelta(seconds=int(elapsed))}",
        flush=True,
    )
    if ok:
        (out_dir / "DONE").write_text(f"{elapsed:.1f}\n")
    return {
        "arm": arm,
        "ok": ok,
        "returncode": proc.returncode,
        "seconds": elapsed,
        "log": str(log_path),
    }


def collect(out_root: Path) -> List[Dict[str, object]]:
    """Read back what each finished arm scored, from its checkpoint."""
    rows = []
    for arm in ARMS:
        ckpt = out_root / arm / "best.pt"
        if not ckpt.exists():
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        rows.append(
            {
                "arm": arm,
                "best_epoch": blob.get("epoch"),
                "val_total_ref": blob.get("val_loss"),
                "cascade": blob.get("cascade"),
                "use_correlation": blob.get("use_correlation"),
                "train_objective": blob.get("train_objective"),
                "checkpoint": str(ckpt),
            }
        )
    rows.sort(key=lambda r: (r["val_total_ref"] is None, r["val_total_ref"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/clement/Documents/data/OcularRigidity/Registration/Dataset/"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/ablation"),
        help="one subdirectory per arm, each with its own best.pt",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument(
        "--pairs-per-video",
        type=int,
        default=SWEEP_PAIRS_PER_VIDEO,
        help="the sweep's main cost knob, and the one thing worth turning down "
        "to fit a time budget: it sets how many frame pairs make an epoch, so "
        "wall-clock is linear in it. reg_cascade_v9 used "
        f"{REFERENCE['pairs-per-video']}; the sweep defaults to "
        f"{SWEEP_PAIRS_PER_VIDEO} to fit a weekend. Lowering it shortens every "
        "arm equally, so the arms stay comparable with each other -- only the "
        "comparison against v9's absolute numbers is lost.",
    )
    ap.add_argument(
        "--gpus",
        nargs="*",
        default=["0"],
        help="CUDA device ids; arms are spread across them, one at a time each",
    )
    ap.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help=f"subset to run (default: all). Available: {', '.join(ARMS)}",
    )
    ap.add_argument(
        "--skip-arms", nargs="*", default=[], help="arms to leave out"
    )
    ap.add_argument("--wandb-project", type=str, default="ocular-rigidity-reg-ablation")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument(
        "--group",
        type=str,
        default=time.strftime("ablation-%Y%m%d"),
        help="W&B group tying this sweep's runs together",
    )
    ap.add_argument(
        "--wandb-checkpoint",
        action="store_true",
        help="also upload each best.pt as a W&B artifact",
    )
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable W&B entirely (checkpoints and logs are still written)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run arms that already completed instead of skipping them",
    )
    ap.add_argument("--python", type=str, default="python")
    ap.add_argument(
        "--hours-per-arm",
        type=float,
        default=None,
        help="measured wall-clock of one arm, used only for the --dry-run estimate",
    )
    ap.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="start even if a gpu looks too full to hold an arm",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--collect-only",
        action="store_true",
        help="just summarise the arms already on disk",
    )
    args = ap.parse_args()

    if args.no_wandb:
        args.wandb_project = None

    if args.collect_only:
        rows = collect(args.out)
        print(json.dumps(rows, indent=2, default=str))
        return

    names = list(args.arms) if args.arms else list(ARMS)
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; available: {list(ARMS)}")
    names = [n for n in names if n not in args.skip_arms]

    pending = []
    for arm in names:
        arm_out = args.out / arm
        if (arm_out / "DONE").exists() and not args.force:
            print(f"{arm}: already done, skipping")
            continue
        pending.append(arm)

    if not pending:
        print("nothing to run.")
        return

    print(f"\n{len(pending)} arm(s) over {len(args.gpus)} gpu(s): {', '.join(pending)}")
    per_arm = args.hours_per_arm or hours_per_arm(
        args.pairs_per_video, int(REFERENCE["batch-size"]), args.epochs
    )
    waves = -(-len(pending) // len(args.gpus))  # ceil
    print(
        f"projected wall-clock: ~{waves * per_arm:.1f} h "
        f"({waves} wave(s) x ~{per_arm:.1f} h/arm, "
        f"{args.pairs_per_video} pairs/video x {args.epochs} epochs)"
    )
    print(
        f"each arm needs ~{VRAM_GB_PER_ARM} GB of VRAM; make sure every gpu in "
        f"{args.gpus} has that free (idle notebook kernels hold on to theirs)."
    )

    problems = check_gpus(args.gpus, VRAM_GB_PER_ARM)
    if problems:
        print("\nnot enough free VRAM:")
        for line in problems:
            print(f"  {line}")
        print(
            "  free the holders (idle notebook kernels are the usual cause) or "
            "pass --gpus with only the cards that are clear; --skip-vram-check "
            "overrides."
        )
        if not (args.dry_run or args.skip_vram_check):
            raise SystemExit(1)

    if args.dry_run:
        for arm in pending:
            print(f"\n--- {arm} ---")
            print(shlex.join(build_command(arm, ARMS[arm], args)))
        return

    work: "queue.Queue[str]" = queue.Queue()
    for arm in pending:
        work.put(arm)
    results: List[Dict[str, object]] = []
    lock = threading.Lock()
    log_dir = args.out / "logs"

    def worker(gpu: str) -> None:
        while True:
            try:
                arm = work.get_nowait()
            except queue.Empty:
                return
            arm_out = args.out / arm
            arm_out.mkdir(parents=True, exist_ok=True)
            res = run_arm(
                arm, build_command(arm, ARMS[arm], args), gpu, arm_out, log_dir
            )
            with lock:
                results.append(res)

    started = time.time()
    threads = [
        threading.Thread(target=worker, args=(g,), daemon=True) for g in args.gpus
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = timedelta(seconds=int(time.time() - started))
    failed = [r["arm"] for r in results if not r["ok"]]
    print(f"\nsweep finished in {total}. {len(results) - len(failed)} ok, "
          f"{len(failed)} failed" + (f": {failed}" if failed else ""))

    summary = {
        "group": args.group,
        "epochs": args.epochs,
        "eval_objective": EVAL_OBJECTIVE,
        "runs": results,
        "results": collect(args.out),
    }
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary -> {path}")

    rows = summary["results"]
    if rows:
        ref = next((r for r in rows if r["arm"] == "reference"), None)
        print(f"\n{'arm':<22} {'val/total (ref objective)':>26} {'delta':>10}")
        for r in rows:
            v = r["val_total_ref"]
            d = (
                f"{v - ref['val_total_ref']:+.4f}"
                if ref and ref["val_total_ref"] is not None and v is not None
                else "-"
            )
            print(f"{r['arm']:<22} {v:>26.4f} {d:>10}")


if __name__ == "__main__":
    main()
