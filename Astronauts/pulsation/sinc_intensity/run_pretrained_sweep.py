# -*- coding: utf-8 -*-
"""
Balayage : encodeur PhysNet de rPPG-Toolbox, preentraine ou non, x deux entrees, pertes SiNC.

    5 poids publies (final_model_release/*PhysNet*.pth) + 1 controle sans preentrainement
    x  entree zscore (intensite) / diffnorm (DiffNormalized de rPPG-Toolbox)
    =  12 entrainements de EPOCHS epoques, puis evaluate_oct.py sur les 23 videos.

Memes donnees, roles, clips, patchs et pertes que le premier entrainement
(``first_training``). Encodeur a lr x ENCODER_LR_SCALE, tete a lr.

Chaque entrainement : ``E:/NASA_Rigidity/Reproducibility/sinc_intensity/pretrained_<poids>_<entree>/``
(``train.log``, ``eval.log``, ``fold0_seed0/...``). Un entrainement deja evalue est saute ; un
dossier inacheve est renomme ``*_incomplete_<horodatage>`` puis relance.
Synthese : ``.../sinc_intensity/pretrained_sweep.csv`` (+ la ligne ``first_training``).

    python run_pretrained_sweep.py            # depuis Astronauts/pulsation/sinc_intensity
    python run_pretrained_sweep.py --summary  # synthese seule
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = Path("E:/NASA_Rigidity/Reproducibility/sinc_intensity")
WEIGHTS = Path("E:/NASA_Rigidity/models/rppg_toolbox")
PRETRAINED = {
    "scamps": "SCAMPS_PhysNet_DiffNormalized.pth",
    "pure": "PURE_PhysNet_DiffNormalized.pth",
    "ubfc": "UBFC-rPPG_PhysNet_DiffNormalized.pth",
    "maubfc": "MA-UBFC_physnet.pth",
    "bp4d": "BP4D_PseudoLabel_PhysNet_DiffNormalized.pth",
    "scratch": None,
}
INPUTS = ("diffnorm", "zscore")
EPOCHS = 60
ENCODER_LR_SCALE = 0.1
MODEL = "physnet_rppg_enc"
COMMON = ["--dataset", "oct_unsupervised", "--channels", "n", "--fpc", "95", "--step", "48",
          "--batch_size", "16", "--fps", "9.57", "--low_hz", "0.667", "--high_hz", "2.0",
          "--augmentation", "sfrg", "--num_workers", "0", "--lr", "0.0001"]
TOL = 0.10


def run_dir(weights, inp):
    return ROOT / f"pretrained_{weights}_{inp}"


def run(cmd, log):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode


def one(weights, inp):
    d = run_dir(weights, inp)
    exp = d / "fold0_seed0"
    if (exp / "predictions" / "predictions.csv").exists():
        print(f"{d.name} : deja fait")
        return
    done = (exp / "best_saved_models" / MODEL).exists() and \
        (exp / "losses_val.csv").exists() and len(pd.read_csv(exp / "losses_val.csv")) >= EPOCHS
    if not done:
        if d.exists():
            d.rename(d.with_name(f"{d.name}_incomplete_{datetime.now():%Y%m%d_%H%M%S}"))
        d.mkdir(parents=True)
        pre = str(WEIGHTS / PRETRAINED[weights]) if PRETRAINED[weights] else "none"
        t0 = time.time()
        rc = run([sys.executable, "-u", "train.py", "--model_type", MODEL, "--pretrained", pre,
                  "--input_norm", inp, "--encoder_lr_scale", str(ENCODER_LR_SCALE),
                  "--epochs", str(EPOCHS), "--experiment_root", str(d), *COMMON], d / "train.log")
        print(f"{d.name} : entrainement rc={rc} en {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            return
    rc = run([sys.executable, "-u", "evaluate_oct.py", "--experiment", str(exp), "--model_type", MODEL,
              "--input_norm", inp], d / "eval.log")
    print(f"{d.name} : evaluation rc={rc}")


def metrics(name, exp, weights, inp):
    p = pd.read_csv(exp / "predictions" / "predictions.csv")
    v = pd.read_csv(exp / "losses_val.csv")
    ok = (p["hr_sinc"] - p["hr_prior_BPM"]).abs() <= TOL * p["hr_prior_BPM"]
    row = {"run": name, "pretrained": weights, "input": inp,
           "epochs": len(v), "best_epoch": int(v.loc[v["total"].idxmin(), "epoch"]),
           "val_loss_min": float(v["total"].min()),
           "hr_sd": float(p["hr_sinc"].std()), "hr_min": float(p["hr_sinc"].min()),
           "hr_max": float(p["hr_sinc"].max()),
           "spearman_prior": float(spearmanr(p["hr_sinc"], p["hr_prior_BPM"]).statistic),
           "mae_prior": float((p["hr_sinc"] - p["hr_prior_BPM"]).abs().mean()),
           "n_10pc_all": int(ok.sum()),
           "abs_r_wave_thickness_median": float(p["r_wave_thickness"].abs().median())}
    for s in ("train", "val", "test"):
        row[f"n_10pc_{s}"] = int(ok[p["split"] == s].sum())
    for r in p[p["split"] == "test"].sort_values("slug").itertuples():
        row[f"hr_{r.slug.split('_')[-1]}"] = round(float(r.hr_sinc), 1)
    return row


def summary():
    rows = []
    ref = ROOT / "first_training" / "fold0_seed0"
    if (ref / "predictions" / "predictions.csv").exists():
        rows.append(metrics("first_training (PhysNet SiNC, 200 ep.)", ref, "aucun", "zscore"))
    for w in PRETRAINED:
        for inp in INPUTS:
            exp = run_dir(w, inp) / "fold0_seed0"
            if (exp / "predictions" / "predictions.csv").exists():
                rows.append(metrics(run_dir(w, inp).name, exp, w, inp))
    df = pd.DataFrame(rows)
    out = ROOT / "pretrained_sweep.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string(index=False))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if not args.summary:
        for w in PRETRAINED:
            for inp in INPUTS:
                one(w, inp)
    summary()


if __name__ == "__main__":
    main()
