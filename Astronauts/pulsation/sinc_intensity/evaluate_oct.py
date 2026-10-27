# -*- coding: utf-8 -*-
"""
Evaluation d'un SiNC entraine sur les crops de choroide OCT (``datasets/OCT.py``).

Pour chaque video de ``datasets/metadata/OCT.csv`` (train, val ET test) :
* clips de ``fpc`` frames au pas ``step`` (les memes que l'entrainement), 7 patchs
  64 x 64 cote a cote, chacun centre-reduit comme a l'entrainement ;
* onde par patch = overlap-add (fenetre de Hann) des clips (``utils/postprocess.overlap_add``),
  onde de la video = overlap-add de la moyenne des ondes standardisees des patchs ;
* FC = pic du periodogramme (``utils/postprocess.estimate_bpm``) dans la bande
  ``--bpm``, a la VRAIE cadence de la video (9,52 ou 9,62 Hz : pas de reechantillonnage) ;
* references : ``hr_BPM`` (consensus par personne) et ``hr_svd_BPM`` (par acquisition)
  de ``hr_prior.csv``, et la FC du pouls d'EPAISSEUR (``pulse_compute.mask_thickness``
  sur le masque cascade_v9 plex_full, colonnes du crop) -- un controle independant de
  l'intensite. Correlation onde / epaisseur filtree a decalage nul.

Sorties dans ``<experiment>/predictions/`` : ``<slug>.npz`` et ``predictions.csv``.

    python evaluate_oct.py --experiment E:/NASA_Rigidity/Reproducibility/sinc_intensity/first_training/fold0_seed0
"""

import argparse
import json
import sys
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas as pd
import torch

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "Astronauts" / "data_visualizer_GUI"))

import pulse_compute  # noqa: E402
from datasets.OCT import normalize_clip  # noqa: E402
from models.PhysNet import PhysNet  # noqa: E402
from models.PhysNetRPPGEncoder import PhysNetRPPGEncoder  # noqa: E402
from utils.postprocess import estimate_bpm, overlap_add  # noqa: E402

from ocularrigidity.data.io import load_mask_frames  # noqa: E402

PATCH = 64
HR_PRIOR = Path("E:/NASA_Rigidity/Reproducibility/hr_prior.csv")
REG = Path("E:/SANSORI/Reproducibility/registered_cascade_v9_plex_full")
NFFT = 5400


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def periodogram(wave, fps, bpm):
    win = signal.get_window("hamming", wave.shape[0])
    f, p = signal.periodogram(wave, window=win, fs=fps, nfft=NFFT)
    keep = (f * 60 >= bpm[0]) & (f * 60 <= bpm[1])
    return f[keep] * 60, p[keep]


def mask_thickness_chunked(path, c0, c1, axial_um, chunk=32):
    """``pulse_compute.mask_thickness`` sur les colonnes c0:c1, masque decode par
    paquets de frames (le masque entier, ~0,5 Go, ne tient pas en RAM pendant un
    entrainement) : meme regle de couverture (>= MIN_COL_COVERAGE des frames)."""
    T = int(np.load(path)["shape"][0])
    colsum, has = [], []
    for i in range(0, T, chunk):
        m = load_mask_frames(path, list(range(i, min(T, i + chunk))))[:, :, c0:c1]
        colsum.append(m.sum(axis=1, dtype=np.float64))
        has.append(m.any(axis=1))
    colsum, has = np.concatenate(colsum), np.concatenate(has)  # (T, nc)
    cover = has.mean(axis=0) >= pulse_compute.MIN_COL_COVERAGE
    if not cover.any():
        cover = has.any(axis=0)
    thick = colsum[:, cover].mean(axis=1) * axial_um
    thick[colsum.sum(axis=1) == 0] = np.nan
    return thick


def thickness_pulse(slug, c0, c1, n, fps, bpm):
    """(epaisseur brute (n,), filtree (n,), FC) sur les colonnes du crop."""
    d = REG / slug
    params = json.loads((d / "registration_params.json").read_text(encoding="utf-8"))
    th = mask_thickness_chunked(d / "mask.npz", c0, c1, params["plex_axial_um"])[:n]
    ok = np.isfinite(th)
    th = np.interp(np.arange(th.size), np.flatnonzero(ok), th[ok])
    b, a = signal.butter(2, [bpm[0] / 60, bpm[1] / 60], btype="band", fs=fps)
    th_bp = signal.filtfilt(b, a, th - th.mean())
    hr = estimate_bpm(th - th.mean(), fps=fps, low_hz=bpm[0] / 60, high_hz=bpm[1] / 60)
    return th, th_bp, hr


def build_model(model_type):
    if model_type == "physnet":
        return PhysNet(input_channels=1)
    if model_type == "physnet_rppg_enc":
        # Architecture seule : les poids viennent du checkpoint de train.py.
        return PhysNetRPPGEncoder(input_channels=1, pretrained=None, drop_p=0.0)
    raise ValueError(f"model_type inconnu : {model_type!r}")


@torch.no_grad()
def predict_video(model, video, fpc, step, device, input_norm="zscore"):
    """(onde de la video (L,), ondes par patch (n_patch, L)).

    Entree ``diffnorm`` : le reseau sort la DERIVEE du pouls (convention de
    rPPG-Toolbox) ; chaque onde de patch est integree (somme cumulee, detendance
    lineaire) avant d'etre combinee. La FC est la meme, la phase devient celle du pouls."""
    T, H, W = video.shape
    if H < PATCH:
        video = np.pad(video, ((0, 0), (0, PATCH - H), (0, 0)), mode="edge")
    n_patch = W // PATCH
    x0 = (W - n_patch * PATCH) // 2
    starts = list(range(0, T - fpc + 1, step))
    per_patch = [[] for _ in range(n_patch)]
    for s in starts:
        clips = np.stack([video[s:s + fpc, :, x0 + k * PATCH:x0 + (k + 1) * PATCH]
                          for k in range(n_patch)]).astype(np.float32)  # (P, T, 64, 64)
        x = torch.stack([normalize_clip(torch.from_numpy(c)[None], input_norm) for c in clips])
        x = x.to(device)  # (P, 1, T, 64, 64)
        y = model(x).cpu().numpy()  # (P, T)
        for k in range(n_patch):
            per_patch[k].append(y[k])
    patches = np.stack([overlap_add(np.stack(w), fpc, step) for w in per_patch])  # (P, L)
    if input_norm == "diffnorm":
        patches = np.stack([signal.detrend(np.cumsum(p)) for p in patches])
    # Signe libre par patch : on aligne chaque patch sur la moyenne avant de moyenner.
    ref = patches.mean(axis=0)
    signs = np.sign([np.corrcoef(p, ref)[0, 1] or 1.0 for p in patches])
    patches = patches * signs[:, None]
    wave = standardize(np.mean([standardize(p) for p in patches], axis=0))
    return wave, patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="dossier fold<K>_seed<s> de train.py")
    ap.add_argument("--metadata", default=str(HERE / "datasets" / "metadata" / "OCT.csv"))
    ap.add_argument("--fpc", type=int, default=95)
    ap.add_argument("--step", type=int, default=48)
    ap.add_argument("--bpm", type=float, nargs=2, default=(40.0, 120.0))
    ap.add_argument("--model_type", default="physnet", choices=("physnet", "physnet_rppg_enc"))
    ap.add_argument("--input_norm", default="zscore", choices=("zscore", "diffnorm"))
    ap.add_argument("--model", default=None, help="defaut : best_saved_models/<model_type>")
    args = ap.parse_args()
    args.model = args.model or f"best_saved_models/{args.model_type}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp = Path(args.experiment)
    # Checkpoint ecrit par notre train.py (perte = scalaire numpy) : weights_only=False.
    ckpt = torch.load(exp / args.model, map_location=device, weights_only=False)
    model = build_model(args.model_type).to(device).float()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"modele : {exp / args.model} (epoque {ckpt['epoch']})")

    meta = pd.read_csv(args.metadata)
    prior = pd.read_csv(HR_PRIOR).set_index("slug")
    out_dir = exp / "predictions"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for _, r in meta.iterrows():
        npz = np.load(r["path"])
        video = npz["video"].astype(np.float32)
        fps = float(npz["fps"])
        c0, c1 = int(npz["c0"]), int(npz["c1"])
        wave, patches = predict_video(model, video, args.fpc, args.step, device, args.input_norm)
        L = wave.size
        t = npz["t_s"][:L]
        lo, hi = args.bpm[0] / 60, args.bpm[1] / 60
        hr = estimate_bpm(wave, fps=fps, low_hz=lo, high_hz=hi)
        hr_patches = np.array([estimate_bpm(standardize(p), fps=fps, low_hz=lo, high_hz=hi) for p in patches])
        f_bpm, psd = periodogram(wave, fps, args.bpm)
        th, th_bp, hr_th = thickness_pulse(r["slug"], c0, c1, L, fps, args.bpm)
        b, a = signal.butter(2, [lo, hi], btype="band", fs=fps)
        wave_bp = signal.filtfilt(b, a, wave)
        r_wave_th = float(np.corrcoef(wave_bp, th_bp)[0, 1])
        np.savez_compressed(out_dir / f"{r['slug']}.npz", t_s=t, fps=fps, wave=wave, wave_bp=wave_bp,
                            patches=patches.astype(np.float32), hr_patches=hr_patches,
                            psd_bpm=f_bpm, psd=psd, thickness=th, thickness_bp=th_bp)
        p = prior.loc[r["slug"]] if r["slug"] in prior.index else {}
        row = {"slug": r["slug"], "participant": r["participant"], "replicate": r["replicate"],
               "split": r["split"], "fps": fps, "n_frames": L,
               "hr_sinc": hr, "hr_patch_median": float(np.median(hr_patches)),
               "hr_patch_min": float(hr_patches.min()), "hr_patch_max": float(hr_patches.max()),
               "hr_thickness": hr_th, "hr_prior_BPM": p.get("hr_BPM", np.nan),
               "hr_svd_BPM": p.get("hr_svd_BPM", np.nan), "r_wave_thickness": r_wave_th}
        rows.append(row)
        print(f"{r['slug']:<26} {r['split']:<5} SiNC {hr:5.1f} | patchs {hr_patches.min():5.1f}-{hr_patches.max():5.1f}"
              f" | epaisseur {hr_th:5.1f} | prior {row['hr_prior_BPM']:5.1f} | svd {row['hr_svd_BPM']:5.1f}"
              f" | r(onde, ep.) {r_wave_th:+.2f}")
    df = pd.DataFrame(rows)
    df.attrs = {}
    df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "evaluate_args.json").write_text(json.dumps({**vars(args), "epoch": int(ckpt["epoch"])},
                                                           indent=2), encoding="utf-8")
    print("->", out_dir / "predictions.csv")


if __name__ == "__main__":
    main()
