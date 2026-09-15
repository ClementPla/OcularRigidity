# -*- coding: utf-8 -*-
"""
register_newmodel.py

Recale les acquisitions de repetabilite avec le RECALAGE APPRIS de Clement
(``ClementP/OCTVideoRegistration@cascade_v9``), et ecrit chaque resultat A COTE
DES DONNEES, a cote des trois variantes deja exportees :

    E:/SANSORI/Reproducibility/<SLUG>/RawImages/registered_cascade_v9[_<H>x<W>]/
        registered_video.mp4  mask.npz  timestamp.txt  transform.npz
        registration_params.json

Memes cinq noms de fichiers que ``export_registered_videos.py`` : l'aval relit
cette variante sans savoir laquelle c'est. La sortie est TOUJOURS a la resolution
native des B-scans, quel que soit le mode : les variantes restent comparables
image par image. Un suivi par acquisition est tenu sous
``E:/NASA_Rigidity/Reproducibility/SegmentationVariations/<variante>/registration_summary.csv``.

Ce que fait le recalage appris
------------------------------
``registration.fused.segment_and_register`` : UNE passe d'encodeur SegFormer
(``ChoroidSegmentationModule@version-2.0.0``) alimente a la fois la tete de
segmentation et un regresseur en cascade qui predit ``dx`` (lateral, un par
frame) et ``dy`` (axial, un par colonne) contre une frame de reference choisie
au milieu de la trajectoire axiale (``reference_selection="motion_medoid"``).
Aucun filtrage temporel n'est applique au transform, par choix de conception.
Le recaleur a ete entraine sur l'encodeur de CE modele de segmentation : les
deux vont par paire, la segmentation n'est donc pas le U-Net des autres
variantes.

La taille d'image : deux modes
------------------------------
Le regresseur a ete entraine sur des frames 1536 x 1024 ; les B-scans
Spectralis font 496 x 768.

  natif (defaut)     les images restent en 496 x 768 et ``reg_model.img_shape``
                     est fixe a cette taille. Mesure sur MODICA_GRAZIANA_OS3 :
                     NCC 0,545, 134 frames sur 348 sous 0,5 -- le modele invente
                     ~53 px de mouvement axial. A ne pas utiliser.

  --resize 1536x1024 le cube est AGRANDI (bilineaire) a la taille d'entrainement
                     et le modele tourne tel qu'entraine. Le transform predit sur
                     la grille agrandie est applique aux frames et masques NATIFS
                     par ``losses.warp`` lui-meme, qui convertit dx et dy de la
                     grille du modele vers celle des images (``sx = w / W``,
                     ``sy = h / H``, dy reechantillonne en largeur). Le masque
                     SegFormer est reduit par moyenne de surface puis seuille a
                     0,5. Mesure : NCC 0,786, aucune frame sous 0,5.

``RegistrationConfig.scale_factor`` n'y joue aucun role : il n'est lu que par le
recalage classique (verifie : resultat identique bit pour bit a 0,5, 1 et 2).

Mesures de qualite, dans le meme processus
------------------------------------------
Pour chaque acquisition, ``registration_metrics.ncc_to_median`` sur la sortie
recalee ET sur l'IDENTITE -- les frames natives et le masque SegFormer natif,
passes par le meme warp a transform nul et le meme filtre de colonnes. Les
B-scans Spectralis sont deja stabilises par l'appareil : sans cette reference,
rien ne dirait ce que le recalage ajoute.

Memoire GPU
-----------
Sous Windows, le pilote deborde en memoire partagee au lieu de lever un OOM :
avec le lot par defaut (8) a 1536 x 1024 sur 12 Go, 8,4 Go partages, et le calcul
rampe sans jamais echouer. En mode agrandi le lot vaut donc 2 par defaut
(6,9 Go, ~190 s par acquisition).

Comme les autres variantes : skip = drop = 0, cube complet, horodatages du XML.

Lancer (kernel pyOR, branche portant le recalage appris, depuis la racine) :
    python Reproducibility/register_newmodel.py --slug MODICA_GRAZIANA_OS3 --resize 1536x1024
    python Reproducibility/register_newmodel.py --all --resize 1536x1024
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import registration_metrics as rm  # noqa: E402
from ocularrigidity.data.io import save_mask  # noqa: E402
from ocularrigidity.pipeline_config import REGISTRATION  # noqa: E402
from ocularrigidity.registration.deep_learning.models.losses import (  # noqa: E402
    _resample_dy,
    warp,
)
from ocularrigidity.registration.fused import segment_and_register  # noqa: E402
from ocularrigidity.registration.postprocess import filter_bad_ascans_per_bms  # noqa: E402
from ocularrigidity.scripts.registration.astronauts import (  # noqa: E402
    build_cube_and_timestamps,
    estimate_fps,
    load_ordered_oct_series,
    write_gray_mp4,
)
from ocularrigidity.segmentation.utils import (  # noqa: E402
    get_choroid_segmentation_model,
    get_registration_model,
)

REPO = Path(__file__).resolve().parent.parent
PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
OUTPUT_SUBDIR = "registered_cascade_v9"
DEVICE = "cuda"
WARP_BATCH = 16
RESIZE_DEFAULT_FUSED_BATCH = 2

SEG_REPO, SEG_REVISION = "ClementP/ChoroidSegmentationModule", "version-2.0.0"
REG_REPO, REG_REVISION = "ClementP/OCTVideoRegistration", "cascade_v9"

RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")

REQUIRED_FILES = (
    "registered_video.mp4",
    "mask.npz",
    "timestamp.txt",
    "transform.npz",
    "registration_params.json",
)
# Meme garde-fou que export_registered_videos.py : au-dela, la sortie est rejetee
# meme si le moteur l'a rendue sans erreur.
MAX_EMPTY_MASK_FRAC = 0.5

SUMMARY_KEYS = [
    "slug", "status", "reason", "n_frames", "fps", "ref_idx", "ref_percentile",
    "fused_batch_size", "dx_ptp", "dx_std", "dy_bulk_ptp", "dy_bulk_std",
    "dy_intra_frame_std_median", "n_bad_columns", "n_empty_mask_frames",
    "ncc_reg_median", "ncc_reg_q1", "ncc_reg_q3", "ncc_reg_p5", "ncc_reg_n_lt_0.5",
    "ncc_identity_median", "ncc_identity_q1", "ncc_identity_q3", "ncc_identity_p5",
    "ncc_identity_n_lt_0.5", "seconds_total", "created",
]


def hub_sha(repo_id: str, revision: str) -> str:
    """SHA de commit du modele sur le Hub : une branche bouge, un SHA non."""
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo_id, revision=revision).sha)
    except Exception as e:  # noqa: BLE001
        return f"inconnu ({type(e).__name__})"


def git_state() -> dict:
    def run(*args):
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return r.stdout.strip()

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", "src")),
    }


def parse_shape(text: str) -> tuple[int, int]:
    h, w = (int(v) for v in text.lower().split("x"))
    return h, w


def list_slugs() -> list[str]:
    return sorted(
        d.name for d in PATH_GENERAL.iterdir()
        if d.is_dir() and RE_SLUG.match(d.name) and (d / "RawImages").is_dir()
    )


def read_params(out_dir: Path) -> dict | None:
    p = out_dir / "registration_params.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def is_complete(out_dir: Path) -> bool:
    """Cinq fichiers, statut ok ET mesures presentes : une sortie d'avant les
    mesures est refaite, pour que toutes les lignes du suivi se comparent."""
    if not all((out_dir / n).exists() for n in REQUIRED_FILES):
        return False
    params = read_params(out_dir)
    return bool(params) and params.get("status") == "ok" and "ncc_reg_median" in params


def patch_largest_cc() -> str:
    """``fused._encode_segment`` nettoie chaque masque par sa plus grande
    composante connexe avec ``keep_largest_connected_component_gpu``, qui
    importe cuCIM (RAPIDS). La passe de sonde l'appelle TOUJOURS, quel que soit
    ``config.keep_largest_cc``. Sans cuCIM, on lui substitue la version CPU
    (cc3d, par frame, 4-connexite), dont la docstring de la version GPU dit
    qu'elle est bit-identique -- meme resultat, seulement plus lent. Le
    remplacement vise le nom tel qu'importe DANS ``fused``, et rien d'autre.
    """
    try:
        import cucim  # noqa: F401

        return "cucim (GPU)"
    except ImportError:
        import ocularrigidity.registration.fused as fused_mod
        from ocularrigidity.segmentation.postprocess.blob import (
            keep_largest_connected_component,
        )

        def _largest_cc_cpu(masks: torch.Tensor) -> torch.Tensor:
            out = keep_largest_connected_component(masks.detach().cpu().numpy())
            return torch.from_numpy(out).to(masks.device)

        fused_mod.keep_largest_connected_component_gpu = _largest_cc_cpu
        return "cc3d (CPU, repli : cucim absent)"


@torch.inference_mode()
def resize_frames(cube: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(T, H, W) uint8 -> (T, *size) uint8, bilineaire."""
    out = np.empty((cube.shape[0], *size), dtype=np.uint8)
    for s in range(0, cube.shape[0], WARP_BATCH):
        e = min(s + WARP_BATCH, cube.shape[0])
        x = torch.from_numpy(cube[s:e]).to(DEVICE).float().unsqueeze(1)
        y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        out[s:e] = y.squeeze(1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out


@torch.inference_mode()
def resize_masks(masks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(T, h, w) bool -> (T, *size) bool : moyenne de surface, seuil 0,5."""
    out = np.empty((masks.shape[0], *size), dtype=bool)
    for s in range(0, masks.shape[0], WARP_BATCH):
        e = min(s + WARP_BATCH, masks.shape[0])
        x = torch.from_numpy(masks[s:e]).to(DEVICE).float().unsqueeze(1)
        y = F.interpolate(x, size=size, mode="area")
        out[s:e] = (y.squeeze(1) >= 0.5).cpu().numpy()
    return out


@torch.inference_mode()
def warp_native(cube, masks, dx_model, dy_model, model_shape):
    """Applique aux frames/masques NATIFS un transform predit sur ``model_shape``.

    ``losses.warp`` prend dx, dy en pixels de ``ref_shape`` et les ramene a la
    resolution de ce qu'il deforme : c'est la conversion utilisee par le
    regresseur pour sa propre pyramide, reprise telle quelle. Le masque voyage en
    canal 0 et le filtre de colonnes suit, comme dans ``segment_and_register``.
    """
    T, H, W = cube.shape
    frames_out = np.empty_like(cube)
    masks_out = np.empty_like(masks)
    dx_t = torch.from_numpy(np.ascontiguousarray(dx_model)).float()
    dy_t = torch.from_numpy(np.ascontiguousarray(dy_model)).float()
    for s in range(0, T, WARP_BATCH):
        e = min(s + WARP_BATCH, T)
        data = torch.stack([
            torch.from_numpy(masks[s:e]).to(DEVICE).float(),
            torch.from_numpy(cube[s:e]).to(DEVICE).float(),
        ], dim=1)
        out, _ = warp(data, dx_t[s:e].to(DEVICE), dy_t[s:e].to(DEVICE), model_shape)
        masks_out[s:e] = (out[:, 0] > 0.5).cpu().numpy()
        frames_out[s:e] = out[:, 1].clamp(0, 255).to(torch.uint8).cpu().numpy()
    bad = filter_bad_ascans_per_bms(masks_out)
    bad_np = bad.cpu().numpy() if hasattr(bad, "cpu") else np.asarray(bad)
    if bool(bad_np.any()):
        frames_out[..., bad_np] = 0
        masks_out[..., bad_np] = 0
    # Le meme transform exprime en pixels natifs, pour transform.npz.
    Hm, Wm = model_shape
    dx_native = (dx_t * (W / Wm)).numpy()
    dy_native = (_resample_dy(dy_t, W) * (H / Hm)).numpy()
    return frames_out, masks_out, bad_np.astype(bool), dx_native, dy_native


def run_fused(frames, seg_model, reg_model, config):
    """``segment_and_register``, en reduisant ``fused_batch_size`` sur OOM. Ne
    protege PAS du debordement en memoire partagee sous Windows (aucune
    exception n'est levee) : c'est le lot choisi en amont qui s'en charge."""
    sizes = [config.fused_batch_size] + [b for b in (4, 2, 1) if b < config.fused_batch_size]
    last = None
    for b in sizes:
        try:
            cfg = replace(config, fused_batch_size=b)
            return segment_and_register(frames, seg_model, reg_model, cfg,
                                        device=DEVICE, verbose=False), cfg
        except torch.cuda.OutOfMemoryError as e:
            last = e
            torch.cuda.empty_cache()
            print(f"  OOM avec fused_batch_size={b}, nouvel essai plus petit", flush=True)
    raise last


def process(slug, seg_model, reg_model, trained_shape, resize, config, cc_backend,
            subdir) -> dict:
    t_all = time.perf_counter()
    raw_dir = PATH_GENERAL / slug / "RawImages"
    out_dir = raw_dir / subdir

    series = load_ordered_oct_series(raw_dir)
    cube, ts_us = build_cube_and_timestamps(raw_dir, series)
    T, H, W = cube.shape

    model_shape = resize if resize else (H, W)
    reg_model.img_shape = model_shape
    model_frames = resize_frames(cube, model_shape) if resize else cube

    result, config = run_fused(model_frames, seg_model, reg_model, config)
    del model_frames
    dx_model = np.asarray(result.transform["dx"], dtype=np.float32)
    dy_model = np.asarray(result.transform["dy"], dtype=np.float32)

    # Le masque brut SegFormer, a la resolution native : sert au recalage en mode
    # agrandi ET a la reference identite dans les deux modes.
    native_raw_masks = (resize_masks(result.raw_masks, (H, W)) if resize
                        else result.raw_masks)
    if resize:
        reg_frames, reg_masks, bad, dx, dy = warp_native(
            cube, native_raw_masks, dx_model, dy_model, model_shape)
    else:
        reg_frames, reg_masks = result.registered_frames, result.registered_masks
        bad = np.asarray(result.transform["bad_columns"], dtype=bool)
        dx, dy = dx_model, dy_model
    dx = np.asarray(dx, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    ref_idx, ref_pct, timings = result.ref_idx, result.ref_percentile, result.timings
    del result
    gc.collect()

    # Identite : meme warp (transform nul), meme filtre de colonnes.
    id_frames, id_masks, _, _, _ = warp_native(
        cube, native_raw_masks, np.zeros(T, np.float32), np.zeros((T, W), np.float32), (H, W))
    metrics = {**rm.summarize(rm.ncc_to_median(reg_frames, reg_masks), "ncc_reg"),
               **rm.summarize(rm.ncc_to_median(id_frames, id_masks), "ncc_identity")}
    del id_frames, id_masks

    area = reg_masks.sum(axis=(1, 2))
    n_empty = int((area == 0).sum())
    status, reason = "ok", ""
    if reg_frames.shape != (T, H, W):
        status, reason = "error", f"forme {reg_frames.shape} != {(T, H, W)}"
    elif n_empty > MAX_EMPTY_MASK_FRAC * T:
        status, reason = "error", f"masque vide sur {n_empty}/{T} frames recalees"

    out_dir.mkdir(parents=True, exist_ok=True)
    fps = estimate_fps(ts_us)
    if status == "ok":
        write_gray_mp4(reg_frames, out_dir / "registered_video.mp4", fps)
        save_mask(reg_masks, out_dir / "mask.npz")
        (out_dir / "timestamp.txt").write_text(
            "\n".join(str(int(t)) for t in ts_us), encoding="utf-8")
        extra = {}
        if resize:
            # Les sorties brutes du modele, sur sa propre grille, en plus du
            # transform natif : de quoi refaire la conversion sans relancer.
            extra = dict(dx_model=dx_model, dy_model=dy_model,
                         model_shape=np.asarray(model_shape, dtype=np.int64))
        np.savez(out_dir / "transform.npz", dx=dx, dy=dy, bad_columns=bad,
                 ref_idx=np.int64(ref_idx),
                 ref_percentile=np.float32(ref_pct if ref_pct is not None else np.nan),
                 **extra)
    else:
        for name in REQUIRED_FILES[:-1]:
            (out_dir / name).unlink(missing_ok=True)

    params = {
        "variant": subdir,
        "slug": slug,
        "condition": slug,
        "status": status,
        "reason": reason,
        "engine": "ocularrigidity.registration.fused.segment_and_register",
        "seg_repo": SEG_REPO,
        "seg_revision": SEG_REVISION,
        "seg_commit_sha": hub_sha(SEG_REPO, SEG_REVISION),
        "reg_repo": REG_REPO,
        "reg_revision": REG_REVISION,
        "reg_commit_sha": hub_sha(REG_REPO, REG_REVISION),
        "reg_img_shape_trained": list(trained_shape),
        "reg_img_shape_used": list(model_shape),
        "mode": "resize" if resize else "native",
        "resize": None if not resize else {
            "model_input_shape": list(model_shape),
            "native_shape": [H, W],
            "frames_upsample": "F.interpolate bilinear, align_corners=False",
            "mask_downsample": "F.interpolate area, seuil >= 0.5",
            "transform_to_native": "losses.warp (sx = W/Wm, sy = H/Hm, _resample_dy)",
        },
        "registration_config": asdict(config),
        "fused_batch_size": config.fused_batch_size,
        "largest_cc_backend": cc_backend,
        "skip_first_n_frames": 0,
        "drop_last_n_frames": 0,
        "n_frames": T,
        "height": H,
        "width": W,
        "fps": round(float(fps), 3),
        "ref_idx": int(ref_idx),
        "ref_percentile": ref_pct,
        "dx_min": float(dx.min()),
        "dx_max": float(dx.max()),
        "dx_ptp": float(np.ptp(dx)),
        "dx_std": float(dx.std()),
        "dy_bulk_ptp": float(np.ptp(dy.mean(axis=1))),
        "dy_bulk_std": float(dy.mean(axis=1).std()),
        "dy_intra_frame_std_median": float(np.median(dy.std(axis=1))),
        "n_bad_columns": int(bad.sum()),
        "n_empty_mask_frames": n_empty,
        **metrics,
        "metric_definition": "registration_metrics.ncc_to_median",
        "timings_s": {k: round(float(v), 2) for k, v in timings.items()},
        "seconds_total": round(time.perf_counter() - t_all, 1),
        "git": git_state(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "registration_params.json").write_text(
        json.dumps(params, indent=2, default=str), encoding="utf-8")
    return params


def write_summary(subdir: str, slugs: list[str]) -> Path:
    """Relit les JSON de toutes les acquisitions : le suivi reflete le disque,
    y compris les sorties d'un lot precedent."""
    rows = []
    for slug in slugs:
        params = read_params(PATH_GENERAL / slug / "RawImages" / subdir)
        if params:
            rows.append({k: params.get(k) for k in SUMMARY_KEYS} | {"slug": slug})
    out = SEGVAR_ROOT / subdir.removeprefix("registered_") / "registration_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_KEYS).to_csv(out, index=False)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recalage appris cascade_v9.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--slug", action="append", help="ex. MODICA_GRAZIANA_OS3 (repetable)")
    target.add_argument("--all", action="store_true", help="toutes les acquisitions")
    parser.add_argument("--resize", default=None,
                        help="agrandit le cube a HxW avant le modele, ex. 1536x1024")
    parser.add_argument("--fused-batch", type=int, default=None,
                        help=f"impose fused_batch_size (defaut {RESIZE_DEFAULT_FUSED_BATCH} "
                             "en mode agrandi)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("Le recalage appris exige CUDA (autocast bfloat16 sur 'cuda').")
        return 2
    resize = parse_shape(args.resize) if args.resize else None
    subdir = OUTPUT_SUBDIR + (f"_{resize[0]}x{resize[1]}" if resize else "")
    all_slugs = list_slugs()
    slugs = all_slugs if args.all else args.slug
    unknown = [s for s in slugs if s not in all_slugs]
    if unknown:
        print(f"Acquisition(s) introuvable(s) : {', '.join(unknown)}")
        return 2
    todo = [s for s in slugs
            if args.overwrite or not is_complete(PATH_GENERAL / s / "RawImages" / subdir)]
    print(f"{len(slugs)} acquisition(s), {len(slugs) - len(todo)} deja complete(s), "
          f"{len(todo)} a traiter -> RawImages/{subdir}/", flush=True)

    config = REGISTRATION
    batch = args.fused_batch or (RESIZE_DEFAULT_FUSED_BATCH if resize else None)
    if batch:
        config = replace(config, fused_batch_size=batch)
    if todo:
        print(f"modeles : {SEG_REPO}@{SEG_REVISION}, {REG_REPO}@{REG_REVISION} | "
              f"mode {'agrandi ' + args.resize if resize else 'natif'} | "
              f"fused_batch_size {config.fused_batch_size}", flush=True)
        seg_model = get_choroid_segmentation_model().to(DEVICE)
        reg_model = get_registration_model().to(DEVICE)
        trained_shape = tuple(int(v) for v in reg_model.img_shape)
        cc_backend = patch_largest_cc()
        print(f"plus grande composante connexe : {cc_backend}\n", flush=True)
        torch.backends.cudnn.benchmark = True

    t_batch = time.perf_counter()
    n_ok = n_err = 0
    for i, slug in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            p = process(slug, seg_model, reg_model, trained_shape, resize, config,
                        cc_backend, subdir)
        except Exception as e:  # noqa: BLE001 - une acquisition ne doit pas tout arreter
            traceback.print_exc()
            out_dir = PATH_GENERAL / slug / "RawImages" / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            p = {"variant": subdir, "slug": slug, "status": "error",
                 "reason": f"{type(e).__name__}: {str(e)[:300]}",
                 "created": datetime.now().isoformat(timespec="seconds")}
            (out_dir / "registration_params.json").write_text(
                json.dumps(p, indent=2), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
        dt = time.perf_counter() - t0
        if p.get("status") == "ok":
            n_ok += 1
            print(f"[{i:>2}/{len(todo)}] {slug:28} ok | NCC recale {p['ncc_reg_median']:.3f} "
                  f"(identite {p['ncc_identity_median']:.3f}) | <0,5 : {p['ncc_reg_n_lt_0.5']:3d} | "
                  f"dx ptp {p['dx_ptp']:5.2f} | dy global ptp {p['dy_bulk_ptp']:5.2f} | "
                  f"{dt:.0f} s", flush=True)
        else:
            n_err += 1
            print(f"[{i:>2}/{len(todo)}] {slug:28} {p.get('status')} : {p.get('reason')} | "
                  f"{dt:.0f} s", flush=True)
        write_summary(subdir, all_slugs)

    out = write_summary(subdir, all_slugs)
    print(f"\n{n_ok} ok, {n_err} en echec ({(time.perf_counter() - t_batch) / 60:.1f} min)")
    print(f"suivi : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
