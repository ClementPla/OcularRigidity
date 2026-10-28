# -*- coding: utf-8 -*-
"""
Segmentation SegFormer + recalage cascade_v9 sur la GRILLE PLEX -- moteur partage
par ``Astronauts/register_cascade_v9_plex.py`` et
``Reproducibility/register_cascade_v9_plex.py``.

Pourquoi la grille PLEX
-----------------------
Les deux reseaux ont ete entraines sur des B-scans PLEX (1,95 um x 5,9 um par
pixel). La variante precedente (``registered_cascade_v9_1536x1024``) etirait le
B-scan Spectralis a 1536 x 1024, soit x3,1 en profondeur et x1,3 en largeur :
une choroide aux mauvaises proportions. ``registration.plex_grid.to_plex_grid``
reechantillonne a la taille de pixel PLEX exacte (ScaleX / ScaleY lus dans le
XML, PAR CONDITION), puis recadre au centre.

Variantes, chacune dans son dossier (a cote des donnees : ``RawImages/<variante>/``
sur la cohorte SANS, ``<racine>/<variante>/<slug>/`` sur la repetabilite -- c'est
``out_dir_of`` qui le dit)
-------------------------------------------------------------------------
    registered_cascade_v9_plex_w1024          grille PLEX, 1024 colonnes au centre (~6 mm)
    registered_cascade_v9_plex_w1024_native   la meme passe ramenee a 496 x 768
    registered_cascade_v9_plex_full           grille PLEX, toute la largeur (~1500 colonnes)
    registered_cascade_v9_plex_full_native    la meme passe ramenee a 496 x 768

Une passe du reseau par largeur (w1024, full) ; la version ``_native`` de chacune
n'est pas une nouvelle passe : son masque, ses probabilites et son transform
sont ceux de la passe PLEX, ramenes a la grille native (``plex_grid``) et le
transform est applique aux frames NATIVES (aucun double reechantillonnage des
frames). Colonnes natives hors du recadrage : vides (frames, masques,
probabilites a 0) et marquees dans ``bad_columns``.

Fichiers de chaque dossier : les cinq habituels (``registered_video.mp4``,
``mask.npz``, ``timestamp.txt``, ``transform.npz``, ``registration_params.json``)
-- l'aval relit une variante sans savoir laquelle c'est -- plus ``prob.npz`` :
probabilites de segmentation recalees, uint8 (0..255 <-> 0..1), AVANT le filtre
de plus grande composante, compressees zstd (``save_probs`` / ``load_probs``).
Elles permettent de regler seuil et lissage plus tard sans relancer le lot.

Jeux de modeles (``MODELS``)
----------------------------
Le nom du moteur est historique : il recale avec n'importe quel couple
(segmentation, recalage) de ``MODELS``, chacun sous son propre prefixe de dossier.

    cascade_v9  SegFormer version-2.0.0 + cascade_v9         registered_cascade_v9_plex_*
    upernet     UPerNet ConvNeXt-t version-3.0.0
                + reg_upernet_convnext_t_bmlines              registered_upernet_plex_*

``transform.npz`` porte aussi ``frame_iou`` : IoU de chaque masque recale avec
celui de la reference (``RegistrationConfig.identify_failed_frames``).
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zstandard as zstd

from ocularrigidity.data.io import save_mask
from ocularrigidity.registration.deep_learning.models.losses import warp
from ocularrigidity.registration.fused import segment_and_register
from ocularrigidity.registration.plex_grid import (
    PLEX_WIDTH,
    PlexGeometry,
    native_coverage,
    plex_to_native,
    to_plex_grid,
    transform_to_native,
)
from ocularrigidity.registration.postprocess import filter_bad_ascans_per_bms
from ocularrigidity.scripts.registration.astronauts import (
    build_cube_and_timestamps,
    estimate_fps,
    load_ordered_oct_series,
    write_gray_mp4,
)

DEVICE = "cuda"
WARP_BATCH = 16
# largeur -> fused_batch_size. Sous Windows, un lot trop gros deborde en memoire
# partagee SANS lever d'OOM et le calcul rampe : 2 a 1536 x 1024 tenait en 6,9 Go.
VARIANTS = {"w1024": (PLEX_WIDTH, 4), "full": (None, 2)}
GRIDS = ("", "_native")
MAX_EMPTY_MASK_FRAC = 0.5
NATIVE_MIN_COVERAGE = 0.999  # colonne native gardee si entierement dans le recadrage

SEG_REPO = "ClementP/ChoroidSegmentationModule"
REG_REPO = "ClementP/OCTVideoRegistration"


@dataclass(frozen=True)
class ModelSpec:
    """Un couple (segmentation, recalage) et les reglages de ``RegistrationConfig``
    qui vont avec. Revisions et reglages EXPLICITES : les defauts de ``consts`` /
    ``RegistrationConfig`` changent avec les livraisons de ``main``, et une
    variante deja calculee ne doit pas changer de sens en douce."""

    prefix: str  # dossiers <prefix>_<variante>[_native]
    seg_revision: str
    reg_revision: str
    config: dict = field(default_factory=dict)  # remplacements dans REGISTRATION


MODELS = {
    # SegFormer + cascade_v9 : la passe d'octobre 2026, a l'identique.
    "cascade_v9": ModelSpec("registered_cascade_v9_plex", "version-2.0.0", "cascade_v9",
                            {"registrator_revision": "cascade_v9",
                             "filter_bad_columns": True, "keep_largest_cc": True,
                             "dy_align_bm": False}),
    # UPerNet ConvNeXt-tiny + regresseur entraine sur les lignes de BM : les
    # reglages de scripts/pulsation/spectralis.py (``registration_config``) --
    # REGISTRATION de main, chaque A-scan aligne sur l'interface haute du masque.
    "upernet": ModelSpec("registered_upernet_plex", "version-3.0.0",
                         "reg_upernet_convnext_t_bmlines",
                         {"registrator_revision": "reg_upernet_convnext_t_bmlines",
                          "filter_bad_columns": False, "keep_largest_cc": False,
                          "dy_align_bm": True}),
}
DEFAULT_MODEL = MODELS["cascade_v9"]
PREFIX = DEFAULT_MODEL.prefix

REQUIRED_FILES = ("registered_video.mp4", "mask.npz", "prob.npz", "timestamp.txt",
                  "transform.npz", "registration_params.json")

SUMMARY_KEYS = [
    "status", "reason", "variant", "grid", "n_frames", "height", "width", "fps",
    "plex_axial_um", "plex_lateral_um", "covered_mm", "native_mm", "ref_idx",
    "ref_percentile", "fused_batch_size", "dx_ptp", "dx_std", "dy_bulk_ptp",
    "dy_bulk_std", "dy_intra_frame_std_median", "n_bad_columns", "n_empty_mask_frames",
    "ncc_reg_median", "ncc_reg_q1", "ncc_reg_q3", "ncc_reg_p5", "ncc_reg_n_lt_0.5",
    "ncc_identity_median", "ncc_identity_q1", "ncc_identity_q3", "ncc_identity_p5",
    "ncc_identity_n_lt_0.5", "iou_ref_median", "iou_ref_p5", "iou_ref_n_lt_0.8",
    "seconds_pass", "created",
]


def subdirs(model: ModelSpec = DEFAULT_MODEL) -> list[str]:
    return [f"{model.prefix}_{v}{g}" for v in VARIANTS for g in GRIDS]


# --------------------------------------------------------------------------- #
# E/S
# --------------------------------------------------------------------------- #
def save_probs(probs: np.ndarray, path) -> None:
    """(T, H, W) uint8, zstd. Ecrit via un fichier temporaire."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    comp = zstd.ZstdCompressor(level=6, threads=-1).compress(
        np.ascontiguousarray(probs, np.uint8).tobytes())
    np.savez(tmp, compressed=np.frombuffer(comp, np.uint8),
             shape=np.array(probs.shape, np.int64))
    tmp.replace(path)


def load_probs(path) -> np.ndarray:
    """(T, H, W) uint8 ; diviser par 255 pour une probabilite."""
    d = np.load(path)
    raw = zstd.ZstdDecompressor().decompress(d["compressed"].tobytes())
    return np.frombuffer(raw, np.uint8).reshape(tuple(d["shape"])).copy()


def read_params(out_dir: Path) -> dict | None:
    p = Path(out_dir) / "registration_params.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def is_complete(out_dir: Path) -> bool:
    if not all((Path(out_dir) / n).exists() for n in REQUIRED_FILES):
        return False
    params = read_params(out_dir)
    return bool(params) and params.get("status") == "ok"


def hub_sha(repo_id: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo_id, revision=revision).sha)
    except Exception as e:  # noqa: BLE001
        return f"inconnu ({type(e).__name__})"


def git_state(repo: Path) -> dict:
    def run(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip()

    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "src"))}


# --------------------------------------------------------------------------- #
# Modeles
# --------------------------------------------------------------------------- #
def patch_largest_cc() -> str:
    """cuCIM absent de pyOR : ``fused`` recoit la version CPU (cc3d), bit-identique
    selon la docstring de la version GPU. Meme remplacement que register_newmodel."""
    try:
        import cucim  # noqa: F401

        return "cucim (GPU)"
    except ImportError:
        pass
    try:
        # Sans cuCIM, blob.py etiquette avec cupyx.scipy.ndimage (identique a cc3d
        # au pixel pres sur les masques SANS) : on garde la version GPU si elle tourne.
        from ocularrigidity.segmentation.postprocess.blob import (
            keep_largest_connected_component_gpu,
        )

        keep_largest_connected_component_gpu(torch.ones((1, 4, 4), dtype=torch.bool, device=DEVICE))
        return "cupyx.scipy.ndimage (GPU, cucim absent)"
    except Exception:  # noqa: BLE001
        import ocularrigidity.registration.fused as fused_mod
        from ocularrigidity.segmentation.postprocess.blob import (
            keep_largest_connected_component,
        )

        def _largest_cc_cpu(masks: torch.Tensor) -> torch.Tensor:
            out = keep_largest_connected_component(masks.detach().cpu().numpy())
            return torch.from_numpy(out).to(masks.device)

        fused_mod.keep_largest_connected_component_gpu = _largest_cc_cpu
        return "cc3d (CPU, repli : cucim absent)"


def load_models(model: ModelSpec = DEFAULT_MODEL):
    from ocularrigidity.segmentation.utils import (
        get_choroid_segmentation_model,
        get_registration_model,
    )

    seg = get_choroid_segmentation_model(model.seg_revision).to(DEVICE)
    reg = get_registration_model(model.reg_revision).to(DEVICE)
    cc = patch_largest_cc()
    torch.backends.cudnn.benchmark = True
    return seg, reg, cc


def run_fused(frames, seg_model, reg_model, config):
    """``segment_and_register`` avec probabilites, lot reduit sur OOM."""
    sizes = [config.fused_batch_size] + [b for b in (4, 2, 1) if b < config.fused_batch_size]
    last = None
    for b in sizes:
        try:
            cfg = replace(config, fused_batch_size=b)
            return segment_and_register(frames, seg_model, reg_model, cfg, device=DEVICE,
                                        verbose=False, return_probs=True), cfg
        except torch.cuda.OutOfMemoryError as e:
            last = e
            torch.cuda.empty_cache()
            print(f"  OOM avec fused_batch_size={b}, nouvel essai plus petit", flush=True)
    raise last


# --------------------------------------------------------------------------- #
# Retour a la grille native
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def warp_native(cube, masks, probs, dx, dy):
    """Applique un transform NATIF (dx (T,), dy (T, W)) aux frames, masques et
    probabilites natifs. Renvoie (frames, masks, probs) uint8/bool/uint8."""
    T, H, W = cube.shape
    f_out, m_out, p_out = np.empty_like(cube), np.empty_like(masks), np.empty_like(probs)
    dx_t, dy_t = torch.from_numpy(dx).float(), torch.from_numpy(dy).float()
    for s in range(0, T, WARP_BATCH):
        e = min(s + WARP_BATCH, T)
        data = torch.stack([torch.from_numpy(masks[s:e]).to(DEVICE).float(),
                            torch.from_numpy(cube[s:e]).to(DEVICE).float(),
                            torch.from_numpy(probs[s:e]).to(DEVICE).float()], dim=1)
        out, _ = warp(data, dx_t[s:e].to(DEVICE), dy_t[s:e].to(DEVICE), (H, W))
        m_out[s:e] = (out[:, 0] > 0.5).cpu().numpy()
        f_out[s:e] = out[:, 1].clamp(0, 255).to(torch.uint8).cpu().numpy()
        p_out[s:e] = out[:, 2].round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return f_out, m_out, p_out


def blank_bad_columns(frames, masks, probs, extra_bad=None):
    bad = filter_bad_ascans_per_bms(masks)
    bad = bad.cpu().numpy() if hasattr(bad, "cpu") else np.asarray(bad)
    bad = bad.astype(bool)
    if extra_bad is not None:
        bad |= extra_bad
    if bad.any():
        frames[..., bad] = 0
        masks[..., bad] = 0
        if probs is not None:
            probs[..., bad] = 0
    return bad


# --------------------------------------------------------------------------- #
# Une condition
# --------------------------------------------------------------------------- #
def _pixel_sizes(series) -> tuple[float, float]:
    """(axial = ScaleY, lateral = ScaleX) en mm, medianes sur les B-scans."""
    sy = [s.oct.scale_y for s in series if s.oct is not None and s.oct.scale_y]
    sx = [s.oct.scale_x for s in series if s.oct is not None and s.oct.scale_x]
    if not sy or not sx:
        raise ValueError("ScaleX / ScaleY absents du XML")
    return float(np.median(sy)), float(np.median(sx))


def _write(out_dir: Path, frames, masks, probs, ts_us, fps, transform: dict, params: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registration_params.json").unlink(missing_ok=True)  # incomplet tant qu'absent
    write_gray_mp4(frames, out_dir / "registered_video.mp4", fps)
    save_mask(masks, out_dir / "mask.npz")
    save_probs(probs, out_dir / "prob.npz")
    (out_dir / "timestamp.txt").write_text("\n".join(str(int(t)) for t in ts_us),
                                           encoding="utf-8")
    np.savez(out_dir / "transform.npz", **transform)
    (out_dir / "registration_params.json").write_text(
        json.dumps(params, indent=2, default=str), encoding="utf-8")


def _stats(dx, dy, bad, masks, metrics, iou=None) -> dict:
    area = masks.sum(axis=(1, 2))
    iou_stats = {}
    if iou is not None and np.isfinite(iou).any():
        iou_stats = {"iou_ref_median": float(np.nanmedian(iou)),
                     "iou_ref_p5": float(np.nanpercentile(iou, 5)),
                     "iou_ref_n_lt_0.8": int((np.nan_to_num(iou) < 0.8).sum())}
    return {**iou_stats, "dx_ptp": float(np.ptp(dx)), "dx_std": float(np.std(dx)),
            "dy_bulk_ptp": float(np.ptp(dy.mean(axis=1))),
            "dy_bulk_std": float(dy.mean(axis=1).std()),
            "dy_intra_frame_std_median": float(np.median(dy.std(axis=1))),
            "n_bad_columns": int(bad.sum()), "n_empty_mask_frames": int((area == 0).sum()),
            **metrics}


def in_raw_dir(raw_dir: Path, subdir: str) -> Path:
    """Defaut de ``out_dir_of`` : la variante sous le dossier des donnees brutes."""
    return Path(raw_dir) / subdir


def process_condition(raw_dir: Path, identity: dict, seg_model, reg_model, cc_backend: str,
                      ncc_fn: Callable, summarize_fn: Callable, repo: Path,
                      variants=None, overwrite: bool = False,
                      out_dir_of: Callable[[Path, str], Path] = in_raw_dir,
                      model: ModelSpec = DEFAULT_MODEL) -> list[dict]:
    """Toutes les variantes d'une condition. Renvoie un dict de parametres par dossier ecrit.

    ``out_dir_of(raw_dir, sous_dossier)`` donne le dossier de sortie d'une variante."""
    from ocularrigidity.pipeline_config import REGISTRATION

    raw_dir = Path(raw_dir)
    variants = list(variants or VARIANTS)
    todo = [v for v in variants
            if overwrite or not all(is_complete(out_dir_of(raw_dir, f"{model.prefix}_{v}{g}"))
                                    for g in GRIDS)]
    if not todo:
        return []

    series = load_ordered_oct_series(raw_dir)
    cube, ts_us = build_cube_and_timestamps(raw_dir, series)
    T, H, W = cube.shape
    axial_mm, lateral_mm = _pixel_sizes(series)
    fps = estimate_fps(ts_us)
    common = {
        **identity,
        "engine": "ocularrigidity.scripts.registration.cascade_v9_plex",
        "model_prefix": model.prefix,
        "seg_repo": SEG_REPO, "seg_revision": model.seg_revision,
        "seg_commit_sha": hub_sha(SEG_REPO, model.seg_revision),
        "reg_repo": REG_REPO, "reg_revision": model.reg_revision,
        "reg_commit_sha": hub_sha(REG_REPO, model.reg_revision),
        "largest_cc_backend": cc_backend, "skip_first_n_frames": 0, "drop_last_n_frames": 0,
        "native_shape": [H, W], "native_axial_mm": axial_mm, "native_lateral_mm": lateral_mm,
        "fps": round(float(fps), 3), "prob_encoding": "uint8 0..255 = sigmoid(logit), avant LCC",
        "git": git_state(repo), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    out = []
    for v in todo:
        width, batch = VARIANTS[v]
        t0 = time.perf_counter()
        frames_p, g = to_plex_grid(cube, axial_mm, lateral_mm, width=width)
        reg_model.img_shape = g.shape
        config = replace(REGISTRATION, fused_batch_size=batch, **model.config)
        res, config = run_fused(frames_p, seg_model, reg_model, config)
        dx_p = np.asarray(res.transform["dx"], np.float32)
        dy_p = np.asarray(res.transform["dy"], np.float32)
        bad_p = np.asarray(res.transform["bad_columns"], bool)
        iou = (np.asarray(res.frame_iou, np.float32) if res.frame_iou is not None
               else np.full(T, np.nan, np.float32))
        geom = g.as_dict()
        base = {**common, "variant": v, "plex_geometry": geom,
                "plex_axial_um": geom["plex_axial_um"], "plex_lateral_um": geom["plex_lateral_um"],
                "covered_mm": geom["covered_mm"], "native_mm": geom["native_mm"],
                "ref_idx": int(res.ref_idx), "ref_percentile": res.ref_percentile,
                "registration_config": asdict(config), "fused_batch_size": config.fused_batch_size,
                "timings_s": {k: round(float(x), 2) for k, x in res.timings.items()},
                "n_frames": T}

        # --- grille PLEX ---------------------------------------------------------
        id_bad = filter_bad_ascans_per_bms(res.raw_masks)
        id_bad = (id_bad.cpu().numpy() if hasattr(id_bad, "cpu") else np.asarray(id_bad)).astype(bool)
        id_f = frames_p.copy()
        id_m = res.raw_masks.copy()
        id_f[..., id_bad] = 0
        id_m[..., id_bad] = 0
        metrics = {**summarize_fn(ncc_fn(res.registered_frames, res.registered_masks), "ncc_reg"),
                   **summarize_fn(ncc_fn(id_f, id_m), "ncc_identity")}
        del id_f, id_m
        out.append(_finish(out_dir_of(raw_dir, f"{model.prefix}_{v}"), res.registered_frames,
                           res.registered_masks, res.registered_probs, ts_us, fps,
                           {"dx": dx_p, "dy": dy_p, "bad_columns": bad_p,
                            "ref_idx": np.int64(res.ref_idx), "frame_iou": iou,
                            "pixel_mm": np.array([g.plex_axial_mm, g.plex_lateral_mm])},
                           {**base, "grid": "plex", "height": g.new_h, "width": g.width},
                           _stats(dx_p, dy_p, bad_p, res.registered_masks, metrics, iou), t0))
        raw_masks_p, raw_probs_p = res.raw_masks, res.raw_probs
        del res, frames_p
        gc.collect()

        # --- grille native ---------------------------------------------------------
        t1 = time.perf_counter()
        cover = native_coverage(g) >= NATIVE_MIN_COVERAGE
        m_n = plex_to_native(raw_masks_p, g) >= 0.5
        p_n = np.clip(np.round(plex_to_native(raw_probs_p, g)), 0, 255).astype(np.uint8)
        del raw_masks_p, raw_probs_p
        m_n[..., ~cover] = False
        p_n[..., ~cover] = 0
        dx_n, dy_n = transform_to_native(dx_p, dy_p, g)
        # Identite native : meme filtre de colonnes, transform nul.
        id_f, id_m, _ = cube.copy(), m_n.copy(), None
        blank_bad_columns(id_f, id_m, None, extra_bad=~cover)
        id_metrics = summarize_fn(ncc_fn(id_f, id_m), "ncc_identity")
        del id_f, id_m
        f_n, m_n, p_n = warp_native(cube, m_n, p_n, dx_n, dy_n)
        bad_n = blank_bad_columns(f_n, m_n, p_n, extra_bad=~cover)
        metrics = {**summarize_fn(ncc_fn(f_n, m_n), "ncc_reg"), **id_metrics}
        out.append(_finish(out_dir_of(raw_dir, f"{model.prefix}_{v}_native"), f_n, m_n, p_n,
                           ts_us, fps,
                           {"dx": dx_n, "dy": dy_n, "bad_columns": bad_n,
                            "outside_crop": ~cover, "ref_idx": np.int64(base["ref_idx"]),
                            "frame_iou": iou,
                            "dx_plex": dx_p, "dy_plex": dy_p,
                            "pixel_mm": np.array([axial_mm, lateral_mm])},
                           {**base, "grid": "native", "height": H, "width": W,
                            "native_columns_in_crop": int(cover.sum()),
                            "native_mapping": "masque/probabilites : canvas PLEX recolle puis "
                                              "INTER_AREA ; transform : plex_grid.transform_to_native "
                                              "applique aux frames natives (losses.warp)"},
                           _stats(dx_n, dy_n, bad_n, m_n, metrics, iou), t1))
        del f_n, m_n, p_n
        gc.collect()
        torch.cuda.empty_cache()
    return out


def _finish(out_dir, frames, masks, probs, ts_us, fps, transform, params, stats, t0) -> dict:
    T = frames.shape[0]
    status, reason = "ok", ""
    if stats["n_empty_mask_frames"] > MAX_EMPTY_MASK_FRAC * T:
        status, reason = "error", f"masque vide sur {stats['n_empty_mask_frames']}/{T} frames recalees"
    params = {**params, **stats, "status": status, "reason": reason,
              "subdir": Path(out_dir).name,
              "seconds_pass": round(time.perf_counter() - t0, 1),
              "created": datetime.now().isoformat(timespec="seconds")}
    if status == "ok":
        _write(Path(out_dir), frames, masks, probs, ts_us, fps, transform, params)
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "registration_params.json").write_text(
            json.dumps(params, indent=2, default=str), encoding="utf-8")
    return params


def write_error(raw_dir: Path, identity: dict, reason: str, variants=None,
                out_dir_of: Callable[[Path, str], Path] = in_raw_dir,
                model: ModelSpec = DEFAULT_MODEL) -> None:
    for v in variants or VARIANTS:
        for gsuf in GRIDS:
            d = out_dir_of(Path(raw_dir), f"{model.prefix}_{v}{gsuf}")
            if is_complete(d):
                continue
            d.mkdir(parents=True, exist_ok=True)
            (d / "registration_params.json").write_text(json.dumps(
                {**identity, "variant": v, "grid": gsuf.strip("_") or "plex", "status": "error",
                 "reason": reason, "created": datetime.now().isoformat(timespec="seconds")},
                indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def write_summaries(items, summary_root: Path, id_cols: list[str],
                    out_dir_of: Callable[[Path, str], Path] = in_raw_dir,
                    model: ModelSpec = DEFAULT_MODEL) -> list[Path]:
    """Un ``registration_summary.csv`` par dossier de sortie, relu sur disque."""
    import pandas as pd

    paths = []
    for sub in subdirs(model):
        rows = []
        for _key, raw_dir, identity in items:
            p = read_params(out_dir_of(Path(raw_dir), sub))
            if p:
                rows.append({**{c: identity.get(c) for c in id_cols},
                             **{k: p.get(k) for k in SUMMARY_KEYS}})
        out = Path(summary_root) / sub.removeprefix("registered_") / "registration_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=id_cols + SUMMARY_KEYS).to_csv(out, index=False)
        paths.append(out)
    return paths


def run_batch(items, summary_root: Path, id_cols: list[str], ncc_fn, summarize_fn,
              repo: Path, variants=None, overwrite=False, limit=None,
              out_dir_of: Callable[[Path, str], Path] = in_raw_dir,
              model: ModelSpec = DEFAULT_MODEL) -> int:
    """``items`` : liste de ``(nom, raw_dir, identity)``, dans l'ordre de traitement.

    ``out_dir_of(raw_dir, sous_dossier)`` : ou ecrire chaque variante (defaut :
    sous ``raw_dir``, l'arborescence SANS)."""
    import traceback

    variants = list(variants or VARIANTS)
    todo = [it for it in items
            if overwrite or not all(is_complete(out_dir_of(Path(it[1]), f"{model.prefix}_{v}{g}"))
                                    for v in variants for g in GRIDS)]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(items)} condition(s), {len(todo)} a traiter | variantes {variants} "
          f"-> {out_dir_of(Path('<raw>'), model.prefix + '_<variante>[_native]')}", flush=True)
    if not todo:
        write_summaries(items, summary_root, id_cols, out_dir_of, model)
        return 0
    seg, reg, cc = load_models(model)
    print(f"modeles : {SEG_REPO}@{model.seg_revision}, {REG_REPO}@{model.reg_revision} | "
          f"composante connexe : {cc}\n", flush=True)
    t_batch = time.perf_counter()
    n_ok = n_err = 0
    for i, (name, raw_dir, identity) in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            rows = process_condition(raw_dir, identity, seg, reg, cc, ncc_fn, summarize_fn,
                                     repo, variants=variants, overwrite=overwrite,
                                     out_dir_of=out_dir_of, model=model)
        except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
            traceback.print_exc()
            write_error(raw_dir, identity, f"{type(e).__name__}: {str(e)[:300]}", variants,
                        out_dir_of, model)
            rows = [{"status": "error", "subdir": "toutes", "reason": str(e)[:200]}]
        gc.collect()
        torch.cuda.empty_cache()
        bits = []
        for p in rows:
            if p.get("status") == "ok":
                n_ok += 1
                bits.append(f"{p['subdir'].removeprefix(model.prefix + '_'):12} NCC {p['ncc_reg_median']:.3f} "
                            f"(id {p['ncc_identity_median']:.3f}) <0,5 {p['ncc_reg_n_lt_0.5']:3d}")
            else:
                n_err += 1
                bits.append(f"{p.get('subdir', '?')} ERREUR {p.get('reason')}")
        print(f"[{i:>3}/{len(todo)}] {name} | {time.perf_counter() - t0:.0f} s\n    "
              + "\n    ".join(bits), flush=True)
        write_summaries(items, summary_root, id_cols, out_dir_of, model)
    paths = write_summaries(items, summary_root, id_cols, out_dir_of, model)
    print(f"\n{n_ok} dossier(s) ok, {n_err} en echec "
          f"({(time.perf_counter() - t_batch) / 60:.1f} min)\nsuivi : {paths[0].parent.parent}")
    return 0
