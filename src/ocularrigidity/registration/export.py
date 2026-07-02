"""
Export d'une video OCT recalee a partir des images brutes (.tif) d'une condition.

Logique partagee par :
  - le script de lot ``Astronauts/register_files.py`` ;
  - le bouton "Enregistrer la video recalee" de l'app
    ``testing_app/first_cc_registration.py``.

Pour une condition (dossier ``RawImages`` contenant les .tif + l'export XML
Spectralis), on empile les images dans l'ordre des horodatages, on segmente la
choroide, on recale la video via ``RegisteredVideo`` (motion/registered_video.py)
en reutilisant une ``RegistrationConfig``, puis on enregistre dans
``<RawImages>/registered/`` :
  - ``registered_video.mp4``  : la video recalee (frames uint8) ;
  - ``mask.npz``              : les masques de choroide recales ;
  - ``timestamp.txt``         : un horodatage (microsecondes) par frame ;
  - ``transform.npz``         : la transformation appliquee (dx, dy) ;
  - ``registration_params.json`` : tracabilite (config + skip/drop + meta).

La video, les masques et les horodatages sont deja rognes du skip/drop de la
config (le recalage prend la 1re frame conservee comme reference) ; lancer
``pulsation.py`` avec ``skip_first_n_frames=0`` / ``drop_last_n_frames=0`` dessus.
"""

from __future__ import annotations

from pathlib import Path
import os
import json
import shutil
import subprocess
import dataclasses
from datetime import datetime

import numpy as np
import imageio.v3 as iio

from ocularrigidity.data.spectralis import SpectralisStudy
from ocularrigidity.data.io import save_mask
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.motion.registered_video import RegisteredVideo
from ocularrigidity.pipeline_config import RegistrationConfig

# Importer registered_video -> compression.py force IMAGEIO_FFMPEG_EXE vers un
# chemin Linux code en dur ; on le neutralise (on appelle ffmpeg directement).
os.environ.pop("IMAGEIO_FFMPEG_EXE", None)

DEFAULT_OUTPUT_SUBDIR = "registered"


# --------------------------------------------------------------------------- #
# Ecriture de la video (ffmpeg, libx264 logiciel -> pas de dependance GPU/nvenc)
# --------------------------------------------------------------------------- #
def resolve_ffmpeg() -> str:
    """Chemin de l'executable ffmpeg (PATH systeme, sinon imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def estimate_fps(ts_us: np.ndarray) -> float:
    """Cadence mediane (Hz) a partir des horodatages (microsecondes)."""
    dts = np.diff(np.asarray(ts_us))
    dts = dts[dts > 0]
    if dts.size == 0:
        return 30.0
    fps = 1e6 / float(np.median(dts))
    return float(min(max(fps, 1.0), 240.0))


def write_gray_mp4(cube: np.ndarray, out_path, fps: float,
                   ffmpeg_exe: str | None = None, crf: int = 18) -> None:
    """Encode un cube (T, H, W) uint8 en mp4 gris (libx264, quasi sans perte).

    La cadence reelle des frames etant irreguliere, ``fps`` n'est qu'une
    metadonnee d'affichage : le timing exact vit dans ``timestamp.txt``.
    """
    if ffmpeg_exe is None:
        ffmpeg_exe = resolve_ffmpeg()
    cube = np.ascontiguousarray(cube, dtype=np.uint8)
    T, H, W = cube.shape
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", f"{float(fps):.6f}",
        "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(memoryview(cube))
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg a echoue en ecrivant {out_path}")


# --------------------------------------------------------------------------- #
# Lecture des images brutes (.tif) dans l'ordre des horodatages du XML
# --------------------------------------------------------------------------- #
def _load_gray_u8(path) -> np.ndarray:
    """Charge une image .tif en niveaux de gris (H x W, uint8)."""
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]  # les .tif sont du gris (eventuellement replique x3)
    return img.astype(np.uint8)


def load_ordered_oct_series(raw_dir) -> list:
    """Series OCT exploitables, triees par horodatage croissant.

    Garde les series ayant un B-scan OCT, un .tif present sur disque et un
    ``AcquisitionTime`` (necessaire pour l'ordre et le timestamp.txt).
    """
    raw_dir = Path(raw_dir)
    xml_files = sorted(raw_dir.glob("*.xml"))
    if not xml_files:
        return []
    study = SpectralisStudy.from_file(xml_files[0])
    series = [
        s
        for s in study.series
        if s.oct is not None
        and s.oct_file_name
        and s.acquisition_time is not None
        and (raw_dir / s.oct_file_name).exists()
    ]
    series.sort(key=lambda s: s.acquisition_time.seconds_of_day)
    return series


def build_cube_and_timestamps(raw_dir, series):
    """Empile les .tif (ordre des horodatages) -> cube (T, H, W) uint8 + ts (us).

    Le cube est rogne a des dimensions paires pour que l'encodage yuv420p
    (libx264) soit valide ; masque et mp4 restent ainsi alignes.
    """
    raw_dir = Path(raw_dir)
    cube = np.stack([_load_gray_u8(raw_dir / s.oct_file_name) for s in series], axis=0)
    ts_us = np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )
    H, W = cube.shape[1], cube.shape[2]
    cube = cube[:, : H - (H % 2), : W - (W % 2)]
    return cube, ts_us


def fill_empty_columns(masks: np.ndarray) -> np.ndarray:
    """Comble les colonnes sans masque par la colonne valide la plus proche.

    Sur un B-scan Spectralis, quelques colonnes de bord n'ont pas de choroide :
    leurs frontieres seraient NaN, et ``ref_bm.mean()`` (utilise par le flatten
    dans register_masks_by_displacement) propagerait le NaN -> deplacement NUL
    partout, donc AUCUN recalage vertical. On les comble pour que le recalage
    opere vraiment (meme correctif que ``fill_empty_columns`` de l'app interactive).
    """
    out = np.asarray(masks).copy()
    cols = np.arange(out.shape[2])
    for i in range(out.shape[0]):
        has = out[i].any(0)
        if has.all():
            continue
        idx = np.where(has)[0]
        if idx.size == 0:
            continue
        nearest = idx[np.abs(cols[:, None] - idx[None, :]).argmin(1)]
        out[i] = out[i][:, nearest]
    return out


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


# --------------------------------------------------------------------------- #
# Pipeline complet pour une condition
# --------------------------------------------------------------------------- #
def export_registered_video(
    raw_dir,
    cfg: RegistrationConfig,
    model,
    *,
    device: str = "cuda",
    out_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    suffix: str = "",
    overwrite: bool = False,
    scale_factor: float = 2.0,
    seg_batch_size: int = 8,
    verbose: bool = True,
    extra_meta: dict | None = None,
) -> dict:
    """Segmente, recale et enregistre la video recalee d'une condition.

    Parameters
    ----------
    raw_dir : Path
        Dossier ``RawImages`` (ou ``RawData``) contenant les .tif et le .xml.
    cfg : RegistrationConfig
        Parametres de recalage (flatten / horizontal_alignment / lateral_method /
        subpixel + skip/drop + batch_size). Reutilisee de l'experience de l'app.
    model
        Modele de segmentation de la choroide (ChoroidSegmentationModule).
    suffix : str, optional
        Suffixe ajoute a TOUS les fichiers de sortie (``registered_video{suffix}.mp4``,
        ``mask{suffix}.npz``, ``timestamp{suffix}.txt``, ``transform{suffix}.npz``,
        ``registration_params{suffix}.json``). Vide = sortie standard ; ``"_ascan"``
        pour la variante recalee A-scan par A-scan, sans ecraser la sortie de base.
    extra_meta : dict, optional
        Metadonnees ajoutees a ``registration_params.json`` (tracabilite).

    Returns
    -------
    dict
        ``{"status": "ok", "video", "out_dir", "n_frames", "fps"}`` en cas de
        succes, ou ``{"status": "skipped", "reason", ...}`` (video deja presente,
        pas d'images, trop peu de frames apres rognage). Leve sur erreur dure.
    """
    raw_dir = Path(raw_dir)
    out_dir = raw_dir / out_subdir
    out_video = out_dir / f"registered_video{suffix}.mp4"
    if out_video.exists() and not overwrite:
        return {"status": "skipped", "reason": "exists",
                "out_dir": out_dir, "video": out_video}

    series = load_ordered_oct_series(raw_dir)
    if len(series) < 2:
        return {"status": "skipped", "reason": "no_series", "out_dir": out_dir}

    cube_full, ts_full = build_cube_and_timestamps(raw_dir, series)

    # Rognage skip/drop de la config (reference = 1re frame conservee).
    end = None if cfg.drop_last_n_frames == 0 else -cfg.drop_last_n_frames
    sl = slice(cfg.skip_first_n_frames, end)
    cube, ts_us = cube_full[sl], ts_full[sl]
    if cube.shape[0] < 2:
        return {"status": "skipped", "reason": "too_few_frames",
                "n_total": int(cube_full.shape[0]), "n_kept": int(cube.shape[0]),
                "out_dir": out_dir}

    # Segmentation de la choroide (memes reglages que first_cc_registration.py).
    masks = np.asarray(
        infer(model, cube, scale_factor=scale_factor,
              batch_size=seg_batch_size, device=device),
        dtype=bool,
    )
    # Indispensable : combler les colonnes vides, sinon ref_bm.mean() = NaN et le
    # recalage vertical (surtout flatten) devient un no-op (cf. fill_empty_columns).
    masks = fill_empty_columns(masks)

    # Recalage via RegisteredVideo : frames/masques deja rognes fournis en
    # memoire (skip/drop=0 ici pour ne pas re-rogner, cache_dir=None).
    registrator = RegisteredVideo(
        video=Path(out_subdir),
        root_data=raw_dir,
        root_masks=raw_dir,
        skip_first_n_frames=0,
        drop_last_n_frames=0,
        flatten=cfg.flatten,
        horizontal_alignment=cfg.horizontal_alignment,
        lateral_method=cfg.lateral_method,
        subpixel=cfg.subpixel,
        crop_w_x=cfg.crop_w_x,
        bp_lo=cfg.bp_lo,
        bp_hi=cfg.bp_hi,
        median_registration=cfg.median_registration,
        median_max_vshift=cfg.median_max_vshift,
        median_use_shadow=cfg.median_use_shadow,
        median_use_log=cfg.median_use_log,
        median_shadow_n=cfg.median_shadow_n,
        median_shadow_a=cfg.median_shadow_a,
        median_log_kernel_size=cfg.median_log_kernel_size,
        median_log_sigma=cfg.median_log_sigma,
        verbose=verbose,
        device=device,
        batch_size=cfg.batch_size,
        cache_dir=None,
    )
    registrator._raw_frames = cube
    registrator._raw_masks = masks
    registrator.compute_registration()

    reg_frames = np.asarray(registrator.registered_frames, dtype=np.uint8)
    reg_masks = np.asarray(registrator.registered_masks, dtype=bool)
    transform = registrator.transform

    out_dir.mkdir(parents=True, exist_ok=True)
    fps = estimate_fps(ts_us)
    write_gray_mp4(reg_frames, out_video, fps)
    save_mask(reg_masks, out_dir / f"mask{suffix}.npz")
    (out_dir / f"timestamp{suffix}.txt").write_text(
        "\n".join(str(int(t)) for t in ts_us), encoding="utf-8"
    )
    transform_arrays = {
        "dx": _to_numpy(transform.get("dx")),
        "dy": _to_numpy(transform.get("dy")),
    }
    # Deplacement par A-scan de la 2e passe (recalage sur la mediane), si activee.
    if transform.get("dy_median") is not None:
        transform_arrays["dy_median"] = _to_numpy(transform.get("dy_median"))
    np.savez(out_dir / f"transform{suffix}.npz", **transform_arrays)

    meta = {
        "registration_config": dataclasses.asdict(cfg),
        "skip_first_n_frames": cfg.skip_first_n_frames,
        "drop_last_n_frames": cfg.drop_last_n_frames,
        "n_frames_total": int(cube_full.shape[0]),
        "n_frames_registered": int(reg_frames.shape[0]),
        "fps": fps,
        "seg_scale_factor": scale_factor,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / f"registration_params{suffix}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    return {"status": "ok", "video": out_video, "out_dir": out_dir,
            "n_frames": int(reg_frames.shape[0]), "fps": fps}
