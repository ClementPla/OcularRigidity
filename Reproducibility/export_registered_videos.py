# -*- coding: utf-8 -*-
"""
export_registered_videos.py

Ecrit les VIDEOS RECALEES des acquisitions de repetabilite A COTE DE LEURS
DONNEES, en TROIS variantes qui ne font varier qu'un facteur a la fois :

    E:/SANSORI/Reproducibility/<SLUG>/RawImages/
        registered_segformer2_flatten_choroid_xcorr/
        registered_model1_flatten_choroid_xcorr/
        registered_model1_fullframe/

    variante                              segmentation      recalage
    ------------------------------------  ----------------  ---------------------------
    segformer2_flatten_choroid_xcorr      SegFormer mit_b2  chaine en 4 temps (actuelle)
    model1_flatten_choroid_xcorr          U-Net se_resnet50 chaine en 4 temps (actuelle)
    model1_fullframe                      U-Net se_resnet50 register_videos (fullframe)

La premiere isole L'EFFET DU MODELE (meme recalage, poids differents), la
troisieme isole L'EFFET DU RECALAGE (meme modele, moteur different) ; la
deuxieme est la reference commune -- et c'est exactement la variante deja
calculee sous ``E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
model1_scale_1.0_flatten_choroid_xcorr/``, ce qui donne un ancrage gratuit pour
verifier que ce script appelle bien la chaine de production.

Les trois tournent avec skip = drop = 0 et une segmentation a scale_factor 1.0 :
meme nombre de frames, memes horodatages, comparables image par image. Le script
REFUSE d'ecrire une variante dont le nombre de frames ou les dimensions ne
correspondent pas a celles deja ecrites pour la meme acquisition.

Chaque dossier de sortie porte les MEMES CINQ NOMS DE FICHIERS que
``export_registered_video``, pour que l'aval puisse relire n'importe laquelle
des variantes sans savoir laquelle :

    registered_video.mp4  mask.npz  timestamp.txt  transform.npz
    registration_params.json

Pourquoi importer ``Astronauts/compute_registration.py`` plutot que reecrire la
chaine : ``process_condition`` prend DEJA le modele en argument et renvoie ses
statistiques -- la seule chose qui distingue les variantes 1 et 2 est ce modele.
L'importer tel quel, sans en toucher une ligne, est la seule garantie solide que
les sorties de la cohorte SANS restent reproductibles par le meme code. Ses
constantes de module (SEG_SCALE_FACTOR=1.0, SEG_BATCH_SIZE=8, FOVEA_CORRECTION,
MAX_LATERAL_SHIFT, ...) sont figees a l'import et sont exactement celles voulues
ici ; les variables d'environnement sont donc posees AVANT cet import.

Lancer (kernel pyOR, depuis la racine du depot) :
    python Reproducibility/export_registered_videos.py                 # les 3 variantes
    python Reproducibility/export_registered_videos.py --variants b    # une seule
    python Reproducibility/export_registered_videos.py --slug BELANGER_CHARLES_OD1
    python Reproducibility/export_registered_videos.py --check         # bilan, rien ne tourne

Interruptible : une acquisition n'est refaite que si ses CINQ fichiers ne sont
pas tous presents (--overwrite pour tout refaire).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Environnement AVANT tout import du depot : ``compute_registration`` fige ses
# constantes de module a l'import, et ``batch_layout`` lit ces variables.
# --------------------------------------------------------------------------- #
os.environ.setdefault("OR_LAYOUT", "flat")
os.environ.setdefault("OR_PATH_GENERAL", "E:/SANSORI/Reproducibility")
os.environ.setdefault(
    "OR_SEGVAR_ROOT", "E:/NASA_Rigidity/Reproducibility/SegmentationVariations"
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ocularrigidity.data.io import load_mask  # noqa: E402
from ocularrigidity.registration.config import RegistrationConfig  # noqa: E402
from ocularrigidity.scripts import batch_layout as layout  # noqa: E402
from ocularrigidity.scripts.registration.astronauts import (  # noqa: E402
    DEFAULT_OUTPUT_SUBDIR,
    export_registered_video,
    load_ordered_oct_series,
)
from ocularrigidity.segmentation.utils import (  # noqa: E402
    DEFAULT_SEGMENTATION_REPO,
    get_choroid_segmentation_model,
)

REPO = Path(__file__).resolve().parent.parent
PATH_GENERAL = layout.env_path("OR_PATH_GENERAL", "E:/SANSORI/Reproducibility")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segmentation : identique pour les trois variantes, sinon la comparaison ne
# porterait plus seulement sur ce qu'on veut comparer.
SEG_SCALE_FACTOR = 1.0
SEG_BATCH_SIZE = 8

REQUIRED_FILES = (
    "registered_video.mp4",
    "mask.npz",
    "timestamp.txt",
    "transform.npz",
    "registration_params.json",
)

# Au-dela de cette fraction de frames recalees au masque VIDE, une sortie est
# rejetee meme si le moteur l'a declaree reussie. Le seuil n'est pas arbitraire :
# les deux moteurs prennent pour reference la frame dont l'aire de masque est la
# plus proche de la MEDIANE, et au-dela de la moitie de frames vides cette
# reference est elle-meme vide. Constate sur MARTINEZ_ALEJANDRA_OS3 (238 frames
# brutes sur 348 quasi noires) : la variante fullframe a rendu "ok" une video au
# masque vide sur les 348 frames, avec un dx constant que ``dx_all_zero`` ne voit pas.
MAX_EMPTY_MASK_FRAC = 0.5

# --------------------------------------------------------------------------- #
# Ancienne methode : le moteur ``register_videos``, en clair
# --------------------------------------------------------------------------- #
# Rien n'est laisse aux defauts de ``RegistrationConfig`` : ``correct_transversal``
# y vaut False, et le moteur renvoie alors un dx IDENTIQUEMENT NUL -- la variante
# s'appellerait "fullframe" sans faire le moindre recalage lateral.
#
# ``fovea_correction_enabled=False`` est delibere : ``lateral/dx.py::fovea_correction``
# applique le decalage de fovee SANS rejet temporel, SANS garde-fou d'amplitude et
# SANS garde NaN, puis caste le masque avec ``.to(torch.bool)`` -- une fovee NaN
# donne donc une frame entierement noire avec un masque entierement vrai, en
# silence. Sur cette cohorte ce n'est pas theorique : la chaine actuelle ecarte la
# trace de fovee sur 5 acquisitions, dont ROY_JOANNIE_OD2 et ses 114 frames sur
# 348 sans fovee exploitable. La variante est donc "correlation plein-cadre sur
# les frames brutes + BM alignee sur la courbe de la frame de reference", ce que
# la docstring de compute_registration.py designe comme l'ancienne methode.
OLD_ENGINE_CONFIG = RegistrationConfig(
    skip_first_n_frames=0,
    drop_last_n_frames=0,
    correct_transversal=True,
    correct_axial=True,
    flatten_rpe=False,
    axial_refinement=True,
    fovea_correction_enabled=False,
    lateral_method="fullframe",
    max_lateral_shift=16,
    smooth_transversal=False,
    smooth_transversal_sigma=2.0,
    crop_factor=0.66,
    scale_factor=1.0,
    transversal_bandpass=(0.02, 0.5),
    axial_bandpass=(0.02, 0.5),
    max_axial_shift=7,
    subpixel=True,
    batch_size=128,
)


@dataclass(frozen=True)
class Variant:
    key: str
    subdir: str
    engine: str  # "chain" (compute_registration) ou "fullframe" (register_videos)
    revision: str | None  # revision HuggingFace du modele de segmentation
    label: str


VARIANTS = {
    "a": Variant(
        key="a",
        subdir="registered_segformer2_flatten_choroid_xcorr",
        engine="chain",
        revision="version-2.0.0",
        label="SegFormer mit_b2 + chaine fovee/flatten/xcorr/A-scan",
    ),
    "b": Variant(
        key="b",
        subdir="registered_model1_flatten_choroid_xcorr",
        engine="chain",
        revision=None,
        label="U-Net se_resnet50 + chaine fovee/flatten/xcorr/A-scan",
    ),
    "c": Variant(
        key="c",
        subdir="registered_model1_fullframe",
        engine="fullframe",
        revision=None,
        label="U-Net se_resnet50 + moteur register_videos (fullframe)",
    ),
}
# ``b`` d'abord : c'est la seule variante comparable a une sortie deja sur E:,
# donc la seule qui permette de verifier le pilote avant de depenser des heures.
DEFAULT_ORDER = ["b", "a", "c"]


# --------------------------------------------------------------------------- #
# Import de la chaine de production, telle quelle
# --------------------------------------------------------------------------- #
def load_compute_registration():
    """``Astronauts/compute_registration.py`` importe comme module, sans copie.

    ``Astronauts/`` n'est pas un package : on charge le fichier par son chemin.
    Aucun effet de bord a l'import (``write_params_json`` n'est appele que depuis
    ``main()``, protege par le garde ``__main__``).
    """
    path = REPO / "Astronauts" / "compute_registration.py"
    spec = importlib.util.spec_from_file_location("or_compute_registration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Tracabilite
# --------------------------------------------------------------------------- #
def resolve_model_sha(repo_id: str, revision: str | None) -> str:
    """SHA de commit du modele sur le Hub -- ``main`` bouge, et c'est justement
    le modele qui distingue une variante de l'autre."""
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo_id, revision=revision).sha)
    except Exception as e:  # noqa: BLE001  (hors ligne, cache local, etc.)
        return f"inconnu ({type(e).__name__})"


def transform_stats(out_dir: Path) -> dict:
    """Statistiques relues de ``transform.npz`` (communes aux trois variantes)."""
    with np.load(out_dir / "transform.npz") as tr:
        keys = sorted(tr.files)
        dx = np.asarray(tr["dx"], dtype=np.float64)
        dy = np.asarray(tr["dy"], dtype=np.float64)
        stats = {
            "transform_keys": keys,
            "dx_ptp": float(np.ptp(dx)),
            "dx_std": float(dx.std()),
            # Attrape une regression ``correct_transversal=False`` : le moteur
            # renvoie alors un dx nul et ne recale rien lateralement.
            "dx_all_zero": bool(np.all(dx == 0)),
            "dy_std": float(np.nanstd(dy)),
        }
        if "dy_median" in tr.files:
            stats["dy_median_std"] = float(np.nanstd(np.asarray(tr["dy_median"])))
        if "bad_columns" in tr.files:
            stats["n_bad_columns"] = int(np.asarray(tr["bad_columns"]).sum())
    return stats


def mask_shape(out_dir: Path):
    """(T, H, W) lu de l'entete de ``mask.npz``, sans decompresser le masque."""
    with np.load(out_dir / "mask.npz") as data:
        t, h, w = (int(v) for v in data["shape"])
    return t, h, w


def write_params(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registration_params.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def base_meta(variant: Variant, slug: str, model_sha: str) -> dict:
    return {
        "variant": variant.subdir,
        "variant_key": variant.key,
        "variant_label": variant.label,
        "condition": slug,
        "engine": (
            "Astronauts/compute_registration.py::process_condition"
            if variant.engine == "chain"
            else "ocularrigidity.registration.rigid.register_videos"
        ),
        "seg_repo": DEFAULT_SEGMENTATION_REPO,
        "seg_revision": variant.revision or "main",
        "seg_commit_sha": model_sha,
        "seg_scale_factor": SEG_SCALE_FACTOR,
        "seg_batch_size": SEG_BATCH_SIZE,
        "seg_use_graphcut": True,  # defaut de ``infer``, il change les masques
        "skip_first_n_frames": 0,
        "drop_last_n_frames": 0,
        "device": DEVICE,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "created": datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Etat des sorties
# --------------------------------------------------------------------------- #
def is_complete(out_dir: Path) -> bool:
    """Les CINQ fichiers presents.

    Tester le seul mp4 ne suffit pas : ``export_registered_video`` ecrit la video
    AVANT le masque, et un plantage entre les deux laisserait une sortie
    incomplete que la reprise accepterait pour toujours. Il faut aussi un JSON au
    statut "ok" : une sortie rejetee est retentee au lot suivant, exactement comme
    un echec leve par la chaine.
    """
    if not all((out_dir / name).exists() for name in REQUIRED_FILES):
        return False
    params = read_params(out_dir)
    return bool(params) and params.get("status") == "ok"


def read_params(out_dir: Path):
    path = out_dir / "registration_params.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def check_consistency(raw_dir: Path, variant: Variant, shape):
    """Compare (T, H, W) aux variantes DEJA ecrites pour la meme acquisition.

    C'est ce qui donne son sens a "comparable image par image" : deux variantes
    qui n'ont pas le meme nombre de frames ne se comparent pas frame a frame.
    Retourne le message de desaccord, ou None.
    """
    for other in VARIANTS.values():
        if other.key == variant.key:
            continue
        params = read_params(raw_dir / other.subdir)
        if not params or params.get("status") != "ok":
            continue
        ref = (params.get("n_frames"), params.get("height"), params.get("width"))
        if None in ref:
            continue
        if tuple(ref) != tuple(shape):
            return f"{tuple(shape)} != {tuple(ref)} de la variante '{other.subdir}'"
    return None


# --------------------------------------------------------------------------- #
# Une acquisition x une variante
# --------------------------------------------------------------------------- #
def timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages (us) des B-scans, dans l'ordre -- sans relire les .tif.

    Meme valeur que ``build_cube_and_timestamps``, mais celui-ci empile aussi les
    349 images, ce qui serait paye deux fois ici.
    """
    series = load_ordered_oct_series(raw_dir)
    return np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )


def run_chain(cr, raw_dir: Path, out_dir: Path, model) -> dict:
    """Variantes a/b : la chaine en 4 temps de ``compute_registration.py``.

    ``process_condition`` ecrit ``cube.mp4`` / ``mask.npz`` / ``transform.npz``
    dans les deux dossiers qu'on lui donne -- ici le meme. Restent a poser le nom
    de fichier commun aux trois variantes et le ``timestamp.txt`` que ce lot-la
    n'ecrit pas (il recalcule les horodatages depuis le XML a chaque lecture).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    row = cr.process_condition(raw_dir, model, out_dir, out_dir)
    if row.get("status") != "ok":
        return row

    (out_dir / "cube.mp4").replace(out_dir / "registered_video.mp4")

    ts_us = timestamps_us(raw_dir)
    if ts_us.size != row["n_frames"]:
        # skip = drop = 0 : une frame ecrite <-> un B-scan du XML. Un desaccord
        # signifierait que le timestamp.txt ne decrit pas cette video.
        return {
            "status": "error",
            "reason": f"{ts_us.size} horodatages pour {row['n_frames']} frames",
        }
    (out_dir / "timestamp.txt").write_text(
        "\n".join(str(int(t)) for t in ts_us), encoding="utf-8"
    )
    return row


def run_fullframe(raw_dir: Path, out_dir: Path, model) -> dict:
    """Variante c : le moteur d'origine, via ``export_registered_video``.

    Il ecrit deja les cinq noms de fichiers voulus. ``overwrite=True`` parce que
    la decision de refaire ou non appartient a l'appelant (``is_complete``), pas
    a son propre test "la video existe".
    """
    result = export_registered_video(
        raw_dir,
        OLD_ENGINE_CONFIG,
        model,
        device=DEVICE,
        out_subdir=out_dir.name,
        overwrite=True,
        scale_factor=SEG_SCALE_FACTOR,  # defaut 2.0 : le passer est indispensable
        seg_batch_size=SEG_BATCH_SIZE,
        verbose=False,
    )
    if result.get("status") != "ok":
        return {
            "status": result.get("status", "error"),
            "reason": result.get("reason", ""),
        }

    t, h, w = mask_shape(out_dir)
    return {
        "status": "ok",
        "reason": "",
        "n_frames": int(result["n_frames"]),
        "fps": round(float(result["fps"]), 3),
        "height": h,
        "width": w,
    }


def discard_outputs(out_dir: Path) -> None:
    """Retire les fichiers de donnees d'une sortie rejetee ; seul reste le JSON qui
    dit pourquoi -- le meme etat qu'un echec leve par la chaine, qui n'ecrit rien.
    Une video "recalee" laissee a cote des donnees serait prise pour bonne."""
    for name in REQUIRED_FILES:
        if name != "registration_params.json":
            (out_dir / name).unlink(missing_ok=True)
    (out_dir / "cube.mp4").unlink(missing_ok=True)


def validate_output(raw_dir: Path, out_dir: Path, variant: Variant, row: dict) -> dict:
    """Controles d'une sortie que le moteur declare reussie ; rejet -> fichiers retires."""
    shape = (row["n_frames"], row["height"], row["width"])
    mismatch = check_consistency(raw_dir, variant, shape)
    if mismatch:
        discard_outputs(out_dir)
        return {"status": "error", "reason": f"dimensions incoherentes : {mismatch}"}

    area = load_mask(out_dir / "mask.npz").sum(axis=(1, 2))
    n_empty = int((area == 0).sum())
    if n_empty > MAX_EMPTY_MASK_FRAC * area.size:
        discard_outputs(out_dir)
        return {
            "status": "error",
            "reason": f"masque vide sur {n_empty}/{area.size} frames recalees",
            "n_empty_mask_frames": n_empty,
        }

    row = dict(row)
    row.update(transform_stats(out_dir))
    row["n_frames_registered"] = row["n_frames"]
    row["n_empty_mask_frames"] = n_empty
    return row


def process(cr, condition_dir: Path, variant: Variant, model, model_sha: str) -> dict:
    meta = base_meta(variant, condition_dir.name, model_sha)
    raw_dir = cr.find_raw_dir(condition_dir)
    if raw_dir is None:
        out_dir = condition_dir / "RawImages" / variant.subdir
        row = {"status": "skipped", "reason": "no_raw_dir"}
    else:
        out_dir = raw_dir / variant.subdir
        try:
            if variant.engine == "chain":
                row = run_chain(cr, raw_dir, out_dir, model)
            else:
                row = run_fullframe(raw_dir, out_dir, model)
        except Exception as e:  # noqa: BLE001
            print(f"  [erreur] {e}")
            traceback.print_exc()
            row = {"status": "error", "reason": str(e)[:300]}

        if row.get("status") == "ok":
            row = validate_output(raw_dir, out_dir, variant, row)

    if variant.engine == "fullframe":
        meta["registration_config"] = asdict(OLD_ENGINE_CONFIG)
        meta["fovea_correction_note"] = (
            "desactivee : lateral/dx.py::fovea_correction n'a ni rejet temporel, "
            "ni garde-fou d'amplitude, ni garde NaN (une fovee NaN noircit la "
            "frame et met le masque a True)"
        )
    else:
        meta["chain"] = {
            "fovea_correction": cr.FOVEA_CORRECTION,
            "max_fovea_failed_frac": cr.MAX_FOVEA_FAILED_FRAC,
            "max_fovea_spread": cr.MAX_FOVEA_SPREAD,
            "col_frac": list(cr.COL_FRAC),
            "max_lateral_shift": cr.MAX_LATERAL_SHIFT,
            "subpixel": cr.SUBPIXEL,
            "smooth_transversal": cr.SMOOTH_TRANSVERSAL,
            "axial_refinement": cr.AXIAL_REFINEMENT,
            "max_axial_shift": cr.MAX_AXIAL_SHIFT,
            "axial_bandpass": list(cr.AXIAL_BANDPASS),
            "flatten_config": asdict(cr.FLATTEN_CONFIG),
        }

    meta.update(row)
    write_params(out_dir, meta)
    return row


# --------------------------------------------------------------------------- #
# Bilan de cohorte
# --------------------------------------------------------------------------- #
def report(conditions) -> None:
    """Par variante : ce qui a abouti -- et surtout l'INTERSECTION des trois.

    Une acquisition peut echouer sous un modele et passer sous l'autre
    (``choroid_roi`` leve quand le tiers central ne contient aucun pixel de
    choroide, et la variante 'fullframe' ne l'appelle jamais). Comparer les
    variantes sans cette intersection reviendrait a comparer trois cohortes.
    """
    print("\n" + "=" * 78)
    print("  Bilan")
    print("=" * 78)
    ok_sets = {}
    for key in DEFAULT_ORDER:
        variant = VARIANTS[key]
        ok, bad = [], []
        for cond in conditions:
            out_dir = cond / "RawImages" / variant.subdir
            params = read_params(out_dir)
            if params and params.get("status") == "ok" and is_complete(out_dir):
                ok.append(cond.name)
            elif params:
                bad.append((cond.name, params.get("reason", "")))
        ok_sets[key] = set(ok)
        print(f"\n{variant.subdir} : {len(ok)} ok, {len(bad)} en echec")
        for name, reason in bad:
            print(f"  [echec] {name} : {reason}")

    inter = set.intersection(*ok_sets.values())
    union = set.union(*ok_sets.values())
    print(f"\nIntersection des variantes : {len(inter)} acquisition(s) comparables")
    for name in sorted(union - inter):
        manque = [VARIANTS[k].subdir for k in DEFAULT_ORDER if name not in ok_sets[k]]
        print(f"  [hors intersection] {name} : manque {', '.join(manque)}")


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Exporte les videos recalees des acquisitions, une variante par dossier."
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_ORDER),
        help="cles des variantes a produire, dans l'ordre (defaut : b,a,c)",
    )
    parser.add_argument(
        "--slug", action="append", help="ne traiter que cette acquisition (repetable)"
    )
    parser.add_argument(
        "--limit", type=int, help="s'arreter apres N acquisitions par variante"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="refaire meme si la sortie est complete"
    )
    parser.add_argument(
        "--check", action="store_true", help="afficher le bilan et sortir"
    )
    args = parser.parse_args(argv)

    if not PATH_GENERAL.exists():
        print(f"Dossier introuvable : {PATH_GENERAL}")
        return 2

    conditions = list(layout.iter_condition_dirs(PATH_GENERAL))
    if args.slug:
        wanted = set(args.slug)
        conditions = [c for c in conditions if c.name in wanted]
        missing = wanted - {c.name for c in conditions}
        if missing:
            print(f"Acquisition(s) introuvable(s) : {', '.join(sorted(missing))}")
            return 2

    if args.check:
        report(conditions)
        return 0

    keys = [k.strip() for k in args.variants.split(",") if k.strip()]
    unknown = [k for k in keys if k not in VARIANTS]
    if unknown:
        print(f"Variante(s) inconnue(s) : {', '.join(unknown)}  (connues : a, b, c)")
        return 2
    if DEFAULT_OUTPUT_SUBDIR in {VARIANTS[k].subdir for k in keys}:
        # Garde-fou : ``registered`` nu est lu par testing_app et par
        # compute_rigidity_compare_mask_model comme la sortie de reference.
        print(f"Le nom '{DEFAULT_OUTPUT_SUBDIR}' est reserve.")
        return 2

    print(f"{len(conditions)} acquisition(s) sous {PATH_GENERAL}")
    print(f"Variantes : {', '.join(VARIANTS[k].subdir for k in keys)}")
    print(f"Peripherique : {DEVICE}\n")

    cr = load_compute_registration()

    for key in keys:
        variant = VARIANTS[key]
        print("=" * 78)
        print(f"  {variant.subdir}\n  {variant.label}")
        print("=" * 78)

        todo = [
            cond
            for cond in conditions
            if args.overwrite or not is_complete(cond / "RawImages" / variant.subdir)
        ]
        if args.limit is not None:
            todo = todo[: args.limit]
        print(f"{len(todo)} a traiter, {len(conditions) - len(todo)} deja complete(s)\n")
        if not todo:
            continue

        model_sha = resolve_model_sha(DEFAULT_SEGMENTATION_REPO, variant.revision)
        model = get_choroid_segmentation_model(revision=variant.revision)
        hp = dict(getattr(model, "hparams", {}) or {})
        print(f"modele : {DEFAULT_SEGMENTATION_REPO}@{variant.revision or 'main'}")
        print(
            f"  arch={hp.get('arch')} encoder={hp.get('encoder_name')} "
            f"sha={model_sha[:12]}\n"
        )

        n_ok = n_err = 0
        for cond in todo:
            print(cond.name, flush=True)
            t0 = time.time()
            row = process(cr, cond, variant, model, model_sha)
            dt = time.time() - t0
            if row.get("status") == "ok":
                n_ok += 1
                print(
                    f"  -> {row['n_frames']} frames {row['height']}x{row['width']} "
                    f"@ {row['fps']:.1f} fps | dx ptp {row['dx_ptp']:.2f} px"
                    f"{' [dx NUL]' if row['dx_all_zero'] else ''} | {dt:.0f} s",
                    flush=True,
                )
            else:
                n_err += 1
                print(
                    f"  [{row.get('status')}] {row.get('reason')} | {dt:.0f} s",
                    flush=True,
                )

        print(f"\n{variant.subdir} : {n_ok} ecrite(s), {n_err} en echec.\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report(conditions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
