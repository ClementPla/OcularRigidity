"""
quantify_image_noise.py

Quantification du BRUIT D'IMAGE sur toutes les donnees recalees, exprimee dans
la parametrisation du modele de bruit de ``simulate_svd_radial_gaussians.py``,
pour pouvoir regler ``noise_level`` de la simulation sur du mesure plutot que
sur une valeur posee a la main.

Principe (celui demande) : comparer chaque frame du cube a la MOYENNE
TEMPORELLE du cube. Le cube etant deja recale, le meme pixel voit la meme
structure d'une frame a l'autre ; ce qui reste autour de sa moyenne est du
bruit (plus la pulsation choroidienne et le residu de recalage -- voir
"Biais connus" plus bas).

Modele de bruit de la simulation (``sim.add_image_noise``) :

    I = R * G + n,   G ~ Gamma(L, 1/L)  (moyenne 1, variance 1/L),
                     n ~ N(0, floor)

d'ou, a l'echelle d'un pixel dont la reflectivite R est constante dans le
temps :

    E[I]   = R
    Var[I] = R^2 / L + floor^2

On mesure donc, par pixel de la ROI, la moyenne temporelle ``mu`` (estimateur
de R) et la variance temporelle ``var``, on agrege par casiers de ``mu``, puis
on ajuste ``var = a * mu^2 + b``. Il vient ``L = 1/a`` et ``floor = sqrt(b)``,
et -- puisque la simulation pose ``L = N_LOOKS_REALISTIC / noise_level^2`` et
``floor = BRUIT_ADDITIF_REALISTIC * noise_level`` -- deux estimations
independantes du ``noise_level`` equivalent :

    noise_level_speckle = sqrt(N_LOOKS_REALISTIC * a)
    noise_level_floor   = floor_en_unites_sim / BRUIT_ADDITIF_REALISTIC

``a`` est sans dimension donc directement comparable a la simulation ; ``b``
est en unites d'intensite (uint8 ici) et doit etre converti en unites sim, ce
qui se fait en ancrant l'intensite reelle sur l'intensite d'une frame sim
propre (voir ``SimReference``).

DEUX MODELES DE PLUS sont ajustes en parallele, parce que le modele speckle
ci-dessus ne decrit PAS ce qu'on observe : l'exposant mesure vaut ~0.5-1, pas
2. Les B-scans exportes sont compresses logarithmiquement, le speckle
multiplicatif n'y est donc plus multiplicatif. Les deux modeles ajoutes :

  - Poisson :   var = k * mu + floor^2      (bruit de comptage)
  - loi puissance libre : var = c * mu^p    (ajustee en log-log, p diagnostique)

Les trois sont dans le CSV avec leur qualite d'ajustement (``*_r2``), donc on
peut voir lequel decrit reellement les donnees avant de choisir comment bruiter
la simulation.

AVEC ET SANS NORMALISATION -- deux variantes par cube :

  - ``brut``      : intensites telles quelles.
  - ``normalise`` : chaque frame divisee par sa moyenne spatiale, exactement
    comme ``PixelTraceConfig.normalize_intensity`` le fait en amont de la SVD
    (``motion/pulsation/traces/pixel.py``), puis remise a l'echelle par la
    moyenne globale des moyennes de frame pour que ``b`` reste en unites
    d'intensite comparables entre les deux variantes (une constante
    multiplicative laisse ``a`` inchange et multiplie ``b`` par son carre).

L'ecart entre les deux isole la part du "bruit" qui est en fait une
fluctuation de GAIN globale par frame (colonne ``frame_gain_cv``) : elle est
multiplicative et commune a toute l'image, donc elle gonfle ``a`` sans etre du
speckle. C'est aussi un terme que la simulation n'a pas du tout aujourd'hui.

Biais connus (les estimations sont a lire comme des bornes) :
  - MAJORANT : la pulsation cardiaque et le residu de recalage font varier
    l'intensite d'un pixel dans le temps sans etre du bruit. La colonne
    ``var_lag1`` (variance estimee sur les differences entre frames
    consecutives, ``E[(I_t - I_{t-1})^2]/2``) attenue fortement ces
    composantes lentes : tout ajustement existe en version ``mean`` (demande)
    et ``lag1`` (controle).
  - MINORANT : les cubes lus sont encodes en x264 CRF 18 (``write_gray_mp4``)
    et sont issus d'une interpolation bilineaire de recalage -- les deux
    lissent un peu le bruit.

Entree (meme arborescence que ``compute_one_cycle_pixel_svd.py``) :
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz

Sorties (sous ``Astronauts/simulation_output/image_noise/``) :
  - ``image_noise_summary.csv`` : une ligne par (condition, variante).
  - ``<variante_masque>/<astro>/<moment>/<condition>.npz`` : courbes
    variance-vs-moyenne par casier, histogramme des residus standardises et
    moyennes de frame, pour la figure du notebook
    ``simulate_svd_radial_gaussians_viewer.ipynb``.

Lancer :  python Astronauts/quantify_image_noise.py
"""

from __future__ import annotations

import csv
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch  # noqa: F401 -- doit etre importe AVANT decord (via data.compression),
# sinon le chargement des DLL de torch echoue (WinError 1114) sur cette machine.

from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_svd_radial_gaussians as sim  # noqa: E402

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANTS = ("model1_scale_1.0",)  # le bruit est une propriete de l'image,
# pas du masque : une seule variante suffit. En ajouter une ici si on veut
# verifier que la ROI (donc le modele de segmentation) ne change rien.
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"

OUT_ROOT = Path(__file__).parent / "simulation_output" / "image_noise"
SUMMARY_CSV = OUT_ROOT / "image_noise_summary.csv"

LIMIT_CONDITIONS: Optional[int] = None  # None = tout ; un entier pour un essai
OVERWRITE = True

#: Casiers de ``mu`` (quantiles) sur lesquels la variance est moyennee avant
#: l'ajustement : sans ca les pixels sombres, tres majoritaires, imposent seuls
#: la pente.
N_BINS = 32
MIN_PIX_PER_BIN = 50

#: Rejet de frames aberrantes (recalage parti, frame noire) : ecart robuste de
#: la moyenne d'intensite sur la ROI.
FRAME_OUTLIER_MAD = 6.0

#: ROI = intersection du masque dans le temps, comme ``PixelTraceSource.base_roi``.
#: Sur une dizaine de conditions cette intersection est VIDE (masque trop mobile
#: ou segmentation qui decroche sur quelques frames) : plutot que de perdre ces
#: cubes, on retombe alors sur les pixels masques dans au moins cette fraction
#: des frames. La colonne ``roi_frame_fraction`` du CSV dit laquelle des deux
#: definitions a servi.
ROI_MIN_PIXELS = 500
ROI_FALLBACK_FRAME_FRACTION = 0.9

#: Histogramme des residus standardises (I - mu)/sigma, pour la forme de la
#: distribution de bruit (Gaussienne vs Gamma).
RESID_HIST_RANGE = (-6.0, 6.0)
RESID_HIST_BINS = 121
RESID_MAX_SAMPLES = 4_000_000

VARIANTS = ("brut", "normalise")


# --------------------------------------------------------------------------- #
# Reference d'intensite de la simulation : une frame propre (sans bruit, sans
# jitter, sans pulsation) rendue avec la meme calibration que le balayage, dont
# on lit deux niveaux d'ancrage sur le masque de la simulation. Sert a convertir
# ``floor`` (unites d'intensite reelles) en unites sim.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimReference:
    R_mean: float  # moyenne de R = cube + FOND sur le masque
    R_p90: float  # 90e centile de R sur le masque (ancrage "structures claires")


def sim_reference() -> Optional[SimReference]:
    try:
        calib = sim.load_real_calibration()
        anatomy = sim.build_ring_anatomy()
        shape = sim.crop_shape(calib)
        cube = sim.render_cube(
            anatomy,
            calib,
            np.zeros(1),
            1.0,
            0.0,  # amplitude nulle : anneau au repos
            np.zeros(1),
            np.zeros((1, shape[1])),
            shape,
        )
        R = cube[0] + sim.FOND
        mask = sim.build_fixed_mask(shape)
        return SimReference(
            R_mean=float(R[mask].mean()), R_p90=float(np.percentile(R[mask], 90))
        )
    except Exception as exc:  # noqa: BLE001 -- XML de calibration absent, etc.
        print(f"[avert] reference d'intensite simulee indisponible : {exc}")
        return None


# --------------------------------------------------------------------------- #
# Ajustements var = f(mu)
# --------------------------------------------------------------------------- #
def _wls(A: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Moindres carres ponderes + R^2 pondere."""
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


@dataclass
class BinnedCurve:
    """Statistiques par casier de ``mu``, base de tous les ajustements."""

    mu: np.ndarray  # moyenne de mu dans le casier
    mu2: np.ndarray  # moyenne de mu^2, DEBIAISEE de la variance d'echantillonnage
    var: np.ndarray  # moyenne de la variance temporelle par pixel
    count: np.ndarray  # nombre de pixels


def binned_curve(mu: np.ndarray, var: np.ndarray, n_frames: int) -> BinnedCurve:
    """Casiers en quantiles de ``mu``.

    ``mu`` est lui-meme bruite (variance ``var/n_frames``), donc la moyenne de
    ``mu^2`` surestime celle de ``R^2`` d'exactement ``var/n_frames`` : on la
    retranche, sinon l'ajustement speckle recupere ce biais dans ``floor``.
    """
    edges = np.quantile(mu, np.linspace(0.0, 1.0, N_BINS + 1))
    idx = np.clip(np.digitize(mu, edges[1:-1]), 0, N_BINS - 1)

    rows = []
    for k in range(N_BINS):
        sel = idx == k
        n = int(sel.sum())
        if n < MIN_PIX_PER_BIN:
            continue
        v = float(var[sel].mean())
        rows.append((float(mu[sel].mean()), float((mu[sel] ** 2).mean()) - v / n_frames, v, n))

    if not rows:
        raise ValueError("aucun casier de mu suffisamment peuple")
    arr = np.array(rows, dtype=float)
    return BinnedCurve(mu=arr[:, 0], mu2=arr[:, 1], var=arr[:, 2], count=arr[:, 3])


def fit_speckle(c: BinnedCurve) -> dict:
    """var = mu^2 / L + floor^2 -- le modele de la simulation."""
    (a, b), r2 = _wls(np.stack([c.mu2, np.ones_like(c.mu2)], axis=1), c.var, c.count)
    return {
        "speckle_a": a,
        "speckle_var0": b,
        "speckle_n_looks": 1.0 / a if a > 0 else float("nan"),
        "speckle_contrast": np.sqrt(a) if a > 0 else float("nan"),
        "speckle_floor": np.sqrt(b) if b > 0 else 0.0,
        "speckle_r2": r2,
    }


def fit_poisson(c: BinnedCurve) -> dict:
    """var = k * mu + floor^2 -- bruit de comptage."""
    (k, b), r2 = _wls(np.stack([c.mu, np.ones_like(c.mu)], axis=1), c.var, c.count)
    return {
        "poisson_k": k,
        "poisson_var0": b,
        "poisson_floor": np.sqrt(b) if b > 0 else 0.0,
        "poisson_r2": r2,
    }


def fit_power(c: BinnedCurve) -> dict:
    """var = c * mu^p, ajustee en log-log : ``p`` dit quel modele est le bon
    (1 = Poisson, 2 = speckle multiplicatif, 0 = additif pur)."""
    ok = (c.mu > 0) & (c.var > 0)
    if ok.sum() < 3:
        return {"power_p": float("nan"), "power_c": float("nan"), "power_r2": float("nan")}
    x, y, w = np.log(c.mu[ok]), np.log(c.var[ok]), c.count[ok]
    (p, log_c), r2 = _wls(np.stack([x, np.ones_like(x)], axis=1), y, w)
    return {"power_p": p, "power_c": float(np.exp(log_c)), "power_r2": r2}


def to_sim_units(fits: dict, mu_mean: float, mu_p90: float, ref: Optional[SimReference]) -> dict:
    """``noise_level`` equivalent de la simulation.

    - Le terme multiplicatif est sans dimension : ``a`` se lit directement.
    - Le plancher additif est en unites d'intensite : on le convertit en
      ancrant l'intensite reelle sur celle d'une frame sim propre. Deux
      ancrages (moyenne de ROI et 90e centile) encadrent l'incertitude de
      cette mise a l'echelle, les supports n'etant pas identiques (la ROI
      reelle est la choroide, le masque sim couvre aussi son fond).
    """
    a = fits["speckle_a"]
    out = {
        "noise_level_speckle": (
            float(np.sqrt(sim.N_LOOKS_REALISTIC * a)) if a > 0 else float("nan")
        ),
        "floor_rel_mean": fits["speckle_floor"] / mu_mean if mu_mean > 0 else float("nan"),
        "floor_rel_p90": fits["speckle_floor"] / mu_p90 if mu_p90 > 0 else float("nan"),
    }
    if ref is None:
        out["noise_level_floor_mean"] = float("nan")
        out["noise_level_floor_p90"] = float("nan")
        out["poisson_k_sim"] = float("nan")
        return out
    out["noise_level_floor_mean"] = (
        out["floor_rel_mean"] * ref.R_mean / sim.BRUIT_ADDITIF_REALISTIC
    )
    out["noise_level_floor_p90"] = (
        out["floor_rel_p90"] * ref.R_p90 / sim.BRUIT_ADDITIF_REALISTIC
    )
    # var = k*mu est homogene a une intensite : k se convertit comme mu.
    scale = ref.R_p90 / mu_p90 if mu_p90 > 0 else float("nan")
    out["poisson_k_sim"] = fits["poisson_k"] * scale
    return out


# --------------------------------------------------------------------------- #
# Chargement / preparation d'un cube
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    variant: str  # variante de masque (SegmentationVariations)
    astro: str
    moment: str
    condition: str
    frames_path: Path
    mask_path: Path


def iter_conditions(variant: str) -> Iterator[Condition]:
    root = SEGVAR_ROOT / variant
    for frames_path in sorted((root / FRAMES_SUBDIR).rglob("cube.mp4")):
        rel = frames_path.relative_to(root / FRAMES_SUBDIR).parent
        parts = rel.parts
        if len(parts) != 3:
            continue
        mask_path = root / MASKS_SUBDIR / rel / "mask.npz"
        if not mask_path.exists():
            continue
        yield Condition(variant, parts[0], parts[1], parts[2], frames_path, mask_path)


def keep_frames(roi_means: np.ndarray) -> np.ndarray:
    """Masque des frames a garder : moyenne d'intensite sur la ROI dans
    ``FRAME_OUTLIER_MAD`` ecarts robustes de la mediane."""
    med = np.median(roi_means)
    mad = 1.4826 * np.median(np.abs(roi_means - med))
    if mad <= 0:
        return np.ones(len(roi_means), dtype=bool)
    return np.abs(roi_means - med) <= FRAME_OUTLIER_MAD * mad


def normalize_traces(x: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Normalisation d'intensite par frame, comme ``PixelTraceSource``.

    ``pixel.py`` divise chaque frame par la moyenne de la frame ENTIERE (bords
    recales a zero compris) : ``gains`` vient donc du cube complet, meme si on
    ne l'applique qu'aux colonnes de la ROI -- un facteur scalaire par frame
    commute avec l'extraction de la ROI. La remise a l'echelle par la moyenne
    globale des moyennes de frame ne change que les unites (``a`` invariant,
    ``b`` multiplie par une constante), et rend ``floor`` comparable a la
    variante brute.
    """
    g = (gains / gains.mean()).astype(np.float32)
    return x / g[:, None]


# --------------------------------------------------------------------------- #
# Statistiques de bruit d'un cube
# --------------------------------------------------------------------------- #
def analyze_variant(x: np.ndarray, n_frames_total: int) -> dict:
    """``x`` : (T, N) intensites des pixels de la ROI, frames deja filtrees."""
    T = x.shape[0]
    mu = x.mean(axis=0)
    var_mean = x.var(axis=0, ddof=1)
    d = np.diff(x, axis=0)
    var_lag1 = (d * d).mean(axis=0) / 2.0

    ok = (mu > 0) & (mu < 255) & np.isfinite(mu) & np.isfinite(var_mean)
    mu, var_mean, var_lag1 = mu[ok], var_mean[ok], var_lag1[ok]
    if mu.size < MIN_PIX_PER_BIN:
        # Arrive quand la ROI tombe entierement sur des colonnes mises a zero
        # par le recalage : le cube est noir la ou on voulait mesurer.
        raise ValueError(
            f"seulement {mu.size} pixels exploitables dans la ROI "
            "(intensite nulle ou saturee partout)"
        )

    curves, fits = {}, {}
    for kind, var in (("mean", var_mean), ("lag1", var_lag1)):
        c = binned_curve(mu, var, T)
        curves[kind] = c
        f = {**fit_speckle(c), **fit_poisson(c), **fit_power(c)}
        fits[kind] = f

    # Forme de la distribution du bruit : residus standardises, sous-echantillonnes.
    sigma = np.sqrt(np.where(var_mean > 0, var_mean, np.nan))
    resid = (x[:, ok] - mu[None, :]) / sigma[None, :]
    resid = resid[np.isfinite(resid)]
    if resid.size > RESID_MAX_SAMPLES:
        # Sous-echantillonnage regulier. Un pas ENTIER pourrait tomber sur un
        # diviseur du nombre de pixels et ne visiter que quelques colonnes ;
        # ``linspace`` derive d'un pixel a l'autre et balaie donc frames ET
        # pixels. (Un tirage aleatoire sans remise coute une permutation de
        # dizaines de millions d'elements pour le meme histogramme.)
        resid = resid[
            np.linspace(0, resid.size - 1, RESID_MAX_SAMPLES).astype(np.int64)
        ]
    hist, edges = np.histogram(resid, bins=RESID_HIST_BINS, range=RESID_HIST_RANGE)
    skew = float(np.mean(resid**3))
    kurt = float(np.mean(resid**4) - 3.0)

    return {
        "n_pixels": int(mu.size),
        "n_frames_used": T,
        "n_frames_total": n_frames_total,
        "mu_mean": float(mu.mean()),
        "mu_median": float(np.median(mu)),
        "mu_p90": float(np.percentile(mu, 90)),
        "resid_skew": skew,
        "resid_excess_kurtosis": kurt,
        # skew d'une Gamma(L) = 2/sqrt(L) : lecture independante de l'ajustement
        "n_looks_from_skew": float((2.0 / skew) ** 2) if skew > 0 else float("nan"),
        "curves": curves,
        "fits": fits,
        "hist": hist,
        "hist_edges": edges,
    }


def process_condition(cond: Condition, ref: Optional[SimReference]) -> list[dict]:
    frames = read_gray(str(cond.frames_path))  # (T, H, W) uint8, garde tel quel
    masks = np.asarray(load_mask(cond.mask_path), dtype=bool)
    if frames.shape != masks.shape:
        raise ValueError(f"frames {frames.shape} != masques {masks.shape}")

    roi = masks.all(axis=0)  # meme ROI que PixelTraceSource.base_roi (sans col/row_frac)
    roi_frame_fraction = 1.0
    if roi.sum() < ROI_MIN_PIXELS:
        roi_frame_fraction = ROI_FALLBACK_FRAME_FRACTION
        roi = masks.mean(axis=0) >= roi_frame_fraction
        print(
            f"  [repli] intersection stricte vide -> ROI = pixels masques dans "
            f">= {roi_frame_fraction:.0%} des frames ({int(roi.sum())} px)"
        )
    if roi.sum() < ROI_MIN_PIXELS:
        raise ValueError(f"ROI trop petite ({int(roi.sum())} px)")

    n_total = frames.shape[0]
    x_all = frames[:, roi].astype(np.float32)  # (T, N) -- seul gros tableau garde
    gains_all = frames.mean(axis=(1, 2))  # moyenne de frame ENTIERE, cf. pixel.py

    keep = keep_frames(x_all.mean(axis=1))
    x, gains = x_all[keep], gains_all[keep]
    del x_all
    if x.shape[0] < 20:
        raise ValueError(f"trop peu de frames retenues ({x.shape[0]}/{n_total})")

    gain_cv = float(gains.std(ddof=1) / gains.mean())

    per_variant = {
        "brut": analyze_variant(x, n_total),
        "normalise": analyze_variant(normalize_traces(x, gains), n_total),
    }

    rows = []
    npz = {
        "astro": cond.astro,
        "moment": cond.moment,
        "condition": cond.condition,
        "mask_variant": cond.variant,
        "frame_means": gains_all.astype(np.float32),
        "frame_kept": keep,
        "frame_gain_cv": gain_cv,
        "n_frames_total": n_total,
        "n_frames_used": int(keep.sum()),
        "roi_n_pixels": int(roi.sum()),
        "roi_frame_fraction": roi_frame_fraction,
        "sim_R_mean": ref.R_mean if ref else float("nan"),
        "sim_R_p90": ref.R_p90 if ref else float("nan"),
    }

    for variant, res in per_variant.items():
        for kind in ("mean", "lag1"):
            c = res["curves"][kind]
            npz[f"{variant}_{kind}_bin_mu"] = c.mu.astype(np.float32)
            npz[f"{variant}_{kind}_bin_mu2"] = c.mu2.astype(np.float32)
            npz[f"{variant}_{kind}_bin_var"] = c.var.astype(np.float32)
            npz[f"{variant}_{kind}_bin_count"] = c.count.astype(np.int64)
            for key, value in res["fits"][kind].items():
                npz[f"{variant}_{kind}_{key}"] = float(value)
        npz[f"{variant}_resid_hist"] = res["hist"]
        npz[f"{variant}_resid_hist_edges"] = res["hist_edges"].astype(np.float32)

        row = {
            "mask_variant": cond.variant,
            "astro": cond.astro,
            "moment": cond.moment,
            "condition": cond.condition,
            "variant": variant,
            "n_frames_total": n_total,
            "n_frames_used": res["n_frames_used"],
            "n_pixels": res["n_pixels"],
            "roi_frame_fraction": roi_frame_fraction,
            "frame_gain_cv": gain_cv,
            "mu_mean": res["mu_mean"],
            "mu_median": res["mu_median"],
            "mu_p90": res["mu_p90"],
            "resid_skew": res["resid_skew"],
            "resid_excess_kurtosis": res["resid_excess_kurtosis"],
            "n_looks_from_skew": res["n_looks_from_skew"],
        }
        for kind in ("mean", "lag1"):
            for key, value in res["fits"][kind].items():
                row[f"{kind}_{key}"] = value
            row.update(
                {
                    f"{kind}_{k}": v
                    for k, v in to_sim_units(
                        res["fits"][kind], res["mu_mean"], res["mu_p90"], ref
                    ).items()
                }
            )
        row["npz_path"] = str(npz_path_for(cond))
        rows.append(row)

    out_path = npz_path_for(cond)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **npz)
    return rows


def npz_path_for(cond: Condition) -> Path:
    return (
        OUT_ROOT / cond.variant / cond.astro / cond.moment / f"{cond.condition}.npz"
    )


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def write_summary(rows: list[dict]) -> None:
    if not rows:
        print("Aucune condition traitee : CSV non ecrit.")
        return
    fields = list(rows[0].keys())
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} lignes ecrites dans {SUMMARY_CSV}")


def main() -> None:
    ref = sim_reference()
    if ref is not None:
        print(
            f"Reference sim (frame propre, masque sim) : R_mean={ref.R_mean:.4f} "
            f"R_p90={ref.R_p90:.4f} (FOND={sim.FOND}, "
            f"L_realiste={sim.N_LOOKS_REALISTIC}, floor_realiste={sim.BRUIT_ADDITIF_REALISTIC})"
        )

    rows: list[dict] = []
    n_done = 0
    for variant in MASK_VARIANTS:
        for cond in iter_conditions(variant):
            if LIMIT_CONDITIONS is not None and n_done >= LIMIT_CONDITIONS:
                break
            label = f"{cond.variant}/{cond.astro}/{cond.moment}/{cond.condition}"
            if not OVERWRITE and npz_path_for(cond).exists():
                print(f"  [skip] deja traite : {label}")
                continue
            print(label)
            try:
                new_rows = process_condition(cond, ref)
            except Exception as exc:  # noqa: BLE001
                print(f"  [erreur] {exc}")
                traceback.print_exc()
                continue
            rows.extend(new_rows)
            n_done += 1
            for row in new_rows:
                print(
                    f"  {row['variant']:>10s} : "
                    f"L={row['mean_speckle_n_looks']:7.1f} "
                    f"floor={row['mean_speckle_floor']:5.2f} "
                    f"(R2={row['mean_speckle_r2']:.3f}) | "
                    f"Poisson k={row['mean_poisson_k']:5.2f} "
                    f"(R2={row['mean_poisson_r2']:.3f}) | "
                    f"p={row['mean_power_p']:.2f} | "
                    f"noise_level_speckle={row['mean_noise_level_speckle']:.2f}"
                )

    write_summary(rows)

    if rows:
        import statistics

        for variant in VARIANTS:
            sub = [r for r in rows if r["variant"] == variant]
            if not sub:
                continue
            print(
                f"\nMediane sur {len(sub)} conditions [{variant}] : "
                f"p={statistics.median(r['mean_power_p'] for r in sub):.2f} · "
                f"L_speckle={statistics.median(r['mean_speckle_n_looks'] for r in sub):.1f} · "
                f"noise_level_speckle="
                f"{statistics.median(r['mean_noise_level_speckle'] for r in sub):.2f} · "
                f"floor/mu={statistics.median(r['mean_floor_rel_mean'] for r in sub):.3f} · "
                f"gain_cv={statistics.median(r['frame_gain_cv'] for r in sub):.3f}"
            )


if __name__ == "__main__":
    main()
