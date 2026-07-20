"""
compute_rigidity_time_series.py

Calcul par lot de la rigidite oculaire a partir des B-scans OCT enregistres.
Etend le notebook ``test_computation_rigidity.ipynb`` a l'ensemble du jeu SANSORI.

Pour chaque condition :
  1. (si ``mask.npz`` present) calcule ``deltaY`` a partir des masques de choroide
     et le sauvegarde dans ``deltaY.npz`` ;
  2. charge le vecteur temps filtre ``t_filt`` (``timeSeries.mat`` + ``idsFilteredMmi``),
     la HR, l'IOP et l'OPA depuis ``visit_data.csv`` et l'AL depuis la base
     ``sansori_db.db`` (table ``Measurements``) ;
  3. estime l'amplitude pulsatile par A-scan via la voie PASSE-BANDE + HILBERT
     (Methode 1 du notebook) ;
  4. en deduit CT, delta_CT, delta_V et le coefficient de rigidite k (Sayah et al. 2020) ;
  5. enregistre 3 figures par condition + les tableaux .npz pour les regenerer.

L'AL est appariee a la condition par (code astronaute, oeil, moment), deduits de
l'arborescence des dossiers. Lorsqu'une valeur (AL/IOP/OPA) est absente, la colonne
correspondante recoit ``<NA>`` et ``k`` n'est alors pas calcule (``<NA>``).

Sortie globale : une table CSV avec, par condition,
``hr`` (BPM), ``AL`` (mm), ``IOP`` (mmHg), ``OPA`` (mmHg),
``CT`` (mm), ``delta_CT`` (mm), ``delta_V`` (mm^3), ``k`` (mm^-3),
precedees de colonnes d'identification pour la tracabilite. Les valeurs
manquantes sont ecrites ``<NA>``.
"""

from pathlib import Path
import csv
import re
import sqlite3
import traceback

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, medfilt

import matplotlib
matplotlib.use("Agg")          # backend non interactif : sauvegarde sur disque
import matplotlib.pyplot as plt

from ocularrigidity.thickness.features import compute_deltaY_masks
from ocularrigidity.data.io import load_mask


# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")
PIX_Y_SIZE = 0.0039          # mm pix-1
AL_FACTOR = 0.87             # facteur d'ajustement axial (Sayah et al. 2020)
HR_HALF_BAND_BPM = 7.0       # demi-largeur de bande autour de la HR (BPM)
OUTPUT_CSV = PATH_GENERAL / "rigidity_time_series.csv"

# Base de donnees clinique (AL)
DB_PATH = PATH_GENERAL / "sansori_db.db"
AL_DESCRIPTION = "Biometry"  # longueur axiale (mm) dans Measurements.description


# --------------------------------------------------------------------------- #
# Rigidite oculaire (Sayah et al. 2020)
# --------------------------------------------------------------------------- #
def sayah_k(deltaCT_mm, CT_mm, AL, IOP, OPA, al_factor=AL_FACTOR):
    """
    Coefficient de rigidite oculaire k (Sayah et al. 2020), en mm^-3.

    Toutes les longueurs sont en UNITES PHYSIQUES (mm), pas en pixels — la
    conversion pixel -> mm (``PIX_Y_SIZE``) se fait cote appelant, cette
    fonction est independante de la source de ``deltaCT_mm``/``CT_mm``
    (video complete ou video one-cycle).

    Parameters
    ----------
    deltaCT_mm : amplitude choroidienne crete-a-crete (mm).
    CT_mm      : epaisseur choroidienne de base (mm).
    AL, IOP, OPA : longueur axiale (mm) et pressions cliniques (mmHg).

    Returns
    -------
    (k, deltaV) : k en mm^-3, deltaV en mm^3 ; ``nan`` si une donnee necessaire
    est absente (AL, IOP, OPA) ou si IOP == 0.
    """
    if np.isfinite(AL):
        AL_adj = AL * al_factor
        deltaV = np.pi / 2.0 * (AL_adj + CT_mm) ** 2 * deltaCT_mm
    else:
        deltaV = np.nan

    if np.isfinite(IOP) and np.isfinite(OPA) and np.isfinite(deltaV) and IOP != 0:
        IOP_IOP0 = (IOP + OPA) / IOP
        k = np.log(IOP_IOP0) / deltaV
    else:
        k = np.nan
    return k, deltaV


# --------------------------------------------------------------------------- #
# Donnees cliniques depuis la base sansori_db.db
# --------------------------------------------------------------------------- #
def _date_to_moment(date):
    """Convertit un libelle de date SANSORI (ex. 'L-21_18m') en moment de vol."""
    d = str(date)
    if "L-" in d:
        return "before"
    if "L+" in d or "R-" in d:
        return "during"
    if "R+" in d:
        return "after"
    return None


def load_clinical_db(db_path):
    """
    Charge une fois la table ``Measurements`` et le mapping astronaute Id -> Code.

    Returns
    -------
    meas : DataFrame
        Colonnes ``astronaut`` (= Code), ``Eye``, ``moment`` (before/during/after),
        ``description`` et ``value_num`` (valeur numerique, NaN si non convertible).
    id2code : dict[int, int | None]
        Numero d'astronaute (prefixe de dossier) -> Code present dans Measurements.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Base introuvable : {db_path}")

    con = sqlite3.connect(str(db_path))
    try:
        astro = pd.read_sql_query("SELECT Id, Code FROM Astronauts", con)
        meas = pd.read_sql_query(
            "SELECT astronaut, date, description, value, Eye FROM Measurements", con)
    finally:
        con.close()

    meas["value_num"] = pd.to_numeric(meas["value"], errors="coerce")
    meas["moment"] = meas["date"].map(_date_to_moment)

    id2code = {}
    for r in astro.itertuples(index=False):
        code = str(r.Code).strip()
        id2code[int(r.Id)] = int(code) if code.isdigit() else None
    return meas, id2code


def lookup_clinical(meas, code, eye, moment, description):
    """
    Moyenne des valeurs (code, oeil, moment) pour une ``description`` donnee.

    Renvoie ``np.nan`` si l'identification est incomplete ou si aucune valeur
    exploitable n'est trouvee (donnee absente de la base).
    """
    if code is None or eye is None or moment is None:
        return np.nan

    sel = ((meas["astronaut"] == code) & (meas["Eye"] == eye)
           & (meas["moment"] == moment) & (meas["description"] == description))

    vals = meas.loc[sel, "value_num"].to_numpy(dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))


def astro_code_from_folder(name, id2code):
    """'01_210713001' -> Code de l'astronaute (via le prefixe Id), ou None."""
    try:
        astro_id = int(name.split("_")[0])
    except (ValueError, IndexError):
        return None
    return id2code.get(astro_id)


def parse_eye(name):
    """'..._rigidity_OS2' -> 'OS' (le suffixe numerique est ignore), ou None."""
    m = re.search(r"(OD|OS)\d*$", name)
    return m.group(1) if m else None


def parse_moment(name):
    """'...before_rigidity' -> 'before' ; '...post_rigidity' -> 'after'."""
    low = name.lower()
    if "before" in low:
        return "before"
    if "post" in low or "after" in low:
        return "after"
    if "during" in low:
        return "during"
    return None


# --------------------------------------------------------------------------- #
# Chargement du vecteur temps (issu du notebook)
# --------------------------------------------------------------------------- #
def load_time_vector(mat_path, struct_name="timeSeries"):
    """
    Parameters
    ----------
    mat_path : str | Path
        Chemin du fichier .mat.
    struct_name : str
        Nom de la struct (array) MATLAB a lire.

    Returns
    -------
    time : (N,) float64 ndarray
        Temps ecoule en secondes, premier echantillon a t = 0.
    """
    mat = loadmat(mat_path)
    if struct_name not in mat:
        available = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"'{struct_name}' introuvable. Variables disponibles : {available}")

    s = mat[struct_name].ravel()  # aplatit la struct array en 1-D (longueur N)

    def field(name):
        # Chaque entree est un array 1x1 : on prend le scalaire en float64.
        # Le cast est essentiel : hour/minute sont en uint8, donc 3600*hour
        # deborderait si on le laissait en entier non signe.
        return np.array([np.ravel(e)[0] for e in s[name]], dtype=np.float64)

    hour = field("hour")
    minute = field("minute")
    second = field("second")

    time = 3600.0 * hour + 60.0 * minute + second
    time = time - time.min()
    return time


# --------------------------------------------------------------------------- #
# Methode 1 : passe-bande fixe + Hilbert (enveloppe ET frequence instantanee)
# --------------------------------------------------------------------------- #
def _weighted_median(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    return v[np.searchsorted(cw, 0.5 * cw[-1])]


def method1_bandpass_hilbert(ts, dY, band_hz, fc, oversample=20, order=4, edge=0.05):
    """
    Filtre passe-bande chaque A-scan puis extrait l'enveloppe analytique (Hilbert).

    Returns
    -------
    t_uni : (M,) ndarray         grille temporelle uniforme (rognee aux bords)
    A     : (M, n_ascan) ndarray enveloppe d'amplitude par A-scan
    f_hil : (M,) ndarray         frequence instantanee (mediane ponderee, Hz)
    S     : (M, n_ascan) ndarray signal filtre passe-bande
    PHI   : (M, n_ascan) ndarray phase instantanee (unwrapped)
    """
    lo, hi = band_hz
    fs = oversample * fc
    if hi >= fs / 2:
        raise ValueError("highcut >= Nyquist : augmente oversample")

    t_uni = np.linspace(ts[0], ts[-1], int(np.ceil((ts[-1] - ts[0]) * fs)) + 1)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")

    A = np.zeros((t_uni.size, dY.shape[1]))
    F = np.zeros((t_uni.size, dY.shape[1]))
    PHI = np.zeros((t_uni.size, dY.shape[1]))
    S = np.zeros((t_uni.size, dY.shape[1]))
    for i in range(dY.shape[1]):
        bp = sosfiltfilt(sos, np.interp(t_uni, ts, dY[:, i]))
        z = hilbert(bp)
        S[:, i] = bp
        A[:, i] = np.abs(z)
        PHI[:,i] = np.unwrap(np.angle(z))
        F[:, i] = np.gradient(np.unwrap(np.angle(z)), t_uni) / (2 * np.pi)

    f_hil = np.array([_weighted_median(F[k], A[k]) for k in range(t_uni.size)])
    good = np.isfinite(f_hil)
    if good.sum() > 1:
        f_hil = np.interp(np.arange(f_hil.size), np.flatnonzero(good), f_hil[good])
    win = int(round(fs / fc)) | 1
    f_hil = medfilt(f_hil, win)

    m = int(edge * t_uni.size)
    if m > 0:
        return t_uni[m:-m], A[m:-m], f_hil[m:-m], S[m:-m], PHI[m:-m], F[m:-m]
    return t_uni, A, f_hil, S, PHI, F


# --------------------------------------------------------------------------- #
# Figures + donnees par condition
# --------------------------------------------------------------------------- #
def save_condition_outputs(reg, fd):
    """Enregistre 3 figures (.png) + les tableaux (.npz) qui les produisent."""
    # --- donnees brutes pour reproduire les figures ---
    np.savez_compressed(
        reg / "rigidity_figdata.npz",
        # figure 1 : A-scan median
        ts=fd["ts"], dY_med=fd["dY_med"],
        t1=fd["t1"], S1_med=fd["S1_med"], A1_med=fd["A1_med"],
        imed=fd["imed"],
        # figure 2 : amplitude par A-scan
        ascan_index=fd["ascan_index"], amp1=fd["amp1"],
        # figure 3 : HR Hilbert
        f_hil_bpm=fd["f_hil_bpm"], hr=fd["hr"], band_bpm=fd["band_bpm"],
    )

    # --- figure 1 : ΔY brut + signal filtre + enveloppe de Hilbert ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(fd["ts"], fd["dY_med"], color="0.8", lw=0.6, label="ΔY brut")
    ax.plot(fd["t1"], fd["S1_med"], color="C0", lw=0.9, alpha=0.5, label="filtre (passe-bande)")
    ax.plot(fd["t1"], fd["A1_med"], color="C0", lw=1.4, label="enveloppe de Hilbert")
    ax.plot(fd["t1"], -fd["A1_med"], color="C0", lw=1.4)
    ax.set(xlabel="t (s)", ylabel="ΔY (pix)",
           title=f"A-scan median #{fd['imed']} : brut + filtre + enveloppe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reg / "rigidity_fig1_deltaY_ascan.png", dpi=150)
    plt.close(fig)

    # --- figure 2 : amplitude crete-a-crete par A-scan ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(fd["ascan_index"], fd["amp1"], color="C0", lw=1.0)
    ax.axhline(np.median(fd["amp1"]), color="r", ls="--", lw=0.8,
               label=f"mediane = {np.median(fd['amp1']):.2f} pix")
    ax.set(xlabel="A-scan (pixel)", ylabel="amplitude crete-a-crete (pix)",
           title="Amplitude pulsatile par A-scan (passe-bande + Hilbert)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reg / "rigidity_fig2_amplitude_par_ascan.png", dpi=150)
    plt.close(fig)

    # --- figure 3 : trace temporelle de la HR (Hilbert) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(fd["t1"], fd["f_hil_bpm"], color="C0", lw=1.2, label="HR Hilbert")
    ax.axhspan(fd["band_bpm"][0], fd["band_bpm"][1], color="red", alpha=0.1,
               label="bande HR ± {:.0f} BPM".format(HR_HALF_BAND_BPM))
    ax.axhline(fd["hr"], color="r", lw=1.0, label=f"HR mesuree = {fd['hr']:.0f} BPM")
    ax.set(xlabel="t (s)", ylabel="HR (BPM)",
           title="Frequence cardiaque instantanee (Hilbert)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reg / "rigidity_fig3_hr_hilbert.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(path_condi, meas, id2code):
    """
    Calcule hr, AL, IOP, OPA, CT, delta_CT, delta_V, k pour une condition,
    et enregistre les 3 figures + donnees associees.

    L'AL provient de la base ``sansori_db.db`` (``meas``), appariee par
    (code astronaute, oeil, moment) deduits de l'arborescence ; l'IOP et l'OPA
    sont lus dans ``visit_data.csv``. Une valeur absente vaut ``np.nan``
    (ecrite ``<NA>``) et empeche le calcul de ``k``.

    Returns
    -------
    dict | None : ligne de resultats, ou None si la condition est incomplete.
    """
    reg = path_condi / "RawImages" / "registeredBscans"
    path_deltaY = reg / "deltaY.npz"
    path_mask = reg / "mask.npz"
    path_idx2keep = reg / "detection_outliers_median_threshold.mat"
    path_heartbeat = path_condi / "Data Files" / "visit_data.csv"   # HR uniquement
    path_timeseries = path_condi / "RawImages" / "timeSeries.mat"

    # --- deltaY : (re)calcule depuis les masques si disponible, sinon charge ---
    if path_mask.exists():
        mask = load_mask(path_mask)
        deltaY = compute_deltaY_masks(mask)
        np.savez_compressed(path_deltaY, deltaY=deltaY)
    elif path_deltaY.exists():
        deltaY = np.load(path_deltaY)["deltaY"]
    else:
        print(f"  [skip] ni mask.npz ni deltaY.npz : {path_condi}")
        return None

    # --- fichiers requis pour l'analyse temporelle ---
    for p in (path_idx2keep, path_heartbeat, path_timeseries):
        if not p.exists():
            print(f"  [skip] fichier manquant {p.name} : {path_condi}")
            return None

    # --- HR / IOP / OPA depuis visit_data.csv (HR indispensable pour la bande) ---
    df = pd.read_csv(path_heartbeat, quoting=csv.QUOTE_NONE)
    hr = np.nanmean(pd.to_numeric(df["HR"], errors="coerce"))          # BPM
    if not np.isfinite(hr):
        print(f"  [skip] HR non exploitable : {path_condi}")
        return None
    band = (hr - HR_HALF_BAND_BPM, hr + HR_HALF_BAND_BPM)              # BPM
    IOP = np.nanmean(pd.to_numeric(df["PascalIOP"], errors="coerce"))  # mmHg
    OPA = np.nanmean(pd.to_numeric(df["PascalOPA"], errors="coerce"))  # mmHg

    # --- AL depuis la base, par (code astronaute, oeil, moment) ---
    code = astro_code_from_folder(path_condi.parent.parent.name, id2code)
    eye = parse_eye(path_condi.name)
    moment = parse_moment(path_condi.parent.name)
    AL = lookup_clinical(meas, code, eye, moment, AL_DESCRIPTION)   # mm

    # --- vecteur temps filtre par les indices conserves ---
    ids = loadmat(path_idx2keep)["idsFilteredMmi"].ravel()
    t = load_time_vector(path_timeseries)
    t_filt = t[ids.astype(int) - 1]   # MATLAB 1-based -> Python 0-based

    if t_filt.shape[0] != deltaY.shape[0]:
        print(f"  [skip] taille t_filt ({t_filt.shape[0]}) != deltaY "
              f"({deltaY.shape[0]}) : {path_condi}")
        return None

    # --- delta_CT par PASSE-BANDE + HILBERT (Methode 1) ---
    flo, fhi = band[0] / 60.0, band[1] / 60.0
    fc0 = 0.5 * (flo + fhi)

    order_sort = np.argsort(t_filt)
    ts = t_filt[order_sort]
    dY = deltaY[order_sort] - deltaY[order_sort].mean(0)

    t1, A1, f_hil, S1 = method1_bandpass_hilbert(ts, dY, (flo, fhi), fc0)
    amp1 = 2.0 * np.median(A1, axis=0)        # amplitude crete-a-crete par A-scan (pix)
    imed = int(np.argsort(amp1)[amp1.size // 2])   # A-scan median (par amplitude)

    # --- rigidite (Sayah et al. 2020) ---
    deltaCT = np.median(amp1) * PIX_Y_SIZE    # mm
    CT = np.median(deltaY) * PIX_Y_SIZE       # mm
    k, deltaV = sayah_k(deltaCT, CT, AL, IOP, OPA)

    # --- figures + donnees ---
    save_condition_outputs(reg, {
        "ts": ts, "dY_med": dY[:, imed],
        "t1": t1, "S1_med": S1[:, imed], "A1_med": A1[:, imed], "imed": imed,
        "ascan_index": np.arange(amp1.size), "amp1": amp1,
        "f_hil_bpm": f_hil * 60.0, "hr": hr, "band_bpm": np.array(band),
    })

    return {
        "hr": hr,             # BPM
        "AL": AL,             # mm
        "IOP": IOP,           # mmHg
        "OPA": OPA,           # mmHg
        "CT": CT,             # mm
        "delta_CT": deltaCT,  # mm
        "delta_V": deltaV,    # mm^3
        "k": k,               # mm^-3
    }


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def _fmt(x):
    """Formatage console : valeur ou '<NA>' si non finie."""
    return f"{x:.4f}" if np.isfinite(x) else "<NA>"


def main():
    meas, id2code = load_clinical_db(DB_PATH)   # base clinique chargee une fois

    rows = []
    for path_astro in PATH_GENERAL.iterdir():
        if not path_astro.is_dir():
            continue
        for path_moment in path_astro.iterdir():
            if not path_moment.match("*rigidity"):
                continue
            for path_condi in path_moment.iterdir():
                if not path_condi.is_dir():
                    continue
                print(path_condi)
                try:
                    result = process_condition(path_condi, meas, id2code)
                except Exception as e:  # noqa: BLE001
                    print(f"  [erreur] {path_condi} : {e}")
                    traceback.print_exc()
                    continue
                if result is None:
                    continue
                result = {
                    "patient": path_astro.name,
                    "moment": path_moment.name,
                    "condition": path_condi.name,
                    "path": str(path_condi),
                    **result,
                }
                rows.append(result)
                print(f"  -> hr={result['hr']:.1f} BPM | AL={_fmt(result['AL'])} mm "
                      f"| CT={result['CT']:.4f} mm | delta_CT={result['delta_CT']:.5f} mm "
                      f"| delta_V={_fmt(result['delta_V'])} mm^3 | k={_fmt(result['k'])}")

    if not rows:
        print("Aucune condition traitee : table CSV non ecrite.")
        return

    columns = ["patient", "moment", "condition", "path",
               "hr", "AL", "IOP", "OPA", "CT", "delta_CT", "delta_V", "k"]
    out = pd.DataFrame(rows)[columns]
    out.to_csv(OUTPUT_CSV, index=False, na_rep="<NA>")   # valeurs manquantes -> <NA>
    print(f"\n{len(out)} conditions ecrites dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()