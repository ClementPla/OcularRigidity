# -*- coding: utf-8 -*-
"""
Inventaire des PARAMETRES D'ACQUISITION, lus dans les exports XML HEYEX
(Heidelberg Spectralis) qui accompagnent les .tif bruts.

Rien n'est calcule a partir des images : tout vient du XML, sauf la frequence
d'acquisition, deduite des horodatages des B-scans.

C'est le MOTEUR, partage par les lots qui l'appellent :

  - ``Astronauts/compute_acquisition_params.py``      cohorte SANS
  - ``Reproducibility/compute_acquisition_params.py`` etude de repetabilite

Les deux ne different que par trois choses -- ou sont les conditions, comment
elles s'identifient, et ou vont les CSV -- et c'est exactement ce que
:func:`run_batch` prend en argument. Toute correction de fond (un champ oublie,
une frequence mal deduite) se fait ICI et profite aux deux.

Trois sorties, qui repondent aux trois questions posees :

  - ``constants.csv``  les parametres IDENTIQUES sur toutes les conditions --
    ceux qu'on peut sortir de la grande table et enoncer une fois pour toutes ;
  - ``conditions.csv`` 1 ligne / condition, tous les parametres, y compris ceux
    qui ne varient pas (la separation constant/variable est refaite a la
    lecture, pour qu'un ajout de condition ne demande pas de relancer autre
    chose) ;
  - ``series.csv``     1 ligne / B-scan OCT retenu -- c'est le grain auquel
    varient la qualite d'image, le gain du capteur et l'intervalle entre
    images ; les histogrammes des pages Quarto en viennent.

Ce que « une condition » veut dire
----------------------------------
Un dossier portant un sous-dossier ``RawImages`` (ou ``RawData``) avec les .tif
et leur .xml. Les series retenues sont celles de ``load_ordered_oct_series`` --
B-scan OCT present sur disque et horodate -- c'est-a-dire EXACTEMENT les images
qui composent la video traitee par le reste de la chaine. Les series du XML sans
.tif sont comptees a part (``n_series_xml`` vs ``n_frames``) plutot que
silencieusement ignorees.

Frequence d'acquisition
-----------------------
Spectralis n'ecrit aucun taux d'images : il horodate chaque B-scan
(``AcquisitionTime``, a la milliseconde). La frequence moyenne d'une video est
donc ``(n_frames - 1) / (t_dernier - t_premier)``, soit l'inverse de
l'intervalle MOYEN. Sont aussi rapportes l'ecart-type et les extremes des
intervalles : l'acquisition n'est pas uniforme, et c'est la raison pour laquelle
tout le reste de la chaine interpole sur une grille reguliere avant Hilbert/SSA.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ocularrigidity.data.spectralis import SpectralisStudy
from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def eye_of(condition: str) -> str:
    """Oeil lu dans le NOM du dossier -- ``..._OD2`` -> ``OD``.

    La lateralite du XML (``R`` / ``L``) est lue aussi et rapportee a part : les
    deux doivent concorder, et une discordance signale un dossier mal range.
    """
    tail = condition.upper().rsplit("_", 1)[-1]
    if tail.startswith("OD"):
        return "OD"
    if tail.startswith("OS"):
        return "OS"
    return "?"


# --------------------------------------------------------------------------- #
# Lecture des champs que le modele type ne porte pas
# --------------------------------------------------------------------------- #
def _txt(node, path):
    if node is None:
        return None
    el = node.find(path)
    if el is None or el.text is None:
        return None
    return el.text.strip() or None


def read_header_fields(xml_path: Path) -> dict:
    """Champs d'en-tete absents de ``SpectralisStudy`` : logiciels, equipement,
    champ de balayage, description d'etude.

    ``SpectralisStudy`` s'arrete aux feuilles de ``OphthalmicAcquisitionContext``
    et du niveau Series ; ``GeneralEquipment``, ``SWVersion`` et ``OCTFieldSize``
    sont des noeuds intermediaires, donc invisibles pour ``_leaf_dict``. Ils sont
    lus ici en une passe ET separee, sur la PREMIERE serie -- ils sont constants
    d'une serie a l'autre a l'interieur d'un fichier.

    Le parseur est celui de ``spectralis.read_root`` et non ``ET.parse`` direct :
    certains exports n'ont aucune declaration d'encodage et sont en cp1252, et
    ``ET.parse`` echoue des le premier nom accentue.
    """
    from ocularrigidity.data.spectralis import read_root

    root = read_root(xml_path)
    body = root.find("BODY")
    study = body.find("Patient/Study") if body is not None else None
    series0 = study.find("Series") if study is not None else None
    eq = series0.find("GeneralEquipment") if series0 is not None else None
    # La source lumineuse du localisateur vit sous <ImageType>, pas sous
    # <OphthalmicAcquisitionContext> : elle echappe donc a `Image.context`.
    localizer = None
    for img in (series0.findall("Image") if series0 is not None else []):
        if (_txt(img, "ImageType/Type") or "").upper() == "LOCALIZER":
            localizer = img
            break
    return {
        "ir_light_source": _txt(localizer, "ImageType/LightSource"),
        "export_type": _txt(body, "ExportType"),
        "sw_viewing": _txt(body, "SWVersion/Name"),
        "sw_viewing_version": _txt(body, "SWVersion/Version"),
        "study_description": _txt(study, "StudyDescription"),
        "operator": _txt(study, "Operator"),
        "manufacturer": _txt(eq, "Manufacturer"),
        "model_name": _txt(eq, "ManufacturerModelName"),
        "aqm_name": _txt(eq, "AQMVersion/Name"),
        "aqm_version": _txt(eq, "AQMVersion/Version"),
        "oct_field_width_deg": _txt(series0, "OCTFieldSize/Width"),
        "series_type": _txt(series0, "Type"),
        "examined_structure": _txt(series0, "ExaminedStructure"),
        "modality": _txt(series0, "Modality"),
        "modality_procedure": _txt(series0, "ModalityProcedure"),
        "images_per_series": _txt(series0, "NumImages"),
    }


# --------------------------------------------------------------------------- #
# Resume d'une colonne de valeurs par serie
# --------------------------------------------------------------------------- #
def _num(values):
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna()


def _mode(values):
    """Valeur la plus frequente, en texte -- pour les champs categoriels.

    ``None`` quand tout est vide. Les egalites sont tranchees par l'ordre
    d'apparition (``value_counts`` est stable), ce qui suffit : les champs
    concernes (``AutoSensorGain``, ``FixationTarget``) sont quasi constants a
    l'interieur d'une condition.
    """
    s = pd.Series([v for v in values if v not in (None, "")], dtype="object")
    if s.empty:
        return None
    return str(s.value_counts().index[0])


def _uniq(values) -> str:
    """Toutes les valeurs distinctes, triees, jointes par « / ».

    Une condition dont ``NumAve`` vaut 9 sur une image et 10 sur les autres doit
    le montrer, pas afficher une moyenne de 9,97 qui n'a jamais ete un reglage.
    """
    s = sorted({str(v) for v in values if v not in (None, "")})
    return " / ".join(s)


def _ctx_values(images, tag):
    return [img.context.get(tag) for img in images if img is not None]


def _valeur_exacte(values):
    """Valeur d'un reglage numerique CONSTANT sur la condition, sans erreur d'arrondi.

    La mediane, et non la moyenne : ``mean`` d'une colonne de 496 fois 0,0039
    ne rend pas exactement 0,0039 (l'erreur depend du nombre de termes), si bien
    que deux conditions au MEME reglage se retrouvent avec deux flottants
    differents -- et qu'un parametre reellement constant sur la cohorte cesse
    d'etre reconnu comme tel. La mediane, elle, renvoie une valeur observee.
    """
    s = _num(values)
    return float(s.median()) if len(s) else float("nan")


# --------------------------------------------------------------------------- #
# Une condition
# --------------------------------------------------------------------------- #
def process(path_condi: Path, identity: dict) -> tuple[dict, pd.DataFrame] | None:
    """Une condition -> (ligne de ``conditions.csv``, table de ``series.csv``).

    ``identity`` porte les colonnes d'identite, DANS L'ORDRE ou elles doivent
    apparaitre en tete des deux tables. Elles varient d'un lot a l'autre (la
    cohorte SANS a un ``moment``, la repetabilite un ``replicate``), et c'est la
    seule chose que ce moteur ne sait pas deduire d'un dossier : elle depend de
    la forme de l'arborescence qui l'entoure. ``path`` en est retire pour
    ``series.csv``, ou il se repeterait sur des milliers de lignes.
    """
    slug = identity.get("slug", path_condi.name)
    condition = identity.get("condition", path_condi.name)

    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        print(f"  ! {slug} : ni RawImages ni RawData")
        return None
    xmls = sorted(raw_dir.glob("*.xml"))
    if not xmls:
        print(f"  ! {slug} : aucun .xml dans {raw_dir.name}")
        return None

    study = SpectralisStudy.from_file(xmls[0])
    series = load_ordered_oct_series(raw_dir)
    if not series:
        print(f"  ! {slug} : aucune serie OCT exploitable")
        return None

    octs = [s.oct for s in series]
    funds = [s.fundus for s in series]

    # --- horodatages -> frequence d'acquisition ----------------------------- #
    t = np.array([s.acquisition_time.seconds_of_day for s in series], dtype=float)
    dt = np.diff(t)
    # Un intervalle nul ou negatif ne peut pas exister dans une acquisition
    # continue : ce serait un passage de minuit (Spectralis ne stocke que
    # l'heure du jour) ou deux series exportees avec le meme horodatage. On les
    # ecarte du calcul plutot que de laisser une frequence infinie.
    dt_ok = dt[dt > 0]
    duree = float(t[-1] - t[0]) if len(t) > 1 else float("nan")
    fs_moy = float(len(dt_ok) / dt_ok.sum()) if dt_ok.size else float("nan")

    # --- geometrie du B-scan ------------------------------------------------ #
    # Start/End sont les extremites de la ligne de balayage sur le fond d'oeil,
    # en mm. Elles donnent trois choses que ni ScaleX ni le champ de 30 degres
    # ne portent : ou la ligne commence, ou elle finit, et son ORIENTATION.
    #
    # La distance entre les deux est la LONGUEUR reellement balayee -- pas
    # ScaleX x Width, qui ne vaudrait que pour un B-scan horizontal.
    #
    # L'angle est ``atan(dy / dx)`` et non ``atan2`` : ce qu'on veut est
    # l'inclinaison d'une DROITE, definie a 180 degres pres et donc a valeurs
    # dans (-90, +90], et non la direction d'un vecteur. Les deux coincident ici
    # -- dx est positif sur toutes les series lues -- mais atan2 renverrait
    # +160 degres la ou atan renvoie -20 si un export inversait un jour les deux
    # extremites, ce qui casserait toute comparaison entre conditions.
    coords, longueurs, angles = [], [], []
    for o in octs:
        if o is not None and o.start_xy and o.end_xy:
            (x0, y0), (x1, y1) = o.start_xy, o.end_xy
            if None in (x0, y0, x1, y1):
                continue
            coords.append((x0, y0, x1, y1))
            longueurs.append(float(np.hypot(x1 - x0, y1 - y0)))
            if x1 != x0:
                angles.append(float(np.degrees(np.arctan((y1 - y0) / (x1 - x0)))))
    longueur_mm = float(np.mean(longueurs)) if longueurs else float("nan")
    # Mediane et non moyenne, pour la meme raison que `_valeur_exacte` : ces
    # quatre coordonnees sont constantes d'une image a l'autre a l'interieur
    # d'une condition, et la mediane rend la valeur observee au lieu d'un
    # flottant approche.
    xy = np.array(coords, dtype=float) if coords else np.full((1, 4), np.nan)
    start_x, start_y, end_x, end_y = (float(np.median(xy[:, i])) for i in range(4))
    angle_deg = float(np.median(angles)) if angles else float("nan")
    # Amplitude de l'angle A L'INTERIEUR de la video : non nulle, elle voudrait
    # dire que la ligne de balayage a bouge en cours d'acquisition.
    angle_etendue = (float(np.max(angles) - np.min(angles)) if angles
                     else float("nan"))

    entete = read_header_fields(xmls[0])

    qualite = _num([o.quality for o in octs])
    gain = _num(_ctx_values(funds, "SensorGain"))
    focus = _num(_ctx_values(funds, "Focus"))
    # NumAve d'un B-scan Spectralis = nombre d'images REELLEMENT moyennees pour
    # celui-la : le compteur monte 1, 2, 3 ... jusqu'a la consigne ART puis y
    # reste. Le reglage de l'operateur est donc le MAXIMUM, pas la moyenne (qui
    # ne serait qu'une facon detournee de mesurer la duree de la montee).
    num_ave = _num(_ctx_values(octs, "NumAve"))

    lat_xml = _uniq([s.laterality for s in series])
    eye = identity.get("eye", eye_of(condition))

    row = {
        **identity,
        # --- identite de l'acquisition -------------------------------------- #
        "study_date": study.study_date.isoformat() if study.study_date else None,
        "laterality_xml": lat_xml,
        "laterality_matches_folder": {"R": "OD", "L": "OS"}.get(lat_xml, lat_xml) == eye,
        "sex": study.patient.sex,
        "n_series_xml": len(study.series),
        "n_frames": len(series),
        # --- temps ----------------------------------------------------------- #
        "duree_s": duree,
        "fs_moy_hz": fs_moy,
        "dt_moy_ms": float(dt_ok.mean() * 1e3) if dt_ok.size else float("nan"),
        "dt_sd_ms": float(dt_ok.std(ddof=1) * 1e3) if dt_ok.size > 1 else float("nan"),
        "dt_min_ms": float(dt_ok.min() * 1e3) if dt_ok.size else float("nan"),
        "dt_max_ms": float(dt_ok.max() * 1e3) if dt_ok.size else float("nan"),
        "t_debut": str(series[0].acquisition_time),
        "utc_bias_min": series[0].acquisition_time.utc_bias,
        # --- geometrie OCT ---------------------------------------------------- #
        "oct_width_px": _uniq([o.width for o in octs]),
        "oct_height_px": _uniq([o.height for o in octs]),
        "oct_scale_x_mm": _valeur_exacte([o.scale_x for o in octs]),
        "oct_scale_y_mm": _valeur_exacte([o.scale_y for o in octs]),
        "oct_longueur_mm": longueur_mm,
        "start_x_mm": start_x,
        "start_y_mm": start_y,
        "end_x_mm": end_x,
        "end_y_mm": end_y,
        "delta_x_mm": end_x - start_x,
        "delta_y_mm": end_y - start_y,
        "angle_deg": angle_deg,
        "angle_etendue_deg": angle_etendue,
        "oct_field_width_deg": entete["oct_field_width_deg"],
        "oct_resolution": _uniq(_ctx_values(octs, "Resolution")),
        "oct_num_ave_max": float(num_ave.max()) if len(num_ave) else float("nan"),
        "oct_edi": _uniq(_ctx_values(octs, "EDI")),
        "oct_evi": _uniq(_ctx_values(octs, "EVI")),
        "oct_position_tolerance": _uniq(_ctx_values(octs, "PositionWithinTolerance")),
        # --- qualite d'image --------------------------------------------------- #
        "quality_moy": float(qualite.mean()) if len(qualite) else float("nan"),
        "quality_sd": float(qualite.std(ddof=1)) if len(qualite) > 1 else float("nan"),
        "quality_min": float(qualite.min()) if len(qualite) else float("nan"),
        "quality_max": float(qualite.max()) if len(qualite) else float("nan"),
        # --- localisateur infrarouge -------------------------------------------- #
        "ir_width_px": _uniq([f.width for f in funds if f is not None]),
        "ir_height_px": _uniq([f.height for f in funds if f is not None]),
        "ir_scale_mm": _valeur_exacte(_ctx_values(funds, "ScaleX")),
        "ir_angle_deg": _uniq(_ctx_values(funds, "Angle")),
        "ir_num_ave": _valeur_exacte(_ctx_values(funds, "NumAve")),
        "ir_resolution": _uniq(_ctx_values(funds, "Resolution")),
        "ir_auto_sensor_gain": _mode(_ctx_values(funds, "AutoSensorGain")),
        "sensor_gain_moy": float(gain.mean()) if len(gain) else float("nan"),
        "sensor_gain_min": float(gain.min()) if len(gain) else float("nan"),
        "sensor_gain_max": float(gain.max()) if len(gain) else float("nan"),
        "focus_moy_d": float(focus.mean()) if len(focus) else float("nan"),
        "focus_min_d": float(focus.min()) if len(focus) else float("nan"),
        "focus_max_d": float(focus.max()) if len(focus) else float("nan"),
        "fixation_target": _mode(_ctx_values(funds, "FixationTarget")),
    }
    row.update({k: v for k, v in entete.items() if k not in row})

    # --- table par B-scan --------------------------------------------------- #
    df_series = pd.DataFrame({
        **{k: v for k, v in identity.items() if k != "path"},
        "frame": np.arange(len(series)),
        "t_s": t - t[0],
        "dt_ms": np.concatenate([[np.nan], dt * 1e3]),
        "quality": [o.quality if o else np.nan for o in octs],
        "num_ave": pd.to_numeric(pd.Series(_ctx_values(octs, "NumAve")),
                                 errors="coerce").to_numpy(),
        "scale_x_mm": [o.scale_x if o else np.nan for o in octs],
        "sensor_gain": pd.to_numeric(pd.Series(_ctx_values(funds, "SensorGain")),
                                     errors="coerce").to_numpy(),
        "focus_d": pd.to_numeric(pd.Series(_ctx_values(funds, "Focus")),
                                 errors="coerce").to_numpy(),
    })
    return row, df_series


# --------------------------------------------------------------------------- #
# Constantes de la cohorte
# --------------------------------------------------------------------------- #
def split_constants(df: pd.DataFrame,
                    cols_id: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    """Colonnes dont TOUTES les conditions partagent la valeur -> table a part.

    Un NaN/None compte comme une valeur : une colonne vide partout est bien une
    constante (« champ absent de tous les exports »), et une colonne renseignee
    sur une seule condition ne l'est pas. Les colonnes d'identite sont exclues
    d'office : elles varient par construction et n'ont rien a faire dans un
    histogramme.
    """
    cols_id = set(cols_id)
    constantes, variables = [], []
    for col in df.columns:
        if col in cols_id:
            continue
        vals = df[col].astype("object").where(pd.notna(df[col]), None)
        if vals.nunique(dropna=False) <= 1:
            constantes.append(col)
        else:
            variables.append(col)
    tab = pd.DataFrame({
        "parametre": constantes,
        "valeur": [df[c].astype("object").where(pd.notna(df[c]), None).iloc[0]
                   for c in constantes],
    })
    return tab, variables


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def run_batch(conditions: list[Path],
              identity_of: Callable[[Path], dict],
              out_dir: Path,
              label_of: Callable[[Path], str] | None = None) -> pd.DataFrame:
    """Traite toutes les conditions et ecrit les trois CSV sous ``out_dir``.

    ``identity_of`` construit les colonnes d'identite d'une condition ; ses cles
    servent aussi de ``cols_id`` pour :func:`split_constants`, si bien qu'un lot
    n'a pas a declarer deux fois la meme liste.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label_of = label_of or (lambda p: f"{p.parent.name} / {p.name}")

    rows, series_frames = [], []
    print(f"{len(conditions)} conditions")
    for i, path_condi in enumerate(conditions, 1):
        print(f"[{i}/{len(conditions)}] {label_of(path_condi)}")
        try:
            out = process(path_condi, identity_of(path_condi))
        except Exception as exc:  # une condition illisible ne doit pas tout arreter
            print(f"  ! echec : {type(exc).__name__}: {exc}")
            continue
        if out is None:
            continue
        row, df_series = out
        rows.append(row)
        series_frames.append(df_series)

    if not rows:
        print("\naucune condition exploitable -- rien n'est ecrit")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "conditions.csv", index=False, encoding="utf-8")
    print(f"\n{out_dir / 'conditions.csv'}  ({len(df)} conditions)")

    df_all = pd.concat(series_frames, ignore_index=True)
    df_all.to_csv(out_dir / "series.csv", index=False, encoding="utf-8")
    print(f"{out_dir / 'series.csv'}  ({len(df_all)} B-scans)")

    cols_id = list(identity_of(conditions[0]))
    tab, variables = split_constants(df, cols_id)
    tab.to_csv(out_dir / "constants.csv", index=False, encoding="utf-8")
    print(f"{out_dir / 'constants.csv'}  ({len(tab)} parametres constants, "
          f"{len(variables)} variables)")
    print("\nvariables : " + ", ".join(variables))
    return df
