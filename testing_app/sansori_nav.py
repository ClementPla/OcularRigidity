"""Navigation dans l'arborescence SANSORI, partagee par les pages de l'app.

    E:/SANSORI/<NN_id>/<...>_rigidity/<...>_rigidity_<OD|OS><rep?>/
        experiments/                 <- experiment_*.json (first_cc_registration)
        RawImages/ (ou RawData/)     <- .tif + export .xml
        RawImages/registered/        <- sorties de register_files / bouton app

  - patient  : dossier prefixe par le numero zero-padde (01_ .. 14_)
  - moment   : "before" sur disque = "before" ; "after" choisi = "post"
  - oeil+rep : suffixe du dossier condition ; numero de replicat ABSENT s'il n'y
               en a qu'un (..._OD), sinon numerote (..._OS1, ..._OS2, ...).
"""

import re
from pathlib import Path

PATH_GENERAL = Path("E:/SANSORI")


def _moment_matches(folder_name: str, moment: str) -> bool:
    """Le moment 'after' choisi par l'utilisateur correspond au dossier 'post'."""
    low = folder_name.lower()
    if moment == "before":
        return "before" in low
    if moment == "after":
        return "post" in low or "after" in low
    return False


def find_patient_dir(patient_id: int) -> Path | None:
    """ID 1..14 -> dossier patient prefixe '01_' .. '14_'."""
    prefix = f"{patient_id:02d}_"
    if not PATH_GENERAL.exists():
        return None
    for p in PATH_GENERAL.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            return p
    return None


def find_moment_dir(patient_dir: Path, moment: str) -> Path | None:
    """Dossier '*rigidity' du patient correspondant au moment choisi."""
    for m in patient_dir.iterdir():
        if m.is_dir() and m.match("*rigidity") and _moment_matches(m.name, moment):
            return m
    return None


def list_replicate_dirs(moment_dir: Path, eye: str) -> list[Path]:
    """Dossiers condition pour un oeil donne, tries par numero de replicat.

    Le suffixe peut etre nu (..._OD = un seul replicat) ou numerote (..._OD2).
    """
    pat = re.compile(rf"{eye}(\d*)$")
    found = []
    for c in moment_dir.iterdir():
        if not c.is_dir():
            continue
        m = pat.search(c.name)
        if m:
            num = int(m.group(1)) if m.group(1) else 0
            found.append((num, c))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les images brutes (.tif) + l'export XML Spectralis.

    Nomme 'RawImages' dans le jeu SANSORI (anciennement 'RawData').
    """
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def format_acq_time(t) -> str:
    """AcquisitionTime -> 'HH:MM:SS.mmm' (millisecondes)."""
    return f"{t.hour:02d}:{t.minute:02d}:{t.second:06.3f}"
