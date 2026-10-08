# -*- coding: utf-8 -*-
"""
estimate_median_hr.py

Une frequence cardiaque par PERSONNE : la mediane de TOUTES les FC trouvees sur
ses replicats par le test sans a priori (``compute_hr_bandpass.py``, FIR fixe
50-85 BPM), intensite ET epaisseur reunies.

Pourquoi remplacer le consensus
-------------------------------
``hr_prior.csv`` (``estimate_consensus_hr.py``) est la mediane des estimations
SVD brutes apres repli d'octave. Sur l'etude de repetabilite, il s'ecarte
nettement de ce que retrouvent deux signaux independants sans rien leur
souffler : DESCOVICH 82,9 contre ~67 BPM, MCNEIL 102,2 contre ~70. La mediane
ecrite ici sert d'a priori au lot de pouls, comme le consensus, via
``OR_HR_PRIOR``.

Ce que ce fichier n'est PAS
---------------------------
Une mesure. Et il est BORNE par construction : les FC d'entree sortent d'une
recherche de pic dans 50-85 BPM, une FC reelle hors de cette bande ne peut pas
en ressortir. La valeur d'origine de chaque acquisition et l'ancien consensus
sont recopies pour que l'ecart reste lisible.

Entree  : E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
              cascade_v9_1536x1024/hr_bandpass_50_85/conditions.csv
Sortie  : E:/NASA_Rigidity/Reproducibility/hr_prior_mediane_bandpass_50_85_cascade_v9.csv

Lancer (depuis la racine du depot) :
    python Reproducibility/estimate_median_hr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEGVAR = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
SOURCE = SEGVAR / "cascade_v9_1536x1024" / "hr_bandpass_50_85" / "conditions.csv"
CONSENSUS = Path("E:/NASA_Rigidity/Reproducibility/hr_prior.csv")
OUT_CSV = Path("E:/NASA_Rigidity/Reproducibility/hr_prior_mediane_bandpass_50_85_cascade_v9.csv")
SIGNAUX = ("hr_intensite_BPM", "hr_epaisseur_BPM")


def main() -> int:
    if not SOURCE.exists():
        print(f"{SOURCE} absent -- lancer d'abord compute_hr_bandpass.py sur cascade_v9")
        return 2
    d = pd.read_csv(SOURCE)
    d = d[d["status"] == "ok"].copy()

    # Toutes les valeurs d'une personne, les deux signaux reunis : la mediane
    # porte sur 2 x n replicats, et une valeur aberrante d'un seul signal ne la
    # deplace pas.
    long = d.melt(id_vars=["participant"], value_vars=list(SIGNAUX), value_name="hr")
    g = long.groupby("participant")["hr"]
    par_personne = pd.DataFrame({
        "hr_mediane": g.median(),
        "n_valeurs": g.size(),
        "etendue_participant_BPM": g.max() - g.min(),
    })
    d = d.merge(par_personne, left_on="participant", right_index=True)
    d["hr_BPM"] = d["hr_mediane"]

    cons = pd.read_csv(CONSENSUS)[["slug", "hr_BPM"]].rename(columns={"hr_BPM": "hr_consensus_BPM"})
    d = d.merge(cons, on="slug", how="left")
    d["ecart_au_consensus_BPM"] = d["hr_BPM"] - d["hr_consensus_BPM"]

    cols = ["slug", "participant", "eye", "replicate", "hr_BPM", "n_valeurs",
            "hr_intensite_BPM", "hr_epaisseur_BPM", "etendue_participant_BPM",
            "hr_consensus_BPM", "ecart_au_consensus_BPM"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d[cols].sort_values("slug").to_csv(OUT_CSV, index=False, encoding="utf-8")

    t = (d.groupby("participant")
          .agg(n_acq=("slug", "size"), n_valeurs=("n_valeurs", "first"),
               mediane=("hr_BPM", "first"), consensus=("hr_consensus_BPM", "first"),
               etendue=("etendue_participant_BPM", "first")))
    t["ecart"] = t["mediane"] - t["consensus"]
    print(t.round(1).to_string())
    print(f"\n{OUT_CSV}  ({len(d)} acquisitions, {t.shape[0]} participants)")
    print("\nRelancer le pouls avec cet a priori :\n"
          f"    OR_HR_PRIOR={OUT_CSV} ... python Astronauts/compute_pulse_from_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
