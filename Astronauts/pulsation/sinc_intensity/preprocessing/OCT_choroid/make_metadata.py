# -*- coding: utf-8 -*-
"""
Metadonnees du jeu OCT pour SiNC : ``datasets/metadata/OCT.csv``.

Lit le resume du lot de crops (``make_dataset.py``) et ne garde que les videos a
9,5-9,6 Hz : pas de reechantillonnage, donc une seule cadence (a 1 % pres) pour la
FFT des pertes. ART fixe la cadence ; a 9,5-9,6 Hz il n'y a que des OD.

Roles, par PARTICIPANT (jamais par replicat : le meme oeil a quelques minutes
d'intervalle fuirait d'un role a l'autre) :
    test  = MODICA_GRAZIANA
    val   = ROY_JOANNIE
    train = les autres

    python Astronauts/pulsation/sinc_intensity/preprocessing/OCT_choroid/make_metadata.py
"""

from pathlib import Path

import pandas as pd

SUMMARY = Path("E:/NASA_Rigidity/Reproducibility/choroid_crops/summary.csv")
CROPS = Path("E:/SANSORI/Reproducibility/choroid_crops")
OUT = Path(__file__).resolve().parents[2] / "datasets" / "metadata" / "OCT.csv"
FPS_RANGE = (9.0, 10.0)
TEST = {"MODICA_GRAZIANA"}
VAL = {"ROY_JOANNIE"}


def split_of(participant: str) -> str:
    return "test" if participant in TEST else "val" if participant in VAL else "train"


def main():
    d = pd.read_csv(SUMMARY)
    d = d[(d["status"] == "ok") & d["fps"].between(*FPS_RANGE)].copy()
    d["path"] = [str(CROPS / s / "crop.npz") for s in d["slug"]]
    d["split"] = d["participant"].map(split_of)
    d = d[["slug", "participant", "eye", "replicate", "fps", "T", "path", "split"]].sort_values("slug")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.groupby("split").agg(videos=("slug", "size"), frames=("T", "sum")))
    print("->", OUT)


if __name__ == "__main__":
    main()
