# -*- coding: utf-8 -*-
"""
estimate_consensus_hr.py

Fabrique un A PRIORI de frequence cardiaque par acquisition, a partir des
acquisitions repetees du MEME participant.

Pourquoi il faut en passer par la
---------------------------------
La cohorte SANS lit sa FC dans ``visit_data.csv`` : une mesure, qui ancre la
recherche du pic cardiaque. L'etude de repetabilite n'a AUCUNE mesure de ce
genre, et le lot retombe alors sur une estimation par SVD cherchee sur
20-240 BPM. Cette recherche part sur une harmonique environ une fois sur deux :

    ROY_JOANNIE_OD     78,6  ->  53,2  ->  142,0 BPM   (meme oeil, 12 minutes)
    PLAYOUT_CLEMENT_OS 64,1  ->  61,3  ->   32,9  ->  143,5 BPM

142 vaut ~2 x 71, 32,9 vaut ~1/2 x 66 : ce sont des harmoniques et des
sous-harmoniques de la meme frequence, pas des battements differents. Comme la
phase -- donc le repliement, donc le bin de reference, donc le strain -- en
depend entierement, un ICC calcule la-dessus mesurerait surtout « l'estimateur
a-t-il choisi deux fois la meme harmonique », et non la repetabilite du strain.

Ce que fait ce script
---------------------
1. REPLIE chaque estimation sur l'octave de la MEDIANE des estimations brutes
   du participant -- l'operation exacte qui defait une erreur d'octave, et rien
   d'autre : une FC de 84 BPM reste 84.
2. Prend la MEDIANE par PARTICIPANT. Le coeur est unique : les deux yeux et
   tous les replicats d'une meme seance partagent la meme FC a la derive pres.
   La mediane, et non la moyenne, pour qu'une estimation restee fausse apres
   repliement ne deplace pas le consensus.
3. Ecrit ``hr_prior.csv`` (une ligne par acquisition), que le lot de pouls
   relira via ``OR_HR_PRIOR``.

Le controle qui dit si l'on a le droit d'en faire un consensus : chez les
participants ou l'estimation ne derape pas (Modica, Playout OD), l'etendue
intra-oeil vaut 1 a 2 BPM. Une FC de seance EST donc stable ; c'est bien
l'estimateur qui bouge, pas le sujet.

Ce que ce fichier n'est PAS
---------------------------
Ce n'est pas une mesure clinique et il ne se fait pas passer pour telle : il
n'est pas ecrit sous le nom ``visit_data.csv``, et la colonne ``hr_source`` du
lot distingue ``prior`` de ``visit_data`` et de ``svd``. Les valeurs d'origine
sont conservees dans le fichier, colonne ``hr_svd_BPM``, pour que l'ecart entre
les deux reste lisible -- c'est un resultat de l'etude, pas un brouillon.

Lancer (apres ``run_batch.py pulse``, avant de le relancer avec l'a priori) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \\
        Reproducibility/estimate_consensus_hr.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT_ROOT = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations"
                    "/model1_scale_1.0_flatten_choroid_xcorr")
CSV_PULSE = VARIANT_ROOT / "pulse_from_data" / "conditions.csv"
# Les estimations BRUTES, d'avant tout ancrage. Elles sont la seule source
# valable : une fois ancree, ``hr_BPM`` vaut le consensus par construction, et
# le recalculer dessus reviendrait a le confirmer par lui-meme. Les conditions
# absentes de cette table (un lot arrive apres l'ancrage) sont reprises dans
# ``CSV_PULSE``, ou elles portent encore ``hr_source = svd``.
CSV_PULSE_BRUT = VARIANT_ROOT / "pulse_from_data_sans_ancrage" / "conditions.csv"
OUT_CSV = Path("E:/NASA_Rigidity/Reproducibility/hr_prior.csv")

# Au-dela de cette etendue residuelle (apres repliement) chez un participant, le
# consensus n'est plus un consensus : c'est le choix arbitraire d'un paquet
# contre un autre. La valeur est ecrite quand meme -- le lot a besoin d'un
# nombre -- mais la colonne ``consensus_fiable`` la marque, et la page doit le
# reporter. 15 BPM : le double de la derive physiologique attendue sur une
# seance de vingt minutes.
SEUIL_ETENDUE = 15.0

RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


def replier_sur(hr: float, f: float) -> float:
    """``hr`` ramene sur l'octave de ``f``.

    ``hr / 2^round(log2(hr/f))`` : 37,8 face a 102 donne 75,6, 102,9 reste
    102,9. C'est la seule erreur qu'on corrige -- celle d'octave, que commet un
    chercheur de pic spectral en accrochant une harmonique ou une
    sous-harmonique.
    """
    if not np.isfinite(hr) or hr <= 0 or not np.isfinite(f) or f <= 0:
        return float("nan")
    return float(hr / 2.0 ** np.round(np.log2(hr / f)))


def ancre(hrs: np.ndarray) -> float:
    """L'octave sur laquelle replier : la MEDIANE des estimations brutes.

    Il faut une ancre, et elle ne peut pas venir d'un critere d'octave : le
    cout d'octave est degenere par construction -- ``f`` et ``2f`` ont
    exactement le meme -- si bien qu'une recherche libre choisit l'un ou
    l'autre par accident numerique. Essayee, elle a double MODICA_GRAZIANA,
    dont les six estimations brutes valaient pourtant 60 a 62 BPM.

    Replier une bande FIXE ne marche pas non plus : c'est la borne qui decide,
    et [45, 100] ecrasait a 52 les cinq acquisitions de MCNEIL_THOMAS estimees
    a ~104 BPM, valeur plausible telle quelle.

    La mediane des valeurs brutes, elle, ne suppose rien : elle dit ou
    l'estimateur atterrit le plus souvent chez CE participant. Les octaves
    ratees sont minoritaires par construction (sinon il n'y aurait pas de
    consensus a chercher), donc la mediane tombe sur la bonne octave.
    """
    hrs = np.asarray([h for h in hrs if np.isfinite(h) and h > 0], dtype=float)
    return float(np.median(hrs)) if hrs.size else float("nan")


def main() -> int:
    if not CSV_PULSE.exists():
        print(f"{CSV_PULSE} absent -- lancer d'abord :\n"
              f"    python Reproducibility/run_batch.py pulse")
        return 2

    d = pd.read_csv(CSV_PULSE)
    d = d[d["status"] == "ok"].copy()

    # Remplacer, pour les conditions qui y figurent, la FC par son estimation
    # d'ORIGINE. Sans cela le consensus se calculerait sur ses propres sorties.
    n_brut = 0
    if CSV_PULSE_BRUT.exists():
        b = pd.read_csv(CSV_PULSE_BRUT)
        b = b[b["status"] == "ok"]
        brut = dict(zip(b["slug"], pd.to_numeric(b["hr_BPM"], errors="coerce")))
        d["hr_BPM"] = [brut.get(s, h) for s, h in zip(d["slug"], d["hr_BPM"])]
        n_brut = int(sum(s in brut for s in d["slug"]))
    print(f"{len(d)} conditions ; {n_brut} FC reprises de "
          f"{CSV_PULSE_BRUT.parent.name}, {len(d) - n_brut} lues telles quelles")
    ancrees = (d.get("hr_source") == "prior").sum() if "hr_source" in d else 0
    if ancrees and n_brut < ancrees:
        print(f"  ! {ancrees - n_brut} condition(s) deja ancree(s) sans "
              f"estimation d'origine disponible -- leur consensus se confirmerait "
              f"lui-meme")
    parts = d["slug"].str.extract(RE_SLUG)
    d["participant"] = parts["participant"]
    d["eye"] = parts["eye"]
    d["replicate"] = pd.to_numeric(parts["replicate"], errors="coerce")
    d["eye_id"] = d["participant"] + "_" + d["eye"]
    d["hr_svd_BPM"] = pd.to_numeric(d["hr_BPM"], errors="coerce")
    # La fondamentale est cherchee par participant sur les valeurs BRUTES ;
    # chaque estimation est ensuite ramenee sur SON octave, ce qui la rend
    # comparable au consensus sans qu'aucune bande arbitraire n'intervienne.
    fond = d.groupby("participant")["hr_svd_BPM"].agg(
        lambda g: ancre(g.to_numpy()))
    d["hr_repliee_BPM"] = [replier_sur(h, fond[p])
                           for h, p in zip(d["hr_svd_BPM"], d["participant"])]

    consensus = d.groupby("participant")["hr_repliee_BPM"].median()
    d["hr_BPM"] = d["participant"].map(consensus)

    # Le repliement ne defait que les erreurs d'OCTAVE. Une erreur de quinte
    # (rapport 3:2) y survit, et se voit alors comme deux paquets de valeurs
    # dont la mediane tombe dans l'un des deux -- arbitrairement. On mesure donc
    # la dispersion RESIDUELLE par participant, et on le dit quand elle est trop
    # grande pour qu'un consensus veuille dire quelque chose.
    disp = (d.groupby("participant")["hr_repliee_BPM"]
              .agg(lambda g: float(g.max() - g.min())))
    d["etendue_participant_BPM"] = d["participant"].map(disp)
    d["consensus_fiable"] = d["etendue_participant_BPM"] <= SEUIL_ETENDUE
    d["ecart_au_consensus_BPM"] = d["hr_repliee_BPM"] - d["hr_BPM"]
    # Combien d'octaves il a fallu defaire : 0 = l'estimation etait deja bonne.
    d["octaves_repliees"] = np.log2(
        d["hr_repliee_BPM"] / d["hr_svd_BPM"]).round().astype("Int64")

    print(f"{'participant':22}{'n':>3}{'consensus':>11}{'etendue':>9}  fiable  FC repliees")
    for p, g in d.groupby("participant"):
        g = g.sort_values(["eye", "replicate"])
        vals = ", ".join(f"{v:.0f}" for v in g["hr_repliee_BPM"])
        ok = "oui" if disp[p] <= SEUIL_ETENDUE else "NON"
        print(f"{p:22}{len(g):>3}{consensus[p]:>10.1f}{disp[p]:>9.1f}  {ok:>6}  {vals}")

    n_repliees = int((d["octaves_repliees"].fillna(0) != 0).sum())
    ecart = d["ecart_au_consensus_BPM"].abs()
    print(f"\n{n_repliees} / {len(d)} estimations etaient a cote d'une octave")
    print(f"ecart au consensus apres repliement : median {ecart.median():.1f} "
          f"BPM, max {ecart.max():.1f}")
    print(f"consensus : {consensus.min():.1f} a {consensus.max():.1f} BPM "
          f"sur {len(consensus)} participants")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    douteux = sorted(disp[disp > SEUIL_ETENDUE].index)
    if douteux:
        print(f"\nCONSENSUS DOUTEUX (etendue > {SEUIL_ETENDUE:.0f} BPM apres"
              f" repliement) : {', '.join(douteux)}")
        print("  Le repliement ne defait que les octaves ; un rapport 3:2 y "
              "survit et scinde les valeurs en deux paquets.")
        print("  L'a priori est ecrit quand meme, marque `consensus_fiable = "
              "False` -- a reporter dans l'analyse.")

    cols = ["slug", "participant", "eye", "replicate", "hr_BPM",
            "hr_svd_BPM", "hr_repliee_BPM", "octaves_repliees",
            "ecart_au_consensus_BPM", "etendue_participant_BPM",
            "consensus_fiable"]
    d[cols].sort_values("slug").to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n{OUT_CSV}  ({len(d)} acquisitions)")
    print("\nRelancer le pouls avec cet a priori :\n"
          f"    OR_HR_PRIOR={OUT_CSV}  OR_OVERWRITE=1  "
          "python Reproducibility/run_batch.py pulse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
