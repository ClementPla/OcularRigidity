# -*- coding: utf-8 -*-
"""
migrate_repeatability.py

Range les acquisitions de l'etude de REPETABILITE, deposees en vrac par le
Spectralis dans un seul dossier, en un dossier par acquisition sous
``E:/SANSORI/Reproducibility/``.

    D:/SANSORI/REPEATABILITY/Raw Images/     11 517 .tif + 34 .xml melanges
        |
        v
    E:/SANSORI/Reproducibility/
        BELANGER_CHARLES_OD1/RawImages/      les .tif de CETTE acquisition + son .xml
        DESCOVICH_DENISE_OD1/RawImages/
        ...

Ce qui decide du regroupement
-----------------------------
Rien n'est devine du nom de fichier : les .tif portent un identifiant opaque
(``7DB83000.tif``). C'est le XML qui dit a quelle acquisition chacun appartient
(``ImageData/ExamURL``), de qui elle est (``LastName`` / ``FirstNames``) et de
quel oeil (``Series/Laterality``, ``R`` -> OD, ``L`` -> OS). Un XML = une
acquisition = un dossier.

Le rang du replicat est l'ORDRE CHRONOLOGIQUE des acquisitions d'un meme oeil,
lu sur l'horodatage du premier B-scan. Il n'est pas plafonne a 3 : un oeil
reacquis quatre fois donne OD1..OD4, et une acquisition interrompue reste un
replicat a part entiere -- son ``n_frames``, porte par le manifeste, permet de
l'ecarter plus tard en connaissance de cause plutot qu'ici en silence.

Les doublons d'export
---------------------
Le meme enregistrement peut avoir ete exporte plusieurs fois : le Spectralis
recopie alors les memes images sous de NOUVEAUX noms de fichiers, ce qui donne
des XML d'apparence independante. Les compter comme des replicats surestimerait
la repetabilite -- c'est meme le pire biais possible pour cette etude, puisque
deux copies du meme enregistrement s'accordent parfaitement.

Ils sont donc detectes en deux temps :

  1. une EMPREINTE gratuite -- (participant, oeil, nombre d'images, vecteur des
     horodatages). Deux acquisitions distinctes ne peuvent pas partager 348
     horodatages a la milliseconde ;
  2. une CONFIRMATION par md5 sur trois fichiers seulement (premiere, mediane et
     derniere image). Hacher les 5 Go pour trancher trois cas n'apporterait
     rien.

Le premier dans le temps est garde et numerote ; les autres sont laisses sur la
source, marques ``duplicate`` au manifeste, avec la preuve dans
``duplicates.csv``.

Ce que la destination apporte au raisonnement
---------------------------------------------
« Des donnees s'ajouteront plus tard » : le script RELIT donc les acquisitions
deja rangees sous ``DEST`` avant de decider quoi que ce soit, et les fait entrer
dans les deux raisonnements ci-dessus.

C'est ce qui rend un second lot correct plutot que seulement inoffensif :

  - un export du lot 2 qui reprend un enregistrement deja range est reconnu
    comme doublon, au lieu de reapparaitre sous un rang de plus ;
  - la numerotation REPREND ou elle s'etait arretee. Les rangs deja attribues
    sont figes -- un dossier existant n'est jamais renomme, meme si une nouvelle
    acquisition lui est chronologiquement anterieure -- et les nouveaux prennent
    les premiers rangs libres.

Ces lignes apparaissent au manifeste avec le statut ``present``.

Sur : par defaut ce script NE DEPLACE RIEN
------------------------------------------
Sans ``--apply``, c'est un essai a blanc : il lit, valide, ecrit le manifeste et
s'arrete. C'est le mode a lancer en premier.

Avec ``--apply``, le deplacement est fait FICHIER PAR FICHIER -- copie, controle
de taille (et de md5 sous ``--verify-hash``), puis suppression de la source. D:
et E: sont deux volumes : il n'y a pas de renommage atomique possible, et une
interruption ne doit pas pouvoir perdre une image. Un dossier destination qui
existe deja n'est jamais touche : la ligne passe ``skipped``.

Lancer (depuis la racine du depot) :
    python Reproducibility/migrate_repeatability.py                  # essai a blanc
    python Reproducibility/migrate_repeatability.py --apply --verify-hash
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ocularrigidity.data.spectralis import SpectralisStudy

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
SOURCE = Path("D:/SANSORI/REPEATABILITY/Raw Images")
DEST = Path("E:/SANSORI/Reproducibility")

# Sous-dossier attendu par le reste de la chaine (``find_raw_dir``,
# ``load_ordered_oct_series``, plus tard ``export_registered_video``).
RAW_SUBDIR = "RawImages"

EYE_OF_LATERALITY = {"R": "OD", "L": "OS", "OD": "OD", "OS": "OS"}

# ``BELANGER_CHARLES_OD1`` -> participant / oeil / rang. Meme expression que
# dans ``Reproducibility/compute_acquisition_params.py`` : c'est le nom de
# dossier qui porte le rang, et les deux scripts doivent le lire pareil.
RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


# --------------------------------------------------------------------------- #
# Nommage
# --------------------------------------------------------------------------- #
def normalize(txt: str | None) -> str:
    """``BELANGER`` a partir de ``Belanger`` -- MAJUSCULES, ASCII, sans separateur.

    Les accents sont retires (decomposition NFKD puis suppression des marques
    combinantes) et tout ce qui n'est pas alphanumerique disparait : un chemin
    Windows non-ASCII a deja pose probleme ailleurs dans la chaine (ffmpeg,
    cp1252), et le nom exact tel qu'il figure dans le XML reste de toute facon
    dans le manifeste.
    """
    if not txt:
        return "INCONNU"
    decomposed = unicodedata.normalize("NFKD", txt)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    kept = "".join(c for c in ascii_only if c.isalnum())
    return kept.upper() or "INCONNU"


# --------------------------------------------------------------------------- #
# Lecture d'un export
# --------------------------------------------------------------------------- #
def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_export(xml_path: Path) -> dict:
    """Tout ce qu'il faut savoir d'un export, en une lecture du XML.

    ``study.series`` porte deja une entree par B-scan, y compris pour les
    exports qui mettent toute la video dans une seule ``<Series>`` : c'est
    ``_parse_series_list`` (voir ``data/spectralis.py``) qui les eclate.
    """
    study = SpectralisStudy.from_file(xml_path)
    series = [s for s in study.series
              if s.oct is not None and s.oct_file_name and s.acquisition_time]
    series.sort(key=lambda s: s.acquisition_time.seconds_of_day)
    if not series:
        raise ValueError("aucun B-scan OCT horodate")

    # Les .tif a deplacer : les B-scans, PLUS le ou les localisateurs infrarouges
    # (un seul dans ces exports, un par serie dans la cohorte historique). Un
    # dict garde l'ordre sans doublonner le localisateur partage.
    files: dict[str, None] = {}
    for s in series:
        files[s.oct_file_name] = None
    for s in study.series:
        if s.fundus_file_name:
            files[s.fundus_file_name] = None

    lat = {s.laterality for s in series if s.laterality}
    laterality = sorted(lat)[0] if lat else None
    eye = EYE_OF_LATERALITY.get((laterality or "").upper(), "?")
    t = [s.acquisition_time.seconds_of_day for s in series]

    return {
        "xml": xml_path.name,
        "xml_path": xml_path,
        # Le dossier qui porte les .tif de CET export. Il vaut la source pour un
        # export a ranger, et le dossier d'arrivee pour un deja range : c'est ce
        # qui permet de comparer les deux sans les distinguer partout ailleurs.
        "dir": xml_path.parent,
        "last_name": study.patient.last_name,
        "first_name": study.patient.first_name,
        "patient_id": study.patient.patient_id,
        "sex": study.patient.sex,
        "laterality_xml": laterality,
        "eye": eye,
        "study_date": study.study_date.isoformat() if study.study_date else None,
        "n_frames": len(series),
        "t_start": str(series[0].acquisition_time),
        "t_start_s": t[0],
        "duration_s": round(t[-1] - t[0], 3),
        "files": list(files),
        "oct_files": [s.oct_file_name for s in series],
        # L'empreinte : le vecteur COMPLET des horodatages, a la milliseconde.
        "fingerprint": (normalize(study.patient.last_name),
                        normalize(study.patient.first_name),
                        eye,
                        len(series),
                        tuple(round(v, 3) for v in t)),
    }


# --------------------------------------------------------------------------- #
# Ce qui est deja range
# --------------------------------------------------------------------------- #
def read_placed(dest: Path) -> list[dict]:
    """Les acquisitions deja presentes sous ``dest``, relues depuis leur dossier.

    Elles ne seront pas deplacees ; elles servent a deux choses que la source
    seule ne peut pas donner : reconnaitre qu'un export d'un lot ulterieur
    reprend un enregistrement deja range, et savoir quels rangs de replicat sont
    pris.
    """
    placed = []
    for xml in sorted(dest.glob(f"*/{RAW_SUBDIR}/*.xml")):
        slug = xml.parent.parent.name
        if slug.startswith("_"):
            continue
        try:
            e = read_export(xml)
        except Exception as exc:
            print(f"  ! {slug} illisible : {type(exc).__name__}: {exc}")
            continue
        e["placed"] = True
        e["slug"] = slug
        m = RE_SLUG.match(slug)
        e["replicate"] = int(m.group("replicate")) if m else None
        placed.append(e)
    return placed


# --------------------------------------------------------------------------- #
# Doublons
# --------------------------------------------------------------------------- #
def confirm_duplicate(a: dict, b: dict) -> tuple[bool, str]:
    """md5 de la premiere, de la mediane et de la derniere image des deux exports.

    L'empreinte des horodatages suffit deja a les rapprocher ; ceci verifie que
    les PIXELS sont bien les memes, c'est-a-dire qu'on a affaire a deux exports
    du meme enregistrement et non a deux enregistrements qu'un hasard aurait
    horodates pareil.
    """
    n = min(len(a["oct_files"]), len(b["oct_files"]))
    preuves = []
    for i in sorted({0, n // 2, n - 1}):
        fa, fb = a["oct_files"][i], b["oct_files"][i]
        ha, hb = md5_of(a["dir"] / fa), md5_of(b["dir"] / fb)
        preuves.append(f"[{i}] {fa}={ha[:8]} {fb}={hb[:8]}")
        if ha != hb:
            return False, "md5 differents : " + " ; ".join(preuves)
    return True, "md5 identiques sur " + " ; ".join(preuves)


def resolve_duplicates(exports: list[dict]) -> list[dict]:
    """Marque ``is_duplicate`` / ``duplicate_of`` ; renvoie les lignes de preuve."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in exports:
        e["is_duplicate"] = False
        e["duplicate_of"] = None
        groups[e["fingerprint"]].append(e)

    preuves = []
    for group in groups.values():
        if len(group) < 2:
            continue
        # Un export DEJA RANGE gagne toujours : le renvoyer au statut de
        # doublon voudrait dire deplacer sa copie source par-dessus lui.
        group.sort(key=lambda e: (not e.get("placed"), e["xml"]))
        garde = group[0]
        for autre in group[1:]:
            same, why = confirm_duplicate(garde, autre)
            preuves.append({
                "xml": autre["xml"],
                "source": str(autre["dir"]),
                # Le nom de fichier ne suffit pas a designer l'export garde : un
                # re-export livre les MEMES noms de .xml, si bien que « xml »
                # et « duplicate_of » se ressemblent et qu'on ne voit plus
                # lequel des deux est deja range. Le slug et le chemin, eux,
                # sont sans ambiguite.
                "duplicate_of": garde["xml"],
                "duplicate_of_slug": garde.get("slug"),
                "duplicate_of_dir": str(garde["dir"]),
                "deja_range": bool(garde.get("placed")),
                "participant": f'{garde["last_name"]} {garde["first_name"]}',
                "eye": garde["eye"], "n_frames": garde["n_frames"],
                "confirme": same, "preuve": why,
            })
            if same:
                autre["is_duplicate"] = True
                autre["duplicate_of"] = garde["xml"]
            else:
                print(f"  ! {autre['xml']} a la meme empreinte que "
                      f"{garde['xml']} mais PAS les memes pixels -- garde comme "
                      f"acquisition distincte ({why})")
    return preuves


# --------------------------------------------------------------------------- #
# Numerotation
# --------------------------------------------------------------------------- #
def assign_slugs(exports: list[dict]) -> None:
    """Rang chronologique par (participant, oeil), doublons exclus du compte."""
    par_oeil: dict[tuple, list[dict]] = defaultdict(list)
    for e in exports:
        e.setdefault("replicate", None)
        if e["is_duplicate"]:
            e["slug"] = None
            continue
        par_oeil[(normalize(e["last_name"]), normalize(e["first_name"]),
                  e["eye"])].append(e)
    for (last, first, eye), group in par_oeil.items():
        # Les rangs deja attribues sont FIGES : renumeroter renommerait un
        # dossier existant, et casserait tout ce qui le reference deja (sorties
        # calculees, manifeste precedent, notes). Les nouvelles acquisitions
        # prennent donc les premiers rangs libres, dans l'ordre chronologique.
        pris = {e["replicate"] for e in group
                if e.get("placed") and e["replicate"] is not None}
        rank = 0
        for e in sorted((e for e in group if not e.get("placed")),
                        key=lambda e: e["t_start_s"]):
            rank += 1
            while rank in pris:
                rank += 1
            e["replicate"] = rank
            e["slug"] = f"{last}_{first}_{eye}{rank}"


# --------------------------------------------------------------------------- #
# Validations
# --------------------------------------------------------------------------- #
def validate(exports: list[dict]) -> bool:
    """Controles prealables. Renvoie False si un seul empeche de deplacer.

    Ne porte que sur les exports A RANGER : ceux deja en place ont ete relus
    depuis leur dossier d'arrivee et ne sont la que pour la numerotation et la
    detection des doublons.
    """
    exports = [e for e in exports if not e.get("placed")]
    ok = True

    manquants = [(e["xml"], f) for e in exports for f in e["files"]
                 if not (e["dir"] / f).exists()]
    if manquants:
        ok = False
        print(f"  ! {len(manquants)} fichier(s) reference(s) absent(s) de la source")
        for xml, f in manquants[:10]:
            print(f"      {xml} -> {f}")

    proprietaire: dict[str, str] = {}
    partages = []
    for e in exports:
        for f in e["files"]:
            if f in proprietaire:
                partages.append((f, proprietaire[f], e["xml"]))
            else:
                proprietaire[f] = e["xml"]
    if partages:
        ok = False
        print(f"  ! {len(partages)} fichier(s) reference(s) par deux XML")
        for f, a, b in partages[:10]:
            print(f"      {f} : {a} et {b}")

    orphelins = {p.name for p in SOURCE.glob("*.tif")} - set(proprietaire)
    if orphelins:
        # Non bloquant : ils restent sur la source, mais il faut le savoir.
        print(f"  ! {len(orphelins)} .tif de la source ne sont references par "
              f"aucun XML -- ils resteront sur place")
        for f in sorted(orphelins)[:10]:
            print(f"      {f}")

    inconnus = [e["xml"] for e in exports if e["eye"] == "?"]
    if inconnus:
        ok = False
        print(f"  ! lateralite illisible : {', '.join(inconnus)}")

    return ok


# --------------------------------------------------------------------------- #
# Deplacement
# --------------------------------------------------------------------------- #
def move_file(src: Path, dst: Path, verify_hash: bool) -> None:
    """Copie, controle, PUIS supprime la source. Jamais l'inverse.

    ``D:`` et ``E:`` sont deux volumes : ``Path.rename`` echouerait et
    ``shutil.move`` ferait la meme copie sans laisser de place a un controle
    entre les deux. Un fichier deja present a destination et de meme taille est
    considere comme deja deplace -- c'est ce qui rend une reprise possible apres
    interruption.
    """
    taille = src.stat().st_size
    if not (dst.exists() and dst.stat().st_size == taille):
        shutil.copy2(src, dst)
        if dst.stat().st_size != taille:
            raise IOError(f"taille differente apres copie : {src} -> {dst}")
    if verify_hash and md5_of(src) != md5_of(dst):
        raise IOError(f"md5 different apres copie : {src} -> {dst}")
    src.unlink()


def apply_move(e: dict, verify_hash: bool) -> str:
    raw_dir = DEST / e["slug"] / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in e["files"]:
        src = e["dir"] / f
        if not src.exists() and (raw_dir / f).exists():
            continue  # deja deplace lors d'un passage interrompu
        move_file(src, raw_dir / f, verify_hash)
    move_file(e["xml_path"], raw_dir / e["xml"], verify_hash)
    return "moved"


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    global SOURCE, DEST

    ap = argparse.ArgumentParser(
        description="Range les exports HEYEX de l'etude de repetabilite en un "
                    "dossier par acquisition.")
    ap.add_argument("--apply", action="store_true",
                    help="deplacer reellement (sinon : essai a blanc)")
    ap.add_argument("--verify-hash", action="store_true",
                    help="comparer le md5 source/destination avant de supprimer")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args(argv)

    SOURCE, DEST = args.source, args.dest
    migration_dir = DEST / "_migration"
    csv_manifest = migration_dir / "manifest.csv"
    csv_duplicates = migration_dir / "duplicates.csv"

    if not SOURCE.is_dir():
        print(f"source introuvable : {SOURCE}")
        return 2

    xmls = sorted(SOURCE.glob("*.xml"))
    print(f"{len(xmls)} export(s) XML dans {SOURCE}")

    # --- lecture ------------------------------------------------------------ #
    exports = []
    for x in xmls:
        try:
            exports.append(read_export(x))
        except Exception as exc:
            print(f"  ! {x.name} illisible : {type(exc).__name__}: {exc}")

    # Ce qui est deja range entre dans le raisonnement (doublons, rangs libres)
    # sans jamais etre deplace.
    placed = read_placed(DEST)
    if placed:
        print(f"{len(placed)} acquisition(s) deja rangee(s) sous {DEST}")
    exports = placed + exports
    if not exports:
        print("rien a lire ni a ranger")
        return 0

    # --- doublons, numerotation, validation --------------------------------- #
    print("\nDoublons d'export")
    preuves = resolve_duplicates(exports)
    n_dup = sum(e["is_duplicate"] for e in exports)
    print(f"  {n_dup} export(s) redondant(s) sur {len(exports)}")

    assign_slugs(exports)

    print("\nValidations")
    ok = validate(exports)

    # Un dossier destination qui existe deja n'est JAMAIS ecrase.
    for e in exports:
        if e.get("placed"):
            e["status"] = "present"
        elif e["is_duplicate"]:
            e["status"] = "duplicate"
        elif (DEST / e["slug"]).exists():
            e["status"] = "skipped"
        else:
            e["status"] = "to_move"
    n_skip = sum(e["status"] == "skipped" for e in exports)
    if n_skip:
        print(f"  {n_skip} dossier(s) destination existent deja -- laisses intacts")
    if ok:
        print("  aucun probleme bloquant")

    # --- recapitulatif ------------------------------------------------------ #
    a_deplacer = [e for e in exports if e["status"] == "to_move"]
    compte: dict[str, dict[str, int]] = defaultdict(lambda: {"OD": 0, "OS": 0})
    for e in exports:
        if not e["is_duplicate"]:
            nom = f'{normalize(e["last_name"])}_{normalize(e["first_name"])}'
            compte[nom][e["eye"]] += 1
    print("\nEffectifs (doublons exclus)")
    print(f"  {'participant':28}{'OD':>4}{'OS':>4}")
    for nom in sorted(compte):
        print(f"  {nom:28}{compte[nom]['OD']:>4}{compte[nom]['OS']:>4}")
    print(f"  {'TOTAL':28}{sum(c['OD'] for c in compte.values()):>4}"
          f"{sum(c['OS'] for c in compte.values()):>4}")
    n_tif = sum(len(e["files"]) for e in a_deplacer)
    n_present = sum(e["status"] == "present" for e in exports)
    print(f"\n  {len(a_deplacer)} dossier(s) a creer, {n_tif} .tif "
          f"+ {len(a_deplacer)} .xml a deplacer"
          + (f" ({n_present} deja en place)" if n_present else ""))

    # --- manifeste (ecrit dans les DEUX modes) ------------------------------ #
    migration_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "xml": e["xml"], "slug": e["slug"], "status": e["status"],
        "last_name": e["last_name"], "first_name": e["first_name"],
        "patient_id": e["patient_id"], "sex": e["sex"],
        "eye": e["eye"], "laterality_xml": e["laterality_xml"],
        "replicate": e["replicate"], "study_date": e["study_date"],
        "t_start": e["t_start"], "duration_s": e["duration_s"],
        "n_frames": e["n_frames"], "n_files": len(e["files"]),
        "duplicate_of": e["duplicate_of"],
        "dest": str(DEST / e["slug"] / RAW_SUBDIR) if e["slug"] else None,
        "source": str(e["xml_path"]),
    } for e in exports]).sort_values(
        ["last_name", "first_name", "eye", "t_start"], kind="stable")
    df.to_csv(csv_manifest, index=False, encoding="utf-8")
    print(f"\n{csv_manifest}")
    if preuves:
        pd.DataFrame(preuves).to_csv(csv_duplicates, index=False, encoding="utf-8")
        print(csv_duplicates)

    # --- deplacement -------------------------------------------------------- #
    if not args.apply:
        print("\nESSAI A BLANC -- rien n'a ete deplace. "
              "Relancer avec --apply pour executer.")
        return 0
    if not ok:
        print("\nValidations en echec : rien n'est deplace.")
        return 1

    print(f"\nDeplacement de {len(a_deplacer)} acquisition(s)")
    for i, e in enumerate(sorted(a_deplacer, key=lambda e: e["slug"]), 1):
        print(f"  [{i}/{len(a_deplacer)}] {e['slug']:26} "
              f"{len(e['files']):>5} fichier(s)", flush=True)
        try:
            e["status"] = apply_move(e, args.verify_hash)
        except Exception as exc:
            e["status"] = "failed"
            print(f"      ! echec : {type(exc).__name__}: {exc}")

    df["status"] = df["xml"].map({e["xml"]: e["status"] for e in exports})
    df.to_csv(csv_manifest, index=False, encoding="utf-8")
    print(f"\n{sum(e['status'] == 'moved' for e in exports)} deplacee(s), "
          f"{sum(e['status'] == 'failed' for e in exports)} en echec, "
          f"{n_dup} doublon(s) laisse(s) sur la source.")
    print(f"{csv_manifest} mis a jour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
