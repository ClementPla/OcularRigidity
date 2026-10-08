# -*- coding: utf-8 -*-
"""
compute_costs_pulse.py

Pouls d'INTENSITE des replicats par BOP-DMD sur fenetre glissante (COSTS :
Coherent Spatio-Temporal Scale separation, Dylewsky, Tao & Kutz 2019 ;
``pydmd.costs.COSTS``), au lieu d'un mode SVD ou de la combinaison optimisee.

Memes pixels que le lot et que ``compute_svd_first_mode.py`` (``charger`` :
meme video cascade_v9, ROI 3/4 centraux x moitie superieure, normalisation,
ancre de FC). Puis :

1. COSTS : un BOP-DMD de rang ``RANG`` dans chaque fenetre de ``N_CYCLES``
   battements (a l'ancre), decalee de ``window / PAS_FRAC`` frames, sur les
   horodatages BRUTS (BOP-DMD n'exige pas un pas constant). La premiere fenetre
   part de la FC d'ancrage et de ses harmoniques (``harmonic_init`` du lot),
   avec des replis si l'ajustement diverge (``ESSAIS``) ;
2. ``cluster_omega`` : k-means des frequences (log10) en ``N_BANDES`` bandes ;
3. bande CARDIAQUE = celle dont la frequence mediane est la plus proche de
   l'ancre ;
4. ``scale_reconstruction`` : la video reconstruite par cette seule bande
   (fenetres ponderees par un noyau gaussien) ;
5. pouls = premiere composante temporelle POD de cette reconstruction, signe
   fixe par la convention du lot (``make_sign_fixer``), puis MEME FIR
   (``bandpass_pulse``) que les autres pouls, pour une comparaison equitable
   avec l'epaisseur.

On garde aussi la frequence suivie fenetre par fenetre (le mode de la bande
cardiaque le plus ample), c'est-a-dire f(t), ce que COSTS apporte en propre.

Sorties, a cote du lot (``<OR_VARIANT>/pulse_from_data/``)
    costs_pulse.csv          1 ligne / acquisition
    costs_pulse_traces.npz   ``<slug>__pulse_costs_fir`` (u_time),
                             ``<slug>__pulse_costs_brut`` (t),
                             ``<slug>__f_suivi_bpm``, ``<slug>__t_fenetres``

Lancer avec le MEME environnement que ``compute_svd_first_mode.py`` (voir son
en-tete : worktree 8fd820d, ``OR_LAYOUT=flat``, ...). ``OR_SLUGS=a,b`` restreint
a quelques acquisitions pour un essai.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydmd.costs import COSTS

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_svd_first_mode as sfm  # noqa: E402

cpd = sfm.cpd

RANG = 8             # rang BOP-DMD dans chaque fenetre (4 paires conjuguees)
N_CYCLES = 5.0       # longueur de fenetre, en battements a l'ancre
PAS_FRAC = 4         # pas = fenetre / PAS_FRAC
N_BANDES = 3         # bandes de frequence (k-means)
TRANSFORM = "log10"
MAXITER = 30         # iterations de la projection variable, par fenetre

OUT_CSV = cpd.OUT_DIR / "costs_pulse.csv"
OUT_NPZ = cpd.OUT_DIR / "costs_pulse_traces.npz"
SLUGS = [s for s in os.environ.get("OR_SLUGS", "").split(",") if s]


# Replis si la projection variable diverge dans une fenetre (``LinAlgError :
# SVD did not converge``) : sans amorce harmonique, puis au rang 6. L'essai
# retenu est ecrit dans la table (colonne ``essai``). Avec la contrainte
# "imag", le premier essai a suffi partout (2026-09-21) ; sans elle, 4
# acquisitions sur 53 echouaient.
ESSAIS = (("harmonique", RANG, True), ("sans_amorce", RANG, False),
          ("harmonique_rang6", 6, True))


def ajuster(X, t, hr, fenetre, pas):
    derniere = None
    for nom, rang, amorce in ESSAIS:
        kw = {"initialize_artificially": True,
              "init_alpha": cpd.harmonic_init(rang, hr / 60.0)} if amorce else {}
        c = COSTS(svd_rank=rang, global_svd=True,
                  # use_proj : COSTS le met a False par defaut -> la projection
                  # variable tourne sur les N pixels (23 s par fenetre au lieu de 0,2).
                  # "imag" (Re omega = 0) : sans elle, ou avec "stable" seul, des
                  # modes tres amortis (Re ~ -16 s^-1) d'amplitude ~1e11 se
                  # compensent ENTRE bandes ; la somme reste juste, mais la bande
                  # cardiaque seule explose (22/52 reconstructions > norme des
                  # donnees, jusqu'a 1e9). Sur 4-5 s, un pouls est une oscillation
                  # d'amplitude constante : la contrainte ne retire rien d'utile.
                  pydmd_kwargs={"eig_constraints": {"imag", "conjugate_pairs"}, "use_proj": True,
                                "varpro_opts_dict": {"maxiter": MAXITER}}, **kw)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c.fit(X.T, t[None, :], window_length=fenetre, step_size=pas)
                c.cluster_omega(n_components=N_BANDES, transform_method=TRANSFORM,
                                kmeans_kwargs={"random_state": 0, "n_init": 10})
            return c, nom
        except np.linalg.LinAlgError as exc:
            derniere = exc
    raise derniere


def process(path_condi: Path) -> tuple[dict, dict]:
    d = sfm.charger(path_condi)
    slug, t, hr, X, ctx = d["slug"], d["t"], d["hr"], d["X"], d["ctx"]
    dt = float(np.median(np.diff(t)))
    fenetre = int(round(N_CYCLES * 60.0 / hr / dt))
    pas = max(1, fenetre // PAS_FRAC)

    t0 = time.perf_counter()
    c, essai = ajuster(X, t, hr, fenetre, pas)
    t_fit = time.perf_counter() - t0

    f_bpm = np.abs(np.asarray(c.omega_array).imag) / (2 * np.pi) * 60.0   # (fenetres, rang)
    amp = np.abs(np.asarray(c.amplitudes_array))
    classes = np.asarray(c.omega_classes)
    med = np.array([np.median(f_bpm[classes == j]) if (classes == j).any() else np.nan
                    for j in range(N_BANDES)])
    jc = int(np.nanargmin(np.abs(med - hr)))

    recon = c.scale_reconstruction()[jc]                                   # (N, T)
    energie = float(np.linalg.norm(recon) / np.linalg.norm(X))
    _, s, vt = np.linalg.svd(recon, full_matrices=False)
    brut = d["fix_sign"](vt[0] * s[0], t)
    pulse = cpd.bandpass_pulse(brut, t, ctx)

    # f(t) : dans chaque fenetre, le mode le plus ample de la bande cardiaque.
    a_c = np.where(classes == jc, amp, -np.inf)
    k = a_c.argmax(axis=1)
    lignes = np.arange(len(k))
    present = np.isfinite(a_c[lignes, k])
    f_suivi = np.where(present, f_bpm[lignes, k], np.nan)
    centres = np.asarray(c.time_array).mean(axis=1)

    en_bande = np.abs(np.nanmedian(f_suivi) - hr) <= cpd.BAND_FRAC * hr
    ligne = {"slug": slug, "hr_BPM": hr, "fenetre_frames": fenetre, "fenetre_s": fenetre * dt,
             "pas_frames": pas, "n_fenetres": int(c.n_slides),
             "bandes_BPM": " ".join(f"{m:.1f}" for m in med), "bande_cardiaque": jc,
             "bande_BPM": float(med[jc]),
             "frac_fenetres_bande": float(present.mean()),
             "f_suivi_med_BPM": float(np.nanmedian(f_suivi)),
             "f_suivi_iqr_BPM": float(np.nanpercentile(f_suivi, 75)
                                      - np.nanpercentile(f_suivi, 25)),
             "energie_rel": energie, "t_fit_s": t_fit, "essai": essai,
             "status_energie": "ok" if energie < 1 else "reconstruction_explose",
             "status": "ok" if en_bande else "bande_hors_ancre"}
    payload = {f"{slug}__pulse_costs_fir": pulse.astype(np.float32),
               f"{slug}__pulse_costs_brut": brut.astype(np.float32),
               f"{slug}__f_suivi_bpm": f_suivi.astype(np.float32),
               f"{slug}__t_fenetres": centres.astype(np.float32)}
    return ligne, payload


def main() -> None:
    conds = sorted(cpd.iter_conditions())
    if SLUGS:
        conds = [p for p in conds if p.name in SLUGS]
    if cpd.LIMIT:
        conds = conds[:cpd.LIMIT]
    print(f"{len(conds)} acquisition(s) -> {OUT_CSV}", flush=True)
    lignes, payload = [], {}
    for i, p in enumerate(conds):
        t0 = time.perf_counter()
        try:
            ligne, pl = process(p)
            payload.update(pl)
        except Exception as exc:  # noqa: BLE001 - une acquisition ne doit pas tout arreter
            ligne = {"slug": p.name, "status": f"echec : {exc}"}
            traceback.print_exc()
        lignes.append(ligne)
        print(f"[{i + 1:>2d}/{len(conds)}] {ligne['slug']}  bande "
              f"{ligne.get('bande_BPM', float('nan')):.1f} BPM (ancre "
              f"{ligne.get('hr_BPM', float('nan')):.1f})  f(t) med "
              f"{ligne.get('f_suivi_med_BPM', float('nan')):.1f} iqr "
              f"{ligne.get('f_suivi_iqr_BPM', float('nan')):.1f}  {ligne['status']}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        pd.DataFrame(lignes).to_csv(OUT_CSV, index=False)
        np.savez_compressed(OUT_NPZ, **payload)


if __name__ == "__main__":
    main()
