# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Pouls ancré sur la FC médiane par personne ».

Ce script NE LIT QUE des tables deja ecrites ; il ne recalcule ni pouls, ni FC.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/
        hr_prior.csv                                      ancien ancrage (consensus)
        hr_prior_mediane_bandpass_50_85_cascade_v9.csv    nouvel ancrage (mediane)
        SegmentationVariations/
            cascade_v9_1536x1024/hr_bandpass_50_85/conditions.csv   FC 50-85 par replicat
            model1_scale_1.0_flatten_choroid_xcorr/pulse_from_data/ reference (existant)
            cascade_v9_1536x1024_hr_consensus/pulse_from_data/      lot B (temoin)
            cascade_v9_1536x1024_hr_mediane/pulse_from_data/        lot A (la demande)
            <lot>/pulse_from_data/ct_pulse.csv, ct_pulse_traces.npz  pouls d'epaisseur
                (Reproducibility/compute_ct_pulse_traces.py)
            <lot>/pulse_from_data/costs_pulse*, svd_first_mode*   pouls COSTS, 1 mode SVD
                (Reproducibility/compute_costs_pulse.py, compute_svd_first_mode.py)

Sorties (fragments inclus par ``repro-hr-mediane.qmd``)
------------------------------------------------------
    tab_prior.qmd, fig_prior.qmd, fig_qualite_pouls.qmd, tab_qualite.qmd,
    fig_pouls.qmd, fig_pouls_int_ct.qmd, fig_accord_int_ct.qmd, tab_int_ct.qmd,
    plotlyjs.html, resume.txt

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_repro_hr_mediane/make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
REPRO = Path("E:/NASA_Rigidity/Reproducibility")
SEGVAR = REPRO / "SegmentationVariations"
PRIOR_OLD = REPRO / "hr_prior.csv"
PRIOR_NEW = REPRO / "hr_prior_mediane_bandpass_50_85_cascade_v9.csv"
HR_BANDPASS = SEGVAR / "cascade_v9_1536x1024" / "hr_bandpass_50_85" / "conditions.csv"
LOTS = {
    "ref": ("référence (U-Net + consensus)",
            SEGVAR / "model1_scale_1.0_flatten_choroid_xcorr" / "pulse_from_data", "#8a94a6"),
    "B": ("cascade_v9 + consensus",
          SEGVAR / "cascade_v9_1536x1024_hr_consensus" / "pulse_from_data", "#c98b1a"),
    "A": ("cascade_v9 + médiane",
          SEGVAR / "cascade_v9_1536x1024_hr_mediane" / "pulse_from_data", "#27a567"),
}
ORDRE = ["ref", "B", "A"]

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_INT = "#2a78d6"
C_CT = "#e08a1e"
C_REF = "#c2453f"
C_NEW = "#27a567"

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
NOMS = ("tab_prior", "fig_prior", "fig_qualite_pouls", "tab_qualite", "fig_pouls",
        "fig_pouls_int_ct", "fig_accord_int_ct", "tab_int_ct")
RESUME: dict = {}
nl = chr(10)


# --------------------------------------------------------------------------- #
# Habillage commun (le meme que figures_repro_cascade)
# --------------------------------------------------------------------------- #
def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        margin={"l": 60, "r": 20, "t": 80, "b": 80 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.12, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    for a in fig.layout.annotations or ():
        if a.text and a.font is not None:
            a.font.size = 11
            a.font.color = COULEUR_TEXTE
    return fig


def enregistrer(fig, nom):
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config=CONFIG, div_id=f"plot-{nom}")
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_repro_hr_mediane/make_figures.py"
              " -- ne pas editer a la main. -->" + nl + nl)
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + nl, encoding="utf-8")
    print(f"  {nom}.qmd")


def absente(nom, pourquoi):
    """Un ``{{< include >}}`` sur un fichier absent fait ECHOUER le rendu du site
    entier : une figure impossible laisse donc un encadre lisible a sa place."""
    ecrire_table(f"::: {{.callout-warning}}{nl}## Figure indisponible{nl}{nl}{pourquoi}{nl}:::",
                 nom)


def fr(x, chiffres=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{{:.{chiffres}f}}".format(x).replace(".", ",")


def med_iqr(s, chiffres=1):
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return "—"
    return (f"{fr(s.median(), chiffres)} [{fr(s.quantile(0.25), chiffres)} – "
            f"{fr(s.quantile(0.75), chiffres)}]")


def essaim(v, largeur=0.12, n_bins=30):
    """Decalages horizontaux : les valeurs proches s'ecartent au lieu de se
    superposer (meme fonction que les autres pages de la section)."""
    v = np.asarray(v, dtype=float)
    x = np.zeros(len(v))
    fini = np.isfinite(v)
    if not fini.any():
        return x
    lo, hi = float(np.min(v[fini])), float(np.max(v[fini]))
    pas = (hi - lo) / n_bins if hi > lo else 1.0
    cle = np.full(len(v), np.nan)
    cle[fini] = np.round((v[fini] - lo) / pas)
    for c in np.unique(cle[fini]):
        idx = np.flatnonzero(cle == c)
        n = len(idx)
        if n > 1:
            x[idx] = np.linspace(-1.0, 1.0, n) * largeur * min(1.0, (n - 1) / 4.0)
    return x


def norm(a):
    a = np.asarray(a, dtype=float)
    s = np.nanstd(a)
    return (a - np.nanmean(a)) / s if s > 0 else a - np.nanmean(a)


def nom_participant(p):
    return str(p).replace("_", " ")


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
entrees = [PRIOR_OLD, PRIOR_NEW, HR_BANDPASS] + [LOTS[k][1] / "conditions.csv" for k in ORDRE]
manquants = [p for p in entrees if not p.exists()]
if manquants:
    pourquoi = ("Le calcul n'a pas encore produit ces tables :" + nl + nl
                + nl.join(f"- `{p}`" for p in manquants) + nl + nl
                + "Lancer `Reproducibility/estimate_median_hr.py`, puis les lots A et B de "
                  "`Astronauts/compute_pulse_from_data.py` (voir la page).")
    for nom in NOMS:
        absente(nom, pourquoi)
    (SORTIE / "plotlyjs.html").write_text("", encoding="utf-8")
    raise SystemExit(0)

RE_PART = r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$"
P_NEW = pd.read_csv(PRIOR_NEW)
P_OLD = pd.read_csv(PRIOR_OLD)
HB = pd.read_csv(HR_BANDPASS)
HB = HB[HB["status"] == "ok"]

COND, METH = {}, {}
for k in ORDRE:
    c = pd.read_csv(LOTS[k][1] / "conditions.csv")
    c = c[c["status"] == "ok"].copy()
    c = pd.concat([c, c["slug"].str.extract(RE_PART)], axis=1)
    c["dmd_ecart_abs"] = (pd.to_numeric(c["dmd_hr_BPM"], errors="coerce") - c["hr_BPM"]).abs()
    COND[k] = c
    m = pd.read_csv(LOTS[k][1] / "methods.csv")
    METH[k] = m[m["methode"] == "1_fir"].set_index("slug")
COMMUNS = sorted(set.intersection(*(set(COND[k]["slug"]) for k in ORDRE)))
PARTICIPANTS = sorted(P_NEW["participant"].unique())
RESUME["n_communs"] = len(COMMUNS)
RESUME["n_ok"] = " | ".join(f"{k} {len(COND[k])}" for k in ORDRE)


# --------------------------------------------------------------------------- #
# 1. L'ancrage : consensus contre mediane
# --------------------------------------------------------------------------- #
def tab_prior():
    t = (P_NEW.groupby("participant")
             .agg(n_acq=("slug", "size"), n_valeurs=("n_valeurs", "first"),
                  mediane=("hr_BPM", "first"), consensus=("hr_consensus_BPM", "first"),
                  etendue=("etendue_participant_BPM", "first")))
    t["ecart"] = t["mediane"] - t["consensus"]
    lignes = ["| participant | acquisitions | valeurs | médiane 50–85 BPM | consensus actuel "
              "| écart | étendue des valeurs |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for p, r in t.iterrows():
        lignes.append(f"| {nom_participant(p)} | {int(r.n_acq)} | {int(r.n_valeurs)} "
                      f"| **{fr(r.mediane)}** | {fr(r.consensus)} | {fr(r.ecart)} "
                      f"| {fr(r.etendue)} |")
        RESUME[f"prior_{p}"] = f"mediane {fr(r.mediane)} | consensus {fr(r.consensus)} | ecart {fr(r.ecart)}"
    RESUME["prior_ecart_abs_med"] = med_iqr(t["ecart"].abs())
    RESUME["prior_ecart_sup_5"] = f"{int((t['ecart'].abs() > 5).sum())} / {len(t)}"
    ecrire_table(nl.join(lignes) + nl + nl
                 + ": FC d'ancrage par participant, en BPM. « médiane » : médiane de toutes "
                   "les FC 50–85 BPM de ses réplicats (intensité et épaisseur réunies, "
                   "recalage cascade_v9). « consensus » : `hr_prior.csv`. {.striped}",
                 "tab_prior")


def fig_prior():
    fig = go.Figure()
    for cle, lib, coul, symb, dec in (("hr_intensite_BPM", "FC intensité", C_INT, "circle", -0.12),
                                      ("hr_epaisseur_BPM", "FC épaisseur", C_CT, "diamond", 0.12)):
        x = np.empty(len(HB))
        for p in HB["participant"].unique():
            sel = (HB["participant"] == p).to_numpy()
            x[sel] = PARTICIPANTS.index(p) + dec + essaim(HB.loc[sel, cle].to_numpy(), largeur=0.08)
        fig.add_trace(go.Scatter(
            x=x, y=HB[cle], mode="markers", name=lib,
            marker={"color": coul, "symbol": symb, "size": 7, "opacity": 0.8},
            text=HB["slug"], hovertemplate="%{text}<br>" + lib + " %{y:.1f} BPM<extra></extra>"))
    for colonne, lib, coul, dash in (("hr_BPM", "médiane retenue", C_NEW, "solid"),
                                     ("hr_consensus_BPM", "consensus actuel", C_REF, "dash")):
        xs, ys = [], []
        for k, p in enumerate(PARTICIPANTS):
            v = P_NEW.loc[P_NEW["participant"] == p, colonne].median()
            if np.isfinite(v):
                xs += [k - 0.4, k + 0.4, None]
                ys += [v, v, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=lib,
                                 line={"color": coul, "width": 2.5, "dash": dash},
                                 hovertemplate=lib + " %{y:.1f} BPM<extra></extra>"))
    for b in (50, 85):
        fig.add_hline(y=b, line={"color": "rgba(150,150,150,0.6)", "width": 1, "dash": "dot"})
    fig.update_xaxes(tickvals=list(range(len(PARTICIPANTS))),
                     ticktext=[nom_participant(p).replace(" ", "<br>", 1) for p in PARTICIPANTS],
                     range=[-0.6, len(PARTICIPANTS) - 0.4], tickfont={"size": 10})
    fig.update_yaxes(title_text="FC (BPM)")
    fig = mise_en_page(fig, "D'où vient la FC de chaque personne<br>points : FC 50–85 BPM de "
                            "ses réplicats · traits : ancrage retenu et ancien", hauteur=520)
    fig.update_layout(margin={"t": 90, "b": 120}, legend={"y": -0.2})
    enregistrer(fig, "fig_prior")


tab_prior()
fig_prior()


# --------------------------------------------------------------------------- #
# 2. Qualite du pouls et de la phase, trois lots
# --------------------------------------------------------------------------- #
# (colonne, source, libelle, sens) -- sens : -1 = plus bas est meilleur
INDICATEURS = [
    ("fneg_1_fir", "cond", "fréquence instantanée négative (FIR)", -1),
    ("HRiqr_1_fir", "cond", "IQR de la FC instantanée (BPM)", -1),
    # Pas de barres verticales dans un libelle : il sert aussi de cellule de
    # tableau Markdown, ou « | » ouvrirait une colonne.
    ("dmd_ecart_abs", "cond", "écart absolu FC DMD − FC d'ancrage (BPM)", -1),
    ("n_channels", "cond", "canaux retenus par la combinaison", 0),
]


def valeurs(k, col, source):
    if source == "cond":
        return COND[k].set_index("slug").loc[COMMUNS, col].astype(float)
    return METH[k].loc[COMMUNS, col].astype(float)


def fig_qualite_pouls():
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=[lib for _, _, lib, _ in INDICATEURS])
    for i, (col, source, lib, _) in enumerate(INDICATEURS):
        r, c = divmod(i, 2)
        for j, k in enumerate(ORDRE):
            v = valeurs(k, col, source)
            nom, _, coul = LOTS[k]
            fig.add_trace(go.Box(x=[j] * len(v), y=v, name=nom, marker={"color": coul},
                                 line={"color": coul}, fillcolor="rgba(0,0,0,0)", width=0.5,
                                 boxpoints=False, hoverinfo="skip", showlegend=False),
                          r + 1, c + 1)
            fig.add_trace(go.Scatter(
                x=j + essaim(v.to_numpy(), largeur=0.2), y=v, mode="markers", showlegend=False,
                marker={"color": coul, "size": 5, "opacity": 0.75}, text=v.index,
                hovertemplate="%{text}<br>%{y:.3g}<extra>" + nom + "</extra>"), r + 1, c + 1)
        fig.update_xaxes(tickvals=[0, 1, 2],
                         ticktext=[LOTS[k][0].replace(" + ", "<br>+ ") for k in ORDRE],
                         tickfont={"size": 9}, row=r + 1, col=c + 1)
    fig = mise_en_page(fig, f"Qualité du pouls et de la phase, sur les {len(COMMUNS)} "
                            "acquisitions communes aux trois lots", hauteur=720, legende=False)
    enregistrer(fig, "fig_qualite_pouls")


def tab_qualite():
    lignes = ["| indicateur | " + " | ".join(LOTS[k][0] for k in ORDRE)
              + " | A − B (ancrage) | B − référence (recalage) |",
              "|:--|" + "--:|" * (len(ORDRE) + 2)]
    rows = INDICATEURS + [("en_bande_frac", "meth", "fraction en bande FC ± 20 % (FIR) ⚠", 1)]
    for col, source, lib, sens in rows:
        v = {k: valeurs(k, col, source) for k in ORDRE}
        dAB = v["A"] - v["B"]
        dBr = v["B"] - v["ref"]

        def delta(d):
            if sens == 0:
                return med_iqr(d, 2)
            mieux = int(((d * sens) > 0).sum())
            return f"{med_iqr(d, 2)} · mieux {mieux}/{len(d)}"

        lignes.append(f"| {lib} | " + " | ".join(med_iqr(v[k], 2) for k in ORDRE)
                      + f" | {delta(dAB)} | {delta(dBr)} |")
        RESUME[f"q_{col}"] = " | ".join(f"{k} {med_iqr(v[k], 3)}" for k in ORDRE)
        RESUME[f"q_{col}_A-B"] = delta(dAB)
        RESUME[f"q_{col}_B-ref"] = delta(dBr)
    ecrire_table(nl.join(lignes) + nl + nl
                 + f": Médiane [IQR] sur les {len(COMMUNS)} acquisitions communes ; deltas "
                   "appariés acquisition par acquisition. « mieux » compte les acquisitions où "
                   "l'indicateur s'améliore. ⚠ La fraction en bande est mesurée dans la bande "
                   "± 20 % autour de l'ancre PROPRE à chaque lot : elle ne se compare pas "
                   "directement d'un lot à l'autre. {.striped}", "tab_qualite")


fig_qualite_pouls()
tab_qualite()


# --------------------------------------------------------------------------- #
# 3. Les pouls, acquisition par acquisition (A contre B)
# --------------------------------------------------------------------------- #
def fig_pouls():
    slugs = [s for s in COMMUNS
             if (LOTS["A"][1] / "traces" / f"{s}.npz").exists()
             and (LOTS["B"][1] / "traces" / f"{s}.npz").exists()]
    if not slugs:
        absente("fig_pouls", "Aucune trace `traces/<slug>.npz` commune aux lots A et B.")
        return
    fig = go.Figure()
    titres = []
    hr_ab = {k: COND[k].set_index("slug")["hr_BPM"] for k in ("A", "B")}
    for i, s in enumerate(slugs):
        for k in ("B", "A"):
            with np.load(LOTS[k][1] / "traces" / f"{s}.npz") as z:
                u, y = z["u_time"], z["pulse_1_fir"]
            fig.add_trace(go.Scatter(
                x=u, y=norm(y), mode="lines", visible=(i == 0), name=LOTS[k][0],
                legendgroup=k, showlegend=True, line={"color": LOTS[k][2], "width": 1.4},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>" + LOTS[k][0] + "</extra>"))
        titres.append(f"{s}<br>ancre consensus {fr(hr_ab['B'].get(s))} BPM · "
                      f"ancre médiane {fr(hr_ab['A'].get(s))} BPM")
    boutons = []
    for i, s in enumerate(slugs):
        vis = [False] * (2 * len(slugs))
        vis[2 * i:2 * i + 2] = [True, True]
        boutons.append({"label": s, "method": "update",
                        "args": [{"visible": vis}, {"title.text": titres[i]}]})
    fig.update_xaxes(title_text="temps (s)")
    fig.update_yaxes(title_text="écart-type")
    fig = mise_en_page(fig, titres[0], hauteur=460)
    fig.update_layout(
        margin={"t": 110},
        updatemenus=[{"buttons": boutons, "direction": "down", "showactive": True,
                      "x": 1.0, "xanchor": "right", "y": 1.0, "yanchor": "bottom",
                      "pad": {"b": 28}, "bgcolor": "rgba(120,120,120,0.25)",
                      "bordercolor": GRILLE, "font": {"size": 11, "color": "#222"}}])
    enregistrer(fig, "fig_pouls")

    # Accord des deux pouls : ils ne different que par l'ancre (bande du FIR et de la
    # combinaison), la video et la ROI etant identiques.
    r = []
    for s in slugs:
        with np.load(LOTS["A"][1] / "traces" / f"{s}.npz") as za, \
                np.load(LOTS["B"][1] / "traces" / f"{s}.npz") as zb:
            a, b, core = za["pulse_1_fir"], zb["pulse_1_fir"], za["core"].astype(bool)
            if len(a) == len(b) and core.sum() > 10:
                r.append(abs(float(np.corrcoef(a[core], b[core])[0, 1])))
    RESUME["pouls_AB_abs_r"] = f"{med_iqr(r, 2)} (n = {len(r)})"


fig_pouls()


# --------------------------------------------------------------------------- #
# 4. Pouls d'intensite (COSTS, un mode SVD, combinaison) contre pouls d'epaisseur
# --------------------------------------------------------------------------- #
# Trois pouls d'intensite tires des memes pixels (colonne ``intensite`` de
# ``ct_pulse.csv``, ``Reproducibility/compute_ct_pulse_traces.py``) :
#   costs        bande cardiaque de COSTS (BOP-DMD sur fenetre glissante,
#                ``Reproducibility/compute_costs_pulse.py``) ;
#   mode         premier mode SVD qui pulse (``compute_svd_first_mode.py``) ;
#   combinaison  ``pulse_1_fir`` du lot.
# L'epaisseur des masques cascade_v9 passe au MEME FIR que chacun.
CT, MODE, COSTS_T = {}, {}, {}
for k in ("A", "B"):
    f = LOTS[k][1] / "ct_pulse.csv"
    if f.exists():
        c = pd.read_csv(f)
        if "intensite" in c.columns:
            c["sig"] = c["plv"] > c["plv_null95"]
            for nom, g in c.groupby("intensite"):
                CT[(k, nom)] = g.set_index("slug")
    f = LOTS[k][1] / "svd_first_mode.csv"
    if f.exists():
        m = pd.read_csv(f)
        MODE[k] = m[m["status"] == "ok"].set_index("slug")
    f = LOTS[k][1] / "costs_pulse.csv"
    if f.exists():
        m = pd.read_csv(f)
        COSTS_T[k] = m[m["status"].isin(["ok", "bande_hors_ancre"])].set_index("slug")
C_COSTS = "#7a4fd6"
SERIES = [(("A", "costs"), "COSTS · ancre médiane", C_COSTS),
          (("B", "costs"), "COSTS · ancre consensus", "#c98b1a"),
          (("A", "mode"), "premier mode SVD · ancre médiane", C_INT),
          (("A", "combinaison"), "combinaison · ancre médiane (référence)", "#8a94a6")]
LIB = {"costs": "COSTS", "mode": "premier mode SVD", "combinaison": "combinaison"}
MANQUE_CT = ("Lancer `Reproducibility/compute_costs_pulse.py` et "
             "`Reproducibility/compute_svd_first_mode.py` (lots A et B), puis "
             "`Reproducibility/compute_ct_pulse_traces.py`.")


def fig_pouls_int_ct():
    racine = LOTS["A"][1]
    fichiers = {"ct": racine / "ct_pulse_traces.npz", "costs": racine / "costs_pulse_traces.npz",
                "mode": racine / "svd_first_mode_traces.npz"}
    if (("A", "costs") not in CT or "A" not in COSTS_T
            or not all(f.exists() for f in fichiers.values())):
        absente("fig_pouls_int_ct", MANQUE_CT)
        return
    Z = {k: np.load(f) for k, f in fichiers.items()}
    cc = CT[("A", "costs")]
    slugs = [s for s in cc.index
             if f"{s}__ct_fir_um" in Z["ct"].files and f"{s}__pulse_costs_fir" in Z["costs"].files
             and f"{s}__pulse_mode_fir" in Z["mode"].files
             and (racine / "traces" / f"{s}.npz").exists()]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32], vertical_spacing=0.1,
                        shared_xaxes=True)
    n_par = 6
    titres, bords = [], []
    for i, s in enumerate(slugs):
        with np.load(racine / "traces" / f"{s}.npz") as z:
            u, y_comb, core = z["u_time"], z["pulse_1_fir"], z["core"].astype(bool)
        ct = Z["ct"][f"{s}__ct_fir_um"].astype(float)
        vis = i == 0
        cache = "legendonly" if vis else False
        k_mode = int(MODE["A"].loc[s, "mode"]) if s in MODE.get("A", pd.DataFrame()).index else -1
        fig.add_trace(go.Scatter(
            x=u, y=norm(Z["costs"][f"{s}__pulse_costs_fir"]), mode="lines", visible=vis,
            name="intensité : COSTS", legendgroup="costs", line={"color": C_COSTS, "width": 1.6},
            hovertemplate="%{x:.2f} s — %{y:.2f} σ<extra>COSTS</extra>"), 1, 1)
        fig.add_trace(go.Scatter(
            x=u, y=norm(ct), customdata=ct - np.mean(ct[core]), mode="lines", visible=vis,
            name="épaisseur (masque)", legendgroup="ct", line={"color": C_CT, "width": 1.5},
            hovertemplate="%{x:.2f} s — %{y:.2f} σ (%{customdata:+.2f} µm)"
                          "<extra>épaisseur</extra>"), 1, 1)
        fig.add_trace(go.Scatter(
            x=u, y=norm(Z["mode"][f"{s}__pulse_mode_fir"]), mode="lines", visible=cache,
            name="intensité : premier mode SVD", legendgroup="mode",
            line={"color": C_INT, "width": 1.0, "dash": "dash"},
            hovertemplate="%{x:.2f} s — %{y:.2f} σ<extra>mode " + str(k_mode) + "</extra>"), 1, 1)
        fig.add_trace(go.Scatter(
            x=u, y=norm(y_comb), mode="lines", visible=cache,
            name="intensité : combinaison", legendgroup="comb",
            line={"color": "#8a94a6", "width": 1.0, "dash": "dot"},
            hovertemplate="%{x:.2f} s — %{y:.2f} σ<extra>combinaison</extra>"), 1, 1)
        tf, ff = Z["costs"][f"{s}__t_fenetres"], Z["costs"][f"{s}__f_suivi_bpm"]
        r = cc.loc[s]
        fig.add_trace(go.Scatter(
            x=tf, y=ff, mode="lines+markers", visible=vis, name="COSTS : FC suivie f(t)",
            legendgroup="f", line={"color": C_COSTS, "width": 1.2}, marker={"size": 5},
            hovertemplate="%{x:.1f} s — %{y:.1f} BPM<extra>f(t)</extra>"), 2, 1)
        fig.add_trace(go.Scatter(
            x=[u[0], u[-1]], y=[r.hr_BPM, r.hr_BPM], mode="lines", visible=vis,
            name="ancre de FC", legendgroup="ancre",
            line={"color": C_REF, "width": 1, "dash": "dash"},
            hovertemplate="ancre %{y:.1f} BPM<extra></extra>"), 2, 1)
        verdict = "au-dessus du hasard" if r.sig else "pas au-dessus du hasard"
        ref = []
        for nom in ("mode", "combinaison"):
            t_ = CT.get(("A", nom))
            if t_ is not None and s in t_.index:
                ref.append(f"{LIB[nom]} {fr(t_.loc[s, 'plv'], 2)}")
        info = COSTS_T["A"].loc[s]
        titres.append(f"{s} · ancre {fr(r.hr_BPM)} BPM · bande COSTS {fr(info.bande_BPM)} BPM "
                      f"(fenêtre {fr(info.fenetre_s)} s)"
                      f"<br>COSTS / épaisseur : r {fr(r.r, 2)} · Δφ {fr(r.dphi_deg, 0)}° · "
                      f"PLV {fr(r.plv, 2)} (seuil {fr(r.plv_null95, 2)}, {verdict})"
                      + (" · PLV " + ", ".join(ref) if ref else ""))
        idx = np.flatnonzero(core)
        gris = {"type": "rect", "xref": "x", "yref": "y domain", "y0": 0, "y1": 1,
                "fillcolor": "rgba(150,150,150,0.15)", "line": {"width": 0}, "layer": "below"}
        bords.append([dict(gris, x0=float(u[0]), x1=float(u[idx[0]])),
                      dict(gris, x0=float(u[idx[-1]]), x1=float(u[-1]))])
    boutons = []
    for i, s in enumerate(slugs):
        vis = [False] * (n_par * len(slugs))
        vis[n_par * i:n_par * i + n_par] = [True, True, "legendonly", "legendonly", True, True]
        boutons.append({"label": s, "method": "update",
                        "args": [{"visible": vis}, {"title.text": titres[i], "shapes": bords[i]}]})
    fig.update_xaxes(title_text="temps (s)", row=2, col=1)
    fig.update_yaxes(title_text="écart-type", row=1, col=1)
    fig.update_yaxes(title_text="FC suivie (BPM)", row=2, col=1)
    fig = mise_en_page(fig, titres[0], hauteur=620)
    fig.update_layout(
        shapes=bords[0], margin={"t": 120, "b": 110}, legend={"y": -0.1},
        updatemenus=[{"buttons": boutons, "direction": "down", "showactive": True,
                      "x": 1.0, "xanchor": "right", "y": 1.0, "yanchor": "bottom",
                      "pad": {"b": 40}, "bgcolor": "rgba(120,120,120,0.25)",
                      "bordercolor": GRILLE, "font": {"size": 11, "color": "#222"}}])
    enregistrer(fig, "fig_pouls_int_ct")


def fig_accord_int_ct():
    if not CT:
        absente("fig_accord_int_ct", MANQUE_CT)
        return
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, shared_xaxes=True,
                        subplot_titles=["PLV intensité / épaisseur (plein : au-dessus du "
                                        "seuil 95 % des décalages circulaires)",
                                        "déphasage moyen Δφ (intensité − épaisseur)"])
    for (cle, nom, coul), dec in zip(SERIES, (-0.3, -0.1, 0.1, 0.3)):
        if cle not in CT:
            continue
        c = CT[cle].reset_index()
        x = np.empty(len(c))
        for p in c["participant"].unique():
            sel = (c["participant"] == p).to_numpy()
            x[sel] = PARTICIPANTS.index(p) + dec + essaim(c.loc[sel, "plv"].to_numpy(),
                                                           largeur=0.07)
        texte = (c["slug"] + "<br>r " + c["r"].round(2).astype(str) + " · Δφ "
                 + c["dphi_deg"].round(0).astype(int).astype(str) + "° · PLV "
                 + c["plv"].round(2).astype(str) + " (seuil " + c["plv_null95"].round(2).astype(str)
                 + ")<br>ancre " + c["hr_BPM"].round(1).astype(str) + " BPM")
        if cle[1] == "mode" and cle[0] in MODE:
            texte = texte + " · mode n° " + c["slug"].map(MODE[cle[0]]["mode"]).astype(
                "Int64").astype(str)
        if cle[1] == "costs" and cle[0] in COSTS_T:
            texte = texte + " · bande COSTS " + c["slug"].map(
                COSTS_T[cle[0]]["bande_BPM"]).round(1).astype(str) + " BPM"
        for sig in (True, False):
            m = (c["sig"] == sig).to_numpy()
            marq = {"color": coul if sig else "rgba(0,0,0,0)", "size": 7,
                    "line": {"color": coul, "width": 1.5}}
            for row, col in ((1, "plv"), (2, "dphi_deg")):
                fig.add_trace(go.Scatter(
                    x=x[m], y=c.loc[m, col], mode="markers", marker=marq, text=texte[m],
                    name=nom, legendgroup=nom, showlegend=sig and row == 1,
                    hovertemplate="%{text}<extra>" + nom + "</extra>"), row, 1)
    fig.update_yaxes(title_text="PLV", range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(title_text="Δφ (°)", range=[-185, 185],
                     tickvals=[-180, -90, 0, 90, 180], row=2, col=1)
    fig.update_xaxes(tickvals=list(range(len(PARTICIPANTS))),
                     ticktext=[nom_participant(p).replace(" ", "<br>", 1) for p in PARTICIPANTS],
                     range=[-0.6, len(PARTICIPANTS) - 0.4], tickfont={"size": 10}, row=2, col=1)
    fig = mise_en_page(fig, "Le pouls d'épaisseur suit-il le pouls d'intensité ?"
                            "<br>COSTS, premier mode SVD et combinaison, vidéos cascade_v9",
                       hauteur=760)
    fig.update_layout(margin={"t": 100, "b": 160}, legend={"y": -0.14})
    enregistrer(fig, "fig_accord_int_ct")


def tab_int_ct():
    if ("A", "costs") not in CT or "A" not in COSTS_T:
        absente("tab_int_ct", MANQUE_CT)
        return
    ca = CT[("A", "costs")]
    lignes = ["| participant | acquisitions | PLV > seuil : COSTS, ancre médiane "
              "| COSTS, ancre consensus | premier mode SVD | combinaison | PLV COSTS [IQR] "
              "| étendue de f(t) COSTS (IQR, BPM) | Δφ COSTS (°) |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]

    def n_sig(cle, p):
        t_ = CT.get(cle)
        if t_ is None:
            return "—"
        g = t_[t_["participant"] == p]
        return f"{int(g.sig.sum())}/{len(g)}"

    for p in PARTICIPANTS:
        a = ca[ca["participant"] == p]
        if a.empty:
            continue
        a = a.join(COSTS_T["A"][["f_suivi_iqr_BPM"]]).sort_values(["eye", "replicate"])
        dphi = " · ".join(f"{e}{int(n)} {d:+.0f}" for e, n, d in
                          a[["eye", "replicate", "dphi_deg"]].itertuples(index=False))
        lignes.append(f"| {nom_participant(p)} | {len(a)} | {n_sig(('A', 'costs'), p)} "
                      f"| {n_sig(('B', 'costs'), p)} | {n_sig(('A', 'mode'), p)} "
                      f"| {n_sig(('A', 'combinaison'), p)} | {med_iqr(a.plv, 2)} "
                      f"| {med_iqr(a.f_suivi_iqr_BPM, 1)} | {dphi} |")
    for (k, nom), c in CT.items():
        cle = f"ct_{k}_{nom}"
        RESUME[f"{cle}_sig"] = f"{int(c.sig.sum())}/{len(c)}"
        RESUME[f"{cle}_plv"] = med_iqr(c.plv, 2)
        RESUME[f"{cle}_abs_r"] = med_iqr(c.abs_r, 2)
        RESUME[f"{cle}_null95"] = med_iqr(c.plv_null95, 2)
        s = c[c.sig]
        RESUME[f"{cle}_sig_slugs"] = ", ".join(f"{i} ({d:+.0f}°)" for i, d in s.dphi_deg.items())
    for k, m in COSTS_T.items():
        RESUME[f"costs_{k}_n"] = f"{len(m)} ({int((m.status == 'ok').sum())} bande dans FC ± 20 %)"
        RESUME[f"costs_{k}_bande_moins_ancre"] = med_iqr(m.bande_BPM - m.hr_BPM, 1)
        RESUME[f"costs_{k}_f_iqr"] = med_iqr(m.f_suivi_iqr_BPM, 1)
        RESUME[f"costs_{k}_frac_fen"] = med_iqr(m.frac_fenetres_bande, 2)
        RESUME[f"costs_{k}_energie"] = med_iqr(m.energie_rel, 3)
        RESUME[f"costs_{k}_fenetre_s"] = med_iqr(m.fenetre_s, 1)
    for k, m in MODE.items():
        RESUME[f"mode_{k}_indices"] = str(m["mode"].astype(int).value_counts().sort_index().to_dict())
    ecrire_table(nl.join(lignes) + nl + nl
                 + ": Accord entre pouls d'intensité et pouls d'épaisseur, par participant. "
                   "« PLV > seuil » : acquisitions dont la PLV dépasse le 95e centile de 500 "
                   "décalages circulaires de l'épaisseur (≈ 5 % attendus par hasard), pour "
                   "COSTS avec chaque ancre, puis pour le premier mode SVD et la combinaison du "
                   "lot (ancre médiane). « étendue de f(t) » : IQR, sur les fenêtres, de la FC "
                   "suivie par COSTS dans une acquisition ; médiane [IQR] sur les acquisitions. "
                   "Δφ : une valeur par acquisition (œil et rang du réplicat) ; le signe du pouls "
                   "étant conventionnel, 0° et ±180° décrivent la même relation au signe "
                   "près. {.striped}", "tab_int_ct")


fig_pouls_int_ct()
fig_accord_int_ct()
tab_int_ct()

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + nl + get_plotlyjs() + nl + "</script>" + nl,
    encoding="utf-8")
print("  plotlyjs.html")
(SORTIE / "resume.txt").write_text(nl.join(f"{k} = {v}" for k, v in RESUME.items()) + nl,
                                   encoding="utf-8")
print("  resume.txt")
