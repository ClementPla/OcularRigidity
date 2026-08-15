"""Figure "bruit d'image mesure sur les vraies donnees" pour le deck reveal.

Reprend la derniere cellule de notebook/simulate_svd_radial_gaussians_viewer.ipynb
(section "Statistiques de bruit mesurees sur les vraies images"), avec la charte
graphique du deck : fond transparent, texte gris clair (theme: dark).

Source des donnees : Astronauts/simulation_output/image_noise/image_noise_summary.csv
produit par `python Astronauts/quantify_image_noise.py`.

Relancer :  python reveal_quarto_presentations/figures_bruit_reel/make_figure_bruit.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
ASTRONAUTS_DIR = RACINE / "Astronauts"
sys.path.insert(0, str(ASTRONAUTS_DIR))
import simulate_svd_radial_gaussians as sim  # noqa: E402

SORTIE = Path(__file__).parent
NOISE_CSV = ASTRONAUTS_DIR / "simulation_output" / "image_noise" / "image_noise_summary.csv"

FIT_KIND = "mean"   # "mean" = ecart a la moyenne du cube ; "lag1" = controle
R2_MIN = 0.5

# --------------------------------------------------------------------------- #
# Style : identique a figures_gaussienne_svd/make_figures.py
# --------------------------------------------------------------------------- #
COULEUR_TEXTE = "#c9c9c9"     # passer a "#3a3a3a" pour un fond clair
C_GRIS = "#9a9a9a"

plt.rcParams.update({
    "savefig.transparent": True,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "text.color": COULEUR_TEXTE,
    "axes.labelcolor": COULEUR_TEXTE,
    "axes.edgecolor": COULEUR_TEXTE,
    "xtick.color": COULEUR_TEXTE,
    "ytick.color": COULEUR_TEXTE,
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "legend.labelcolor": COULEUR_TEXTE,
})

VARIANTES = ["brut", "normalise"]
COULEUR = {"brut": "#2a78d6", "normalise": "#eb6834"}

assert NOISE_CSV.exists(), (
    f"{NOISE_CSV} absent -- lancez d'abord `python Astronauts/quantify_image_noise.py`."
)

dfn = pd.read_csv(NOISE_CSV)
n_cubes = dfn["npz_path"].nunique()
bon = dfn[f"{FIT_KIND}_power_r2"] >= R2_MIN
n_ecartes = int((~bon & (dfn["variant"] == "brut")).sum())
dfn = dfn[bon]
print(f"{n_cubes} cubes analyses, {n_ecartes} ecartes (R2 loi puissance < {R2_MIN})")


def charger(variant):
    out = {}
    for path in dfn.loc[dfn["variant"] == variant, "npz_path"]:
        z = np.load(path)
        p = f"{variant}_{FIT_KIND}_"
        out[path] = {
            "mu": z[p + "bin_mu"].astype(float),
            "mu2": z[p + "bin_mu2"].astype(float),
            "var": z[p + "bin_var"].astype(float),
            "count": z[p + "bin_count"].astype(float),
            "hist": z[f"{variant}_resid_hist"].astype(float),
            "hist_edges": z[f"{variant}_resid_hist_edges"].astype(float),
        }
    return out


courbes = {v: charger(v) for v in VARIANTES}


def fit3(c):
    """Ajustement conjoint : var = c0 + c1*mu + c2*mu^2."""
    A = np.stack([np.ones_like(c["mu"]), c["mu"], c["mu2"]], axis=1)
    sw = np.sqrt(c["count"])
    coef, *_ = np.linalg.lstsq(A * sw[:, None], c["var"] * sw, rcond=None)
    return coef


def contributions(variant):
    rows = []
    for path, c in courbes[variant].items():
        c0, c1, c2 = fit3(c)
        mu_ref = float(np.average(c["mu"], weights=c["count"]))
        parts = np.array([c0, c1 * mu_ref, c2 * mu_ref ** 2])
        total = parts.sum()
        rows.append({
            "npz_path": path, "mu_ref": mu_ref, "var_ref": total,
            "frac_const": parts[0] / total,
            "frac_poisson": parts[1] / total,
            "frac_speckle": parts[2] / total,
        })
    return pd.DataFrame(rows)


contrib = {v: contributions(v) for v in VARIANTES}
dfb = dfn[dfn["variant"] == "brut"]

# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

# (1) var vs mu
ax = axes[0, 0]
for variant in VARIANTES:
    for c in courbes[variant].values():
        ax.plot(c["mu"], c["var"], color=COULEUR[variant], lw=0.4, alpha=0.25)
grille = np.geomspace(
    np.median([c["mu"][0] for c in courbes["brut"].values()]),
    np.median([c["mu"][-1] for c in courbes["brut"].values()]), 40,
)
medianes = {}
for variant in VARIANTES:
    ys = [np.interp(np.log(grille), np.log(c["mu"]), np.log(c["var"]),
                    left=np.nan, right=np.nan) for c in courbes[variant].values()]
    medianes[variant] = np.exp(np.nanmedian(np.vstack(ys), axis=0))
    ax.plot(grille, medianes[variant], color=COULEUR[variant], lw=2.4,
            label=f"mediane -- {variant}")
milieu = len(grille) // 2
ancre = medianes["brut"][milieu]
for pente, style, nom in ((1.0, "--", "Poisson"), (2.0, ":", "speckle")):
    ax.plot(grille, ancre * (grille / grille[milieu]) ** pente,
            color=C_GRIS, ls=style, lw=1.1, label=f"pente {pente:g} ({nom})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("intensite moyenne du pixel, mu (niveaux)")
ax.set_ylabel("variance temporelle (niveaux^2)")
ax.set_title("Variance vs moyenne, un trait par cube")
ax.legend(fontsize=7)

# (2) exposant de la loi puissance
ax = axes[0, 1]
bins = np.linspace(-0.5, 2.2, 34)
for variant in VARIANTES:
    ax.hist(dfn.loc[dfn["variant"] == variant, f"{FIT_KIND}_power_p"], bins=bins,
            color=COULEUR[variant], alpha=0.55, label=variant)
for x, nom in ((0.0, "additif"), (1.0, "Poisson"), (2.0, "speckle")):
    ax.axvline(x, color=C_GRIS, ls="--", lw=1.0)
    ax.text(x, ax.get_ylim()[1] * 0.96, nom, rotation=90, fontsize=7,
            ha="right", va="top", color=C_GRIS)
ax.set_xlabel("exposant p de var = c * mu^p")
ax.set_ylabel("nombre de cubes")
ax.set_title("Quel modele de bruit ?")
ax.legend(fontsize=8)

# (3) part de variance par terme
ax = axes[0, 2]
etiquettes = ["constante", "~ mu\n(Poisson)", "~ mu^2\n(speckle)"]
donnees = [contrib["brut"][k].clip(-0.2, 1.2)
           for k in ("frac_const", "frac_poisson", "frac_speckle")]
bp = ax.boxplot(donnees, showfliers=False, patch_artist=True)
for patch, couleur in zip(bp["boxes"], (C_GRIS, "#3fa34d", "#d1495b")):
    patch.set_facecolor(couleur); patch.set_alpha(0.6)
for partie in ("whiskers", "caps", "medians"):
    for art in bp[partie]:
        art.set_color(COULEUR_TEXTE)
ax.axhline(0.0, color=C_GRIS, lw=0.8)
ax.set_xticks(range(1, 4)); ax.set_xticklabels(etiquettes, fontsize=8)
ax.set_ylabel("part de la variance a mu typique (brut)")
ax.set_title("Ou est la variance ? (ajustement conjoint)")

# (4) noise_level equivalent
ax = axes[1, 0]
for col, couleur, nom in (
    (f"{FIT_KIND}_noise_level_speckle", "#d1495b", "depuis le terme speckle"),
    (f"{FIT_KIND}_noise_level_floor_mean", C_GRIS, "depuis le plancher (ancre moyenne)"),
    (f"{FIT_KIND}_noise_level_floor_p90", "#9b5de5", "depuis le plancher (ancre p90)"),
):
    v = dfb[col].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(v, bins=np.geomspace(0.01, 20, 40), color=couleur, alpha=0.5, label=nom)
ax.axvline(sim.REALISTIC_NOISE_LEVEL, color=COULEUR_TEXTE, lw=1.6,
           label=f"noise_level = {sim.REALISTIC_NOISE_LEVEL:g} (pose par la simulation)")
ax.set_xscale("log")
ax.set_xlabel("noise_level equivalent")
ax.set_ylabel("nombre de cubes")
ax.set_title("Le noise_level de la simulation, vu par la mesure")
ax.legend(fontsize=7)

# (5) ce que la normalisation par frame enleve
ax = axes[1, 1]
fusion = contrib["brut"].merge(contrib["normalise"], on="npz_path",
                               suffixes=("_brut", "_norm"))
gain_cv_col = (dfn[dfn["variant"] == "brut"].set_index("npz_path")["frame_gain_cv"]
               .reindex(fusion["npz_path"]).to_numpy())
ax.scatter(gain_cv_col, fusion["var_ref_norm"] / fusion["var_ref_brut"],
           s=18, color=COULEUR["normalise"], alpha=0.7)
ax.axhline(1.0, color=C_GRIS, lw=1.0)
ax.set_xlabel("fluctuation de gain par frame (ecart-type relatif)")
ax.set_ylabel("variance normalisee / variance brute")
ax.set_title("Part de la variance qui n'est qu'un gain global")

# (6) forme de la distribution du bruit
ax = axes[1, 2]
for variant in VARIANTES:
    hists = np.vstack([c["hist"] / c["hist"].sum() for c in courbes[variant].values()])
    edges = next(iter(courbes[variant].values()))["hist_edges"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    largeur = edges[1] - edges[0]
    ax.plot(centres, np.median(hists, axis=0) / largeur, color=COULEUR[variant],
            lw=1.6, label=f"mesure -- {variant}")
x = np.linspace(-6, 6, 400)
ax.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color=C_GRIS, ls="--", lw=1.2,
        label="N(0, 1)")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1.0)
ax.set_xlabel("residu standardise (I - mu) / sigma")
ax.set_ylabel("densite")
ax.set_title(f"Forme du bruit (asymetrie mediane = {dfb['resid_skew'].median():+.2f})")
ax.legend(fontsize=8)

fig.suptitle(
    f"Bruit d'image mesure sur {dfb['npz_path'].nunique()} cubes recales "
    f"(variante brut, variance = {FIT_KIND})", fontsize=12, color=COULEUR_TEXTE,
)
fig.tight_layout()
chemin = SORTIE / "bruit_reel_mesure.png"
fig.savefig(chemin)
plt.close(fig)
print("ecrit :", chemin)


# --------------------------------------------------------------------------- #
# Chiffres a citer sur la slide
# --------------------------------------------------------------------------- #
def med_iqr(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return f"{s.median():.3g} [{s.quantile(.25):.3g}, {s.quantile(.75):.3g}]"


parts = contrib["brut"][["frac_const", "frac_poisson", "frac_speckle"]].median()
ref = np.load(dfb["npz_path"].iloc[0])
R_mean_sim = float(ref["sim_R_mean"])
L_mesure = dfb[f"{FIT_KIND}_speckle_n_looks"].median()
floor_rel = dfb[f"{FIT_KIND}_floor_rel_mean"].median()

print(f"""
=== A CITER SUR LA SLIDE ===
cubes retenus            : {dfn['npz_path'].nunique()} / {n_cubes}
exposant p (brut)        : {med_iqr(dfb[f'{FIT_KIND}_power_p'])}
part const/Poisson/speckle: {parts['frac_const']:.0%} / {parts['frac_poisson']:.0%} / {parts['frac_speckle']:.0%}
gain par frame (CV)      : {med_iqr(dfb['frame_gain_cv'])}
asymetrie des residus    : {dfb['resid_skew'].median():+.2f}
noise_level (speckle)    : {med_iqr(dfb[f'{FIT_KIND}_noise_level_speckle'])}
noise_level (plancher p90): {med_iqr(dfb[f'{FIT_KIND}_noise_level_floor_p90'])}
facteur de divergence    : {dfb[f'{FIT_KIND}_noise_level_floor_p90'].median() / dfb[f'{FIT_KIND}_noise_level_speckle'].median():.0f}
--- simulation actuelle ---
N_LOOKS_REALISTIC        = {sim.N_LOOKS_REALISTIC:g}  -> mesure {L_mesure:.0f}
BRUIT_ADDITIF_REALISTIC  = {sim.BRUIT_ADDITIF_REALISTIC:g}  -> mesure {floor_rel * R_mean_sim:.3f}
""")
