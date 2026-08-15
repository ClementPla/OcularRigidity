"""Figures de la simulation "anneau de gaussiennes radiales" pour le deck reveal.

Reprend notebook/simulate_svd_radial_gaussians_viewer.ipynb (cellules 3 et 7)
avec la charte du deck : fond transparent, texte gris clair (theme: dark).

Produit :
  concept_anneau.png      -- le cercle qui respire, avec ses features de bord
  concept_kymographe.png  -- le mouvement sous-pixel, visible apres retrait de la moyenne
  bruit_sources.png       -- jitter x / jitter y / bruit d'image, image + kymographe
  resultats_phaseA.png    -- balayages 1D : taux de detection et correlation

Relancer (kernel pyOR) :
  C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
      reveal_quarto_presentations/figures_simulation/make_figures_simulation.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "Astronauts"))
import simulate_svd_radial_gaussians as sim  # noqa: E402

SORTIE = Path(__file__).parent
SORTIE.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Style : identique aux autres dossiers de figures du deck
# --------------------------------------------------------------------------- #
COULEUR_TEXTE = "#c9c9c9"
C_GRIS = "#9a9a9a"
C_SIGNAL = "#2a78d6"
C_BRUIT = "#eb6834"
C_ROUGE = "#d1495b"

plt.rcParams.update({
    "savefig.transparent": True,
    "savefig.dpi": 190,
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

N_FRAMES_APERCU = 60

CALIB = sim.load_real_calibration()
ANATOMIE = sim.build_ring_anatomy()
FORME = sim.crop_shape(CALIB)
MASQUE = sim.build_fixed_mask(FORME)
T_COMPLET = np.arange(CALIB.n_frames) / CALIB.fs_hz
DUREE = float(T_COMPLET[-1])
ASPECT_PHYSIQUE = CALIB.scale_y_um / CALIB.scale_x_um

AMPL_PX_AXIAL = sim.TYPICAL_AMPLITUDE_UM / CALIB.scale_y_um

print(f"calibration : {CALIB.scale_x_um:.1f} um/px lateral, "
      f"{CALIB.scale_y_um:.1f} um/px axial, {CALIB.fs_hz:.1f} Hz, "
      f"{CALIB.n_frames} frames ({DUREE:.1f} s)")
print(f"amplitude typique {sim.TYPICAL_AMPLITUDE_UM:.0f} um c-a-c = "
      f"{AMPL_PX_AXIAL:.2f} px axial  -> SOUS-PIXEL")


def rendre_cube(amplitude_um=sim.TYPICAL_AMPLITUDE_UM, jitter_x_px=0.0,
                jitter_y_px=0.0, noise_level=0.0, n_frames=N_FRAMES_APERCU, seed=0):
    t = T_COMPLET[:n_frames]
    rng = np.random.default_rng(seed)
    jx, jy = sim.sample_jitter(
        n_frames, FORME[1], sim.JitterParams(jitter_x_px, jitter_y_px), rng
    )
    cube = sim.render_cube(ANATOMIE, CALIB, t, DUREE, amplitude_um, jx, jy, FORME)
    return sim.add_image_noise(cube, noise_level, rng), t


def kymographe(cube):
    return cube[:, :, cube.shape[2] // 2].T


# =========================================================================== #
# 1. Concept : l'anneau et ses features de bord
# =========================================================================== #
cube_ref, t_ref = rendre_cube(noise_level=1.0)
cube_net, _ = rendre_cube(noise_level=0.0)

indices = np.linspace(0, len(t_ref) - 1, 4).astype(int)
fig, axes = plt.subplots(1, 6, figsize=(17, 3.6))
vmax = np.percentile(cube_ref, 99.5)
for ax, i in zip(axes, indices):
    ax.imshow(cube_ref[i], cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"t = {t_ref[i]:.2f} s", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[4].imshow(cube_ref.mean(axis=0), cmap="magma")
axes[4].contour(MASQUE, levels=[0.5], colors="cyan", linewidths=0.8)
axes[4].set_title("moyenne + ROI", fontsize=9)
axes[4].set_xticks([]); axes[4].set_yticks([])
axes[5].imshow(cube_ref.mean(axis=0), cmap="magma", aspect=ASPECT_PHYSIQUE)
axes[5].set_title("moyenne, aspect physique", fontsize=9)
axes[5].set_xticks([]); axes[5].set_yticks([])
fig.suptitle("Anneau de gaussiennes radiales : en pixels a gauche, en microns a droite",
             fontsize=11, color=COULEUR_TEXTE)
fig.tight_layout()
fig.savefig(SORTIE / "concept_anneau.png")
plt.close(fig)
print("ecrit : concept_anneau.png")

# --------------------------------------------------------------------------- #
# 2. Concept : le mouvement est sous-pixel
# --------------------------------------------------------------------------- #
k_net = kymographe(cube_net)
bord = int(np.argmax(k_net.mean(axis=1)))
fenetre = slice(max(bord - 12, 0), bord + 13)
lignes_zoom = np.arange(k_net.shape[0])[fenetre]
etendue = [t_ref[0], t_ref[-1], lignes_zoom[-1], lignes_zoom[0]]


def kymo_centre(cube_):
    k = kymographe(cube_)[fenetre]
    return k - k.mean(axis=1, keepdims=True)


fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
axes[0].imshow(k_net[fenetre], cmap="magma", aspect="auto", extent=etendue)
axes[0].set_title(f"kymographe brut (bord, ligne {bord}) : on ne voit rien", fontsize=10)
axes[0].set_ylabel("ligne de l'image")

kc = kymo_centre(cube_net)
v = np.abs(kc).max()
axes[1].imshow(kc, cmap="RdBu_r", aspect="auto", vmin=-v, vmax=v, extent=etendue)
axes[1].set_title("moins la moyenne temporelle : la pulsation apparait", fontsize=10)

axes[2].plot(t_ref, sim.radial_displacement(t_ref, DUREE, sim.TYPICAL_AMPLITUDE_UM),
             lw=1.5, color=C_SIGNAL)
axes[2].set_ylabel("deplacement radial (um)")
axes[2].set_title("le mouvement impose, pour comparaison", fontsize=10)
axes[2].grid(alpha=0.25, color=C_GRIS)
for ax in axes:
    ax.set_xlabel("temps (s)")
fig.suptitle(
    f"Le deplacement est SOUS-PIXEL : {sim.TYPICAL_AMPLITUDE_UM:.0f} um crete a crete "
    f"= {AMPL_PX_AXIAL:.2f} px en axial",
    fontsize=11, color=COULEUR_TEXTE,
)
fig.tight_layout()
fig.savefig(SORTIE / "concept_kymographe.png")
plt.close(fig)
print("ecrit : concept_kymographe.png")

# =========================================================================== #
# 3. Sources de bruit : jitter + image
# =========================================================================== #
N_FRAMES_COMPARAISON = 40
conditions = (
    ("net", dict(noise_level=0.0)),
    ("bruit d'image x1", dict(noise_level=1.0)),
    ("jitter x = 2 px\n(commun a l'image)", dict(noise_level=0.0, jitter_x_px=2.0)),
    ("jitter y = 2 px\n(par A-scan)", dict(noise_level=0.0, jitter_y_px=2.0)),
)

fig, axes = plt.subplots(2, len(conditions), figsize=(3.6 * len(conditions), 6.6))
for col, (nom, kwargs) in enumerate(conditions):
    cube_c, t_c = rendre_cube(n_frames=N_FRAMES_COMPARAISON, **kwargs)
    axes[0, col].imshow(cube_c[0], cmap="magma", vmin=0,
                        vmax=np.percentile(cube_c, 99.5))
    axes[0, col].set_title(nom, fontsize=10)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    k = kymographe(cube_c)[fenetre]
    k = k - k.mean(axis=1, keepdims=True)
    v = np.abs(k).max()
    axes[1, col].imshow(k, cmap="RdBu_r", aspect="auto", vmin=-v, vmax=v,
                        extent=[t_c[0], t_c[-1], lignes_zoom[-1], lignes_zoom[0]])
    axes[1, col].set_xlabel("temps (s)")
    if col == 0:
        axes[1, col].set_ylabel("ligne (coupe verticale, centree)")
fig.suptitle("Une image (haut) et son kymographe centre (bas), par type de degradation",
             fontsize=11, color=COULEUR_TEXTE)
fig.tight_layout()
fig.savefig(SORTIE / "bruit_sources.png")
plt.close(fig)
print("ecrit : bruit_sources.png")

# =========================================================================== #
# 4. Resultats : balayages 1D (phase A), depuis le summary.csv courant
# =========================================================================== #
FREQ_ERR_THRESHOLD = 0.10

df = pd.read_csv(sim.SUMMARY_CSV)
df["abs_corr"] = df["corr_combined_vs_truth"].abs()
df["detected"] = df["confidence"].isin(["high", "medium"]) & (
    df["freq_rel_err"] <= FREQ_ERR_THRESHOLD
)
print(f"\n{len(df)} runs ; detectes : {df['detected'].mean():.1%} "
      f"({int(df['detected'].sum())}/{len(df)})")

BALAYAGES = (
    ("A_amplitude", "amplitude_um", "amplitude radiale (um c-a-c)"),
    ("A_noise", "noise_level", "bruit d'image (1 = realiste)"),
    ("A_jitter_x", "jitter_x_px", "sigma jitter x (px, commun)"),
    ("A_jitter_y", "jitter_y_px", "sigma jitter y (px, par A-scan)"),
)

FIXES = {  # ce qui est tenu constant pendant chaque balayage
    "amplitude_um": "um",
    "noise_level": "x",
    "jitter_x_px": "px",
    "jitter_y_px": "px",
}


def legende_fixes(sub, param):
    """Les autres facteurs, tenus constants -- a afficher pour ne pas
    surinterpreter un balayage fait dans des conditions favorables."""
    bouts = []
    for autre, unite in FIXES.items():
        if autre == param:
            continue
        vals = sub[autre].unique()
        if len(vals) == 1:
            nom = autre.replace("_um", "").replace("_px", "").replace("_level", "")
            bouts.append(f"{nom} = {vals[0]:g} {unite}".replace(" x", "x"))
    return ", ".join(bouts)


fig, axes = plt.subplots(2, 4, figsize=(16, 6.6), sharey="row")
for col, (phase, param, xlabel) in enumerate(BALAYAGES):
    sub = df[df["phase"] == phase]
    x = np.array(sorted(sub[param].unique()))
    g = sub.groupby(param)

    ax = axes[0, col]
    ax.plot(x, g["detected"].mean().reindex(x), "o-", color=C_SIGNAL, lw=1.8, ms=5)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color=C_GRIS, ls=":", lw=1.0)
    ax.set_title(f"{phase.replace('A_', '')}  (n = {len(sub)})\n"
                 f"{legende_fixes(sub, param)}", fontsize=9)
    if col == 0:
        ax.set_ylabel("taux de detection")

    ax = axes[1, col]
    m = g["abs_corr"].mean().reindex(x)
    s = g["abs_corr"].std().reindex(x)
    ax.errorbar(x, m, yerr=s, fmt="o-", color=C_BRUIT, lw=1.8, ms=5,
                capsize=3, ecolor=C_GRIS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    if col == 0:
        ax.set_ylabel("|corr(combine, verite)|")

fig.suptitle(
    f"Phase A -- un facteur a la fois ; detecte = confiance high/medium "
    f"ET erreur de frequence <= {FREQ_ERR_THRESHOLD:.0%}",
    fontsize=11, color=COULEUR_TEXTE,
)
fig.tight_layout()
fig.savefig(SORTIE / "resultats_phaseA.png")
plt.close(fig)
print("ecrit : resultats_phaseA.png")

# --------------------------------------------------------------------------- #
# Chiffres a citer sur la slide
# --------------------------------------------------------------------------- #
print("\n=== A CITER SUR LA SLIDE (taux de detection par niveau) ===")
for phase, param, xlabel in BALAYAGES:
    sub = df[df["phase"] == phase]
    g = sub.groupby(param)["detected"].agg(["mean", "size"])
    detail = "  ".join(f"{i:g}:{r['mean']:.0%}" for i, r in g.iterrows())
    print(f"  {param:14s} {detail}")
