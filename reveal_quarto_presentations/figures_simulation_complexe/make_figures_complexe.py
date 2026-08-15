"""Figures du balayage combine (phase B) -- section "Simulation results" du deck.

Source : E:/SANSORI_simulation_output/svd_radial_gaussians/summary.csv (1015 runs)
         + les .npz de sweep_combined/ pour les runs individuels.

Produit :
  methodes_design.png     -- plan d'experience : grille, replicats, bruit tire, pouls impose
  amplitude_detection.png -- detection et correlation vs amplitude
  jitter_xy.png           -- jitter x vs jitter y : detection et correlation
  run_jx2_jy0.png         -- 3 um, jx=2 jy=0 : cartes spatiales + pouls reconstruit
  run_jx0_jy2.png         -- 3 um, jx=0 jy=2 : idem
  run_jx2_jy2_phase.png   -- 8 um, jx=2 jy=2 : cartes + pouls + phase et FC instantanee

Relancer (kernel pyOR) :
  C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
      reveal_quarto_presentations/figures_simulation_complexe/make_figures_complexe.py
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
NPZ_DIR = Path("E:/SANSORI_simulation_output/svd_radial_gaussians/sweep_combined")

FREQ_ERR_THRESHOLD = 0.10

COULEUR_TEXTE = "#c9c9c9"
C_GRIS = "#9a9a9a"
C_SIGNAL = "#2a78d6"
C_BRUIT = "#eb6834"
C_ROUGE = "#d1495b"
C_VERT = "#3fa34d"
C_VIOLET = "#9b5de5"

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

# --------------------------------------------------------------------------- #
df = pd.read_csv(sim.SUMMARY_CSV)
df["abs_corr"] = df["corr_combined_vs_truth"].abs()
df["detected"] = df["confidence"].isin(["high", "medium"]) & (
    df["freq_rel_err"] <= FREQ_ERR_THRESHOLD
)
B = df[df["phase"] == "B_combined"]
print(f"{len(df)} runs, phase B = {len(B)}  ({int(df['confidence'].eq('failed').sum())} failed)")

CALIB = sim.load_real_calibration()


def charger(tag):
    return np.load(NPZ_DIR / f"{tag}.npz", allow_pickle=True)


def rep_median(ampl, jx, jy):
    """Le replicat dont la correlation est la plus proche de la mediane de sa
    cellule -- evite de choisir le plus flatteur des 15."""
    s = B[(B["amplitude_um"] == ampl) & (B["jitter_x_px"] == jx)
          & (B["jitter_y_px"] == jy)].copy()
    s["ecart"] = (s["abs_corr"] - s["abs_corr"].median()).abs()
    return s.nsmallest(1, "ecart").iloc[0]


def cartes_spatiales(ax_list, d, n_show):
    """V reshape sur le masque de l'anneau ; rouge = composante retenue."""
    V = d["V"]
    uniform_time = d["uniform_time"]
    calib_partial = sim.RealCalibration(
        scale_x_um=float(d["scale_x_um"]), scale_y_um=float(d["scale_y_um"]),
        fs_hz=float(d["fs_hz"]), n_frames=len(uniform_time),
        duration_s=(len(uniform_time) - 1) / float(d["fs_hz"]),
    )
    shape = sim.crop_shape(calib_partial)
    mask2d = sim.build_fixed_mask(shape)
    selected = set(d["selected_indices"].tolist())
    vmax = np.nanpercentile(np.abs(V[:, :n_show]), 99.5)
    for k, ax in enumerate(ax_list):
        spatial = np.full(mask2d.shape, np.nan)
        spatial[mask2d] = V[:, k]
        ax.imshow(spatial, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"$v_{{{k}}}$", fontsize=11,
                     color=C_ROUGE if k in selected else COULEUR_TEXTE)
        ax.set_xticks([]); ax.set_yticks([])
    return selected


ZOOM_S = 6.0  # duree du panneau de detail


def pouls_reconstruit(ax_full, ax_zoom, d):
    """Verite vs reconstruction, toutes deux CENTREES puis mises a la meme
    echelle (la SVD ne fixe ni le niveau continu, ni le gain, ni le signe)."""
    combined = d["combined_uniform"]
    r_true = d["r_true_uniform"]
    t = d["uniform_time"]
    kept = d["kept_mask"]

    vrai = r_true - np.nanmean(r_true[kept])
    comb = combined - np.nanmean(combined[kept])
    sign = np.sign(np.corrcoef(comb[kept], vrai[kept])[0, 1]) or 1.0
    recon = sign * comb / np.nanstd(comb[kept]) * np.nanstd(vrai[kept])

    for ax in (ax_full, ax_zoom):
        ax.plot(t, vrai, color=C_GRIS, lw=1.8, label="deplacement impose (verite)")
        ax.plot(t, recon, color=C_SIGNAL, lw=1.1, ls="--", label="pouls reconstruit")
        ax.set_xlabel("temps (s)")
        ax.axhline(0, color=C_GRIS, lw=0.6, alpha=0.5)
    ax_full.set_ylabel("deplacement radial (um)")
    ax_full.set_title(f"duree complete ({t[-1]:.0f} s)", fontsize=9)
    ax_zoom.set_xlim(0, ZOOM_S)
    ax_zoom.set_title(f"zoom sur les {ZOOM_S:g} premieres secondes", fontsize=9)
    ax_zoom.legend(fontsize=8, loc="upper right")


# =========================================================================== #
# 1. Methodes : plan d'experience
# =========================================================================== #
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))

t = np.arange(CALIB.n_frames) / CALIB.fs_hz
duree = float(t[-1])
axes[0].plot(t, sim.radial_displacement(t, duree, 3.0), color=C_SIGNAL, lw=1.2)
axes[0].set_xlabel("temps (s)"); axes[0].set_ylabel("deplacement radial (um)")
axes[0].set_title(f"Pouls impose : chirp, {CALIB.n_frames} frames, {duree:.1f} s", fontsize=10)
axes[0].grid(alpha=0.25, color=C_GRIS)

f_true = sim.chirp_freq(t, duree) * 60.0
axes[1].plot(t, f_true, color=C_VERT, lw=1.6)
axes[1].set_xlabel("temps (s)"); axes[1].set_ylabel("BPM")
axes[1].set_title("Frequence instantanee imposee", fontsize=10)
axes[1].grid(alpha=0.25, color=C_GRIS)

axes[2].hist(B["noise_level"], bins=28, color=C_BRUIT, alpha=0.75)
axes[2].axvline(1.0, color=COULEUR_TEXTE, lw=1.5, label="niveau realiste = 1")
axes[2].set_xlabel("noise_level tire par run")
axes[2].set_ylabel("nombre de runs")
axes[2].set_title("Bruit d'image : variable de nuisance, tiree au sort", fontsize=10)
axes[2].legend(fontsize=8)

fig.suptitle(
    f"Plan d'experience -- {len(B)} runs = 5 amplitudes x 3 jitter_x x 3 jitter_y "
    f"x 15 replicats", fontsize=11, color=COULEUR_TEXTE)
fig.tight_layout()
fig.savefig(SORTIE / "methodes_design.png")
plt.close(fig)
print("ecrit : methodes_design.png")

# =========================================================================== #
# 2. Amplitude
# =========================================================================== #
g = B.groupby("amplitude_um").agg(
    n=("detected", "size"), det=("detected", "mean"),
    corr=("abs_corr", "median"),
    corr_q1=("abs_corr", lambda s: s.quantile(0.25)),
    corr_q3=("abs_corr", lambda s: s.quantile(0.75)),
)
x = g.index.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].bar(np.arange(len(x)), g["det"] * 100, color=C_SIGNAL, alpha=0.85, width=0.62)
for i, v in enumerate(g["det"] * 100):
    axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9, color=COULEUR_TEXTE)
axes[0].set_xticks(np.arange(len(x)))
axes[0].set_xticklabels([f"{v:g}" for v in x])
axes[0].set_ylim(0, 112)
axes[0].set_xlabel("amplitude radiale (um crete a crete)")
axes[0].set_ylabel("taux de detection (%)")
axes[0].set_title("Detection vs amplitude", fontsize=10)

axes[1].plot(x, g["corr"], "o-", color=C_BRUIT, lw=1.8, ms=6)
axes[1].fill_between(x, g["corr_q1"], g["corr_q3"], color=C_BRUIT, alpha=0.2)
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("amplitude radiale (um crete a crete)")
axes[1].set_ylabel("|corr(reconstruit, verite)|")
axes[1].set_title("Correlation vs amplitude (mediane, IQR)", fontsize=10)
axes[1].grid(alpha=0.25, color=C_GRIS)

fig.suptitle("Amplitude : n = 135 runs par niveau, tous jitters et bruits confondus",
             fontsize=11, color=COULEUR_TEXTE)
fig.tight_layout()
fig.savefig(SORTIE / "amplitude_detection.png")
plt.close(fig)
print("ecrit : amplitude_detection.png")

print("\n=== TABLE AMPLITUDE (slide 2) ===")
for a, r in g.iterrows():
    print(f"  {a:g} um | n={int(r['n'])} | detection {r['det']:.0%} | "
          f"corr {r['corr']:.2f} [{r['corr_q1']:.2f}, {r['corr_q3']:.2f}]")

# =========================================================================== #
# 3. jitter x vs jitter y
# =========================================================================== #
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
for col, (nom, autre, couleur) in enumerate(
    [("jitter_x_px", "jitter_y_px", C_SIGNAL), ("jitter_y_px", "jitter_x_px", C_ROUGE)]
):
    sub = B[B[autre] == 0.0]
    g2 = sub.groupby(nom).agg(det=("detected", "mean"), corr=("abs_corr", "median"),
                              n=("detected", "size"))
    axes[0].plot(g2.index, g2["det"] * 100, "o-", color=couleur, lw=2.0, ms=7,
                 label=f"{nom.replace('_px','')} (autre = 0)")
    axes[1].plot(g2.index, g2["corr"], "o-", color=couleur, lw=2.0, ms=7,
                 label=f"{nom.replace('_px','')} (autre = 0)")
    print(f"\n=== {nom} (l'autre a 0) ===")
    print(g2.round(3).to_string())

axes[0].set_ylim(0, 105); axes[0].set_ylabel("taux de detection (%)")
axes[0].set_title("Detection", fontsize=10)
axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("|corr(reconstruit, verite)|")
axes[1].set_title("Correlation avec la verite", fontsize=10)
for ax in axes:
    ax.set_xlabel("sigma du jitter (px)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, color=C_GRIS)
fig.suptitle("Jitter x (commun a l'image) vs jitter y (par A-scan) -- n = 75 par point",
             fontsize=11, color=COULEUR_TEXTE)
fig.tight_layout()
fig.savefig(SORTIE / "jitter_xy.png")
plt.close(fig)
print("ecrit : jitter_xy.png")

# =========================================================================== #
# 4 & 5. Runs individuels : jx seul, puis jy seul
# =========================================================================== #
N_SHOW = 6
for ampl, jx, jy, nom_fichier in [
    (3.0, 2.0, 0.0, "run_jx2_jy0.png"),
    (3.0, 0.0, 2.0, "run_jx0_jy2.png"),
]:
    r = rep_median(ampl, jx, jy)
    d = charger(r["tag"])
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, N_SHOW, height_ratios=[1.25, 1.0], hspace=0.34)
    axes_v = [fig.add_subplot(gs[0, k]) for k in range(N_SHOW)]
    sel = cartes_spatiales(axes_v, d, N_SHOW)
    ax_full = fig.add_subplot(gs[1, :3])
    ax_zoom = fig.add_subplot(gs[1, 3:])
    pouls_reconstruit(ax_full, ax_zoom, d)
    fig.suptitle(
        f"amplitude {ampl:g} um, jitter_x = {jx:g} px, jitter_y = {jy:g} px, "
        f"bruit {float(d['noise_level']):.2f}   |   "
        f"corr = {abs(float(d['corr_combined_vs_truth'])):.3f}, "
        f"confiance = {str(d['rate_confidence'])}   |   "
        f"composantes retenues : {sorted(sel)} (rouge si affichee)",
        fontsize=11, color=COULEUR_TEXTE,
    )
    fig.savefig(SORTIE / nom_fichier)
    plt.close(fig)
    print(f"ecrit : {nom_fichier}  (tag {r['tag']}, composantes retenues {sorted(sel)})")

# =========================================================================== #
# 6. 8 um, jx=2 jy=2 : cartes + pouls + phase et FC instantanee
# =========================================================================== #
r = rep_median(8.0, 2.0, 2.0)
d = charger(r["tag"])
t = d["uniform_time"]
good = d["good_uniform"]

fig = plt.figure(figsize=(15, 8.8))
gs = fig.add_gridspec(3, N_SHOW, height_ratios=[1.2, 0.95, 0.95], hspace=0.5)
axes_v = [fig.add_subplot(gs[0, k]) for k in range(N_SHOW)]
sel = cartes_spatiales(axes_v, d, N_SHOW)

ax_full = fig.add_subplot(gs[1, :3])
ax_zoom = fig.add_subplot(gs[1, 3:])
pouls_reconstruit(ax_full, ax_zoom, d)

ax_ph = fig.add_subplot(gs[2, :3])
true_phase = sim.chirp_phase(t, float(t[-1])) % (2 * np.pi)
ax_ph.plot(t, d["phase_uniform"], color=C_VERT, lw=1.4, label="phase mesuree (Hilbert)")
ax_ph.plot(t, true_phase, color=C_GRIS, lw=1.6, ls=":", label="phase vraie (chirp)")
ax_ph.fill_between(t, 0, 2 * np.pi, where=~good, color=C_GRIS, alpha=0.22, step="mid")
ax_ph.set_xlim(0, ZOOM_S)
ax_ph.set_ylim(0, 2 * np.pi * 1.28)
ax_ph.set_xlabel("temps (s)"); ax_ph.set_ylabel("phase (rad, mod 2$\\pi$)")
ax_ph.set_title(f"zoom {ZOOM_S:g} s -- decalage constant de "
                f"{np.degrees(np.median(np.angle(np.exp(1j * (d['phase_uniform'] - true_phase)))[good])):+.0f} deg",
                fontsize=9)
ax_ph.legend(fontsize=8, ncol=2, loc="upper center")

ax_f = fig.add_subplot(gs[2, 3:])
f_true_bpm = d["f_true_uniform"] * 60.0
ax_f.plot(t, d["inst_bpm"], color=C_VIOLET, lw=1.1, label="FC instantanee mesuree")
ax_f.plot(t, f_true_bpm, color=C_GRIS, lw=1.6, ls=":", label="FC instantanee vraie")
ax_f.fill_between(t, np.nanmin(f_true_bpm), np.nanmax(f_true_bpm), where=~good,
                  color=C_GRIS, alpha=0.22, step="mid")
ax_f.set_ylim(0.7 * np.nanmin(f_true_bpm), 1.45 * np.nanmax(f_true_bpm))
ax_f.set_xlabel("temps (s)"); ax_f.set_ylabel("BPM instantane")
ax_f.set_title("duree complete -- noter les artefacts de bord (Hilbert)", fontsize=9)
ax_f.legend(fontsize=8, ncol=2, loc="upper center")

fig.suptitle(
    f"amplitude 8 um, jitter_x = 2 px, jitter_y = 2 px, bruit {float(d['noise_level']):.2f}"
    f"   |   corr = {abs(float(d['corr_combined_vs_truth'])):.3f}, "
    f"{len(sel)} composantes retenues   |   zones grisees = phase jugee non fiable",
    fontsize=11, color=COULEUR_TEXTE,
)
fig.savefig(SORTIE / "run_jx2_jy2_phase.png")
plt.close(fig)
print(f"ecrit : run_jx2_jy2_phase.png  (tag {r['tag']}, composantes {sorted(sel)})")

# --- mesure du dephasage, pour pouvoir l'affirmer ou non sur la slide ------
comb = d["combined_uniform"]
r_true = d["r_true_uniform"]
kept = d["kept_mask"]
c = comb[kept] - np.nanmean(comb[kept])
v = r_true[kept] - np.nanmean(r_true[kept])
c /= np.nanstd(c); v /= np.nanstd(v)
n = len(c)
lags = np.arange(-n + 1, n)
xc = np.correlate(c, v, mode="full") / n
lag_best = lags[np.argmax(np.abs(xc))]
dt = float(np.median(np.diff(t)))
periode = 1.0 / float(d["rate_freq"])
print(f"\n=== DEPHASAGE (8 um, jx=2, jy=2) ===")
print(f"  decalage au max de correlation : {lag_best} echantillons = "
      f"{lag_best * dt * 1000:+.0f} ms = {360 * lag_best * dt / periode:+.0f} deg")
print(f"  correlation a decalage nul  : {xc[n - 1]:+.3f}")
print(f"  correlation au max          : {xc[np.argmax(np.abs(xc))]:+.3f}")
ecart_phase = np.angle(np.exp(1j * (d["phase_uniform"] - true_phase)))
print(f"  ecart de phase Hilbert median (points fiables) : "
      f"{np.degrees(np.median(ecart_phase[good])):+.0f} deg")
