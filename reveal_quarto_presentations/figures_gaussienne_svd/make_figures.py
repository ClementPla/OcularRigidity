# -*- coding: utf-8 -*-
"""
Figures pour la presentation reveal/quarto -- SVD d'une gaussienne qui se deplace.

Genere des PNG INDEPENDANTS (un par panneau) a fond transparent, pensés pour le
theme `dark` de `SANS_meca_predictors_REPORT.qmd` : la mise en page en lignes /
colonnes est faite cote Quarto, pas ici. Le fichier `slides.qmd` ecrit a cote
contient les blocs prets a coller.

Simulation (identique a la section 2 de `notebook/understanding_SVD.ipynb`) :
une gaussienne 2-D qui oscille verticalement, observee comme une pile d'images.
`A` a une ligne par pixel et une colonne par image, donc `U` porte des cartes
spatiales et `V` des formes d'onde.

Trois parties : (1) frequence fixe, effet de chaque bruit isole puis des deux ;
(2) chirp 1.0 -> 1.3 Hz, sans bruit puis avec les deux ; (3) frequence fixe mais
amplitude modulee par paliers (8 -> 3 -> 2 px, 10 periodes chacun), sans bruit
puis avec les deux.

Lancer :  python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Parametres de simulation
# --------------------------------------------------------------------------- #
TAILLE = 50               # image TAILLE x TAILLE pixels
SIGMA = 7.0               # ecart-type de la gaussienne (px)
# Amplitude TOTALE, crete a crete : le deplacement va de -1.5 a +1.5 px.
# (meme convention que le notebook ; mettre 6.0 pour une demi-amplitude de 3 px)
AMPLITUDE_TOTALE = 3.0
BRUIT_DEPL = 1.0          # ecart-type du bruit de deplacement (px)
F0 = 1.0                  # Hz
FS = 15.0                 # Hz
N_OSC = 10                # -> 150 images, 10 s

# Bruit d'image, facon OCT : fond + speckle multiplicatif + plancher additif.
FOND = 0.15
N_LOOKS = 4               # contraste de speckle = 1 / sqrt(N_LOOKS) = 0.50
BRUIT_ADDITIF = 0.02
GRAINE_IMAGE = 1
GRAINE_DEPL = 0

# Partie 2 : chirp
F_DEB, F_FIN = 1.0, 1.3   # Hz
PHI0 = np.pi / 3          # phase initiale

# Partie 3 : modulation d'amplitude. Frequence FIXE a F0, mais l'amplitude
# change par paliers, sur trois portions de duree egale.
AMPLITUDES_P3 = (8.0, 3.0, 2.0)   # px crete a crete, un palier par tiers
# 10 periodes PAR PALIER, donc 30 au total (30 s a 1 Hz, 450 images) -- chaque
# palier est a lui seul aussi long que toute la partie 1, et porte donc autant
# d'information qu'elle. Le total etant un multiple de 3, chaque palier contient
# un nombre ENTIER de cycles : les changements d'amplitude tombent sur un zero
# du sinus, donc le deplacement reste CONTINU -- seule sa pente saute. Avec un
# total non multiple de 3 les paliers changeraient en plein milieu d'une
# oscillation, et le saut de valeur qui en resulterait etalerait un artefact
# large bande dans tous les spectres, qu'on prendrait pour un effet de la
# modulation.
N_OSC_PAR_PALIER_P3 = 10
N_OSC_P3 = 3 * N_OSC_PAR_PALIER_P3

N_VECTEURS = 5            # U0..U4 et V0..V4

# --------------------------------------------------------------------------- #
# Style : fond transparent, annotations lisibles sur le theme dark de reveal
# --------------------------------------------------------------------------- #
COULEUR_TEXTE = "#c9c9c9"     # passer a "#3a3a3a" pour un fond clair
C_SIGNAL = "#2a78d6"          # bleu : meme palette que les figures existantes
C_BRUIT = "#eb6834"           # orange
C_REF = "#9a9a9a"

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

SORTIE = Path(__file__).parent
FICHIERS: list[str] = []


def enregistrer(fig, nom):
    chemin = SORTIE / nom
    fig.savefig(chemin)
    plt.close(fig)
    FICHIERS.append(nom)
    return chemin


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
T = int(round(N_OSC * FS / F0))
t = np.arange(T) / FS
DUREE = T / FS
yy, xx = np.mgrid[0:TAILLE, 0:TAILLE]
CENTRE = (TAILLE - 1) / 2
DEMI = AMPLITUDE_TOTALE / 2


def gaussiennes(deplacement):
    """Cube (T, TAILLE, TAILLE) : gaussienne translatee verticalement."""
    return np.exp(
        -(
            ((yy[None] - CENTRE - deplacement[:, None, None]) ** 2
             + (xx[None] - CENTRE) ** 2)
            / (2 * SIGMA**2)
        )
    )


def bruiter_image(cube, graine=GRAINE_IMAGE):
    """Fond + speckle multiplicatif (gamma, moyenne 1) + plancher additif."""
    rng = np.random.default_rng(graine)
    R = cube + FOND
    return R * rng.gamma(N_LOOKS, 1.0 / N_LOOKS, R.shape) + rng.normal(
        0.0, BRUIT_ADDITIF, R.shape
    )


def matrice(cube):
    """Une ligne par pixel, une colonne par image."""
    return cube.reshape(len(cube), -1).T


def aligner_signes(U, Vt, U_ref):
    """Fixe le signe de chaque couple (u_k, v_k) sur une condition de reference.

    Le signe des vecteurs singuliers est arbitraire : sans cela une meme carte
    pourrait apparaitre en bleu dans une colonne et en rouge dans la suivante,
    ce qui rendrait la comparaison illisible.
    """
    U, Vt = U.copy(), Vt.copy()
    for k in range(min(U.shape[1], U_ref.shape[1])):
        if float(U[:, k] @ U_ref[:, k]) < 0:
            U[:, k] *= -1
            Vt[k] *= -1
    return U, Vt


# --- deplacements ---------------------------------------------------------- #
rng_depl = np.random.default_rng(GRAINE_DEPL)
bruit_depl = rng_depl.normal(0.0, BRUIT_DEPL, T)

dy_net = DEMI * np.sin(2 * np.pi * F0 * t)
dy_bruite = dy_net + bruit_depl

# --- chirp (partie 2) ------------------------------------------------------- #
f_chirp = F_DEB + (F_FIN - F_DEB) * t / DUREE
phi_chirp = 2 * np.pi * (F_DEB * t + (F_FIN - F_DEB) * t**2 / (2 * DUREE)) + PHI0
dy_chirp = DEMI * np.sin(phi_chirp)
dy_chirp_bruite = dy_chirp + bruit_depl

# --- les quatre conditions de la partie 1 ---------------------------------- #
cubes = {
    "net": gaussiennes(dy_net),
    "bruit_depl": gaussiennes(dy_bruite),
    "bruit_image": bruiter_image(gaussiennes(dy_net)),
    "deux_bruits": bruiter_image(gaussiennes(dy_bruite)),
}
deplacements = {
    "net": dy_net,
    "bruit_depl": dy_bruite,
    "bruit_image": dy_net,
    "deux_bruits": dy_bruite,
}
CONDITIONS_P1 = ["net", "bruit_depl", "bruit_image", "deux_bruits"]
TITRES_P1 = {
    "net": "sans bruit",
    "bruit_depl": "bruit dans le deplacement",
    "bruit_image": "bruit dans l'image",
    "deux_bruits": "les deux bruits",
}

cubes_p2 = {
    "net": gaussiennes(dy_chirp),
    "bruite": bruiter_image(gaussiennes(dy_chirp_bruite)),
}
deplacements_p2 = {"net": dy_chirp, "bruite": dy_chirp_bruite}
CONDITIONS_P2 = ["net", "bruite"]

# --- modulation d'amplitude (partie 3) -------------------------------------- #
# Base de temps PROPRE a la partie 3 : elle est trois fois plus longue que les
# deux premieres (30 s au lieu de 10), il lui faut donc son propre `t`, son
# propre tirage de bruit de deplacement et sa propre grille de frequences.
T3 = int(round(N_OSC_P3 * FS / F0))
t3 = np.arange(T3) / FS
DUREE3 = T3 / FS
freqs3 = np.fft.rfftfreq(T3, 1 / FS)

# Bornes des trois paliers, en images. T3 est un multiple de 3 par construction
# (N_OSC_P3 est un multiple de 3, FS / F0 = 15), la division est donc exacte.
TIERS = T3 // 3
BORNES_P3 = [(0, TIERS), (TIERS, 2 * TIERS), (2 * TIERS, T3)]
SEPARATIONS_P3 = (TIERS / FS, 2 * TIERS / FS)  # en secondes, pour les figures

# Enveloppe en escalier : demi-amplitude, une valeur par image.
enveloppe_p3 = np.repeat(np.array(AMPLITUDES_P3) / 2.0, TIERS)
dy_am = enveloppe_p3 * np.sin(2 * np.pi * F0 * t3)

rng_depl3 = np.random.default_rng(GRAINE_DEPL)
dy_am_bruite = dy_am + rng_depl3.normal(0.0, BRUIT_DEPL, T3)

cubes_p3 = {
    "net": gaussiennes(dy_am),
    "bruite": bruiter_image(gaussiennes(dy_am_bruite)),
}
deplacements_p3 = {"net": dy_am, "bruite": dy_am_bruite}
CONDITIONS_P3 = ["net", "bruite"]

# --- SVD ------------------------------------------------------------------- #
svd = {}
U_ref = None
for cond in CONDITIONS_P1:
    U, S, Vt = np.linalg.svd(matrice(cubes[cond]), full_matrices=False)
    if U_ref is None:
        U_ref = U
    U, Vt = aligner_signes(U, Vt, U_ref)
    svd[cond] = (U, S, Vt)

svd_p2 = {}
U_ref2 = None
for cond in CONDITIONS_P2:
    U, S, Vt = np.linalg.svd(matrice(cubes_p2[cond]), full_matrices=False)
    if U_ref2 is None:
        U_ref2 = U
    U, Vt = aligner_signes(U, Vt, U_ref2)
    svd_p2[cond] = (U, S, Vt)

svd_p3 = {}
U_ref3 = None
for cond in CONDITIONS_P3:
    U, S, Vt = np.linalg.svd(matrice(cubes_p3[cond]), full_matrices=False)
    if U_ref3 is None:
        U_ref3 = U
    U, Vt = aligner_signes(U, Vt, U_ref3)
    svd_p3[cond] = (U, S, Vt)

freqs = np.fft.rfftfreq(T, 1 / FS)


# --------------------------------------------------------------------------- #
# Briques graphiques
# --------------------------------------------------------------------------- #
def figure_image(img, nom, vmin=None, vmax=None):
    """Une image de la pile, sans axes."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    if vmin is None:
        vmin, vmax = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for cote in ax.spines.values():
        cote.set_visible(False)
    return enregistrer(fig, nom)


def separateurs(ax, instants):
    """Traits verticaux aux changements de palier (partie 3)."""
    for instant in instants:
        ax.axvline(instant, color=COULEUR_TEXTE, alpha=0.35, lw=0.8, ls="--")


def figure_deplacement(signal, nom, ylim, couleur, n_osc_affichees=5, temps=None,
                       separations=(), figsize=(4.4, 2.2)):
    """Le deplacement impose, en pixels."""
    temps = t if temps is None else temps
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(temps, signal, lw=1.2, color=couleur)
    ax.axhline(0, color=C_REF, lw=0.6, ls=":")
    separateurs(ax, separations)
    ax.set_xlim(0, n_osc_affichees / F0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("deplacement (px)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return enregistrer(fig, nom)


def figure_carte(u, nom, vmax=None):
    """Un vecteur singulier spatial, remis en image.

    Le centre de `RdBu_r` est blanc : tel quel, chaque carte serait un pave
    clair sur un theme reveal sombre. On rend donc l'opacite proportionnelle a
    |u|, ce qui laisse le fond du slide traverser les zones nulles et marche
    aussi bien sur fond clair.

    Chaque carte est normalisee par son PROPRE 99.5e centile. Une echelle
    commune a toute une ligne serait fixee par les conditions bruitees, dont
    quelques pixels isoles montent tres haut, et les cartes propres
    deviendraient invisibles. Les u_k etant tous de norme 1, c'est de toute
    facon la FORME qui se compare d'une colonne a l'autre, pas l'amplitude --
    les amplitudes relatives sont dans `recapitulatif.txt`.
    """
    carte = u.reshape(TAILLE, TAILLE)
    vmax = np.percentile(np.abs(carte), 99.5) if vmax is None else vmax
    opacite = np.clip(np.abs(carte) / vmax, 0.0, 1.0) ** 0.5
    fig, ax = plt.subplots(figsize=(2.1, 2.1))
    ax.imshow(carte, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=opacite)
    ax.set_xticks([]); ax.set_yticks([])
    for cote in ax.spines.values():
        cote.set_visible(True)
        cote.set_color(COULEUR_TEXTE)
        cote.set_alpha(0.35)
        cote.set_linewidth(0.6)
    return enregistrer(fig, nom)


def figure_courbe(v, nom, ylim, n_osc_affichees=5, temps=None, separations=(),
                  figsize=(3.6, 1.5)):
    """Un vecteur singulier temporel."""
    temps = t if temps is None else temps
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(temps, v, lw=1.0, color=C_SIGNAL)
    ax.axhline(0, color=C_REF, lw=0.6, ls=":")
    separateurs(ax, separations)
    ax.set_xlim(0, n_osc_affichees / F0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("temps (s)")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return enregistrer(fig, nom)


def figure_spectre(v, nom, f_max=7.5, freqs_axe=None):
    """Spectre de puissance d'un vecteur temporel, normalise par son maximum."""
    freqs_axe = freqs if freqs_axe is None else freqs_axe
    p = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)))) ** 2
    fig, ax = plt.subplots(figsize=(3.6, 1.6))
    # Reperes aux harmoniques de f0 : c'est sur eux que tombent les pics quand
    # le signal est propre, et leur absence saute aux yeux quand il ne l'est pas.
    for h in range(1, int(f_max / F0) + 1):
        ax.axvline(h * F0, color=COULEUR_TEXTE, alpha=0.22, lw=0.7, ls=":")
    ax.semilogy(freqs_axe, p / p.max(), lw=1.0, color=C_SIGNAL)
    ax.set_xlim(0, f_max)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("frequence (Hz)")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return enregistrer(fig, nom)


def remise_a_echelle(v1, reference):
    """`v1` (norme 1, signe arbitraire) ramene en pixels sur `reference`.

    Le facteur est un ecart-type global : c'est donc la FORME de la modulation
    qui se lit, pas une mesure d'amplitude absolue -- l'echelle est calee une
    fois pour tout l'enregistrement, pas palier par palier.
    """
    signe = np.sign(np.corrcoef(v1, reference)[0, 1]) or 1.0
    return signe * v1 / v1.std() * reference.std()


def figure_superposition(sinus, depl, v1, nom, n_osc_affichees=5, temps=None,
                         label_ref="sinusoide", enveloppe=None, separations=(),
                         figsize=(5.2, 2.6), xlim=None):
    """Signal de reference, deplacement impose et v1 remis a l'echelle, superposes."""
    temps = t if temps is None else temps
    v1_px = remise_a_echelle(v1, depl)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(temps, depl, lw=1.0, color=C_BRUIT, label="deplacement impose")
    if not np.allclose(depl, sinus):
        ax.plot(temps, sinus, lw=1.8, color=C_REF, label=label_ref)
    else:
        ax.plot([], [], lw=1.8, color=C_REF, label=f"{label_ref} (= deplacement)")
    ax.plot(temps, v1_px, lw=1.2, ls="--", color=C_SIGNAL, label="$v_1$ remis a l'echelle")
    if enveloppe is not None:
        ax.plot(temps, enveloppe, lw=0.9, ls=":", color=C_REF, label="enveloppe imposee")
        ax.plot(temps, -enveloppe, lw=0.9, ls=":", color=C_REF)
    separateurs(ax, separations)
    ax.set_xlim(*(xlim if xlim is not None else (0, n_osc_affichees / F0)))
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("deplacement (px)")
    # Legende au-dessus du cadre : a l'interieur elle recouvrait les courbes.
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return enregistrer(fig, nom)


# --------------------------------------------------------------------------- #
# PARTIE 1 -- effet du bruit
# --------------------------------------------------------------------------- #
# Slide 1 : methodes. Colonne 1 = bruit dans l'image, colonne 2 = bruit dans le
# deplacement. Les deux images partagent la meme echelle de couleur, les deux
# signaux les memes bornes : sinon la comparaison serait faussee.
img_nette = cubes["net"][0]
img_bruitee = cubes["bruit_image"][0]
v_lo, v_hi = np.percentile(np.concatenate([img_nette.ravel(), img_bruitee.ravel()]),
                           [0.5, 99.5])
figure_image(img_nette, "p1_s1_image_sans_bruit.png", v_lo, v_hi)
figure_image(img_bruitee, "p1_s1_image_avec_bruit.png", v_lo, v_hi)

marge = 1.15 * max(np.abs(dy_net).max(), np.abs(dy_bruite).max())
figure_deplacement(dy_net, "p1_s1_deplacement_sans_bruit.png", (-marge, marge), C_SIGNAL)
figure_deplacement(dy_bruite, "p1_s1_deplacement_avec_bruit.png", (-marge, marge), C_BRUIT)

# Slides 2, 3, 4 : une image par (vecteur, condition).
for k in range(N_VECTEURS):
    ylim_v = 1.15 * max(np.abs(svd[c][2][k]).max() for c in CONDITIONS_P1)
    for cond in CONDITIONS_P1:
        U, S, Vt = svd[cond]
        figure_carte(U[:, k], f"p1_s2_U{k}_{cond}.png")
        figure_courbe(Vt[k], f"p1_s3_V{k}_{cond}.png", (-ylim_v, ylim_v))
        figure_spectre(Vt[k], f"p1_s4_spectre_V{k}_{cond}.png")

# Slide 5 : superposition sinus / deplacement / v1.
for cond in CONDITIONS_P1:
    figure_superposition(dy_net, deplacements[cond], svd[cond][2][1],
                         f"p1_s5_superposition_{cond}.png")

# --------------------------------------------------------------------------- #
# PARTIE 2 -- chirp f0 -> 1.3 f0
# --------------------------------------------------------------------------- #
img_p2_net = cubes_p2["net"][0]
img_p2_bruite = cubes_p2["bruite"][0]
v_lo2, v_hi2 = np.percentile(
    np.concatenate([img_p2_net.ravel(), img_p2_bruite.ravel()]), [0.5, 99.5]
)
figure_image(img_p2_net, "p2_s1_image_net.png", v_lo2, v_hi2)
figure_image(img_p2_bruite, "p2_s1_image_bruite.png", v_lo2, v_hi2)

marge2 = 1.15 * max(np.abs(dy_chirp).max(), np.abs(dy_chirp_bruite).max())
figure_deplacement(dy_chirp, "p2_s1_signal_net.png", (-marge2, marge2), C_SIGNAL,
                   n_osc_affichees=N_OSC)
figure_deplacement(dy_chirp_bruite, "p2_s1_signal_bruite.png", (-marge2, marge2),
                   C_BRUIT, n_osc_affichees=N_OSC)

for k in range(N_VECTEURS):
    ylim_v = 1.15 * max(np.abs(svd_p2[c][2][k]).max() for c in CONDITIONS_P2)
    for cond in CONDITIONS_P2:
        U, S, Vt = svd_p2[cond]
        figure_carte(U[:, k], f"p2_s2_U{k}_{cond}.png")
        figure_courbe(Vt[k], f"p2_s3_V{k}_{cond}.png", (-ylim_v, ylim_v),
                      n_osc_affichees=N_OSC)
        figure_spectre(Vt[k], f"p2_s4_spectre_V{k}_{cond}.png")

# --------------------------------------------------------------------------- #
# PARTIE 3 -- modulation d'amplitude (8 -> 3 -> 2 px, trois tiers de la video)
# --------------------------------------------------------------------------- #
img_p3_net = cubes_p3["net"][0]
img_p3_bruite = cubes_p3["bruite"][0]
v_lo3, v_hi3 = np.percentile(
    np.concatenate([img_p3_net.ravel(), img_p3_bruite.ravel()]), [0.5, 99.5]
)
figure_image(img_p3_net, "p3_s1_image_net.png", v_lo3, v_hi3)
figure_image(img_p3_bruite, "p3_s1_image_bruite.png", v_lo3, v_hi3)

# L'enregistrement dure trois fois plus longtemps que ceux des parties 1 et 2 :
# aux proportions de celles-ci, les 30 oscillations formeraient une bande pleine
# ou l'on ne distinguerait plus rien. Les panneaux temporels de la partie 3 sont
# donc PLUS ALLONGES -- la largeur a l'ecran etant fixee par la mise en page, ce
# sont des bandeaux plus bas, et le PNG lui-meme garde assez de pixels par cycle
# pour rester lisible en zoomant. La superposition, elle, passe sur toute la
# largeur (une par ligne) plutot qu'en deux colonnes.
FIGSIZE_DEPL_P3 = (7.0, 2.0)
FIGSIZE_COURBE_P3 = (6.4, 1.4)
FIGSIZE_SUPERPOSITION_P3 = (10.0, 2.6)

marge3 = 1.15 * max(np.abs(dy_am).max(), np.abs(dy_am_bruite).max())
figure_deplacement(dy_am, "p3_s1_signal_net.png", (-marge3, marge3), C_SIGNAL,
                   n_osc_affichees=N_OSC_P3, temps=t3, separations=SEPARATIONS_P3,
                   figsize=FIGSIZE_DEPL_P3)
figure_deplacement(dy_am_bruite, "p3_s1_signal_bruite.png", (-marge3, marge3),
                   C_BRUIT, n_osc_affichees=N_OSC_P3, temps=t3,
                   separations=SEPARATIONS_P3, figsize=FIGSIZE_DEPL_P3)

for k in range(N_VECTEURS):
    ylim_v = 1.15 * max(np.abs(svd_p3[c][2][k]).max() for c in CONDITIONS_P3)
    for cond in CONDITIONS_P3:
        U, S, Vt = svd_p3[cond]
        figure_carte(U[:, k], f"p3_s2_U{k}_{cond}.png")
        figure_courbe(Vt[k], f"p3_s3_V{k}_{cond}.png", (-ylim_v, ylim_v),
                      n_osc_affichees=N_OSC_P3, temps=t3,
                      separations=SEPARATIONS_P3, figsize=FIGSIZE_COURBE_P3)
        figure_spectre(Vt[k], f"p3_s4_spectre_V{k}_{cond}.png", freqs_axe=freqs3)

for cond in CONDITIONS_P3:
    figure_superposition(
        dy_am, deplacements_p3[cond], svd_p3[cond][2][1],
        f"p3_s5_superposition_{cond}.png",
        n_osc_affichees=N_OSC_P3, temps=t3, label_ref="signal impose (net)",
        enveloppe=enveloppe_p3, separations=SEPARATIONS_P3,
        figsize=FIGSIZE_SUPERPOSITION_P3,
    )
    # Zoom sur le dernier palier (le plus faible) : c'est la que la mesure
    # decroche, et a 30 periodes le panneau d'ensemble ne le montre plus.
    figure_superposition(
        dy_am, deplacements_p3[cond], svd_p3[cond][2][1],
        f"p3_s6_zoom_{cond}.png",
        n_osc_affichees=N_OSC_P3, temps=t3, label_ref="signal impose (net)",
        enveloppe=enveloppe_p3, separations=SEPARATIONS_P3,
        figsize=FIGSIZE_SUPERPOSITION_P3, xlim=(2 * DUREE3 / 3, DUREE3),
    )


# --------------------------------------------------------------------------- #
# Recapitulatif chiffre, a citer sur les slides
# --------------------------------------------------------------------------- #
def concentration(v, freqs_axe=None):
    freqs_axe = freqs if freqs_axe is None else freqs_axe
    p = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)))) ** 2
    return p.max() / p.sum(), freqs_axe[p.argmax()]


def amplitudes_par_tiers(v1):
    """Amplitude crete a crete lue sur `v1`, palier par palier, en pixels.

    `v1` est d'abord ramene en pixels par un facteur UNIQUE cale sur le
    deplacement impose net (cf. `remise_a_echelle`) : les trois valeurs se
    comparent donc entre elles et a (8, 3, 2), mais elles ne constituent pas
    une mesure absolue independante. Pour une sinusoide pure, ecart-type =
    amplitude crete a crete / (2*sqrt(2)), d'ou le facteur.
    """
    v1_px = remise_a_echelle(v1, dy_am)
    return [2 * np.sqrt(2) * v1_px[a:b].std() for a, b in BORNES_P3]


lignes = ["condition          | s1/s0  | corr(v1, depl.) | corr(v1, sinus) | "
          "conc(v1) | pic v1"]
for cond in CONDITIONS_P1:
    U, S, Vt = svd[cond]
    conc, pic = concentration(Vt[1])
    lignes.append(
        f"{TITRES_P1[cond]:18s} | {S[1] / S[0]:.4f} |      {abs(np.corrcoef(Vt[1], deplacements[cond])[0, 1]):.4f}     "
        f"|     {abs(np.corrcoef(Vt[1], dy_net)[0, 1]):.4f}      |  {conc:.3f}   | {pic:.2f} Hz"
    )
lignes.append("")
lignes.append("partie 2 (chirp)   | s1/s0  | corr(v1, depl.) | corr(v1, chirp) | "
              "conc(v1) | pic v1")
for cond in CONDITIONS_P2:
    U, S, Vt = svd_p2[cond]
    conc, pic = concentration(Vt[1])
    lignes.append(
        f"{cond:18s} | {S[1] / S[0]:.4f} |      {abs(np.corrcoef(Vt[1], deplacements_p2[cond])[0, 1]):.4f}     "
        f"|     {abs(np.corrcoef(Vt[1], dy_chirp)[0, 1]):.4f}      |  {conc:.3f}   | {pic:.2f} Hz"
    )
lignes.append("")
lignes.append("partie 3 (mod. ampl.) | s1/s0  | corr(v1, depl.) | corr(v1, AM net) | "
              "conc(v1) | pic v1")
for cond in CONDITIONS_P3:
    U, S, Vt = svd_p3[cond]
    conc, pic = concentration(Vt[1], freqs3)
    lignes.append(
        f"{cond:18s} | {S[1] / S[0]:.4f} |      {abs(np.corrcoef(Vt[1], deplacements_p3[cond])[0, 1]):.4f}     "
        f"|     {abs(np.corrcoef(Vt[1], dy_am)[0, 1]):.4f}      |  {conc:.3f}   | {pic:.2f} Hz"
    )
lignes.append("")
lignes.append("partie 3 -- amplitude crete a crete lue sur v1 (px), par tiers")
lignes.append("condition          | tiers 1 (8.0) | tiers 2 (3.0) | tiers 3 (2.0) | "
              "rapports vs 8:3:2")
AMP_P3 = {cond: amplitudes_par_tiers(svd_p3[cond][2][1]) for cond in CONDITIONS_P3}
for cond in CONDITIONS_P3:
    a1, a2, a3 = AMP_P3[cond]
    lignes.append(
        f"{cond:18s} |     {a1:5.2f}     |     {a2:5.2f}     |     {a3:5.2f}     | "
        f"{a2 / a1:.3f} vs {AMPLITUDES_P3[1] / AMPLITUDES_P3[0]:.3f}, "
        f"{a3 / a1:.3f} vs {AMPLITUDES_P3[2] / AMPLITUDES_P3[0]:.3f}"
    )
recap = "\n".join(lignes)
(SORTIE / "recapitulatif.txt").write_text(recap + "\n", encoding="utf-8")
print(recap)


# --------------------------------------------------------------------------- #
# Fragment Quarto pret a coller (les noms de fichiers viennent d'ici, donc ils
# ne peuvent pas se desynchroniser des figures reellement ecrites)
# --------------------------------------------------------------------------- #
DOSSIER = SORTIE.name


def img(nom, largeur="100%"):
    assert nom in FICHIERS, f"figure non generee : {nom}"
    return f"![]({DOSSIER}/{nom}){{width={largeur}}}"


def grille(prefixe, conditions, entetes, symbole, n=N_VECTEURS):
    """Tableau markdown : une ligne par vecteur, une colonne par condition."""
    lignes = ["| | " + " | ".join(entetes) + " |",
              "|" + "---|" * (len(entetes) + 1)]
    for k in range(n):
        cellules = [img(f"{prefixe}{k}_{c}.png") for c in conditions]
        lignes.append(f"| ${symbole}_{k}$ | " + " | ".join(cellules) + " |")
    return "\n".join(lignes)


ENTETES_P1 = ["sans bruit", "bruit dans le<br>deplacement",
              "bruit dans<br>l'image", "les deux<br>bruits"]
ENTETES_P2 = ["sans bruit", "bruite"]
ENTETES_P3 = ["sans bruit", "les deux bruits"]


def tableau_amplitudes_p3():
    """Tableau markdown des amplitudes lues palier par palier."""
    entetes = [f"tiers {i + 1}<br>(impose {a:.0f} px)"
               for i, a in enumerate(AMPLITUDES_P3)]
    lignes = ["| | " + " | ".join(entetes) + " |", "|" + "---|" * 4]
    for cond, titre in zip(CONDITIONS_P3, ENTETES_P3):
        cells = " | ".join(f"{a:.2f} px" for a in AMP_P3[cond])
        lignes.append(f"| {titre} | {cells} |")
    return "\n".join(lignes)

qmd = f"""<!--
Fragment a coller dans SANS_meca_predictors_REPORT.qmd (ou a inclure).
Genere par {DOSSIER}/make_figures.py -- ne pas editer a la main, relancer le
script apres toute modification des figures.
Les chemins sont relatifs au dossier PARENT ({DOSSIER}/...), c'est-a-dire au
dossier ou vit le .qmd de la presentation.
-->

# Partie 1 -- effet du bruit

## Presentation des methodes

Gaussienne 2-D ({TAILLE}x{TAILLE} px, $\\sigma$ = {SIGMA:.0f} px) qui oscille
verticalement.

- signal a **{F0:.0f} Hz**, echantillonnage a **{FS:.0f} Hz**, {N_OSC} oscillations
- amplitude totale du deplacement : **{AMPLITUDE_TOTALE:.0f} px** crete a crete
- bruit de deplacement : **{BRUIT_DEPL:.0f} px** (ecart-type)
- bruit d'image : fond {FOND}, speckle multiplicatif de contraste
  {1 / np.sqrt(N_LOOKS):.2f}, plancher additif {BRUIT_ADDITIF}

| | bruit dans l'image | bruit dans le deplacement |
|---|---|---|
| sans bruit | {img("p1_s1_image_sans_bruit.png", "70%")} | {img("p1_s1_deplacement_sans_bruit.png")} |
| avec bruit | {img("p1_s1_image_avec_bruit.png", "70%")} | {img("p1_s1_deplacement_avec_bruit.png")} |

## $U$ -- vecteurs singuliers spatiaux

{grille("p1_s2_U", CONDITIONS_P1, ENTETES_P1, "u")}

## $V$ -- vecteurs singuliers temporels

{grille("p1_s3_V", CONDITIONS_P1, ENTETES_P1, "v")}

## $V$ -- domaine frequentiel

{grille("p1_s4_spectre_V", CONDITIONS_P1, ENTETES_P1, "v")}

::: footer
Pointilles verticaux : harmoniques de $f_0$. Echelle log, chaque spectre
normalise par son maximum.
:::

## Superposition du signal et de $v_1$

::: {{layout-ncol=2}}
{img("p1_s5_superposition_net.png")}

{img("p1_s5_superposition_bruit_depl.png")}

{img("p1_s5_superposition_bruit_image.png")}

{img("p1_s5_superposition_deux_bruits.png")}
:::

# Partie 2 -- chirp de $f_0$ a 1.3 $f_0$

## Methodologie

Meme simulation, mais la frequence derive de **{F_DEB:.1f} a {F_FIN:.1f} Hz**
sur les {N_OSC:.0f} s, avec une phase initiale de $\\pi/3$.

| | sans bruit | avec les deux bruits |
|---|---|---|
| image | {img("p2_s1_image_net.png", "70%")} | {img("p2_s1_image_bruite.png", "70%")} |
| deplacement | {img("p2_s1_signal_net.png")} | {img("p2_s1_signal_bruite.png")} |

## $U$ -- vecteurs singuliers spatiaux

{grille("p2_s2_U", CONDITIONS_P2, ENTETES_P2, "u")}

## $V$ -- vecteurs singuliers temporels

{grille("p2_s3_V", CONDITIONS_P2, ENTETES_P2, "v")}

## $V$ -- domaine frequentiel

{grille("p2_s4_spectre_V", CONDITIONS_P2, ENTETES_P2, "v")}

# Partie 3 -- modulation d'amplitude

## Methodologie

Meme gaussienne, frequence FIXE a **{F0:.0f} Hz**, mais l'amplitude change par
paliers : la video est coupee en **trois portions de duree egale**, de
**{N_OSC_PAR_PALIER_P3} periodes** chacune ({DUREE3 / 3:.0f} s), d'amplitude
crete a crete **{AMPLITUDES_P3[0]:.0f}**, **{AMPLITUDES_P3[1]:.0f}** puis
**{AMPLITUDES_P3[2]:.0f} px**. Soit {N_OSC_P3} periodes et {DUREE3:.0f} s au
total : chaque palier est a lui seul aussi long que toute la partie 1.

Le nombre de periodes par palier est entier, donc les changements d'amplitude
tombent sur un zero du sinus et le deplacement reste **continu** -- seule sa
pente saute. Sinon le saut de valeur etalerait un artefact large bande dans
tous les spectres, qu'on prendrait pour un effet de la modulation.

La condition bruitee cumule les **deux** bruits, comme la partie 2 : bruit de
deplacement {BRUIT_DEPL:.0f} px d'ecart-type + bruit d'image. Sur le dernier
palier l'amplitude imposee vaut {AMPLITUDES_P3[2]:.0f} px crete a crete, soit
+/-{AMPLITUDES_P3[2] / 2:.0f} px : le bruit de deplacement y est de l'ordre du
signal.

| | sans bruit | avec les deux bruits |
|---|---|---|
| image | {img("p3_s1_image_net.png", "70%")} | {img("p3_s1_image_bruite.png", "70%")} |
| deplacement | {img("p3_s1_signal_net.png")} | {img("p3_s1_signal_bruite.png")} |

::: {{.source-note}}
Traits verticaux tiretes : changements de palier.
:::

## $U$ -- vecteurs singuliers spatiaux

{grille("p3_s2_U", CONDITIONS_P3, ENTETES_P3, "u")}

## $V$ -- vecteurs singuliers temporels

{grille("p3_s3_V", CONDITIONS_P3, ENTETES_P3, "v")}

## $V$ -- domaine frequentiel

{grille("p3_s4_spectre_V", CONDITIONS_P3, ENTETES_P3, "v")}

## Superposition du signal et de $v_1$

Sur toute la largeur et non en deux colonnes comme la partie 1 : a
{N_OSC_P3} periodes, deux colonnes ne laisseraient plus qu'une bande pleine.

**Sans bruit**

{img("p3_s5_superposition_net.png")}

**Avec les deux bruits**

{img("p3_s5_superposition_bruite.png")}

## Zoom sur le dernier palier

Les {N_OSC_PAR_PALIER_P3} dernieres periodes, celles d'amplitude
{AMPLITUDES_P3[2]:.0f} px crete a crete -- le palier ou la mesure decroche, et
que le panneau d'ensemble ne montre plus.

**Sans bruit**

{img("p3_s6_zoom_net.png")}

**Avec les deux bruits**

{img("p3_s6_zoom_bruite.png")}

## Amplitude lue palier par palier

Amplitude crete a crete lue sur $v_1$. $v_1$ est ramene en pixels par un facteur
**unique** cale sur le deplacement impose net : les trois valeurs se comparent
entre elles et a (8, 3, 2), ce n'est pas une mesure absolue independante.

{tableau_amplitudes_p3()}
"""

(SORTIE / "slides.qmd").write_text(qmd, encoding="utf-8")

readme = f"""# Figures -- SVD d'une gaussienne qui se deplace

Materiel pour la presentation reveal/quarto. **Tout est regenere par
`python make_figures.py`** : ne pas retoucher les PNG a la main.

- `slides.qmd` : les {len(FICHIERS)} figures deja mises en page, prete a coller
  dans `SANS_meca_predictors_REPORT.qmd` (chemins relatifs au dossier parent).
- `recapitulatif.txt` : les chiffres a citer sur les slides.

## Nommage

`p<partie>_s<slide>_<contenu>_<condition>.png`

| condition | partie 1 | partie 2 | partie 3 |
|---|---|---|---|
| `net` | sans bruit | chirp sans bruit | modulation sans bruit |
| `bruit_depl` | bruit dans le deplacement | -- | -- |
| `bruit_image` | bruit dans l'image | -- | -- |
| `deux_bruits` | les deux | -- | -- |
| `bruite` | -- | chirp + les deux bruits | modulation + les deux bruits |

Contenus : `U<k>` (carte spatiale), `V<k>` (forme d'onde), `spectre_V<k>`,
`superposition`, `zoom` (partie 3, dernier palier), plus les panneaux de methodo
du slide 1.

## Choix graphiques

- fond **transparent**, texte en gris clair : pensé pour `theme: dark`. Pour un
  fond clair, passer `COULEUR_TEXTE` a `"#3a3a3a"` en tete du script.
- les cartes `U` ont une opacite proportionnelle a `|u|` (le centre blanc de
  `RdBu_r` ferait un pave clair sur fond sombre) et sont normalisees
  **individuellement** : c'est la forme qui se compare d'une colonne a l'autre,
  pas l'amplitude.
- le signe de chaque couple `(u_k, v_k)` est aligne sur la condition sans
  bruit, sinon une meme carte changerait de couleur d'une colonne a l'autre.

## Parametres

Gaussienne {TAILLE}x{TAILLE} px, sigma = {SIGMA:.0f} px, amplitude totale
{AMPLITUDE_TOTALE:.0f} px crete a crete, {F0:.0f} Hz echantillonne a {FS:.0f} Hz,
{N_OSC} oscillations. Bruit de deplacement {BRUIT_DEPL:.0f} px ; bruit d'image =
fond {FOND} + speckle de contraste {1 / np.sqrt(N_LOOKS):.2f} + plancher
{BRUIT_ADDITIF}. Partie 2 : {F_DEB:.1f} -> {F_FIN:.1f} Hz, phase initiale pi/3.
Partie 3 : {F0:.0f} Hz fixe, amplitude crete a crete
{AMPLITUDES_P3[0]:.0f} -> {AMPLITUDES_P3[1]:.0f} -> {AMPLITUDES_P3[2]:.0f} px,
{N_OSC_PAR_PALIER_P3} periodes par palier ({DUREE3 / 3:.0f} s), soit
{N_OSC_P3} periodes et {DUREE3:.0f} s au total (nombre entier de periodes par
palier, donc les paliers changent a un zero du sinus : deplacement continu).

```
{recap}
```
"""
(SORTIE / "README.md").write_text(readme, encoding="utf-8")

print(f"\n{len(FICHIERS)} figures + slides.qmd + README.md + recapitulatif.txt")
print(f"dans {SORTIE}")
