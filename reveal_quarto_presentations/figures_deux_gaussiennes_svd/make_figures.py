# -*- coding: utf-8 -*-
"""
Figures -- SVD de DEUX gaussiennes qui oscillent ensemble.

Suite de `figures_gaussienne_svd/make_figures.py` (une seule gaussienne). Meme
style graphique, memes conventions, meme fragment `slides.qmd` regenere a cote ;
seul l'objet simule change.

Trois experiences, toutes SANS BRUIT, toutes a la meme frequence et EN PHASE.
Dans les trois, les deux gaussiennes sont identiques sauf sur UN point, celui
qu'on teste :

  e1  amplitudes differentes (8 px contre 2 px), tout le reste egal
  e2  INTENSITES differentes (rapport 2), meme amplitude de mouvement
  e3  ECART-TYPE deux fois plus petit a droite, meme intensite, meme amplitude

Ce que l'ensemble cherche a montrer :

  - Deux oscillateurs synchrones ne donnent PAS deux composantes. Le premier
    ordre du deplacement s'ecrit A(x) * s(t) * dI/dy avec un SEUL s(t) commun,
    donc il est de rang 1 : les deux gaussiennes partagent u_1/v_1, ponderees
    par leur contribution respective.
  - Cette ponderation n'est PAS le rapport des amplitudes. e1 seul le
    laisserait croire ; e2 montre qu'une gaussienne deux fois plus brillante
    pese deux fois plus a mouvement IDENTIQUE, et e3 qu'un objet deux fois plus
    fin pese autant qu'un large (la norme de dI/dy ne depend pas de sigma).
    Ce qui est pondere, c'est amplitude x intensite -- pas l'amplitude seule.
  - Les composantes suivantes (k >= 2) sont les HARMONIQUES de s(t), et elles
    appartiennent a l'oscillateur le plus non lineaire, c'est-a-dire celui dont
    le deplacement est le plus grand DEVANT SON PROPRE sigma.

D'ou les 20 vecteurs affiches : c'est toute cette echelle qu'on veut voir.

Lancer :  python make_figures.py
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, firwin, hilbert

# --------------------------------------------------------------------------- #
# Geometrie commune aux trois experiences
# --------------------------------------------------------------------------- #
HAUTEUR = 60              # px
LARGEUR = 120             # px : deux gaussiennes cote a cote
SIGMA = 7.0               # ecart-type de reference (px)

# Colonnes des deux centres. 60 px d'ecart = 8.6 sigma : le recouvrement vaut
# exp(-(30/7)^2/2) ~ 1e-4 du pic, donc les deux moities de l'image sont
# separables sans ambiguite -- c'est ce qui autorise a lire une carte u_k
# moitie par moitie (cf. `part_gauche`).
X_GAUCHE, X_DROITE = 30.0, 90.0
Y_CENTRE = (HAUTEUR - 1) / 2.0

F0 = 1.0                  # Hz
FS = 15.0                 # Hz
N_OSC = 20                # -> 300 images, 20 s

#: Les 20 premiers vecteurs : on veut voir jusqu'ou la decroissance des valeurs
#: singulieres reste structuree avant de tomber dans le bruit numerique.
N_VECTEURS = 20

#: Fenetre d'affichage des formes d'onde. Le signal est STATIONNAIRE (meme
#: amplitude du debut a la fin), 5 oscillations sont donc representatives des
#: 20 -- et lisibles, ce que 20 cycles dans une cellule de tableau ne seraient
#: pas. Les spectres, eux, sont calcules sur l'enregistrement ENTIER.
N_OSC_AFFICHEES = 5

#: Nombre de modes filmes (indices 0 a N_MODES_VIDEO - 1). Les videos montrent
#: la reconstruction de RANG 1 de chaque mode, sigma_i * u_i v_i^T, remise en
#: images : c'est la seule facon de voir ce qu'une composante "fait" a l'image,
#: la carte u_i seule etant figee et la courbe v_i seule sans support spatial.
N_MODES_VIDEO = 6

#: Duree des videos, en cycles. Le signal est stationnaire et periodique : 5
#: cycles suffisent a tout montrer et se lisent en boucle sans couture, la ou
#: les 20 de l'enregistrement ne feraient que repeter la meme chose 4 fois de
#: plus pour 4 fois le poids de fichier.
N_OSC_VIDEO = 5

#: Gris de fond des videos, sur lequel les modes sont composes. Neutre a
#: dessein : le site a un theme clair ET un theme sombre, aucun des deux fonds
#: ne peut donc etre suivi, et un gris moyen reste lisible sur les deux.
FOND_VIDEO = 0.55

#: Agrandissement des videos : l'image simulee fait 60x120 px, illisible telle
#: quelle dans un navigateur. Agrandissement par REPETITION de pixels (et non
#: par interpolation) pour ne pas inventer de douceur que la simulation n'a pas.
ZOOM_VIDEO = 4

#: Amplitude commune aux experiences e2 et e3, ou le mouvement doit etre le
#: MEME des deux cotes pour que la seule difference testee soit l'intensite
#: (e2) ou la largeur (e3). Choisie moderee : devant le plus petit sigma de la
#: serie (SIGMA/2 = 3.5 px en e3) le deplacement vaut +/-0.57 sigma, assez
#: petit pour que le premier ordre domine -- donc pour que les predictions
#: analytiques du rapport des normes soient testables -- mais assez grand pour
#: qu'une echelle d'harmoniques existe et se laisse attribuer a l'un des deux
#: oscillateurs.
AMP_COMMUNE = 4.0         # px crete a crete


# --------------------------------------------------------------------------- #
# Extraction de phase : les deux methodes de `Astronauts/compute_one_cycle_
# compare_methods.py` (`1_fir` et `3a_mssa`), transposees sur la simulation.
#
# Les constantes et les recettes sont celles de `compute_pulse_from_data.py`,
# qui est le script ou ces deux phases sont REELLEMENT calculees (le script de
# comparaison, lui, ne fait que les relire dans les .npz). Trois ecarts
# assumes, tous parce que la simulation est plus simple que la donnee :
#
#   - Le peridogramme remplace le Lomb-Scargle. La production echantillonne a
#     des instants NON uniformes (horodatages Spectralis), d'ou Lomb-Scargle ;
#     ici la grille est exactement uniforme, ou les deux coincident. Le
#     sur-echantillonnage frequentiel (x5) reprend le `df = 1/(5*duree)` de
#     `build_ctx`.
#   - Le passe-bande FIR est re-ecrit ici au lieu d'importer
#     `ocularrigidity.motion.filters._1d.spatio_temporal_filter` : les scripts
#     de figures du site sont volontairement autonomes (numpy + matplotlib +
#     scipy), et le paquet n'est installe que dans l'environnement `pyOR`. La
#     recette est reprise a l'identique -- meme regle de nombre de taps, meme
#     `firwin`, meme `filtfilt`, meme `padlen` -- seule la branche spatiale,
#     sans objet sur un signal 1-D, est omise.
#   - Le "pouls combine" qui alimente le FIR est ici sigma_1 v_1 et non la
#     sortie d'`OptimizedSpectralCombination`. Sur une simulation sans bruit,
#     ou v_1 est un ton pur de concentration maximale, l'optimiseur ne peut
#     retenir que lui : le reimplementer n'ajouterait rien.
# --------------------------------------------------------------------------- #
BAND_FRAC = 0.2       # bande cardiaque : f0 +/- 20 % (idem production)
SSA_CYCLES = 3.0      # fenetre de plongement M-SSA, en cycles
SSA_N_COMP = 20       # composantes M-SSA examinees pour le regroupement
EDGE_FRAC = 0.2       # bords exclus des STATISTIQUES (transitoire du FIR)

#: Canaux passes a la M-SSA : les composantes temporelles k = 1..N_CHAN_MSSA.
#: La production les prend de la combinaison optimisee, qui ne selectionne que
#: des composantes portant du mouvement -- jamais k = 0, l'anatomie statique.
#: On reproduit cette exclusion : injecter le mode moyen donnerait a la M-SSA un
#: canal quasi constant, sans rapport avec l'hypothese physique du procede (un
#: meme pouls vu par plusieurs canaux).
N_CHAN_MSSA = 6

PHASE_METHODES = ("1_fir", "3a_mssa")
PHASE_LABELS = {"1_fir": "FIR band-pass", "3a_mssa": "M-SSA"}
PHASE_COULEURS = {"1_fir": "#2a78d6", "3a_mssa": "#e0a80d"}


@dataclass(frozen=True)
class Experience:
    tag: str              # prefixe de fichier
    titre: str            # titre de section
    ce_qui_change: str    # la seule chose qui differe entre les deux gaussiennes
    amp_gauche: float     # px crete a crete
    amp_droite: float
    intensite_gauche: float   # amplitude du pic de la gaussienne
    intensite_droite: float
    sigma_gauche: float   # px
    sigma_droite: float
    prediction: float     # rapport attendu des normes de u_1, gauche / droite
    justification: str    # d'ou vient cette prediction


EXPERIENCES = [
    Experience(
        tag="e1",
        titre="Amplitudes differentes",
        ce_qui_change="l'amplitude du mouvement",
        amp_gauche=8.0, amp_droite=2.0,
        intensite_gauche=1.0, intensite_droite=1.0,
        sigma_gauche=SIGMA, sigma_droite=SIGMA,
        prediction=8.0 / 2.0,
        justification=(
            "le premier ordre vaut $A\\,s(t)\\,\\partial I/\\partial y$ et les "
            "deux gaussiennes sont identiques : le rapport des normes de $u_1$ "
            "est celui des amplitudes"
        ),
    ),
    Experience(
        tag="e2",
        titre="Intensites differentes",
        ce_qui_change="l'intensite (le pic de la gaussienne)",
        amp_gauche=AMP_COMMUNE, amp_droite=AMP_COMMUNE,
        intensite_gauche=1.0, intensite_droite=0.5,
        sigma_gauche=SIGMA, sigma_droite=SIGMA,
        prediction=1.0 / 0.5,
        justification=(
            "$\\partial I/\\partial y$ est proportionnel a l'intensite, donc a "
            "mouvement identique le rapport des normes de $u_1$ est celui des "
            "intensites"
        ),
    ),
    Experience(
        tag="e3",
        titre="Ecart-type deux fois plus petit",
        ce_qui_change="l'ecart-type (la largeur)",
        amp_gauche=AMP_COMMUNE, amp_droite=AMP_COMMUNE,
        intensite_gauche=1.0, intensite_droite=1.0,
        sigma_gauche=SIGMA, sigma_droite=SIGMA / 2,
        prediction=1.0,
        justification=(
            "pour une gaussienne de pic unite, "
            "$\\|\\partial I/\\partial y\\|^2 = \\pi/2$ **quel que soit "
            "$\\sigma$** : le gradient d'un objet fin est plus raide d'exactement "
            "ce qu'il couvre moins de pixels"
        ),
    ),
]

# --------------------------------------------------------------------------- #
# Style : identique a figures_gaussienne_svd (fond transparent, theme dark)
# --------------------------------------------------------------------------- #
COULEUR_TEXTE = "#c9c9c9"
C_SIGNAL = "#2a78d6"          # bleu
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
    fig.savefig(SORTIE / nom)
    plt.close(fig)
    FICHIERS.append(nom)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
T = int(round(N_OSC * FS / F0))
t = np.arange(T) / FS
DUREE = T / FS
yy, xx = np.mgrid[0:HAUTEUR, 0:LARGEUR]
freqs = np.fft.rfftfreq(T, 1 / FS)

# Un seul signal temporel pour tout le monde : c'est la definition de "meme
# frequence, en phase". Ce qui distingue les deux gaussiennes est l'amplitude
# par laquelle chacune le multiplie, pas sa forme.
signal = np.sin(2 * np.pi * F0 * t)

# Frontiere entre les deux moities de l'image, a mi-chemin des deux centres.
COL_FRONTIERE = int(round((X_GAUCHE + X_DROITE) / 2))

#: Nombre d'images par cycle. C'est LUI qui fixe le rang : le deplacement etant
#: periodique et le cycle tombant sur un nombre ENTIER d'images, la trace
#: temporelle de chaque pixel est une suite N_PAR_CYCLE-periodique. Or ces
#: suites forment un espace de dimension exactement N_PAR_CYCLE -- le rang ne
#: peut donc pas le depasser, quels que soient le nombre de pixels, le nombre
#: de cycles enregistres ou la non-linearite de l'objet.
N_PAR_CYCLE = int(round(FS / F0))
#: Rang de l'harmonique la plus haute non repliee.
H_NYQUIST = int(0.5 * FS / F0)


def une_gaussienne(x0, sigma, intensite, deplacement):
    """Cube (T, HAUTEUR, LARGEUR) : une gaussienne translatee verticalement."""
    return intensite * np.exp(
        -(
            ((yy[None] - Y_CENTRE - deplacement[:, None, None]) ** 2
             + (xx[None] - x0) ** 2)
            / (2 * sigma**2)
        )
    )


# --------------------------------------------------------------------------- #
# SSA / Hankel -- transcription de `compute_pulse_from_data.py`
# --------------------------------------------------------------------------- #
L_SSA = int(round(SSA_CYCLES * FS / F0)) | 1  # fenetre de plongement, impaire


def periodogramme(y):
    """(frequences, puissance) d'une trace sur la grille uniforme.

    Remplace le Lomb-Scargle de la production, equivalent ici puisque la grille
    est exactement uniforme. Zero-padding x5 pour retrouver la resolution
    frequentielle `df = 1 / (5 * duree)` de `build_ctx`.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    n = 5 * y.size
    p = np.abs(np.fft.rfft(y * np.hanning(y.size), n=n)) ** 2
    return np.fft.rfftfreq(n, 1 / FS), p


def score_bande(y):
    """(fraction de puissance dans la bande cardiaque, frequence du pic)."""
    f, p = periodogramme(y)
    en_bande = np.abs(f - F0) <= BAND_FRAC * F0
    return (float(p[en_bande].sum() / (p.sum() + 1e-12)), float(f[p.argmax()]))


def hankel_matrix(x, L):
    """Matrice trajectoire (L, K), K = N - L + 1 : H[i, j] = x[i + j]."""
    x = np.asarray(x, dtype=float)
    K = x.size - L + 1
    if K < L:
        raise ValueError(f"L = {L} trop grand pour N = {x.size} (K = {K} < L)")
    return np.lib.stride_tricks.sliding_window_view(x, K)


def diagonal_average(M):
    """Retour Hankel -> 1-D : moyenne de chaque anti-diagonale."""
    L, K = M.shape
    out = np.zeros(L + K - 1)
    cnt = np.zeros(L + K - 1)
    for i in range(L):
        out[i:i + K] += M[i]
        cnt[i:i + K] += 1.0
    return out / cnt


def cardiac_keep(recon_1d_list, tol):
    """Indices des composantes dont le pic tombe a moins de `tol` Hz de f0.

    A defaut, la paire de plus forte puissance en bande -- meme repli que la
    production, qui prefere un groupe explicitement degrade a une exception au
    milieu d'un lot.
    """
    pics, fracs = [], []
    for r in recon_1d_list:
        fr, pk = score_bande(r)
        pics.append(pk)
        fracs.append(fr)
    pics, fracs = np.asarray(pics), np.asarray(fracs)
    keep = np.where(np.abs(pics - F0) <= tol)[0]
    if keep.size == 0:
        keep = np.argsort(fracs)[::-1][:2]
    return np.sort(keep), pics, fracs


def mssa_denoise(chan, L=L_SSA, n_comp=SSA_N_COMP):
    """M-SSA : une seule matrice trajectoire empilant les canaux.

    Les Hankel des canaux sont empiles VERTICALEMENT, si bien que la SVD voit
    un sous-espace temporel PARTAGE -- l'hypothese physique du probleme : un
    meme pouls vu par plusieurs canaux, a des amplitudes et des retards
    differents. C'est exactement la situation des deux gaussiennes.
    """
    tol = BAND_FRAC * F0
    n_ch, N = chan.shape
    H = np.vstack([hankel_matrix(c, L) for c in chan])
    Uh, sh, Vth = np.linalg.svd(H, full_matrices=False)
    n = int(min(n_comp, sh.size))

    recon = np.empty((n, n_ch, N))
    for k in range(n):
        Mk = sh[k] * np.outer(Uh[:, k], Vth[k])
        for c in range(n_ch):
            recon[k, c] = diagonal_average(Mk[c * L:(c + 1) * L])

    # Regroupement sur la MOYENNE des canaux : une composante M-SSA est
    # cardiaque ou ne l'est pas, c'est une propriete du mode partage.
    keep, pics, fracs = cardiac_keep([recon[k].mean(axis=0) for k in range(n)], tol)
    return recon[keep].sum(axis=0), {"sv": sh, "keep": keep, "pics": pics,
                                     "fracs": fracs, "H_shape": H.shape}


def mssa_pulse(chan, L=L_SSA):
    """M-SSA puis 1re composante principale des canaux debruites.

    Les canaux debruites sont, par construction, quasi de rang 2 : leur
    premiere PC EST l'oscillation partagee.
    """
    R, diag = mssa_denoise(chan, L)
    _, Sc, Vtc = np.linalg.svd(R - R.mean(axis=1, keepdims=True), full_matrices=False)
    diag["pc1_var"] = float(Sc[0] ** 2 / (Sc**2).sum())
    return Vtc[0] * Sc[0], diag


def bandpass_fir(y):
    """Passe-bande cardiaque FIR a phase lineaire (methode `1_fir`).

    Transcription 1-D de `motion.filters._1d.spatio_temporal_filter` : meme
    regle de nombre de taps (4 cycles de la coupure basse, borne a n/3 et
    impair), meme `firwin`, meme `filtfilt` a phase nulle, meme `padlen`.
    """
    y = np.asarray(y, dtype=float)
    nyq = 0.5 * FS
    lo, hi = (1 - BAND_FRAC) * F0 / nyq, min((1 + BAND_FRAC) * F0 / nyq, 0.99)
    n_t = y.size
    n_taps = int(round(4.0 * FS / ((1 - BAND_FRAC) * F0)))
    max_taps = max(3, (n_t // 3) | 1)
    n_taps = min(max(n_taps, 31), max_taps)
    if n_taps % 2 == 0:
        n_taps += 1
    if n_taps > max_taps:
        n_taps -= 2
    taps = firwin(n_taps, [lo, hi], pass_zero=False)
    return filtfilt(taps, 1.0, y, padlen=min(3 * n_taps, n_t - 1))


def phase_analytique(y):
    """Phase repliee sur [0, 2pi) par transformee de Hilbert (idem production)."""
    y = np.asarray(y, dtype=float)
    return np.mod(np.unwrap(np.angle(hilbert(y - y.mean()))), 2 * np.pi)


def phaseur_f0(y):
    """Phaseur de `y` a f0 -- sert a fixer la POLARITE, pas la phase."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    return (y @ np.cos(2 * np.pi * F0 * t)) + 1j * (y @ np.sin(2 * np.pi * F0 * t))


def fixer_signe(y, reference):
    """Polarite de `y` alignee sur `reference`, via le phaseur a f0.

    Le signe d'un vecteur singulier est arbitraire, et une inversion se lit
    comme un dephasage de 180 deg : sans cette convention la comparaison des
    phases mesurerait surtout un tirage au sort. La production fait de meme
    (`make_sign_fixer`), a ceci pres qu'elle s'aligne sur la moyenne des traces
    brutes ; ici la reference est le signal impose, ce qui revient a dire que la
    polarite est POSEE et n'est pas un resultat de la mesure.
    """
    aligne = float(np.real(phaseur_f0(y) * np.conj(phaseur_f0(reference))))
    return (-1.0 if aligne < 0 else 1.0) * np.asarray(y, dtype=float)


def ecart_angulaire(a, b):
    """a - b, replie sur (-pi, pi]."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class PhaseMethode:
    nom: str
    pouls: np.ndarray        # trace dont la phase est lue
    phase: np.ndarray        # (T,) repliee sur [0, 2pi)
    erreur: np.ndarray       # (T,) ecart a la phase imposee, radians
    biais_deg: float         # moyenne circulaire de l'erreur, coeur seul
    plv: float               # constance de ce decalage, coeur seul
    dispersion_deg: float    # ecart-type circulaire, coeur seul
    rms_deg: float           # erreur quadratique moyenne, coeur seul
    max_deg: float           # ecart maximal, coeur seul
    rms_bords_deg: float     # la meme, sur les bords exclus
    diag: dict


@dataclass
class Resultat:
    exp: Experience
    cube: np.ndarray
    dy_gauche: np.ndarray
    dy_droite: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    conc: list          # (concentration, frequence du pic) par composante
    part: list          # part de |u_k|^2 sur la gaussienne de gauche
    rapport_u1: float
    corr_v1: float
    k_plancher: int
    phases: dict  # nom de methode -> PhaseMethode


def concentration(v):
    """Concentration spectrale et frequence du pic.

    Pour un ton pur sous fenetre de Hann elle vaut ~0.66, quelle que soit la
    frequence : c'est ce qui en fait un test binaire "forme d'onde ou arrondi".
    """
    p = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)))) ** 2
    return p.max() / p.sum(), freqs[p.argmax()]


def part_gauche(u):
    """Fraction de l'energie de la carte `u` portee par la gaussienne de gauche."""
    carte = u.reshape(HAUTEUR, LARGEUR)
    gauche = float((carte[:, :COL_FRONTIERE] ** 2).sum())
    return gauche / float((carte**2).sum())


def rapport_normes(u):
    """Rapport des normes gauche/droite de la carte `u`.

    C'est la seule facon de retrouver ce qui distingue les deux gaussiennes :
    v_1 seul ne le porte pas, il a fondu les deux en une forme d'onde unique.
    """
    carte = u.reshape(HAUTEUR, LARGEUR)
    gauche = np.linalg.norm(carte[:, :COL_FRONTIERE])
    droite = np.linalg.norm(carte[:, COL_FRONTIERE:])
    return float(gauche / droite) if droite > 0 else float("inf")


#: Phase IMPOSEE, exacte et analytique. Le deplacement vaut (A/2) sin(2 pi f0 t)
#: et la transformee de Hilbert d'un sinus vaut -cosinus, donc le signal
#: analytique est exp(i(2 pi f0 t - pi/2)) : la phase de reference n'est pas
#: estimee sur le signal simule, elle est ECRITE. C'est ce qui fait de cette
#: page un test et non une comparaison de deux estimateurs entre eux.
PHASE_IMPOSEE = np.mod(2 * np.pi * F0 * t - np.pi / 2, 2 * np.pi)

#: Coeur de l'enregistrement : les bords portent le transitoire du FIR (et,
#: pour la M-SSA, la moyenne diagonale y porte sur moins de termes). La
#: production les exclut de ses STATISTIQUES par la meme fraction ; on garde
#: les courbes entieres a l'affichage, mais les chiffres sont lus au coeur.
COEUR = np.zeros(T, dtype=bool)
COEUR[int(EDGE_FRAC * T):T - int(EDGE_FRAC * T)] = True


def mesurer_phase(nom, pouls, diag) -> PhaseMethode:
    pouls = fixer_signe(pouls, signal)
    phase = phase_analytique(pouls)
    err = ecart_angulaire(phase, PHASE_IMPOSEE)
    z = np.exp(1j * err[COEUR]).mean()
    plv = float(np.abs(z))
    return PhaseMethode(
        nom=nom,
        pouls=pouls,
        phase=phase,
        erreur=err,
        biais_deg=float(np.degrees(np.angle(z))),
        plv=plv,
        # Ecart-type circulaire : sqrt(-2 ln R). Il tend vers l'ecart-type
        # ordinaire quand la dispersion est petite, mais reste defini quand
        # elle ne l'est pas -- ce qu'une phase exige.
        dispersion_deg=float(np.degrees(np.sqrt(max(-2 * np.log(max(plv, 1e-15)), 0)))),
        rms_deg=float(np.degrees(np.sqrt(np.mean(err[COEUR] ** 2)))),
        max_deg=float(np.degrees(np.abs(err[COEUR]).max())),
        rms_bords_deg=float(np.degrees(np.sqrt(np.mean(err[~COEUR] ** 2)))),
        diag=diag,
    )


def analyser_phases(U, S, Vt) -> dict:
    """Les deux phases concurrentes, sur les composantes temporelles de la SVD."""
    # `1_fir` : passe-bande sur le pouls combine. Voir l'entete pour le choix
    # de sigma_1 v_1 comme combinaison.
    pouls_fir = bandpass_fir(S[1] * Vt[1])

    # `3a_mssa` : M-SSA multicanal sur les composantes porteuses de mouvement,
    # AUCUNE bande imposee -- c'est le regroupement interne qui trie.
    canaux = np.stack([S[k] * Vt[k] for k in range(1, N_CHAN_MSSA + 1)])
    pouls_mssa, diag_mssa = mssa_pulse(canaux)

    return {
        "1_fir": mesurer_phase("1_fir", pouls_fir, {"n_taps_bande": BAND_FRAC}),
        "3a_mssa": mesurer_phase("3a_mssa", pouls_mssa, diag_mssa),
    }


def simuler(exp: Experience) -> Resultat:
    dy_gauche = (exp.amp_gauche / 2.0) * signal
    dy_droite = (exp.amp_droite / 2.0) * signal
    cube = (
        une_gaussienne(X_GAUCHE, exp.sigma_gauche, exp.intensite_gauche, dy_gauche)
        + une_gaussienne(X_DROITE, exp.sigma_droite, exp.intensite_droite, dy_droite)
    )

    # Une ligne par pixel, une colonne par image : U porte des cartes spatiales,
    # V des formes d'onde.
    U, S, Vt = np.linalg.svd(cube.reshape(T, -1).T, full_matrices=False)

    # Signe arbitraire : on le fixe pour que l'extremum de chaque v_k soit
    # positif, sinon la moitie des courbes serait retournee sans que cela veuille
    # dire quoi que ce soit.
    for k in range(len(S)):
        if Vt[k][np.argmax(np.abs(Vt[k]))] < 0:
            U[:, k] *= -1
            Vt[k] *= -1

    conc = [concentration(Vt[k]) for k in range(N_VECTEURS)]
    # Premier k qui ne decrit plus la simulation. Le critere est SPECTRAL et non
    # une magnitude arbitraire : tant que v_k est une harmonique de f0, sa
    # concentration vaut ~0.66 ; des que v_k n'est plus qu'un arrondi, elle
    # s'effondre sous 0.1. Un seuil sur sigma_k/sigma_0 couperait, lui, des
    # composantes encore parfaitement structurees.
    k_plancher = int(np.argmax(np.array([c for c, _ in conc]) < 0.5))

    return Resultat(
        exp=exp,
        cube=cube,
        dy_gauche=dy_gauche,
        dy_droite=dy_droite,
        U=U, S=S, Vt=Vt,
        conc=conc,
        part=[part_gauche(U[:, k]) for k in range(N_VECTEURS)],
        rapport_u1=rapport_normes(U[:, 1]),
        corr_v1=abs(float(np.corrcoef(Vt[1], signal)[0, 1])),
        k_plancher=k_plancher,
        phases=analyser_phases(U, S, Vt),
    )


# --------------------------------------------------------------------------- #
# Briques graphiques
# --------------------------------------------------------------------------- #
def figure_image(img, nom):
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.imshow(img, cmap="magma", vmin=0.0, vmax=1.0)  # echelle absolue : les
    ax.set_xticks([]); ax.set_yticks([])              # intensites se comparent
    for cote in ax.spines.values():
        cote.set_visible(False)
    enregistrer(fig, nom)


def figure_coupe(img, nom):
    """Coupe horizontale au centre : rend explicites intensite et largeur."""
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    ax.plot(img[int(round(Y_CENTRE))], lw=1.3, color=C_SIGNAL)
    ax.axvline(COL_FRONTIERE, color=COULEUR_TEXTE, alpha=0.30, lw=0.7, ls="--")
    ax.set_xlabel("colonne (px)")
    ax.set_ylabel("intensite")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def figure_deplacements(res: Resultat, nom):
    """Le ou les deplacements imposes."""
    exp = res.exp
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    if np.allclose(res.dy_gauche, res.dy_droite):
        ax.plot(t, res.dy_gauche, lw=1.4, color=C_SIGNAL,
                label=f"les deux ({exp.amp_gauche:.0f} px c.-a-c.)")
    else:
        ax.plot(t, res.dy_gauche, lw=1.3, color=C_SIGNAL,
                label=f"gauche ({exp.amp_gauche:.0f} px c.-a-c.)")
        ax.plot(t, res.dy_droite, lw=1.3, color=C_BRUIT,
                label=f"droite ({exp.amp_droite:.0f} px c.-a-c.)")
    ax.axhline(0, color=C_REF, lw=0.6, ls=":")
    ax.set_xlim(0, N_OSC_AFFICHEES / F0)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("deplacement (px)")
    ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def figure_valeurs_singulieres(res: Resultat, nom):
    k = np.arange(N_VECTEURS)
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.semilogy(k, res.S[:N_VECTEURS] / res.S[0], "o-", ms=3.5, lw=1.0,
                color=C_SIGNAL)
    ax.set_xticks(k[::2])
    ax.set_xlabel("indice $k$")
    ax.set_ylabel(r"$\sigma_k / \sigma_0$")
    ax.grid(axis="y", color=COULEUR_TEXTE, alpha=0.15, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def figure_carte(u, nom):
    """Un vecteur singulier spatial, remis en image (opacite ~ |u|).

    Normalisation par le 99.5e centile de CHAQUE carte : les u_k etant tous de
    norme 1, c'est la forme et la repartition gauche/droite qui se comparent
    d'une ligne a l'autre, pas l'amplitude absolue (elle est dans sigma_k).
    """
    carte = u.reshape(HAUTEUR, LARGEUR)
    vmax = np.percentile(np.abs(carte), 99.5)
    opacite = np.clip(np.abs(carte) / vmax, 0.0, 1.0) ** 0.5
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    ax.imshow(carte, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=opacite)
    ax.axvline(COL_FRONTIERE, color=COULEUR_TEXTE, alpha=0.30, lw=0.7, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    for cote in ax.spines.values():
        cote.set_visible(True)
        cote.set_color(COULEUR_TEXTE)
        cote.set_alpha(0.35)
        cote.set_linewidth(0.6)
    enregistrer(fig, nom)


def figure_courbe(v, nom, ylim):
    fig, ax = plt.subplots(figsize=(3.6, 1.5))
    ax.plot(t, v, lw=1.0, color=C_SIGNAL)
    ax.axhline(0, color=C_REF, lw=0.6, ls=":")
    ax.set_xlim(0, N_OSC_AFFICHEES / F0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("temps (s)")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    enregistrer(fig, nom)


def figure_spectre(v, nom, f_max=7.5):
    """Spectre de puissance, sur l'enregistrement ENTIER, normalise par son max."""
    p = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)))) ** 2
    fig, ax = plt.subplots(figsize=(3.6, 1.6))
    for h in range(1, int(f_max / F0) + 1):
        ax.axvline(h * F0, color=COULEUR_TEXTE, alpha=0.22, lw=0.7, ls=":")
    ax.semilogy(freqs, p / p.max(), lw=1.0, color=C_SIGNAL)
    ax.set_xlim(0, f_max)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("frequence (Hz)")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    enregistrer(fig, nom)


def figure_superposition(res: Resultat, nom):
    """v_1 remis a l'echelle, compare au(x) deplacement(s) impose(s)."""
    exp, v1 = res.exp, res.Vt[1]
    signe = np.sign(np.corrcoef(v1, signal)[0, 1]) or 1.0
    v1_px = signe * v1 / v1.std() * res.dy_gauche.std()
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(t, res.dy_gauche, lw=1.6, color=C_REF,
            label=f"deplacement {exp.amp_gauche:.0f} px")
    if not np.allclose(res.dy_gauche, res.dy_droite):
        ax.plot(t, res.dy_droite, lw=1.6, color=C_BRUIT,
                label=f"deplacement {exp.amp_droite:.0f} px")
    ax.plot(t, v1_px, lw=1.2, ls="--", color=C_SIGNAL,
            label=f"$v_1$ (echelle : {exp.amp_gauche:.0f} px)")
    ax.set_xlim(0, N_OSC_AFFICHEES / F0)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("deplacement (px)")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def figure_pouls(res: Resultat, nom):
    """Signal impose et les deux pouls extraits, chacun a sa propre echelle.

    Les pouls sont normalises a l'ecart-type du signal impose : ce qui se
    compare ici est la FORME et le calage temporel, pas l'amplitude -- aucune
    des deux methodes ne pretend restituer une amplitude physique.
    """
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    ax.plot(t, signal, lw=1.8, color=C_REF, label="signal impose")
    for methode in PHASE_METHODES:
        m = res.phases[methode]
        y = m.pouls / m.pouls.std() * signal.std()
        ax.plot(t, y, lw=1.1, ls="--", color=PHASE_COULEURS[methode],
                label=PHASE_LABELS[methode])
    ax.set_xlim(0, N_OSC_AFFICHEES / F0)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("amplitude (u. a.)")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def figure_phase(res: Resultat, nom):
    """Phase imposee et phases mesurees, en dents de scie sur [0, 360 deg)."""
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    ax.plot(t, np.degrees(PHASE_IMPOSEE), lw=2.2, color=C_REF,
            label="phase imposee")
    for methode in PHASE_METHODES:
        ax.plot(t, np.degrees(res.phases[methode].phase), lw=1.1, ls="--",
                color=PHASE_COULEURS[methode], label=PHASE_LABELS[methode])
    ax.set_xlim(0, N_OSC_AFFICHEES / F0)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("phase (deg)")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def figure_erreur_phase(res: Resultat, nom):
    """Ecart a la phase imposee, sur TOUT l'enregistrement, bords grises.

    C'est la figure qui repond a la question posee : non pas "les deux methodes
    se ressemblent-elles" mais "de combien chacune s'ecarte de la verite".
    """
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bord = EDGE_FRAC * DUREE
    for x0, x1 in ((0, bord), (DUREE - bord, DUREE)):
        ax.axvspan(x0, x1, color=COULEUR_TEXTE, alpha=0.10, lw=0)
    ax.axhline(0, color=C_REF, lw=0.8, ls=":")
    for methode in PHASE_METHODES:
        m = res.phases[methode]
        ax.plot(t, np.degrees(m.erreur), lw=1.1, color=PHASE_COULEURS[methode],
                label=f"{PHASE_LABELS[methode]} "
                      f"(RMS {m.rms_deg:.2f} deg au coeur)")
    ax.set_xlim(0, DUREE)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("erreur de phase (deg)")
    ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


def resoudre_ffmpeg() -> str:
    """Chemin de ffmpeg : PATH d'abord, `imageio_ffmpeg` en secours.

    Meme strategie que `scripts/registration/astronauts.py::resolve_ffmpeg`,
    pour ne pas dependre d'une installation systeme sur une machine qui n'a que
    le paquet Python.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg introuvable : ni dans le PATH, ni via imageio_ffmpeg "
            "(`pip install imageio-ffmpeg`)."
        ) from exc


FFMPEG = resoudre_ffmpeg()
VIDEOS: list[str] = []


def ecrire_mp4(cube, nom, vmax, fps=FS, zoom=ZOOM_VIDEO):
    """Ecrit un cube signe (T, H, W) en .mp4, colore en divergent.

    L'echelle de couleur est FIXE sur toute la video (bornes -vmax/+vmax) : une
    normalisation image par image annulerait visuellement la variation
    temporelle, qui est precisement ce que la video sert a montrer.
    """
    couleurs = plt.get_cmap("RdBu_r")
    niveaux = np.clip(cube / (2 * vmax) + 0.5, 0.0, 1.0)
    rgb = couleurs(niveaux)[..., :3]  # (T, H, W, 3), dans [0, 1]

    # Le centre de RdBu_r est blanc : tel quel, chaque video serait un pave
    # blanc sur le theme sombre du site. Les cartes u_k en PNG reglent cela par
    # une opacite proportionnelle a |u| qui laisse passer le fond -- impossible
    # en h264, qui n'a pas de canal alpha. On compose donc la meme opacite sur
    # un gris neutre : les zones sans signal deviennent grises au lieu de
    # blanches, ce qui tient sur fond clair comme sur fond sombre, et on garde
    # le meme langage visuel que les cartes fixes.
    opacite = np.clip(np.abs(cube) / vmax, 0.0, 1.0)[..., None] ** 0.5
    rgb = (opacite * rgb + (1 - opacite) * FOND_VIDEO) * 255
    rgb = rgb.astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, zoom, axis=1), zoom, axis=2)

    h, w = rgb.shape[1], rgb.shape[2]
    if h % 2 or w % 2:  # yuv420p exige des dimensions paires
        rgb = rgb[:, : h - h % 2, : w - w % 2]
        h, w = rgb.shape[1], rgb.shape[2]

    commande = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart",
        str(SORTIE / nom),
    ]
    proc = subprocess.run(commande, input=rgb.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg a echoue sur {nom} : {proc.stderr.decode(errors='replace')}"
        )
    VIDEOS.append(nom)


def video_mode(res: Resultat, k: int, nom: str):
    """Video de la reconstruction de rang 1 du mode k : sigma_k * u_k v_k^T.

    C'est le terme que le mode ajoute a l'image, pas la carte u_k : on y voit
    donc a la fois OU il agit et COMMENT il varie dans le temps.
    """
    n_frames = int(round(N_OSC_VIDEO * FS / F0))
    mode = res.S[k] * np.outer(res.U[:, k], res.Vt[k][:n_frames])  # (pixels, t)
    mode = mode.T.reshape(n_frames, HAUTEUR, LARGEUR)
    # Borne de couleur commune a toute la video, prise sur le mode entier.
    vmax = float(np.percentile(np.abs(mode), 99.9)) or 1.0
    ecrire_mp4(mode, nom, vmax)


def figure_profil(res: Resultat, nom, n_profils=4):
    """Norme de u_k colonne par colonne, pour les premiers k.

    Lecture la plus directe de "a qui appartient cette composante" : deux bosses
    de hauteurs comparables = les deux gaussiennes contribuent, une seule bosse
    = la composante appartient a un seul oscillateur.
    """
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    couleurs = plt.cm.viridis(np.linspace(0.15, 0.85, n_profils))
    for k in range(n_profils):
        profil = np.linalg.norm(res.U[:, k].reshape(HAUTEUR, LARGEUR), axis=0)
        ax.plot(profil / profil.max(), lw=1.2, color=couleurs[k], label=f"$u_{k}$")
    ax.axvline(COL_FRONTIERE, color=COULEUR_TEXTE, alpha=0.30, lw=0.7, ls="--")
    ax.set_xlabel(f"colonne (px)  --  ce qui change : {res.exp.ce_qui_change}")
    ax.set_ylabel("norme par colonne\n(normalisee)")
    ax.legend(fontsize=7, ncol=n_profils, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    enregistrer(fig, nom)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
RESULTATS = {}
for exp in EXPERIENCES:
    res = simuler(exp)
    RESULTATS[exp.tag] = res
    p = exp.tag

    figure_image(res.cube[0], f"{p}_s1_image.png")
    figure_coupe(res.cube[0], f"{p}_s1_coupe.png")
    figure_deplacements(res, f"{p}_s1_deplacements.png")
    figure_valeurs_singulieres(res, f"{p}_s2_valeurs_singulieres.png")

    for k in range(N_VECTEURS):
        ylim = 1.15 * np.abs(res.Vt[k]).max()
        figure_carte(res.U[:, k], f"{p}_s3_U{k}.png")
        figure_courbe(res.Vt[k], f"{p}_s4_V{k}.png", (-ylim, ylim))
        figure_spectre(res.Vt[k], f"{p}_s5_spectre_V{k}.png")

    figure_superposition(res, f"{p}_s6_superposition.png")
    figure_profil(res, f"{p}_s7_profil_colonnes.png")

    figure_pouls(res, f"{p}_s9_pouls.png")
    figure_phase(res, f"{p}_s9_phase.png")
    figure_erreur_phase(res, f"{p}_s9_erreur_phase.png")

    for k in range(N_MODES_VIDEO):
        video_mode(res, k, f"{p}_s8_mode{k}.mp4")


# --------------------------------------------------------------------------- #
# Recapitulatif chiffre
# --------------------------------------------------------------------------- #
lignes = [
    f"deux gaussiennes en phase a {F0:.0f} Hz, sans bruit -- {T} images, "
    f"{DUREE:.0f} s, image {HAUTEUR}x{LARGEUR} px",
    "",
]
for exp in EXPERIENCES:
    res = RESULTATS[exp.tag]
    lignes += [
        f"=== {exp.tag} : {exp.titre} "
        f"(amplitudes {exp.amp_gauche:.0f}/{exp.amp_droite:.0f} px, "
        f"intensites {exp.intensite_gauche:.2f}/{exp.intensite_droite:.2f}, "
        f"sigma {exp.sigma_gauche:.1f}/{exp.sigma_droite:.1f} px)",
        f"corr(v1, signal impose)                 : {res.corr_v1:.6f}",
        f"norme(u1|gauche) / norme(u1|droite)     : {res.rapport_u1:.3f}  "
        f"(prediction : {exp.prediction:.3f}, ecart "
        f"{100 * (res.rapport_u1 / exp.prediction - 1):+.1f}%)",
        f"rang effectif (concentration < 0.5)     : {res.k_plancher}",
        "-- phase vs phase imposee (statistiques au coeur, bords "
        f"{EDGE_FRAC:.0%} exclus) --",
        "methode      | biais deg | dispersion deg | RMS deg | max deg | "
        "PLV     | RMS bords deg",
    ]
    for methode in PHASE_METHODES:
        m = res.phases[methode]
        lignes.append(
            f"{PHASE_LABELS[methode]:12s} | {m.biais_deg:+9.3f} | "
            f"{m.dispersion_deg:14.3f} | {m.rms_deg:7.3f} | {m.max_deg:7.3f} | "
            f"{m.plv:.5f} | {m.rms_bords_deg:13.3f}"
        )
    lignes += [
        " k | sigma_k/sigma_0 | conc(v_k) | pic v_k | part de |u_k|^2 a gauche",
    ]
    for k in range(N_VECTEURS):
        conc, pic = res.conc[k]
        lignes.append(
            f"{k:2d} |     {res.S[k] / res.S[0]:.3e}   |   {conc:.3f}   | "
            f"{pic:5.2f} Hz |          {res.part[k]:.3f}"
        )
    lignes.append("")
recap = "\n".join(lignes)
(SORTIE / "recapitulatif.txt").write_text(recap + "\n", encoding="utf-8")
print(recap)


# --------------------------------------------------------------------------- #
# Fragment Quarto pret a inclure
# --------------------------------------------------------------------------- #
DOSSIER = SORTIE.name


def img(nom, largeur="100%"):
    assert nom in FICHIERS, f"figure non generee : {nom}"
    return f"![]({DOSSIER}/{nom}){{width={largeur}}}"


def video(nom, largeur="100%"):
    """Balise <video> brute plutot que le shortcode Quarto.

    Le shortcode `{{{{< video >}}}}` vise surtout l'embarquement de plateformes
    externes et impose son propre gabarit ; ici il en faut six par ligne, muets
    et bouclants, donc autant ecrire la balise. Pas d'`autoplay` : six lectures
    simultanees par experience, dix-huit sur la page, sature le processeur pour
    rien -- l'utilisateur lance celle qu'il veut comparer.
    """
    assert nom in VIDEOS, f"video non generee : {nom}"
    # `style` et non l'attribut `width` : celui-ci ne prend que des pixels sur
    # un <video>, un pourcentage y est ignore par les navigateurs.
    return (
        f'<video src="{DOSSIER}/{nom}" style="width:{largeur};height:auto" '
        f'controls loop muted playsinline preload="metadata"></video>'
    )


def grille_videos(res: Resultat):
    """Les N_MODES_VIDEO premiers modes en video, trois par ligne."""
    p = res.exp.tag
    lignes = []
    for debut in range(0, N_MODES_VIDEO, 3):
        indices = range(debut, min(debut + 3, N_MODES_VIDEO))
        entetes = [
            f"$\\sigma_{{{k}}} u_{{{k}}} v_{{{k}}}^{{T}}$ "
            f"({res.conc[k][1]:.0f} Hz)"
            for k in indices
        ]
        lignes += [
            "| " + " | ".join(entetes) + " |",
            "|" + "---|" * len(entetes),
            "| " + " | ".join(video(f"{p}_s8_mode{k}.mp4") for k in indices) + " |",
            "",
        ]
    return "\n".join(lignes)


def grille_composantes(res: Resultat):
    """Une ligne par composante : u_k, v_k, spectre de v_k, et les chiffres."""
    p = res.exp.tag
    lignes = [
        "| composante | $u_k$ (carte spatiale) | $v_k$ (forme d'onde) | "
        "spectre de $v_k$ |",
        "|---|---|---|---|",
    ]
    for k in range(N_VECTEURS):
        _conc, pic = res.conc[k]
        etiquette = (
            f"**$k = {k}$**<br>$\\sigma_k/\\sigma_0$ = {res.S[k] / res.S[0]:.2e}"
            f"<br>pic : {pic:.2f} Hz<br>{res.part[k]:.0%} a gauche"
        )
        lignes.append(
            f"| {etiquette} | {img(f'{p}_s3_U{k}.png')} | {img(f'{p}_s4_V{k}.png')} "
            f"| {img(f'{p}_s5_spectre_V{k}.png')} |"
        )
    return "\n".join(lignes)


EXPOSANTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def deg(v, signe=False) -> str:
    """Un angle en degres, lisible sur SIX ordres de grandeur.

    La M-SSA sort ici des erreurs de l'ordre de 1e-4 degre : les afficher en
    `.2f` donnerait une colonne de `0.00` d'ou l'on ne pourrait pas dire si la
    methode est exacte ou simplement arrondie. On bascule donc en notation
    scientifique sous le centieme de degre.
    """
    if not np.isfinite(v):
        return "--"
    if abs(v) < 1e-2:
        mantisse, exposant = f"{v:.0e}".split("e")
        chiffres = str(int(exposant)).translate(EXPOSANTS)
        return f"{mantisse}·10{chiffres}°"
    return f"{v:+.2f}°" if signe else f"{v:.2f}°"


def tableau_phase(res: Resultat) -> str:
    lignes = [
        "| methode | biais | dispersion | RMS | max | RMS sur les bords |",
        "|---|---|---|---|---|---|",
    ]
    for methode in PHASE_METHODES:
        m = res.phases[methode]
        lignes.append(
            f"| {PHASE_LABELS[methode]} | {deg(m.biais_deg, signe=True)} | "
            f"{deg(m.dispersion_deg)} | {deg(m.rms_deg)} | {deg(m.max_deg)} | "
            f"{deg(m.rms_bords_deg)} |"
        )
    return "\n".join(lignes)


def section_phase(res: Resultat) -> str:
    p = res.exp.tag
    fir, mssa = (res.phases[m] for m in PHASE_METHODES)
    return f"""
## Phase : imposee, FIR et M-SSA

Les deux methodes de `compute_one_cycle_compare_methods.py` appliquees a cette
simulation, ou la phase imposee est **connue exactement** :
$\\varphi(t) = 2\\pi f_0 t - \\pi/2$ (la transformee de Hilbert d'un sinus vaut
moins un cosinus). Ce n'est donc pas une comparaison de deux estimateurs entre
eux, mais de chacun a la verite.

| | |
|---|---|
| {img(f"{p}_s9_pouls.png")} | {img(f"{p}_s9_phase.png")} |

A gauche, le pouls extrait par chaque methode, ramene a l'ecart-type du signal
impose. A droite, les phases en dents de scie. Les deux courbes mesurees se
superposent a l'imposee.

{img(f"{p}_s9_erreur_phase.png", "90%")}

L'ecart a la phase imposee, sur l'enregistrement entier. Les bandes grisees sont
les {EDGE_FRAC:.0%} de bord que la production exclut de ses statistiques (le FIR
y a son transitoire, et la moyenne diagonale de la M-SSA y porte sur moins de
termes) ; les chiffres ci-dessous sont lus **au coeur** uniquement.

{tableau_phase(res)}

**Au coeur, les deux methodes retrouvent la phase imposee.** Le FIR est a
{deg(fir.rms_deg)} de RMS, la M-SSA a {deg(mssa.rms_deg)} : plusieurs ordres de
grandeur sous ce qui compterait pour un repliement, et sous la resolution
temporelle elle-meme (une image vaut {360 / (FS / F0):.0f}° de cycle). Sur un
signal sans bruit, strictement periodique et de frequence constante, aucune des
deux n'a de difficulte -- c'est le resultat attendu, et il vaut surtout comme
verification que les deux chaines sont correctement branchees.

**Toute la difference est aux bords.** Le FIR y monte a
{np.degrees(np.abs(fir.erreur).max()):.0f}° sur le dernier cycle
({deg(fir.rms_bords_deg)} de RMS sur les bandes grisees), la M-SSA reste a
{deg(mssa.rms_bords_deg)}. La raison n'est pas la transformee de Hilbert,
commune aux deux : la M-SSA rend un sinus exactement periodique, sur lequel la
Hilbert par FFT est exacte jusqu'aux extremites, tandis que `filtfilt` laisse un
transitoire qui brise cette periodicite. C'est ce que le
`EDGE_CYCLES` de `compute_one_cycle_compare_methods.py` retire avant de replier
-- et on voit ici pourquoi il le fait **pour les deux** methodes : ce qui est
retire au FIR par necessite ne coute rien a la M-SSA, mais comparer deux
methodes sur des jeux de frames differents serait pire.

La M-SSA n'a pas triche pour autant : la SVD de la matrice trajectoire empilee
({mssa.diag["H_shape"][0]} x
{mssa.diag["H_shape"][1]}) sort ses valeurs singulieres **par
paires** ({" / ".join(f"{v:.3f}" for v in
(res.phases["3a_mssa"].diag["sv"][:4] / res.phases["3a_mssa"].diag["sv"][0]))}),
signature d'un contenu oscillatoire ; le regroupement retient la premiere paire,
la seule dont le pic tombe a {F0:.0f} Hz, et rejette les suivantes, qui sont les
harmoniques. Les canaux debruites sont alors de rang 1 a
{res.phases["3a_mssa"].diag["pc1_var"]:.6f} pres : leur premiere composante
principale **est** l'oscillation partagee.
"""


def section(res: Resultat, numero: int) -> str:
    exp, p = res.exp, res.exp.tag
    ecart = 100 * (res.rapport_u1 / exp.prediction - 1)
    return f"""
# Experience {numero} -- {exp.titre.lower()}

Les deux gaussiennes sont identiques **sauf sur un point** :
{exp.ce_qui_change}.

- amplitude crete a crete : **{exp.amp_gauche:.0f} px** a gauche,
  **{exp.amp_droite:.0f} px** a droite
- intensite (pic) : **{exp.intensite_gauche:.2f}** a gauche,
  **{exp.intensite_droite:.2f}** a droite
- ecart-type : **{exp.sigma_gauche:.1f} px** a gauche,
  **{exp.sigma_droite:.1f} px** a droite

| image (echelle absolue) | coupe horizontale au centre |
|---|---|
| {img(f"{p}_s1_image.png")} | {img(f"{p}_s1_coupe.png")} |

| deplacements imposes | valeurs singulieres |
|---|---|
| {img(f"{p}_s1_deplacements.png")} | {img(f"{p}_s2_valeurs_singulieres.png")} |

## Les {N_VECTEURS} premieres composantes

{grille_composantes(res)}

::: footer
Trait vertical tirete sur les cartes : frontiere entre les deux moities de
l'image. Pointilles verticaux sur les spectres : harmoniques de $f_0$. Echelle
log, chaque spectre normalise par son maximum.
:::

## Les {N_MODES_VIDEO} premiers modes en video

Chaque video montre la reconstruction de **rang 1** du mode,
$\\sigma_k\\,u_k v_k^{{T}}$ remise en images : le terme que ce mode ajoute a
l'image, donc a la fois **ou** il agit et **comment** il varie dans le temps.
Somme des {N_MODES_VIDEO}, on retrouverait presque la simulation ; pris
separement, chacun est un etage du developpement.

{grille_videos(res)}

::: {{.source-note}}
{N_OSC_VIDEO} cycles a {FS:.0f} images/s, soit le temps reel. Echelle de
couleur divergente, **fixe sur toute la video** et propre a chaque mode : une
normalisation image par image annulerait la variation temporelle, qui est
justement ce qu'on veut voir. Le zoom est une repetition de pixels, sans
interpolation.
:::

## A qui appartient chaque composante

{img(f"{p}_s7_profil_colonnes.png", "80%")}

Norme de $u_k$ colonne par colonne, chaque courbe normalisee par son propre
maximum. Deux bosses = les deux gaussiennes contribuent, une seule = la
composante appartient a un seul oscillateur. Attention : c'est l'**aire** sous
la bosse qui vaut contribution, pas sa hauteur -- deux bosses de hauteurs
differentes mais de largeurs differentes aussi peuvent tres bien peser pareil,
et c'est precisement le cas de l'experience 3. Le chiffre a lire est le rapport
des normes ci-dessous.

{img(f"{p}_s6_superposition.png", "80%")}

$v_1$ est **une seule** forme d'onde -- corr($v_1$, signal impose) =
{res.corr_v1:.4f} -- et elle ne porte que la frequence et la phase communes.
Ce qui distingue les deux gaussiennes est dans **$u_1$** : le rapport de ses
normes entre les deux moities de l'image vaut **{res.rapport_u1:.2f}**, pour
**{exp.prediction:.2f}** attendu ({ecart:+.1f} %) -- {exp.justification}.
{section_phase(res)}"""


r1, r2, r3 = (RESULTATS[e.tag] for e in EXPERIENCES)

qmd = f"""<!--
Fragment inclus par svd-simple-simulation-two-gaussians.qmd.
Genere par {DOSSIER}/make_figures.py -- ne pas editer a la main, relancer le
script apres toute modification des figures.
Les chemins sont relatifs au dossier PARENT ({DOSSIER}/...), c'est-a-dire au
dossier ou vit le .qmd de la page.
-->

# Dispositif commun

Deux gaussiennes 2-D dans la meme image ({HAUTEUR}x{LARGEUR} px), centres
distants de {X_DROITE - X_GAUCHE:.0f} px -- soit {(X_DROITE - X_GAUCHE) / SIGMA:.1f}
fois l'ecart-type de reference, donc sans recouvrement mesurable : chaque moitie
de l'image appartient a une seule gaussienne, ce qui autorise a lire une carte
$u_k$ moitie par moitie.

Elles oscillent verticalement a la **meme frequence** et **en phase** dans les
trois experiences. Une seule chose change d'une gaussienne a l'autre a chaque
fois, et c'est ce qui donne son titre a l'experience.

- signal a **{F0:.0f} Hz**, echantillonnage a **{FS:.0f} Hz**,
  **{N_OSC} oscillations** ({T} images, {DUREE:.0f} s)
- **aucun bruit**, ni sur le deplacement ni sur l'image
- **{N_VECTEURS} vecteurs singuliers** affiches, bien au-dela du rang utile

::: {{.source-note}}
Formes d'onde tracees sur les {N_OSC_AFFICHEES} premieres oscillations : le
signal est stationnaire, elles valent pour les {N_OSC}. Les spectres, eux, sont
calcules sur l'enregistrement entier.
:::
{section(r1, 1)}
# Ou s'arrete la simulation, ou commence l'arrondi

Deux ruptures se lisent dans le tableau ci-dessus, et aucune des deux ne dit
quoi que ce soit sur la choroide -- ce sont des proprietes de la mesure. Elles
se retrouvent telles quelles dans les deux experiences suivantes.

- **$k \\leq {H_NYQUIST}$** : chaque composante est l'harmonique $k$ de $f_0$, a
  {F0:.0f}, 2, 3... Hz. **Au-dela**, l'harmonique depasse Nyquist
  ({0.5 * FS:.1f} Hz) et **se replie** : le pic redescend (7, 6, 5... Hz) au lieu
  de continuer a monter. Ce n'est pas une nouvelle structure, c'est la meme
  echelle repliee par l'echantillonnage a {FS:.0f} Hz -- $v_{{13}}$ tombe a 2 Hz
  et $v_{{14}}$ a {F0:.0f} Hz, la ou les harmoniques 13 et 14 se replient.
- **$k \\geq {r1.k_plancher}$** : plus de forme d'onde du tout. La concentration
  spectrale, qui vaut ~0.66 pour toutes les composantes precedentes (la valeur
  d'un ton pur), s'effondre a {r1.conc[r1.k_plancher][0]:.2f} ; les cartes ne
  sont plus que du grain et $\\sigma_k/\\sigma_0$ stagne a
  {r1.S[r1.k_plancher] / r1.S[0]:.0e}, l'epsilon du `float64`.

Le rang effectif est donc **{r1.k_plancher}**, et ce n'est pas la precision du
calcul qui le fixe. Le deplacement est periodique, et un cycle tombe sur un
nombre **entier** d'images ({FS:.0f} Hz / {F0:.0f} Hz = {N_PAR_CYCLE} images par
cycle) : la trace temporelle de chaque pixel est donc une suite
{N_PAR_CYCLE}-periodique, et ces suites forment un espace de dimension
exactement {N_PAR_CYCLE}. Le rang ne peut pas depasser {N_PAR_CYCLE}, quels que
soient le nombre de pixels, le nombre de cycles enregistres ({N_OSC} ici) ou la
non-linearite de l'objet -- et la SVD sature cette borne, exactement.

Attention a ne pas en conclure qu'il y a {N_PAR_CYCLE} harmoniques utiles :
au-dela de Nyquist les harmoniques repliees n'apportent pas de direction
nouvelle, elles retombent dans le plan deja occupe par leur alias. Ce qui compte
est la dimension totale, {N_PAR_CYCLE}, pas le decompte des harmoniques.

{N_PAR_CYCLE} est une **borne**, pas une valeur atteinte a tous les coups :
l'experience 2 s'arrete a {r2.k_plancher}. Son deplacement est plus petit
({r2.exp.amp_gauche:.0f} px contre {r1.exp.amp_gauche:.0f}), ses harmoniques
decroissent donc plus vite et passent sous l'epsilon machine avant d'avoir
epuise l'espace disponible. C'est le meme mecanisme que sur des donnees reelles,
a ceci pres que le plancher y est celui du **bruit** et non de l'arrondi -- et
qu'il arrive alors bien plus tot, apres deux ou trois composantes, pas douze.
{section(r2, 2)}{section(r3, 3)}
# Ce que les trois experiences disent ensemble

| experience | ce qui change | rapport des normes de $u_1$ | attendu |
|---|---|---|---|
| 1 | amplitude ({r1.exp.amp_gauche:.0f} / {r1.exp.amp_droite:.0f} px) | **{r1.rapport_u1:.2f}** | {r1.exp.prediction:.2f} |
| 2 | intensite ({r2.exp.intensite_gauche:.2f} / {r2.exp.intensite_droite:.2f}) | **{r2.rapport_u1:.2f}** | {r2.exp.prediction:.2f} |
| 3 | ecart-type ({r3.exp.sigma_gauche:.1f} / {r3.exp.sigma_droite:.1f} px) | **{r3.rapport_u1:.2f}** | {r3.exp.prediction:.2f} |

Dans les trois cas $v_1$ est la **meme** forme d'onde et ne distingue rien : la
frequence et la phase sont communes, et c'est tout ce qu'un vecteur temporel
peut porter quand les oscillateurs sont synchrones. Toute l'information qui
separe les deux gaussiennes est dans $u_1$.

Mais $u_1$ ne pese pas l'amplitude. Il pese le produit
**amplitude x intensite** : l'experience 2 le montre directement -- a mouvement
strictement identique, la gaussienne deux fois plus brillante occupe deux fois
plus de $u_1$. Lire un rapport d'amplitudes sur une carte $u_1$ n'est donc
legitime qu'entre regions de meme reflectivite. Dans un B-scan reel, ou la
reflectivite varie fortement d'une couche a l'autre, c'est une hypothese forte
et rarement verifiee.

La largeur, elle, ne compte pas : l'experience 3 donne un rapport de
{r3.rapport_u1:.2f} pour deux objets dont l'un est deux fois plus fin et couvre
quatre fois moins de pixels. La norme de $\\partial I/\\partial y$ vaut $\\pi/2$
pour toute gaussienne de pic unite : ce qu'un objet fin perd en surface, il le
regagne exactement en raideur de gradient. Un petit vaisseau bien contraste pese
donc autant qu'une large plage, a mouvement egal.

## La phase, elle, ne distingue rien du tout

| experience | RMS FIR (coeur) | RMS M-SSA (coeur) | RMS FIR (bords) | RMS M-SSA (bords) |
|---|---|---|---|---|
| 1 | {deg(r1.phases["1_fir"].rms_deg)} | {deg(r1.phases["3a_mssa"].rms_deg)} | {deg(r1.phases["1_fir"].rms_bords_deg)} | {deg(r1.phases["3a_mssa"].rms_bords_deg)} |
| 2 | {deg(r2.phases["1_fir"].rms_deg)} | {deg(r2.phases["3a_mssa"].rms_deg)} | {deg(r2.phases["1_fir"].rms_bords_deg)} | {deg(r2.phases["3a_mssa"].rms_bords_deg)} |
| 3 | {deg(r3.phases["1_fir"].rms_deg)} | {deg(r3.phases["3a_mssa"].rms_deg)} | {deg(r3.phases["1_fir"].rms_bords_deg)} | {deg(r3.phases["3a_mssa"].rms_bords_deg)} |

Les trois lignes sont identiques a deux centiemes de degre pres, et c'est
attendu : amplitude, intensite et largeur changent **$u_1$**, pas $v_1$. Or la
phase se lit sur $v_1$, qui est le meme ton pur a {F0:.0f} Hz dans les trois
cas. Autrement dit, tout ce que cette page a montre sur ce qui distingue deux
gaussiennes est invisible a une mesure de phase -- ce qui est la bonne nouvelle
pour l'extraction du pouls (la phase cardiaque ne depend pas de la reflectivite
locale) et la mauvaise pour l'amplitude (aucune phase ne dira jamais laquelle
des deux bouge le plus).

**Ce que ce test ne dit pas.** Le cas est facile : sans bruit, strictement
periodique, a frequence constante, et une SVD qui a deja separe les harmoniques
en composantes distinctes -- de sorte que le regroupement de la M-SSA n'a plus
grand-chose a trier. Les deux methodes y sont exactes, ce qui valide leur
implementation et isole leur seule difference reelle, le transitoire de bord.
Cela ne les departage pas sur des donnees reelles, ou la difficulte est ailleurs
: bruit, derive de la frequence cardiaque, non-stationnarite de l'amplitude
(cf. [la partie 3 de la page precedente](svd-simple-simulation.qmd)). C'est ce
que mesure `compute_one_cycle_compare_methods.py` sur la cohorte.

Les harmoniques, enfin, ne se repartissent pas comme le premier ordre : elles
appartiennent a l'oscillateur le plus non lineaire, c'est-a-dire celui dont le
deplacement est le plus grand **devant son propre $\\sigma$**. En experience 1
c'est la gaussienne de {r1.exp.amp_gauche:.0f} px ({r1.part[2]:.0%} de $u_2$) ;
en experience 3, a deplacement identique, c'est la gaussienne **fine**
({1 - r3.part[2]:.0%} de $u_2$ a droite) alors qu'elle partage $u_1$ a parts
egales avec l'autre.
"""

(SORTIE / "slides.qmd").write_text(qmd, encoding="utf-8")

readme = f"""# Figures -- SVD de deux gaussiennes en phase

Materiel pour la page `svd-simple-simulation-two-gaussians.qmd`. **Tout est
regenere par `python make_figures.py`** : ne pas retoucher les PNG a la main.

- `slides.qmd` : les {len(FICHIERS)} figures et {len(VIDEOS)} videos deja mises
  en page, incluses par la page (chemins relatifs au dossier parent).
- `recapitulatif.txt` : les chiffres a citer.

Les `.mp4` sont les {N_MODES_VIDEO} premiers modes de chaque experience, en
reconstruction de rang 1 ($\\sigma_k u_k v_k^T$), sur {N_OSC_VIDEO} cycles en
temps reel. Ils exigent **ffmpeg** (PATH, ou le paquet `imageio-ffmpeg`) et sont
declares dans `resources:` de `_quarto.yml` -- sans quoi Quarto ne les copierait
pas dans `_site`, les balises `<video>` brutes n'etant pas suivies par son
detecteur de ressources.

## Les trois experiences

Deux gaussiennes cote a cote, meme frequence ({F0:.0f} Hz), meme phase, aucun
bruit. Une seule chose change d'une gaussienne a l'autre a chaque fois :

| tag | ce qui change | gauche | droite | rapport de $u_1$ attendu |
|---|---|---|---|---|
| `e1` | amplitude | {EXPERIENCES[0].amp_gauche:.0f} px | {EXPERIENCES[0].amp_droite:.0f} px | {EXPERIENCES[0].prediction:.2f} |
| `e2` | intensite | {EXPERIENCES[1].intensite_gauche:.2f} | {EXPERIENCES[1].intensite_droite:.2f} | {EXPERIENCES[1].prediction:.2f} |
| `e3` | ecart-type | {EXPERIENCES[2].sigma_gauche:.1f} px | {EXPERIENCES[2].sigma_droite:.1f} px | {EXPERIENCES[2].prediction:.2f} |

## Nommage

`<tag>_s<section>_<contenu>.png` -- `U<k>` (carte spatiale), `V<k>` (forme
d'onde), `spectre_V<k>`, plus les panneaux de methodo, le spectre des valeurs
singulieres, les profils par colonne et la superposition.

## Parametres

Image {HAUTEUR}x{LARGEUR} px, centres distants de
{X_DROITE - X_GAUCHE:.0f} px, ecart-type de reference {SIGMA:.0f} px.
{F0:.0f} Hz echantillonne a {FS:.0f} Hz, {N_OSC} oscillations ({T} images,
{DUREE:.0f} s). Amplitude commune aux experiences 2 et 3 :
{AMP_COMMUNE:.0f} px crete a crete. {N_VECTEURS} vecteurs singuliers affiches.

```
{recap}
```
"""
(SORTIE / "README.md").write_text(readme, encoding="utf-8")

print(f"\n{len(FICHIERS)} figures + {len(VIDEOS)} videos "
      f"+ slides.qmd + README.md + recapitulatif.txt")
print(f"dans {SORTIE}")
