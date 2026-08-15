"""
simulate_svd_radial_gaussians.py

Simulation d'un anneau de gaussiennes (choroide synthetique) pour caracteriser
les limites de detectabilite de la chaine d'extraction du pouls du package
(``motion.pulsation``) : jitter de recalage residuel, bruit d'image, et leur
combinaison.

Geometrie (voir plan) :
  - Un disque imaginaire de rayon R0_UM=500um ; les gaussiennes (sigma tire
    entre 10 et 20um) sont placees SUR SA CIRCONFERENCE, a des angles fixes
    tires une fois (``ANATOMY_SEED``) et geles pour tout le balayage.
  - Le CERCLE COMPLET est rendu et mesure : la frame contient l'anneau entier,
    centre du disque au centre de l'image. (Une version anterieure ne gardait
    qu'un quadrant, l'anneau etant invariant par rotation ; on est revenu au
    champ plein -- le rendu par fenetres locales, voir ``render_cube``, en
    rend le cout inferieur a celui du quadrant d'avant.)
  - Le disque "respire" radialement : R(t) = R0 + (amplitude/2)*sin(phase),
    phase suivant un chirp 1.0 -> 1.3 Hz sur toute la duree de la video
    (meme formule que ``reveal_quarto_presentations/figures_gaussienne_svd/make_figures.py``).
  - Jitter de recalage NON corrige, injecte directement avant la mesure (pas
    de passage par le vrai moteur de recalage) : un decalage x commun a toute
    l'image par frame, un decalage y independant par frame ET par A-scan
    (colonne).
  - Bruit d'image CALIBRE SUR LES DONNEES par
    ``Astronauts/quantify_image_noise.py`` (medianes sur 102 cubes recales) :
    plancher additif + terme en R ("Poisson") + speckle multiplicatif residuel,
    plus un gain global par frame et une echelle de bruit propre a chaque frame
    -- le bruit n'est pas le meme d'une image a l'autre d'un meme cube. Le tout
    mis a l'echelle par un seul ``noise_level`` (0 = net, 1 = bruit MEDIAN
    MESURE, >1 = plus bruite que la realite). Voir le bloc de commentaires
    au-dessus de ``add_image_noise`` pour la decomposition et les valeurs.
    (Ce n'est plus le modele de ``reveal_quarto_presentations/
    figures_gaussienne_svd/make_figures.py``, reste sur l'ancien speckle a
    4 looks.)

Echelle de pixel et cadence : lues depuis un export XML Spectralis reel
(``XML_PATH``) via ``ocularrigidity.data.spectralis``, pas inventees.

Mesure : chaine complete du package, comme
``Astronauts/compute_one_cycle_pixel_svd.py`` --
``PixelTraceSource`` -> ``DecomposedTraceSource(method="svd")`` ->
``LombScargleRateEstimator`` -> ``OptimizedSpectralCombination`` ->
``HilbertPhaseEstimator``, le tout pilote par ``PulseExtractor``. Les frames
synthetiques sont injectees dans un faux "registered_video" minimal (pas de
``VideoRegistrator`` reel), puisque le but est de tester la robustesse de la
mesure elle-meme a un residu de recalage, pas la chaine de recalage.

Balayage en deux phases (voir plan) :
  - Phase A : 1D, un facteur a la fois (amplitude, bruit, jitter_x, jitter_y),
    pour les limites individuelles. Le bruit y est pousse HORS du domaine
    realiste, de l'image nette a 8x l'ecart-type mesure : on cherche ou la
    mesure casse, pas a rester plausible.
  - Phase B : grille combinee ciblee (jitter_x x jitter_y, a trois amplitudes
    representatives) avec un niveau de bruit TIRE AU HASARD a chaque run dans
    la distribution mesuree d'un cube a l'autre -- chaque run est un cube
    plausible tire au sort, la limite combinee obtenue vaut donc pour la
    cohorte et non pour un point de grille arbitraire.

Aucun cube video n'est sauvegarde : chaque run est entierement determine par
ses parametres + graine (anatomie fixe, seed de bruit/jitter par run), donc
regenerable a la demande pour la visualisation a posteriori. Sont sauvegardes
par run (``.npz``) : vecteurs singuliers gauche/droit (U, V) et valeurs
singulieres (S), diagnostics du taux/phase, verite terrain (deplacement
radial impose, frequence instantanee du chirp), plus une ligne recapitulative
dans ``summary.csv``.

Lancer :  python Astronauts/simulate_svd_radial_gaussians.py
"""

from __future__ import annotations

import csv
import itertools
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import filtfilt, firwin

from ocularrigidity.data.spectralis import SpectralisStudy
from ocularrigidity.motion.pulsation import (
    CardiacBand,
    DecompositionConfig,
    DecomposedTraceSource,
    HilbertPhaseConfig,
    HilbertPhaseEstimator,
    LombScargleConfig,
    LombScargleRateEstimator,
    OptimizedSpectralCombination,
    PixelTraceConfig,
    PixelTraceSource,
    PulseExtractor,
    SpectralCombinationConfig,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner

# --------------------------------------------------------------------------- #
# Calibration reelle (XML Spectralis) -- pas de constantes inventees
# --------------------------------------------------------------------------- #
XML_PATH = Path(
    "E:/SANSORI/01_210713001/210713001before_rigidity/210713001before_rigidity_OD/"
    "RawImages/581EC8B0.xml"
)


@dataclass(frozen=True)
class RealCalibration:
    scale_x_um: float  # lateral (A-scan), um/px
    scale_y_um: float  # axial (profondeur), um/px
    fs_hz: float
    n_frames: int
    duration_s: float


def load_real_calibration(xml_path: Path = XML_PATH) -> RealCalibration:
    study = SpectralisStudy.from_file(xml_path)
    oct_series = [
        s for s in study.series if s.oct is not None and s.acquisition_time is not None
    ]
    if not oct_series:
        raise ValueError(f"Aucune serie OCT horodatee dans {xml_path}")
    oct_series.sort(key=lambda s: s.acquisition_time.seconds_of_day)

    t = np.array([s.acquisition_time.seconds_of_day for s in oct_series], dtype=float)
    dt = np.diff(t)
    scale_x, scale_y = oct_series[0].lateral_resolution, oct_series[0].axial_resolution
    if scale_x is None or scale_y is None:
        raise ValueError(f"ScaleX/ScaleY absents dans {xml_path}")

    return RealCalibration(
        scale_x_um=scale_x * 1000.0,
        scale_y_um=scale_y * 1000.0,
        fs_hz=float(1.0 / np.median(dt)),
        n_frames=len(t),
        duration_s=float(t[-1] - t[0]),
    )


# --------------------------------------------------------------------------- #
# Anatomie synthetique : anneau de gaussiennes, fige pour tout le balayage
# --------------------------------------------------------------------------- #
R0_UM = 500.0
#: Nombre de gaussiennes sur la circonference COMPLETE (elles etaient deja
#: tirees sur 2*pi quand seul un quadrant etait rendu -- trois sur quatre
#: tombaient alors hors champ). Densite lineique = N_BLOBS / (2*pi*R0_UM),
#: soit un blob tous les ~17um d'arc a 180, contre ~26um a 120.
N_BLOBS = 180
ANATOMY_SEED = 0
SIGMA_UM_RANGE = (10.0, 20.0)


@dataclass(frozen=True)
class RingAnatomy:
    theta: np.ndarray  # (N_BLOBS,) radians, angle sur la circonference
    sigma_um: np.ndarray  # (N_BLOBS,) ecart-type physique (isotrope en um)


def build_ring_anatomy(
    n_blobs: int = N_BLOBS, seed: int = ANATOMY_SEED
) -> RingAnatomy:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, n_blobs)
    sigma_um = rng.uniform(*SIGMA_UM_RANGE, n_blobs)
    return RingAnatomy(theta=theta, sigma_um=sigma_um)


# --------------------------------------------------------------------------- #
# Pulsation radiale + chirp (meme formule que make_figures.py)
# --------------------------------------------------------------------------- #
F_DEB, F_FIN = 1.0, 1.3  # Hz
PHI0 = np.pi / 3


def chirp_phase(t: np.ndarray, duration_s: float) -> np.ndarray:
    return (
        2 * np.pi * (F_DEB * t + (F_FIN - F_DEB) * t**2 / (2 * duration_s)) + PHI0
    )


def chirp_freq(t: np.ndarray, duration_s: float) -> np.ndarray:
    return F_DEB + (F_FIN - F_DEB) * t / duration_s


def radial_displacement(
    t: np.ndarray, duration_s: float, amplitude_um: float
) -> np.ndarray:
    """R(t) - R0, crete a crete = amplitude_um."""
    return (amplitude_um / 2.0) * np.sin(chirp_phase(t, duration_s))


# --------------------------------------------------------------------------- #
# Jitter (residu de recalage non corrige), en pixels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JitterParams:
    sigma_x_px: float = 0.0  # commun a toute l'image, par frame
    sigma_y_px: float = 0.0  # independant par frame ET par A-scan (colonne)


def sample_jitter(
    n_frames: int, n_cols: int, params: JitterParams, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    jitter_x = (
        rng.normal(0.0, params.sigma_x_px, n_frames)
        if params.sigma_x_px > 0
        else np.zeros(n_frames)
    )
    jitter_y = (
        rng.normal(0.0, params.sigma_y_px, (n_frames, n_cols))
        if params.sigma_y_px > 0
        else np.zeros((n_frames, n_cols))
    )
    return jitter_x, jitter_y


# --------------------------------------------------------------------------- #
# Rendu analytique de l'image (pas de raster-puis-shift : le jitter et la
# pulsation sont appliques en decalant la coordonnee d'evaluation, donc un
# deplacement sous-pixel exact, sans flou d'interpolation parasite)
# --------------------------------------------------------------------------- #
BLOB_TAIL_UM = 3 * SIGMA_UM_RANGE[1] + 5.0  # 3-sigma + demi-amplitude max (10/2)
PIXEL_SAFETY_MARGIN = 10  # px, marge additionnelle pour le jitter max du balayage


#: Rayon de troncature du rendu, en ecarts-type : au-dela, la gaussienne vaut
#: moins de exp(-18) ~ 1.5e-8 du pic, cinq ordres de grandeur sous le plancher
#: de bruit (0.043) et sous la precision du float32 auquel le cube est stocke.
#: C'est ce qui autorise le rendu par fenetres locales de ``render_cube``.
BLOB_CUTOFF_SIGMA = 6.0


def ring_center_rc(shape: tuple[int, int]) -> tuple[float, float]:
    """Centre du disque, en (ligne, colonne) : le centre geometrique de la frame."""
    H, W = shape
    return (H - 1) / 2.0, (W - 1) / 2.0


def frame_shape(calib: RealCalibration) -> tuple[int, int]:
    """Champ plein : l'anneau COMPLET, centre du disque au centre de l'image.

    Demi-hauteur/demi-largeur = rayon du disque + queue des gaussiennes +
    marge pour le jitter, converties dans chaque echelle de pixel (les pixels
    ne sont pas carres : ~3.9um en axial contre ~11.4um en lateral, l'anneau
    circulaire en microns est donc une ellipse allongee dans la grille).
    """
    half_h_px = (
        int(np.ceil((R0_UM + BLOB_TAIL_UM) / calib.scale_y_um)) + PIXEL_SAFETY_MARGIN
    )
    half_w_px = (
        int(np.ceil((R0_UM + BLOB_TAIL_UM) / calib.scale_x_um)) + PIXEL_SAFETY_MARGIN
    )
    return 2 * half_h_px + 1, 2 * half_w_px + 1  # (H, W)


#: Ancien nom (epoque "un seul quadrant rendu"), garde parce que les consommateurs
#: hors de ce fichier l'appellent : ``Astronauts/quantify_image_noise.py`` et
#: ``notebook/simulate_svd_radial_gaussians_viewer.ipynb``.
crop_shape = frame_shape


def render_cube(
    anatomy: RingAnatomy,
    calib: RealCalibration,
    t: np.ndarray,
    duration_s: float,
    amplitude_um: float,
    jitter_x: np.ndarray,
    jitter_y: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rendu analytique du cube, par FENETRES LOCALES autour de chaque blob.

    Une gaussienne de sigma 10-20um ne couvre que ~2px en lateral et ~5px en
    axial : evaluer chacune sur la frame entiere (ce que faisait la version
    precedente) gaspillait plus de 99% du calcul. On n'evalue donc chaque blob
    que sur sa fenetre a +/- ``BLOB_CUTOFF_SIGMA`` sigma, et on somme les
    contributions par ``bincount``. A 6 sigma le resultat est identique a la
    frame pleine a ~2e-9 pres (verifie), pour ~25x moins de temps -- ce qui
    rend le champ plein moins cher que le quadrant de la version d'avant.

    La fenetre est definie en COORDONNEES DECALEES : le jitter x (commun a la
    frame) et le jitter y (propre a chaque colonne) deplacent le centre de la
    fenetre, ils ne sont pas appliques par interpolation apres rendu. Le
    deplacement reste donc sous-pixel exact, sans flou parasite.
    """
    H, W = shape
    cy, cx = ring_center_rc(shape)

    R = R0_UM + radial_displacement(t, duration_s, amplitude_um)  # (T,)
    cos_t, sin_t = np.cos(anatomy.theta), np.sin(anatomy.theta)  # (N,)
    sx_px = anatomy.sigma_um / calib.scale_x_um  # (N,)
    sy_px = anatomy.sigma_um / calib.scale_y_um  # (N,)

    # Demi-fenetre commune a tous les blobs (dimensionnee sur le plus large),
    # pour garder des tableaux rectangulaires donc vectorisables d'un coup.
    hw = int(np.ceil(BLOB_CUTOFF_SIGMA * sx_px.max())) + 1
    hh = int(np.ceil(BLOB_CUTOFF_SIGMA * sy_px.max())) + 1
    off_w = np.arange(-hw, hw + 1)  # (PW,)
    off_h = np.arange(-hh, hh + 1)  # (PH,)

    cube = np.empty((len(t), H, W), dtype=np.float32)
    for i in range(len(t)):
        # Centres, jitter inclus. En x le decalage est commun a la frame ; en y
        # il depend de la colonne, donc le centre de ligne est un (blob, colonne).
        c_center = cx + R[i] * cos_t / calib.scale_x_um + jitter_x[i]  # (N,)
        by = cy + R[i] * sin_t / calib.scale_y_um  # (N,)

        c_idx = np.rint(c_center)[:, None].astype(np.int64) + off_w  # (N, PW)
        c_in = (c_idx >= 0) & (c_idx < W)
        c_safe = np.clip(c_idx, 0, W - 1)  # hors champ -> masque par c_in
        ex = np.exp(-0.5 * ((c_safe - c_center[:, None]) / sx_px[:, None]) ** 2)

        r_center = by[:, None] + jitter_y[i][c_safe]  # (N, PW)
        r_idx = np.rint(r_center)[:, :, None].astype(np.int64) + off_h  # (N, PW, PH)
        r_in = (r_idx >= 0) & (r_idx < H)
        r_safe = np.clip(r_idx, 0, H - 1)
        ey = np.exp(-0.5 * ((r_safe - r_center[:, :, None]) / sy_px[:, None, None]) ** 2)

        keep = c_in[:, :, None] & r_in
        flat = r_safe * W + c_safe[:, :, None]
        cube[i] = np.bincount(
            flat[keep], weights=(ex[:, :, None] * ey)[keep], minlength=H * W
        ).reshape(H, W)
    return cube


# --------------------------------------------------------------------------- #
# Bruit d'image, CALIBRE SUR LES DONNEES par ``Astronauts/quantify_image_noise.py``
# (medianes sur 102 cubes recales de ``SegmentationVariations`` ; cf.
# ``simulation_output/image_noise/image_noise_summary.csv`` et la section
# "Statistiques de bruit mesurees" du notebook viewer).
#
# La variance temporelle d'un pixel de reflectivite R se decompose en trois
# termes, tous mesures :
#
#     Var[I] = FLOOR^2   +   K_POISSON * R   +   R^2 / N_LOOKS
#              (plancher)    (~ comptage)        (speckle multiplicatif)
#
# Le troisieme est le SEUL que cette simulation modelisait auparavant, avec un
# contraste de 0.50 (N_LOOKS = 4). La mesure dit qu'il ne porte quasiment RIEN
# de la variance (~0%, contre ~38% pour le plancher et ~63% pour le terme en R)
# et que son contraste vaut 0.09. C'est attendu : les B-scans exportes sont
# compresses logarithmiquement, le speckle multiplicatif n'y est plus
# multiplicatif. Le terme est garde -- il est mesure, pas suppose -- mais il ne
# domine plus.
#
# S'y ajoutent deux dispersions mesurees, absentes de l'ancien modele :
#
#   - un GAIN global par frame : l'image entiere respire de +/- 6% d'une frame
#     a l'autre. C'est exactement ce que ``PixelTraceConfig.normalize_intensity``
#     cherche a enlever en amont de la SVD, donc c'est un effet que la
#     simulation doit produire si on veut pouvoir juger cette normalisation.
#
#   - une ECHELLE DE BRUIT PROPRE A CHAQUE FRAME : le bruit n'est pas le meme
#     d'une image a l'autre d'un meme cube. Calibree sur le melange d'echelles
#     qu'implique l'exces de kurtosis des residus standardises mesures
#     (K = 2.3 -> sigma_log = sqrt(ln(1 + K/3) / 4) = 0.38, soit un ecart-type
#     relatif d'environ 40%). C'est un MAJORANT : les evenements de mouvement
#     residuel epaississent eux aussi les queues des residus. Seule l'ECHELLE
#     varie d'une frame a l'autre -- la mesure ne dit rien sur une variation de
#     la repartition entre les trois termes, on ne l'invente donc pas.
#
# ``noise_level`` reste un multiplicateur d'ECART-TYPE de bruit : les trois
# termes de variance sont en noise_level^2 et le gain par frame en noise_level,
# donc 0 = image nette, 1 = bruit MEDIAN MESURE, >1 = plus bruite que la
# realite. (Par construction, repasser ``quantify_image_noise.py`` sur un cube
# simule a noise_level = 1 doit redonner ~1 par les deux voies, speckle et
# plancher -- ce qui n'etait pas le cas avant cette calibration : les deux
# voies divergeaient d'un facteur ~14.)
# --------------------------------------------------------------------------- #
# Deux precautions dans le report des mesures vers ces constantes :
#
#  - Les trois termes par pixel sont lus sur la variante NORMALISEE du CSV, pas
#    sur la brute : la variance brute contient DEJA la fluctuation de gain, que
#    le modele reinjecte ensuite explicitement. Les calibrer sur la brute
#    compterait le gain deux fois. Recomposition : bruit par pixel 0.219 +
#    gain 0.060 -> 0.227 de rapport bruit/signal a l'intensite moyenne, contre
#    0.235 mesure sur la donnee brute -- 4% d'ecart, le gain global n'etant
#    qu'un modele approche de ce que la normalisation par frame enleve.
#
#  - Le plancher et le terme en R, qui ont une dimension d'intensite, sont
#    ancres sur la MEME reference : l'intensite moyenne de la ROI cote donnees,
#    celle du masque simule (R_mean = 0.293) cote simulation. Ancrer sur le 90e
#    centile donnerait un K_POISSON ~23% plus haut : l'intensite reelle etant
#    compressee logarithmiquement, la correspondance avec l'echelle simulee
#    n'est pas un simple facteur, et cet ecart est l'incertitude de conversion.
FOND = 0.15
N_LOOKS_REALISTIC = 125.0  # contraste de speckle mesure : 1/sqrt(125) = 0.089
BRUIT_ADDITIF_REALISTIC = 0.043  # plancher = 14.5% de l'intensite moyenne
K_POISSON_REALISTIC = 0.0055  # terme en R (0.0068 avec l'ancrage p90)
GAIN_SIGMA_REALISTIC = 0.060  # gain global par frame, ecart-type relatif
NOISE_FRAME_SIGMA_LOG = 0.38  # dispersion de l'echelle de bruit d'une frame a l'autre


def frame_noise_scales(
    n_frames: int, noise_level: float, rng: np.random.Generator
) -> np.ndarray:
    """Echelle de bruit propre a chaque frame, d'esperance quadratique unite.

    Tirage log-normal renormalise (``E[s^2] = 1`` exactement, d'ou le
    ``- sigma^2``) : la dispersion d'une frame a l'autre ne change donc PAS le
    niveau de bruit moyen du cube, elle ne fait que le repartir inegalement dans
    le temps. C'est bien l'effet a tester -- des frames franchement plus
    bruitees que les autres --, pas une facon detournee de bruiter davantage.
    """
    sigma = NOISE_FRAME_SIGMA_LOG
    return noise_level * np.exp(sigma * rng.normal(0.0, 1.0, n_frames) - sigma**2)


def add_image_noise(
    cube: np.ndarray, noise_level: float, rng: np.random.Generator
) -> np.ndarray:
    if noise_level <= 0:
        return cube

    n_frames = cube.shape[0]
    R = cube.astype(np.float64) + FOND
    s = frame_noise_scales(n_frames, noise_level, rng)[:, None, None]

    # 1. Speckle multiplicatif : L looks, contraste 1/sqrt(L). Marginal aux
    #    niveaux mesures, garde pour rester fidele a la decomposition.
    n_looks = N_LOOKS_REALISTIC / s**2
    noisy = R * rng.gamma(n_looks, 1.0 / n_looks, R.shape)

    # 2. Plancher additif + terme en R : les deux composantes dominantes. Un
    #    seul tirage gaussien, d'ecart-type sqrt(floor^2 + k*R) -- somme des
    #    variances, comme dans l'ajustement conjoint de la mesure.
    sd = s * np.sqrt(BRUIT_ADDITIF_REALISTIC**2 + K_POISSON_REALISTIC * R)
    noisy += rng.normal(0.0, 1.0, R.shape) * sd

    # 3. Gain global par frame : multiplie SIGNAL ET BRUIT, comme un gain de
    #    detecteur. Log-normal (et non gaussien) pour rester positif meme aux
    #    forts noise_level du balayage.
    sigma_g = noise_level * GAIN_SIGMA_REALISTIC
    gain = np.exp(sigma_g * rng.normal(0.0, 1.0, n_frames) - sigma_g**2 / 2)
    noisy *= gain[:, None, None]

    return noisy.astype(np.float32)


# --------------------------------------------------------------------------- #
# Masque fixe (emprise constante dans le temps -> ROI deterministe, isole
# l'effet teste du bruit/jitter plutot que de le confondre avec une variation
# de masque qu'un vrai pipeline n'aurait pas dans ces conditions)
# --------------------------------------------------------------------------- #
def build_fixed_mask(shape: tuple[int, int], border_px: int = 3) -> np.ndarray:
    """Ellipse pleine inscrite dans la frame, centree sur le disque : elle
    contient l'anneau COMPLET et ne laisse dehors que les coins du champ, ou
    aucun blob ne tombe."""
    H, W = shape
    cy, cx = ring_center_rc(shape)
    ry, rx = (H - 1) / 2.0 - border_px, (W - 1) / 2.0 - border_px
    rows = np.arange(H, dtype=np.float64)[:, None]
    cols = np.arange(W, dtype=np.float64)[None, :]
    return (((rows - cy) / ry) ** 2 + ((cols - cx) / rx) ** 2) <= 1.0


# --------------------------------------------------------------------------- #
# Faux "registered_video" : duplique juste l'interface attendue par
# PixelTraceSource/VideoTimelineAligner, sans passer par VideoRegistrator --
# le jitter est un residu NON corrige injecte avant la mesure, pas quelque
# chose que le vrai moteur de recalage a eu la chance de corriger.
# --------------------------------------------------------------------------- #
@dataclass
class FakeRegisteredVideo:
    registered_frames: np.ndarray  # (T, H, W) float
    registered_masks: np.ndarray  # (T, H, W) bool
    skip_first_n_frames: int = 0
    drop_last_n_frames: int = 0


N_SVD_COMPONENTS = 100


# --------------------------------------------------------------------------- #
# Passe-bande avant Hilbert : ni PixelTraceSource (dont `filtered_signal`
# renvoie le signal brut) ni DecomposedTraceSource ne filtrent, donc la trace
# agregee arrive large bande sur la transformee de Hilbert, dont l'hypothese
# bande etroite tombe. On filtre donc la trace agregee -- pas les traces
# individuelles : c'est le seul signal dont on lit la phase, et la SVD continue
# de voir du brut.
# --------------------------------------------------------------------------- #
@dataclass
class BandpassedHilbertConfig(HilbertPhaseConfig):
    # Demi-largeur de bande en fraction de f0 : [(1-f)*f0, (1+f)*f0]. Centrer
    # sur la frequence estimee plutot que sur la CardiacBand (30-180 bpm) garde
    # la 2e harmonique du pouls hors bande, sinon elle fait osciller la phase
    # instantanee. None -> repli sur la CardiacBand.
    rel_bandwidth: Optional[float] = 0.4
    n_taps_cycles: float = 4.0  # longueur du FIR, en cycles cardiaques


class BandpassedHilbertPhaseEstimator(HilbertPhaseEstimator):
    """Hilbert precede d'un passe-bande FIR centre sur la frequence estimee."""

    def __init__(
        self,
        config: Optional[BandpassedHilbertConfig] = None,
        aggregator=None,
        band: Optional[CardiacBand] = None,
    ):
        super().__init__(config or BandpassedHilbertConfig(), aggregator)
        self.band = band

    def _band_hz(self, rate, fs: float) -> Optional[tuple[float, float]]:
        cfg = self.config
        f0 = rate.freq if rate is not None else None
        if cfg.rel_bandwidth and f0 is not None and np.isfinite(f0) and f0 > 0:
            lo, hi = (1 - cfg.rel_bandwidth) * f0, (1 + cfg.rel_bandwidth) * f0
        elif self.band is not None:
            lo, hi = self.band.effective_hz_range
        else:
            return None
        return max(lo, 1e-3), min(hi, 0.99 * 0.5 * fs)

    def phase_from_trace(self, trace, traces, rate):
        x = np.asarray(trace, dtype=float)
        valid = ~np.isnan(x)
        if not valid.any():
            return super().phase_from_trace(trace, traces, rate)

        band = self._band_hz(rate, traces.fs)
        if band is None or band[0] >= band[1]:
            return super().phase_from_trace(trace, traces, rate)

        # filtfilt ne tolere pas les NaN : on bouche les trous, on filtre, puis
        # on les remet -- super() les repercute dans `good`.
        idx = np.arange(len(x))
        filled = x.copy()
        filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
        filled = filled - filled.mean()

        nyq = 0.5 * traces.fs
        n_taps = int(round(self.config.n_taps_cycles * traces.fs / band[0])) | 1
        n_taps = min(n_taps, (len(filled) // 3) | 1)
        if n_taps < 5:  # enregistrement trop court pour ce passe-bande
            return super().phase_from_trace(trace, traces, rate)
        taps = firwin(n_taps, [band[0] / nyq, band[1] / nyq], pass_zero=False)
        y = filtfilt(taps, 1.0, filled, padlen=min(3 * n_taps, len(filled) - 1))

        y[~valid] = np.nan
        return super().phase_from_trace(y, traces, rate)


def build_extractor(
    fake_reg: FakeRegisteredVideo, aligner: VideoTimelineAligner, band: CardiacBand
) -> tuple[PulseExtractor, PixelTraceSource, OptimizedSpectralCombination]:
    pixel_source = PixelTraceSource(
        fake_reg,
        aligner,
        PixelTraceConfig(
            band=band, col_frac=None, row_frac=None, block_sizes=(1,), verbose=False
        ),
    )
    svd_source = DecomposedTraceSource(
        pixel_source,
        DecompositionConfig(
            method="svd", n_components=N_SVD_COMPONENTS, normalize=False, random_state=0
        ),
    )
    rate_estimator = LombScargleRateEstimator(
        LombScargleConfig(band=band, harmonic_correction=True, verbose=False)
    )
    aggregator = OptimizedSpectralCombination(SpectralCombinationConfig())
    phase_estimator = BandpassedHilbertPhaseEstimator(
        BandpassedHilbertConfig(rel_bandwidth=0.4), aggregator=aggregator, band=band
    )
    extractor = PulseExtractor(
        trace_source=svd_source,
        phase_estimator=phase_estimator,
        rate_estimator=rate_estimator,
        registered_video=fake_reg,
        aligner=aligner,
    )
    return extractor, pixel_source, aggregator


# --------------------------------------------------------------------------- #
# Un run : simule le cube, le mesure via le package, sauvegarde diagnostics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunParams:
    phase: str  # "A_amplitude" / "A_noise" / "A_jitter_x" / "A_jitter_y" / "B_combined"
    tag: str  # nom de fichier, unique
    amplitude_um: float
    jitter_x_px: float
    jitter_y_px: float
    noise_level: float
    seed: int


SUMMARY_FIELDS = [
    "phase",
    "tag",
    "amplitude_um",
    "jitter_x_px",
    "jitter_y_px",
    "noise_level",
    "seed",
    "rate_freq_hz",
    "confidence",
    "freq_rel_err",
    "corr_combined_vs_truth",
    "n_selected",
    "objective",
    "npz_path",
]


def append_summary(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one(
    anatomy: RingAnatomy,
    calib: RealCalibration,
    params: RunParams,
    out_dir: Path,
    summary_csv: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{params.tag}.npz"

    n_frames = calib.n_frames
    t = np.arange(n_frames) / calib.fs_hz
    duration_s = float(t[-1])
    shape = frame_shape(calib)

    rng = np.random.default_rng(params.seed)
    jitter_x, jitter_y = sample_jitter(
        n_frames,
        shape[1],
        JitterParams(params.jitter_x_px, params.jitter_y_px),
        rng,
    )

    try:
        cube = render_cube(
            anatomy, calib, t, duration_s, params.amplitude_um, jitter_x, jitter_y, shape
        )
        cube = add_image_noise(cube, params.noise_level, rng)
        mask2d = build_fixed_mask(shape)
        masks = np.tile(mask2d, (n_frames, 1, 1))

        fake_reg = FakeRegisteredVideo(registered_frames=cube, registered_masks=masks)
        timestamps_us = (t * 1e6).astype(np.int64)
        aligner = VideoTimelineAligner(fake_reg, timestamps_us)
        band = CardiacBand(bpm_range=(30.0, 180.0))

        extractor, _pixel_source, aggregator = build_extractor(fake_reg, aligner, band)

        traces = extractor.traces
        rate = extractor.rate
        combined_uniform = aggregator.aggregate(traces, rate)
        phase_estimator = extractor.phase_estimator
        phase_u, good_u = phase_estimator.phase_from_trace(combined_uniform, traces, rate)
        extractor._phase = phase_estimator.build_track(phase_u, good_u, traces, rate)
        phase = extractor.phase

        S = np.linalg.norm(traces.values, axis=0)
        U = traces.values / np.where(S > 0, S, 1.0)
        V = traces.mixing

        r_true_uniform = radial_displacement(
            traces.uniform_time, duration_s, params.amplitude_um
        )
        f_true_uniform = chirp_freq(traces.uniform_time, duration_s)

        kept = traces.kept_mask
        if kept.sum() > 1:
            corr = float(
                np.corrcoef(combined_uniform[kept], r_true_uniform[kept])[0, 1]
            )
        else:
            corr = float("nan")
        f_true_mean = float(np.mean(f_true_uniform))
        freq_rel_err = float(abs(rate.freq - f_true_mean) / f_true_mean)

        last = aggregator.last_result
        np.savez_compressed(
            npz_path,
            phase=params.phase,
            tag=params.tag,
            amplitude_um=params.amplitude_um,
            jitter_x_px=params.jitter_x_px,
            jitter_y_px=params.jitter_y_px,
            noise_level=params.noise_level,
            seed=params.seed,
            scale_x_um=calib.scale_x_um,
            scale_y_um=calib.scale_y_um,
            fs_hz=calib.fs_hz,
            U=U.astype(np.float32),
            S=S.astype(np.float32),
            V=V.astype(np.float32),
            uniform_time=traces.uniform_time.astype(np.float32),
            kept_mask=kept,
            r_true_uniform=r_true_uniform.astype(np.float32),
            f_true_uniform=f_true_uniform.astype(np.float32),
            combined_uniform=combined_uniform.astype(np.float32),
            rate_freq=rate.freq,
            rate_confidence=rate.confidence,
            rate_freqs=np.asarray(rate.diagnostics["freqs"], dtype=np.float32),
            rate_power=np.asarray(rate.diagnostics["power"], dtype=np.float32),
            rate_peak_freq=np.asarray(rate.diagnostics["peak_freq"], dtype=np.float32),
            rate_concentration=np.asarray(
                rate.diagnostics["concentration"], dtype=np.float32
            ),
            selected_indices=last.selected_indices,
            weights=last.weights,
            objective=last.objective,
            phase_uniform=phase.phase_uniform.astype(np.float32),
            good_uniform=phase.good_uniform,
            inst_bpm=phase.inst_bpm.astype(np.float32),
            corr_combined_vs_truth=corr,
            freq_rel_err=freq_rel_err,
            notes=np.array(extractor.notes, dtype=object),
        )

        row = {
            "phase": params.phase,
            "tag": params.tag,
            "amplitude_um": params.amplitude_um,
            "jitter_x_px": params.jitter_x_px,
            "jitter_y_px": params.jitter_y_px,
            "noise_level": params.noise_level,
            "seed": params.seed,
            "rate_freq_hz": rate.freq * 60.0,
            "confidence": rate.confidence,
            "freq_rel_err": freq_rel_err,
            "corr_combined_vs_truth": corr,
            "n_selected": int(last.selected_indices.size),
            "objective": last.objective,
            "npz_path": str(npz_path),
        }
        append_summary(summary_csv, row)
        print(
            f"  {params.tag}: {rate.freq * 60.0:.1f} bpm, conf={rate.confidence}, "
            f"corr={corr:.3f}, freq_err={freq_rel_err:.1%}"
        )

    except Exception as exc:  # noqa: BLE001 -- expected past the detection limit
        # Le .npz de diagnostic est lui-meme faillible (fichier verrouille par un
        # kernel Jupyter qui l'a ouvert, disque plein...). Un balayage de plusieurs
        # heures ne doit pas mourir la-dessus : on note l'echec dans summary.csv,
        # qui est de toute facon la source de verite du balayage, et on continue.
        try:
            np.savez_compressed(
                npz_path,
                phase=params.phase,
                tag=params.tag,
                amplitude_um=params.amplitude_um,
                jitter_x_px=params.jitter_x_px,
                jitter_y_px=params.jitter_y_px,
                noise_level=params.noise_level,
                seed=params.seed,
                error=str(exc),
            )
        except Exception as save_exc:  # noqa: BLE001
            print(f"  [avert] diagnostic non ecrit pour {params.tag} : {save_exc}")
        row = {
            "phase": params.phase,
            "tag": params.tag,
            "amplitude_um": params.amplitude_um,
            "jitter_x_px": params.jitter_x_px,
            "jitter_y_px": params.jitter_y_px,
            "noise_level": params.noise_level,
            "seed": params.seed,
            "rate_freq_hz": "",
            "confidence": "failed",
            "freq_rel_err": "",
            "corr_combined_vs_truth": "",
            "n_selected": "",
            "objective": "",
            "npz_path": str(npz_path),
        }
        append_summary(summary_csv, row)
        print(f"  [FAILED] {params.tag}: {exc}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Plan de balayage
# --------------------------------------------------------------------------- #
#: Sortie sur E: et non a cote du code : un balayage complet pese ~9 Go (chaque
#: .npz porte V, la carte spatiale, qui suit la taille de la ROI -- passee de
#: ~7000 a ~27000 pixels avec le cercle complet), et C: n'a pas cette place.
#: E: est deja le disque des donnees SANSORI.
OUT_ROOT = Path("E:/SANSORI_simulation_output") / "svd_radial_gaussians"
SUMMARY_CSV = OUT_ROOT / "summary.csv"

#: Replicats par point de grille en phase A. A parametres fixes, la seule chose
#: qui change d'un replicat a l'autre est la graine (bruit + jitter) : 10
#: tirages donnent une mediane et un ecart interquartile lisibles par point,
#: au lieu des 3 valeurs d'avant dont on ne pouvait pas dire grand-chose.
N_REPEATS = 10

AMPLITUDE_GRID_UM = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#: Phase A -- le bruit est balaye BIEN AU-DELA du realisme, de l'image
#: parfaitement nette (0) a 8x l'ecart-type mesure : le but d'un balayage 1D est
#: de trouver ou la mesure casse, pas de rester dans le plausible. La zone
#: plausible (cf. NOISE_LEVEL_CUBE_SIGMA_LOG : ~0.6 a 1.6) est echantillonnee
#: plus finement, puisque c'est la que se joue la lecture des vraies donnees.
NOISE_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
JITTER_X_GRID_PX = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
JITTER_Y_GRID_PX = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

TYPICAL_AMPLITUDE_UM = 6.0
REALISTIC_NOISE_LEVEL = 1.0  # = bruit median mesure (cf. add_image_noise)

# Grille combinee (phase B), resserree autour des limites individuelles
# attendues -- a ajuster une fois la phase A regardee.
COMBINED_JITTER_X_GRID_PX = [0.0, 1.0, 2.0]
COMBINED_JITTER_Y_GRID_PX = [0.0, 1.0, 2.0]
COMBINED_AMPLITUDE_GRID_UM = [1.0, 2.0, 3.0, 6.0, 8.0]  # un cas difficile, un cas facile

#: Phase B -- le bruit n'est PLUS une dimension de grille : il est TIRE AU
#: HASARD a chaque run dans la distribution mesuree d'un cube a l'autre,
#: log-normale de mediane 1 (= bruit median mesure) et d'ecart-type relatif 25%
#: (dispersion inter-cube de sigma_bruit/mu dans image_noise_summary.csv). Un
#: run de phase B est donc "un cube plausible tire au sort", pas un point de
#: grille : c'est ce qu'on veut pour une limite combinee qui doit valoir sur la
#: cohorte, et le niveau tire est enregistre (colonne ``noise_level``) donc
#: reste exploitable comme regresseur a l'analyse.
NOISE_LEVEL_CUBE_SIGMA_LOG = 0.25
#: Le bruit etant tire au sort, chaque case de la grille est un echantillon de
#: cubes et non une condition unique : il en faut assez pour que le niveau de
#: bruit reste exploitable comme regresseur A L'INTERIEUR d'une case, pas
#: seulement en moyenne. 15 tirages couvrent l'essentiel de la log-normale.
COMBINED_N_REPEATS = 15


def sample_noise_level(rng: np.random.Generator) -> float:
    """``noise_level`` d'un cube, tire dans la distribution mesuree."""
    return float(np.exp(NOISE_LEVEL_CUBE_SIGMA_LOG * rng.normal()))


_seed_counter = itertools.count(1000)


def run_phase_a(anatomy: RingAnatomy, calib: RealCalibration) -> None:
    print("=== Phase A : balayages 1D ===")

    print("-- amplitude --")
    out_dir = OUT_ROOT / "sweep_amplitude"
    for amplitude_um in AMPLITUDE_GRID_UM:
        for rep in range(N_REPEATS):
            tag = f"amplitude_{amplitude_um:05.2f}um_rep{rep}"
            params = RunParams(
                phase="A_amplitude",
                tag=tag,
                amplitude_um=amplitude_um,
                jitter_x_px=0.0,
                jitter_y_px=0.0,
                noise_level=0.0,
                seed=next(_seed_counter),
            )
            run_one(anatomy, calib, params, out_dir, SUMMARY_CSV)

    print("-- bruit d'image --")
    out_dir = OUT_ROOT / "sweep_noise"
    for noise_level in NOISE_GRID:
        for rep in range(N_REPEATS):
            tag = f"noise_{noise_level:04.2f}_rep{rep}"
            params = RunParams(
                phase="A_noise",
                tag=tag,
                amplitude_um=TYPICAL_AMPLITUDE_UM,
                jitter_x_px=0.0,
                jitter_y_px=0.0,
                noise_level=noise_level,
                seed=next(_seed_counter),
            )
            run_one(anatomy, calib, params, out_dir, SUMMARY_CSV)

    print("-- jitter x (commun a l'image) --")
    out_dir = OUT_ROOT / "sweep_jitter_x"
    for jitter_x_px in JITTER_X_GRID_PX:
        for rep in range(N_REPEATS):
            tag = f"jitterx_{jitter_x_px:04.2f}px_rep{rep}"
            params = RunParams(
                phase="A_jitter_x",
                tag=tag,
                amplitude_um=TYPICAL_AMPLITUDE_UM,
                jitter_x_px=jitter_x_px,
                jitter_y_px=0.0,
                noise_level=REALISTIC_NOISE_LEVEL,
                seed=next(_seed_counter),
            )
            run_one(anatomy, calib, params, out_dir, SUMMARY_CSV)

    print("-- jitter y (independant par A-scan) --")
    out_dir = OUT_ROOT / "sweep_jitter_y"
    for jitter_y_px in JITTER_Y_GRID_PX:
        for rep in range(N_REPEATS):
            tag = f"jittery_{jitter_y_px:04.2f}px_rep{rep}"
            params = RunParams(
                phase="A_jitter_y",
                tag=tag,
                amplitude_um=TYPICAL_AMPLITUDE_UM,
                jitter_x_px=0.0,
                jitter_y_px=jitter_y_px,
                noise_level=REALISTIC_NOISE_LEVEL,
                seed=next(_seed_counter),
            )
            run_one(anatomy, calib, params, out_dir, SUMMARY_CSV)


def run_phase_b(anatomy: RingAnatomy, calib: RealCalibration) -> None:
    print("=== Phase B : grille combinee (jitter_x x jitter_y), bruit tire au sort ===")
    out_dir = OUT_ROOT / "sweep_combined"
    grid = itertools.product(
        COMBINED_AMPLITUDE_GRID_UM,
        COMBINED_JITTER_X_GRID_PX,
        COMBINED_JITTER_Y_GRID_PX,
    )
    for amplitude_um, jitter_x_px, jitter_y_px in grid:
        for rep in range(COMBINED_N_REPEATS):
            # Tirage du niveau de bruit sur un generateur dedie, seme par la
            # graine du run : le cube reste entierement reproductible depuis la
            # seule ligne de summary.csv, comme le reste du balayage.
            seed = next(_seed_counter)
            noise_level = sample_noise_level(np.random.default_rng(seed))
            tag = (
                f"combined_A{amplitude_um:04.1f}_Jx{jitter_x_px:04.2f}_"
                f"Jy{jitter_y_px:04.2f}_N{noise_level:04.2f}_rep{rep}"
            )
            params = RunParams(
                phase="B_combined",
                tag=tag,
                amplitude_um=amplitude_um,
                jitter_x_px=jitter_x_px,
                jitter_y_px=jitter_y_px,
                noise_level=noise_level,
                seed=seed,
            )
            run_one(anatomy, calib, params, out_dir, SUMMARY_CSV)


if __name__ == "__main__":
    calib = load_real_calibration()
    print(
        f"Calibration reelle ({XML_PATH.name}) : "
        f"scale_x={calib.scale_x_um:.2f}um/px scale_y={calib.scale_y_um:.2f}um/px "
        f"fs={calib.fs_hz:.2f}Hz n_frames={calib.n_frames} duration={calib.duration_s:.1f}s"
    )
    anatomy = build_ring_anatomy()

    run_phase_a(anatomy, calib)
    run_phase_b(anatomy, calib)

    print(f"\nTermine. Resume : {SUMMARY_CSV}")
