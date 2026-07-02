"""Page : creation de la/les video(s) one-cycle a partir de la/les video(s) recalee(s).

Une video one-cycle est la reconstruction d'un battement cardiaque moyen : les
frames recalees sont rangees par phase cardiaque puis moyennees (cf.
ocularrigidity.motion.one_cycle_export).

Deux variantes de video recalee peuvent exister dans ``RawImages/registered/``
(cf. page « Video recalee ») :
  - SANS correction A-scan : ``registered_video.mp4``       -> ``one_cycle.mp4`` ;
  - AVEC correction A-scan  : ``registered_video_ascan.mp4`` -> ``one_cycle_ascan.mp4``.

Cette page génère le one-cycle pour CHACUNE des variantes présentes et affiche
les deux résultats.
"""

import json

import streamlit as st

from ocularrigidity.pipeline_config import PULSATION
from ocularrigidity.registration.export import DEFAULT_OUTPUT_SUBDIR
from ocularrigidity.motion.one_cycle_export import export_one_cycle_video

from _registration_common import select_condition, read_video_bytes, DEVICE

st.set_page_config(page_title="Video one-cycle", layout="wide")

# (label, suffixe, video recalee d'entree, one-cycle de sortie).
VARIANTS = [
    ("Sans A-scan", "", "registered_video.mp4", "one_cycle.mp4"),
    ("Avec A-scan", "_ascan", "registered_video_ascan.mp4", "one_cycle_ascan.mp4"),
]

st.title("Video one-cycle")
col_cfg, col_out = st.columns([1, 1], gap="large")

with col_cfg:
    # 1) Donnees (partagees avec les autres pages via session_state).
    ctx = select_condition(with_time_point=False)
    registered_dir = ctx.raw_dir / DEFAULT_OUTPUT_SUBDIR

    # ------------------------------------------------------------------- #
    # 2) Parametres du one-cycle (defauts : pipeline_config.PULSATION)
    # ------------------------------------------------------------------- #
    st.header("Parametres one-cycle")
    expected_bpm = st.number_input(
        "FC attendue (BPM) — vide = detection auto", value=None,
        min_value=1.0, step=1.0, placeholder="auto",
        help="Ancre la recherche de frequence autour de la FC connue (si dispo).",
    )
    method = st.selectbox(
        "Decomposition", ["ICA", "PCA"], index=0,
        help="Separation des composantes temporelles avant scoring Lomb-Scargle.",
    )
    phase_method = st.selectbox(
        "Methode de phase (fold)", ["peak_locked", "iq"], index=0,
        help="peak_locked = phase 0 a chaque pic systolique ; iq = demodulation IQ.",
    )
    c1, c2 = st.columns(2)
    n_bins = c1.number_input(
        "n_bins (frames / cycle)", value=int(PULSATION.n_bins), min_value=2, step=1,
        help="Nombre de casiers de phase = nombre de frames du cycle reconstruit.",
    )
    n_cycle = c2.number_input(
        "n_cycle (cycles moyennes)", value=int(PULSATION.n_cycle), min_value=1, step=1,
        help="Nombre de cycles (tranches temporelles) moyennes et concatenes.",
    )
    c3, c4 = st.columns(2)
    fold_method = c3.selectbox(
        "Repliement (fold)", ["median", "mean"],
        index=0 if PULSATION.one_cycle_fold_method == "median" else 1,
    )
    output_fps = c4.number_input(
        "fps de sortie", value=int(PULSATION.output_fps), min_value=1, step=1,
        help="Cadence d'affichage du .mp4 one-cycle.",
    )

    with st.expander("Parametres avances"):
        sigma_col = st.number_input(
            "sigma_col (lissage spatial)", value=float(PULSATION.sigma_col),
            min_value=0.0, step=0.5,
        )
        cs1, cs2 = st.columns(2)
        col_lo = cs1.number_input(
            "col_slice debut", value=int(PULSATION.col_slice.start), min_value=0, step=1,
        )
        col_hi = cs2.number_input(
            "col_slice fin", value=int(PULSATION.col_slice.stop), min_value=1, step=1,
        )
        target_fpb = st.number_input(
            "target_frames_per_bin", value=25, min_value=1, step=1,
        )
        band_frac = st.number_input(
            "expected_bpm_band_frac", value=float(PULSATION.expected_bpm_band_frac),
            min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        )
        n_comp = st.number_input(
            "n_separable_components", value=16, min_value=2, step=1,
        )
        phase_cycles = st.number_input(
            "phase_smoother_cycles", value=2.0, min_value=0.5, step=0.5,
        )
        harmonic = st.checkbox("harmonic_correction", value=True)
        b1, b2 = st.columns(2)
        bpm_lo = b1.number_input("bpm_range min", value=30.0, min_value=1.0, step=1.0)
        bpm_hi = b2.number_input("bpm_range max", value=180.0, min_value=1.0, step=1.0)

    # Parametres communs aux deux variantes.
    oc_kwargs = dict(
        expected_bpm=(float(expected_bpm) if expected_bpm else None),
        bpm_range=(float(bpm_lo), float(bpm_hi)),
        expected_bpm_band_frac=float(band_frac),
        ICA_or_PCA=method,
        sigma_col=float(sigma_col),
        col_slice=(int(col_lo), int(col_hi)),
        n_separable_components=int(n_comp),
        phase_smoother_cycles=float(phase_cycles),
        harmonic_correction=bool(harmonic),
        phase_method_for_fold=phase_method,
        n_bins=int(n_bins),
        n_cycle=int(n_cycle),
        target_frames_per_bin=int(target_fpb),
        one_cycle_fold_method=fold_method,
        output_fps=int(output_fps),
    )

    st.caption(
        "La video recalee est deja rognee (skip/drop du recalage) : le one-cycle "
        "est calcule sans re-rogner ni re-recaler."
    )

    # ------------------------------------------------------------------- #
    # 3) Generation : un bouton par variante de video recalee presente
    # ------------------------------------------------------------------- #
    st.header("Génération")
    for label, suffix, in_video, out_video in VARIANTS:
        st.markdown(f"**{label}** · entrée `{in_video}` → sortie `{out_video}`")
        in_path = registered_dir / in_video
        if not in_path.exists():
            st.caption(
                f"Vidéo recalée absente (`{in_video}`). Génère-la sur « Video recalee »."
            )
            continue
        if st.button(f"Générer one-cycle ({label})", key=f"gen_oc_{suffix or 'base'}"):
            with st.spinner("Extraction du cycle cardiaque + repliement..."):
                try:
                    result = export_one_cycle_video(
                        registered_dir,
                        suffix=suffix,
                        overwrite=True,
                        device=DEVICE,
                        verbose=False,
                        extra_meta={
                            "variant": "ascan" if suffix else "base",
                            "patient": ctx.patient_dir.name,
                            "moment": ctx.moment,
                            "eye": ctx.eye,
                            "replicate": int(ctx.r),
                        },
                        **oc_kwargs,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Echec de la generation one-cycle ({label}) : {e}")
                else:
                    if result["status"] == "ok":
                        st.success(
                            f"{label} : {result['video']}  ·  "
                            f"{result['n_frames']} frames  ·  FC {result['cardiac_bpm']:.1f} BPM "
                            f"(confiance {result['confidence']})"
                        )
                    else:
                        st.warning(
                            f"{label} non généré (raison : {result.get('reason')})."
                        )

# --------------------------------------------------------------------------- #
# Resultat : affiche les deux videos one-cycle enregistrees (si presentes).
# --------------------------------------------------------------------------- #
with col_out:
    st.header("Resultat")
    for label, suffix, in_video, out_video in VARIANTS:
        st.subheader(label)
        out_path = registered_dir / out_video
        if out_path.exists():
            st.video(read_video_bytes(str(out_path), out_path.stat().st_mtime))
            st.caption(str(out_path))
            params_path = registered_dir / f"one_cycle_params{suffix}.json"
            if params_path.exists():
                with st.expander(f"Parametres du one-cycle ({label})"):
                    st.json(json.loads(params_path.read_text(encoding="utf-8")))
        else:
            st.info("Aucune video one-cycle enregistrée pour cette variante.")
