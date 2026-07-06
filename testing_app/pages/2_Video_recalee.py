"""Page : generation de la/les video(s) recalee(s) de toute la condition + apercu.

Recale TOUTES les images de la condition (ordre des horodatages du XML) avec la
``RegistrationConfig`` reglee sur la page « Recalage » (partagee via
``st.session_state`` — source de verite = ``pipeline_config.py``), et enregistre
la sortie dans ``RawImages/registered/``.

Deux variantes, enregistrees sous des noms distincts (elles ne s'ecrasent pas) :
  - SANS correction A-scan : ``registered_video.mp4`` (``axial_refinement=False``) ;
  - AVEC correction A-scan  : ``registered_video_ascan.mp4`` (``axial_refinement=True``).

Les deux vidéos sont affichées côte à côte si elles existent.
"""

import dataclasses

import streamlit as st

from ocularrigidity.scripts.registration.astronauts import (
    export_registered_video,
    DEFAULT_OUTPUT_SUBDIR,
)

from _registration_common import (
    select_condition,
    registration_config_from_state,
    get_seg_model,
    read_video_bytes,
    DEVICE,
)

# Noms de sortie des deux variantes (cf. export_registered_video(suffix=...)).
BASE_VIDEO = "registered_video.mp4"
ASCAN_SUFFIX = "_ascan"
ASCAN_VIDEO = f"registered_video{ASCAN_SUFFIX}.mp4"

st.set_page_config(page_title="Video recalee", layout="wide")
col_cfg, col_out = st.columns([1, 1], gap="large")


def _run_export(raw_dir, cfg, suffix, ctx):
    """Lance ``export_registered_video`` et affiche le resultat."""
    with st.spinner("Recalage de la video complete (segmentation + recalage)..."):
        try:
            result = export_registered_video(
                raw_dir,
                cfg,
                get_seg_model(),
                device=DEVICE,
                out_subdir=DEFAULT_OUTPUT_SUBDIR,
                suffix=suffix,
                overwrite=True,
                scale_factor=2.0,
                seg_batch_size=8,
                verbose=False,
                extra_meta={
                    "source": "pages/2_Video_recalee.py",
                    "variant": "ascan" if suffix else "base",
                    "patient": ctx.patient_dir.name,
                    "moment": ctx.moment,
                    "eye": ctx.eye,
                    "replicate": int(ctx.r),
                },
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Echec du recalage de la video : {e}")
            return
    if result["status"] == "ok":
        st.success(
            f"Video enregistree : {result['video']}  ·  "
            f"{result['n_frames']} frames @ {result['fps']:.1f} fps"
        )
    else:
        st.warning(f"Video non enregistree (raison : {result.get('reason')}).")


with col_cfg:
    # 1) Donnees (partagees via session_state) — pas de point de temps ici.
    ctx = select_condition(with_time_point=False)
    raw_dir = ctx.raw_dir

    # ------------------------------------------------------------------- #
    # 2) RegistrationConfig courante (reglee sur la page « Recalage »).
    # ------------------------------------------------------------------- #
    cfg = registration_config_from_state()
    # Deux configs derivees : la base (sans A-scan) et la variante A-scan
    # (axial_refinement force ON) — independamment du reglage courant, pour
    # toujours pouvoir generer/comparer les deux.
    cfg_base = dataclasses.replace(cfg, axial_refinement=False)
    cfg_ascan = dataclasses.replace(cfg, axial_refinement=True)

    st.header("Parametres de recalage")
    st.caption(
        "Repris de la page « Recalage » (partagés via la session ; modifiez-les "
        "sur cette page)."
    )
    st.write(
        f"- Horizontal (X) : **{cfg.lateral_method if cfg.correct_transversal else 'desactive'}**\n"
        f"- Vertical (Y) : **{'oui' if cfg.correct_axial else 'non'}"
        f"{' (flatten)' if (cfg.correct_axial and cfg.flatten_rpe) else ''}**\n"
        f"- Sous-pixel : **{'oui' if cfg.subpixel else 'non'}**\n"
        f"- Correction A-scan : max_axial_shift = **{cfg.max_axial_shift} px** · "
        f"axial_bandpass = **{cfg.axial_bandpass}**"
    )
    with st.expander("RegistrationConfig (base / A-scan)"):
        st.caption("Sans A-scan :")
        st.json(dataclasses.asdict(cfg_base), expanded=False)
        st.caption("Avec A-scan :")
        st.json(dataclasses.asdict(cfg_ascan), expanded=False)

    # ------------------------------------------------------------------- #
    # 3) Generation : deux boutons (sans / avec correction A-scan)
    # ------------------------------------------------------------------- #
    st.header("Génération")
    st.caption(
        f"Le skip/drop de la config ({cfg.skip_first_n_frames}/"
        f"{cfg.drop_last_n_frames}) est appliqué. Sorties dans "
        f"`{raw_dir.name}/registered/`."
    )
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**Sans A-scan** → `{BASE_VIDEO}`")
        if st.button("Enregistrer (sans A-scan)"):
            _run_export(raw_dir, cfg_base, "", ctx)
    with b2:
        st.markdown(f"**Avec A-scan** → `{ASCAN_VIDEO}`")
        if st.button("Enregistrer (A-scan)"):
            _run_export(raw_dir, cfg_ascan, ASCAN_SUFFIX, ctx)

# --------------------------------------------------------------------------- #
# Apercu : les deux videos recalees enregistrees (si presentes).
# --------------------------------------------------------------------------- #
with col_out:
    st.header("Aperçu des videos recalees")
    reg_dir = raw_dir / DEFAULT_OUTPUT_SUBDIR

    def _show(path, title):
        st.subheader(title)
        if path.exists():
            st.video(
                read_video_bytes(str(path), path.stat().st_mtime),
                loop=True,
                autoplay=True,
            )
            st.caption(str(path))
        else:
            st.info("Non enregistrée pour cette condition.")

    _show(reg_dir / BASE_VIDEO, "Sans correction A-scan")
    _show(reg_dir / ASCAN_VIDEO, "Avec correction A-scan")
