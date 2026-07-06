"""Page : apercu (avant / apres) de la 2e passe A-scan/RPE sur une paire.

Les parametres (``max_axial_shift``, ``axial_bandpass``, ...) se reglent sur la
page « Recalage » (partages via ``st.session_state``, source de verite =
``pipeline_config.RegistrationConfig``) ; cette page se contente de comparer le
recalage SANS puis AVEC la 2e passe, sur la meme paire (reference, point
choisi), avec le meme moteur (``VideoRegistrator``) que l'export video complet.
"""

import dataclasses

import numpy as np
import streamlit as st

from _registration_common import (
    select_condition,
    registration_config_from_state,
    register_pair,
    make_overlay,
)

st.set_page_config(page_title="Correction A-scan", layout="wide")
col_image, col_params = st.columns([2, 1], gap="large")

with col_params:
    # 1) Donnees (partagees avec la page « Recalage » via session_state).
    ctx = select_condition(with_time_point=True)
    best, chosen = ctx.best, ctx.chosen

    # 2) Parametres de recalage courants (regles sur la page « Recalage »).
    cfg = registration_config_from_state()
    st.header("Correction par A-scan (RPE)")
    st.caption(
        "Repris de la page « Recalage » (réglez-les là-bas) : "
        f"max_axial_shift = **{cfg.max_axial_shift} px** · "
        f"axial_bandpass = **{cfg.axial_bandpass}**."
    )
    if not cfg.axial_refinement:
        st.info(
            "axial_refinement est désactivée sur la page « Recalage ». L'aperçu "
            "ci-dessous force la comparaison ON/OFF ; l'export « Video recalee » "
            "produit toujours les deux variantes (boutons dédiés), indépendamment "
            "de ce réglage."
        )

    # ------------------------------------------------------------------- #
    # 3) Recalage de la paire, sans puis avec la 2e passe.
    # ------------------------------------------------------------------- #
    try:
        cfg_before = dataclasses.replace(cfg, axial_refinement=False)
        cfg_after = dataclasses.replace(cfg, axial_refinement=True)
        reg_before = register_pair(ctx.best_image_path, ctx.chosen_image_path, cfg_before)
        reg_after = register_pair(ctx.best_image_path, ctx.chosen_image_path, cfg_after)
    except Exception as e:  # noqa: BLE001
        st.error(f"Recalage impossible : {e}")
        st.stop()

    dy_median = reg_after.transform.get("dy_median")
    if dy_median is not None:
        dy_last = np.asarray(dy_median)[-1]
        st.metric(
            "Correction A-scan : |dy| médian (px)",
            f"{np.nanmedian(np.abs(dy_last)):.2f}",
        )

# --------------------------------------------------------------------------- #
# Panneau de gauche : comparaison avant / apres.
# --------------------------------------------------------------------------- #
with col_image:
    st.subheader("Sans correction A-scan")
    before_frames = np.asarray(reg_before.registered_frames)
    st.image(
        make_overlay(before_frames[0], before_frames[-1]),
        caption=(
            f"magenta : {best.oct_file_name} (reference)  ·  "
            f"vert : {chosen.oct_file_name} (recalage initial seul)"
        ),
        use_container_width=True,
    )

    st.subheader("Avec correction A-scan (2e passe)")
    after_frames = np.asarray(reg_after.registered_frames)
    st.image(
        make_overlay(after_frames[0], after_frames[-1]),
        caption=(
            f"magenta : {best.oct_file_name} (reference)  ·  "
            f"vert : {chosen.oct_file_name} (+ recalage A-scan)"
        ),
        use_container_width=True,
    )
