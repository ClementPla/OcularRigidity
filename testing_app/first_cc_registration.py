"""Page d'accueil : reglage de TOUS les parametres de recalage + apercu sur une paire.

Les widgets ci-dessous couvrent exactement les champs de
``ocularrigidity.pipeline_config.RegistrationConfig`` — la SEULE source de verite
pour les parametres de recalage de tout le pipeline SANSORI (cette app,
``Astronauts/register_files.py``, ...). Il n'y a plus de fichier d'experience par
condition : « Enregistrer » reecrit directement les valeurs par defaut de
``RegistrationConfig`` dans ``pipeline_config.py`` (cf.
``_registration_common.save_registration_config``) — aucun ``.json`` n'est ecrit.

L'apercu (paire reference / point choisi) utilise le meme moteur de recalage
(``VideoRegistrator``) que l'export video complet, donc ce qui est affiche ici
est representatif de ce que produira « Video recalee ».

Les etapes suivantes sont sur des pages dediees :
  - « Correction A-scan » : apercu avant/apres de la 2e passe (A-scan/RPE) ;
  - « Video recalee »     : generation de la video recalee complete.
"""

import dataclasses

import numpy as np
import streamlit as st

from _registration_common import (
    select_condition,
    registration_config_widgets,
    save_registration_config,
    register_pair,
    segment_images,
    make_overlay,
    overlay_mask,
    load_gray,
    format_acq_time,
)

st.set_page_config(page_title="Recalage", layout="wide")
col_image, col_params = st.columns([2, 1], gap="large")

with col_params:
    # 1) Donnees : patient / moment / oeil / replicat / point de temps.
    ctx = select_condition(with_time_point=True)
    best, chosen = ctx.best, ctx.chosen
    best_image_path, chosen_image_path = ctx.best_image_path, ctx.chosen_image_path

    # ------------------------------------------------------------------- #
    # 2) Parametres de recalage = champs de RegistrationConfig.
    # ------------------------------------------------------------------- #
    cfg = registration_config_widgets()

    show_mask = st.checkbox(
        "Afficher le masque (choroide) sur les images", value=True,
        help="Segmentation de la choroide superposee sur la reference et le point choisi.",
    )

    # ------------------------------------------------------------------- #
    # 3) Recalage de la paire (meme moteur que l'export video complet).
    # ------------------------------------------------------------------- #
    try:
        registrator = register_pair(best_image_path, chosen_image_path, cfg)
    except Exception as e:  # noqa: BLE001
        st.error(f"Recalage impossible : {e}")
        st.stop()

    reg_frames = np.asarray(registrator.registered_frames)
    reg_masks = np.asarray(registrator.registered_masks)
    reg_ref, reg_mov = reg_frames[0], reg_frames[-1]
    reg_overlay = make_overlay(reg_ref, reg_mov)

    dx = float(np.asarray(registrator.transform["dx"])[-1])
    st.metric(
        "Decalage lateral estime dx (px)",
        f"{dx:.3f}" if cfg.subpixel else f"{dx:.0f}",
    )
    if cfg.correct_axial:
        dy_col = np.asarray(registrator.transform["dy"])[-1]
        st.caption(
            f"Decalage vertical (Y) median |dy| = {np.nanmedian(np.abs(dy_col)):.2f} px "
            "(par colonne, membrane de Bruch)."
        )
    if cfg.axial_refinement and registrator.transform.get("dy_median") is not None:
        dy_med = np.asarray(registrator.transform["dy_median"])[-1]
        st.caption(
            f"Correction A-scan (2e passe) median |dy| = {np.nanmedian(np.abs(dy_med)):.2f} px."
        )

    # ------------------------------------------------------------------- #
    # 4) Enregistrement : ecrit DIRECTEMENT pipeline_config.py.
    # ------------------------------------------------------------------- #
    st.header("Enregistrer")
    st.caption(
        "Ecrit les valeurs ci-dessus dans RegistrationConfig, directement dans "
        "pipeline_config.py — aucun autre fichier n'est cree."
    )
    with st.expander("RegistrationConfig actuelle", expanded=False):
        st.json(dataclasses.asdict(cfg), expanded=False)
    if st.button("Enregistrer les parametres de recalage"):
        saved_path = save_registration_config(cfg)
        st.success(f"RegistrationConfig mise a jour : {saved_path}")

    st.caption(
        "Etapes suivantes (menu de gauche) : « Correction A-scan » puis « Video recalee »."
    )

# --------------------------------------------------------------------------- #
# Panneau de gauche : reference, point choisi, resultat recale (paire).
# --------------------------------------------------------------------------- #
with col_image:
    ref_gray = load_gray(best_image_path)
    mov_gray = load_gray(chosen_image_path)
    masks_display = segment_images(str(best_image_path), str(chosen_image_path)) if show_mask else None

    sub_best, sub_time = st.columns(2)
    mask_note = "  ·  masque choroide (rouge)" if masks_display is not None else ""

    with sub_best:
        st.subheader(f"Reference — qualite {best.oct.quality:g}")
        ref_disp = (
            overlay_mask(ref_gray, masks_display[0])
            if masks_display is not None else ref_gray.astype(np.uint8)
        )
        st.image(ref_disp, caption=f"{best.oct_file_name}{mask_note}",
                 use_container_width=True)

    with sub_time:
        st.subheader(f"Point de temps {format_acq_time(chosen.acquisition_time)}")
        mov_disp = (
            overlay_mask(mov_gray, masks_display[1])
            if masks_display is not None else mov_gray.astype(np.uint8)
        )
        st.image(mov_disp, caption=f"{chosen.oct_file_name}{mask_note}",
                 use_container_width=True)

    st.subheader("Recalage — reference (magenta) + image recalee (vert)")
    st.image(
        reg_overlay,
        caption=(
            f"dx = {dx:.3f} px  ·  Y = {'oui' if cfg.correct_axial else 'non'}"
            f"{' (flatten)' if (cfg.correct_axial and cfg.flatten_rpe) else ''}"
            f"{' + A-scan' if cfg.axial_refinement else ''}  |  "
            f"magenta : {best.oct_file_name} (reference)  ·  "
            f"vert : {chosen.oct_file_name} (recalee)"
        ),
        use_container_width=True,
    )
