"""Page d'accueil : recalage initial (X/Y) d'une paire reference / point de temps.

Cette page couvre le RECALAGE INITIAL et son apercu par paire, puis la sauvegarde
de l'experience (parametres). Les etapes suivantes sont sur des pages dediees :
  - « Correction A-scan » : correction par A-scan sur la mediane (RPE) — apercu
    image par image + parametres de correction ;
  - « Video recalee »     : generation de la video recalee + apercu.

Les selections (patient/moment/oeil) et TOUS les parametres de recalage sont
partages entre les pages via ``st.session_state`` (cf. _registration_common).
"""

import dataclasses
import json
from datetime import datetime

import streamlit as st
import numpy as np

from _registration_common import (
    select_condition,
    read_median_params,
    build_reg_cfg,
    load_gray,
    make_overlay,
    apply_dx,
    overlay_mask,
    estimate_dx,
    register_y,
    segment_images,
    format_acq_time,
)

# =========================================================================== #
# Interface : parametres a DROITE, images a GAUCHE
# =========================================================================== #
st.set_page_config(page_title="Recalage initial", layout="wide")
col_image, col_params = st.columns([2, 1], gap="large")

with col_params:
    # 1) Donnees : patient / moment / oeil / replicat / point de temps.
    ctx = select_condition(with_time_point=True)
    best, chosen = ctx.best, ctx.chosen
    best_image_path, chosen_image_path = ctx.best_image_path, ctx.chosen_image_path

    # ------------------------------------------------------------------- #
    # 2) Recalage horizontal (X) : methode + parametres associes
    # ------------------------------------------------------------------- #
    st.header("Recalage horizontal (X)")
    x_method = st.selectbox(
        "Methode",
        options=["fullframe", "xcorr", "aucun"],
        key="w_x_method",
        help=(
            "fullframe = correlation de phase 2D plein champ "
            "(estimate_lateral_shift_fullframe) ; "
            "xcorr = correlation 1D des profils lateraux ; "
            "aucun = pas de recalage horizontal (dx = 0)."
        ),
    )

    x_params: dict = {}
    if x_method == "fullframe":
        # Defaut a 64 px (et non 16 comme en video) : deux points de temps
        # distincts peuvent deriver lateralement de bien plus qu'entre deux
        # frames consecutives.
        x_params["max_shift"] = st.number_input(
            "max_shift (px)", min_value=1, step=1, key="w_ff_max_shift",
            help="Decalage lateral maximal teste.",
        )
        with st.expander("Parametres avances (fullframe)"):
            x_params["max_vshift"] = st.number_input(
                "max_vshift (px)", min_value=1, step=1, key="w_ff_max_vshift",
                help="Bande verticale sommee autour du centre pour le profil X.",
            )
            x_params["downsample"] = st.number_input(
                "downsample (px)", min_value=64, step=64, key="w_ff_downsample",
                help="Cote de la grille de calcul FFT (carre).",
            )
            cc1, cc2 = st.columns(2)
            x_params["bp_lo"] = cc1.number_input(
                "bandpass bas", min_value=0.0, max_value=0.5,
                step=0.01, format="%.3f", key="w_ff_bp_lo",
            )
            x_params["bp_hi"] = cc2.number_input(
                "bandpass haut", min_value=0.01, max_value=1.0,
                step=0.01, format="%.3f", key="w_ff_bp_hi",
            )
    elif x_method == "xcorr":
        x_params["max_shift"] = st.number_input(
            "max_shift (px)", min_value=1, step=1, key="w_xc_max_shift",
            placeholder="None -> W // 4",
            help="Decalage maximal teste. Vide => None (W // 4 en interne).",
        )
        x_params["drop_edges"] = st.number_input(
            "drop_edges (px)", min_value=0, step=1, key="w_xc_drop_edges",
            help="Marge laterale ignoree de chaque cote du profil.",
        )

    # ------------------------------------------------------------------- #
    # 3) Recalage vertical (Y) : register_masks_by_displacement
    # ------------------------------------------------------------------- #
    st.header("Recalage vertical (Y)")
    y_enabled = st.checkbox(
        "Activer le recalage vertical (Y)", key="w_y_enabled",
        help=(
            "register_masks_by_displacement : segmentation de la choroide puis "
            "alignement des frontieres (membrane de Bruch)."
        ),
    )
    flatten = st.checkbox(
        "flatten (aplatir la membrane de Bruch)", key="w_flatten",
        disabled=not y_enabled,
    )

    # ------------------------------------------------------------------- #
    # 4) Options globales
    # ------------------------------------------------------------------- #
    st.header("Options")
    subpixel = st.checkbox(
        "subpixel", key="w_subpixel",
        help="Interpolation sous-pixel du pic (s'applique en X et en Y).",
    )
    show_mask = st.checkbox(
        "Afficher le masque (choroide) sur les images", value=True,
        help="Segmentation de la choroide superposee sur la reference et le point choisi.",
    )

    # ------------------------------------------------------------------- #
    # 5) Calcul du recalage
    # ------------------------------------------------------------------- #
    ref_gray = load_gray(best_image_path)
    mov_gray = load_gray(chosen_image_path)

    masks_display = (
        segment_images(str(best_image_path), str(chosen_image_path))
        if show_mask else None
    )

    try:
        dx = estimate_dx(x_method, ref_gray, mov_gray, x_params, subpixel)
    except Exception as e:  # ex. drop_edges trop grand pour la largeur de l'image
        st.error(f"Recalage horizontal impossible : {e}")
        st.stop()
    st.metric(
        "Decalage estime dx (px, resolution originale)",
        f"{dx:.0f}" if not subpixel else f"{dx:.3f}",
    )

    if x_method == "fullframe":
        crop_w = int(ref_gray.shape[1] * 3 / 4)
        ds = int(x_params["downsample"])
        optimal_ds = 1 << (crop_w - 1).bit_length()
        st.caption(
            f"FFT fullframe : grille {ds}×{ds} · granularite laterale = "
            f"crop_w/downsample = {crop_w}/{ds} = {crop_w / ds:.3f} px · "
            f"downsample optimal (FFT, ≤ 1 px) ≈ {optimal_ds}"
        )

    if y_enabled:
        try:
            reg_frames = register_y(
                str(best_image_path), str(chosen_image_path), dx, flatten, subpixel
            )
        except Exception as e:
            st.error(f"Recalage vertical impossible : {e}")
            st.stop()
        reg_overlay = make_overlay(reg_frames[0], reg_frames[1])
        reg_title = "Recalage X + Y"
        reg_caption = (
            f"dx = {dx:.3f} px  ·  flatten = {flatten}  |  "
            f"magenta : {best.oct_file_name} (reference)  ·  "
            f"vert : {chosen.oct_file_name} (recalee X+Y)"
        )
    else:
        reg_overlay = make_overlay(ref_gray, apply_dx(mov_gray, dx))
        reg_title = "Recalage X"
        reg_caption = (
            f"dx = {dx:.3f} px  |  magenta : {best.oct_file_name} (reference)  ·  "
            f"vert : {chosen.oct_file_name} (recalee X)"
        )

    # ------------------------------------------------------------------- #
    # 6) Parametres d'experience (src/pipeline_config.py) + sauvegarde
    #
    # Les parametres de correction A-scan (page dediee) sont lus depuis
    # session_state, de sorte que l'experience enregistree reste complete.
    # ------------------------------------------------------------------- #
    st.header("Experience")
    median_enabled = bool(st.session_state.get("w_median_enabled", False))
    median_params = read_median_params()
    reg_cfg = build_reg_cfg(
        x_method, y_enabled, flatten, subpixel, median_enabled, median_params
    )
    experiment = {
        "patient": ctx.patient_dir.name,
        "moment": ctx.moment,
        "eye": ctx.eye,
        "replicate": int(ctx.r),
        "reference_image": best.oct_file_name,
        "reference_quality": float(best.oct.quality),
        "moving_image": chosen.oct_file_name,
        "moving_quality": float(chosen.oct.quality),
        "moving_time": format_acq_time(chosen.acquisition_time),
        "x_method": x_method,
        "x_params": x_params,
        "lateral_dx_px": round(float(dx), 4),
        "y_enabled": bool(y_enabled),
        "flatten": bool(flatten and y_enabled),
        "subpixel": bool(subpixel),
        "median_enabled": bool(median_enabled),
        "median_params": median_params if median_enabled else {},
        "registration_config": dataclasses.asdict(reg_cfg),
    }
    if median_enabled:
        st.caption(
            "Correction A-scan ACTIVEE (reglee sur la page « Correction A-scan ») : "
            "incluse dans l'experience et la video recalee."
        )
    st.json(experiment, expanded=False)
    if st.button("Enregistrer les parametres d'experience"):
        out_dir = ctx.data_dir / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_path.write_text(
            json.dumps(experiment, indent=2, default=str), encoding="utf-8"
        )
        st.success(f"Parametres enregistres : {out_path}")

    st.caption(
        "Etapes suivantes (menu de gauche) : « Correction A-scan » puis "
        "« Video recalee »."
    )

# --------------------------------------------------------------------------- #
# Panneau de gauche : reference, point choisi, resultat recale (paire).
# --------------------------------------------------------------------------- #
with col_image:
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

    st.subheader(f"{reg_title} — reference (magenta) + image recalee (vert)")
    st.image(reg_overlay, caption=reg_caption, use_container_width=True)
