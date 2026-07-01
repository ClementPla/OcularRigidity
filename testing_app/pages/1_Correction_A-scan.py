"""Page : correction par A-scan sur la mediane (RPE), posterieure au recalage initial.

Apres le recalage initial (X/Y, page d'accueil), chaque A-scan est recale
verticalement sur la mediane du volume : compensation d'ombres (correct_shadow)
et/ou LoG (laplacian_of_gaussian) pour rehausser la RPE, puis correlation de
phase spectrale par colonne (registration/axial/median_registration).

Cette page regroupe :
  - les PARAMETRES de correction (compensation / LoG decouples, max_vshift, ...) ;
  - l'APERCU image par image de la correction sur la paire (reference, point
    choisi). La cible de l'apercu est la reference (la mediane n'existe qu'au
    niveau video, cf. page « Video recalee »).

Le recalage initial (X/Y) est repris depuis ``st.session_state`` (regle sur la
page d'accueil) ; les parametres de correction y sont aussi ecrits, donc la video
recalee et l'experience enregistree en tiennent compte.
"""

import streamlit as st
import numpy as np

from _registration_common import (
    select_condition,
    read_x_params,
    base_registration,
    rpe_enhance,
    prep_label,
    apply_ascan_vshift,
    estimate_chosen_ascan_dy,
    make_overlay,
    to_display,
    save_ascan_params,
    load_ascan_params,
)

st.set_page_config(page_title="Correction A-scan", layout="wide")
col_image, col_params = st.columns([2, 1], gap="large")

with col_params:
    # 1) Donnees (partagees avec la page d'accueil via session_state).
    ctx = select_condition(with_time_point=True)

    # Recalage initial (X/Y) repris de session_state (regle sur la page d'accueil).
    x_method = st.session_state.get("w_x_method", "fullframe")
    x_params = read_x_params(x_method)
    y_enabled = bool(st.session_state.get("w_y_enabled", True))
    flatten = bool(st.session_state.get("w_flatten", False))
    subpixel = bool(st.session_state.get("w_subpixel", False))
    st.caption(
        f"Recalage initial repris : X = {x_method} · Y = "
        f"{'oui' if y_enabled else 'non'}{' (flatten)' if (y_enabled and flatten) else ''} "
        f"· subpixel = {'oui' if subpixel else 'non'}  (réglé sur « Recalage initial »)."
    )

    # ------------------------------------------------------------------- #
    # 2) Parametres de la correction par A-scan (RPE)
    # ------------------------------------------------------------------- #
    st.header("Correction par A-scan (RPE)")
    median_enabled = st.checkbox(
        "Marquer la correction A-scan comme choisie (expérience)", key="w_median_enabled",
        help=(
            "Enregistre ce choix dans l'expérience (registration_config). La vidéo "
            "A-scan se génère via le bouton dédié « Enregistrer (A-scan) » de la page "
            "« Video recalee », qui produit toujours les deux variantes (sans / avec) "
            "pour comparaison — indépendamment de cette case. L'aperçu ci-dessous est "
            "toujours affiché pour régler les paramètres."
        ),
    )

    # Compensation d'ombres et LoG DECOUPLES : l'un, l'autre, les deux, ou aucun.
    cse1, cse2 = st.columns(2)
    use_shadow = cse1.checkbox(
        "Compensation d'ombres", key="w_median_use_shadow",
        help="correct_shadow (Girard 2011). Indépendant du filtre LoG.",
    )
    use_log = cse2.checkbox(
        "Filtre LoG", key="w_median_use_log",
        help="laplacian_of_gaussian. Indépendant de la compensation.",
    )
    if not use_shadow and not use_log:
        st.caption(
            "Aucun prétraitement : la corrélation de phase opère sur les pixels bruts."
        )
    max_vshift = st.number_input(
        "max_vshift (px)", min_value=1, step=1, key="w_median_max_vshift",
        help="Déplacement vertical maximal testé pour chaque A-scan.",
    )
    with st.expander("Paramètres avancés (compensation + LoG)"):
        cc1, cc2 = st.columns(2)
        shadow_n = cc1.number_input(
            "shadow n (exposant)", min_value=0.1, step=0.1, format="%.2f",
            key="w_median_shadow_n", help="Exposant I**n de correct_shadow.",
            disabled=not use_shadow,
        )
        shadow_a = cc2.number_input(
            "shadow a (échelle)", min_value=0.01, step=0.05, format="%.2f",
            key="w_median_shadow_a", help="Facteur d'échelle du dénominateur.",
            disabled=not use_shadow,
        )
        cc3, cc4 = st.columns(2)
        log_k = cc3.number_input(
            "LoG taille noyau", min_value=3, step=2, key="w_median_log_k",
            help="Côté du noyau LoG carré (impair).", disabled=not use_log,
        )
        log_sigma = cc4.number_input(
            "LoG sigma (lissage)", min_value=0.1, step=0.5, format="%.2f",
            key="w_median_log_sigma", help="Écart-type de la Gaussienne du LoG.",
            disabled=not use_log,
        )

    st.caption(
        "La vidéo A-scan (`registered_video_ascan.mp4`) se génère sur la page "
        "« Video recalee » à partir des paramètres ENREGISTRÉS ci-dessous."
    )

    # Enregistrement dedie : la page « Video recalee » charge ce fichier.
    ascan_params_to_save = {
        "enabled": bool(median_enabled),
        "use_shadow": bool(use_shadow),
        "use_log": bool(use_log),
        "max_vshift": int(max_vshift),
        "shadow_n": float(shadow_n),
        "shadow_a": float(shadow_a),
        "log_kernel_size": int(log_k),
        "log_sigma": float(log_sigma),
    }
    if st.button("Enregistrer les paramètres de correction A-scan"):
        saved_path = save_ascan_params(ctx.data_dir, ascan_params_to_save)
        st.success(f"Paramètres A-scan enregistrés : {saved_path}")
    existing = load_ascan_params(ctx.data_dir)
    if existing and existing.get("saved"):
        st.caption(f"Dernier enregistrement A-scan : {existing['saved']}.")
    else:
        st.caption("Aucun paramètre A-scan enregistré pour cette condition.")

    # ------------------------------------------------------------------- #
    # 3) Base = recalage initial X(+Y), puis correction A-scan (apercu)
    # ------------------------------------------------------------------- #
    try:
        base_ref, base_mov, dx = base_registration(
            ctx, x_method, x_params, y_enabled, flatten, subpixel
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Recalage initial impossible : {e}")
        st.stop()

    median_preview = None
    try:
        dy_col = estimate_chosen_ascan_dy(
            base_ref, base_mov, use_shadow, shadow_n, shadow_a,
            use_log, log_k, log_sigma, max_vshift, subpixel,
        )
        chosen_axial = apply_ascan_vshift(base_mov, dy_col)
        median_preview = {
            "ref_rpe": to_display(
                rpe_enhance(base_ref, use_shadow, shadow_n, shadow_a, use_log, log_k, log_sigma)),
            "mov_rpe": to_display(
                rpe_enhance(base_mov, use_shadow, shadow_n, shadow_a, use_log, log_k, log_sigma)),
            "overlay": make_overlay(base_ref, chosen_axial),
            "dy": dy_col,
            "label": prep_label(use_shadow, use_log),
        }
    except Exception as e:  # noqa: BLE001
        st.warning(f"Aperçu de la correction A-scan impossible : {e}")

# --------------------------------------------------------------------------- #
# Panneau de gauche : apercu image par image de la correction.
# --------------------------------------------------------------------------- #
with col_image:
    st.subheader("Correction par A-scan (RPE) — aperçu image par image")
    if median_preview is None:
        st.info("Aperçu indisponible pour ce réglage.")
    else:
        _plabel = median_preview["label"]
        st.caption(
            f"Prétraitement : {_plabel}, puis recalage de chaque A-scan. Cible de "
            "l'aperçu = image de référence ; la vidéo exportée aligne sur la médiane "
            "du volume."
        )
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.image(median_preview["ref_rpe"], caption=f"Référence : {_plabel}",
                     use_container_width=True)
        with mcol2:
            st.image(median_preview["mov_rpe"], caption=f"Point choisi : {_plabel}",
                     use_container_width=True)
        st.image(
            median_preview["overlay"],
            caption=(
                "Après correction par A-scan  ·  |dy| médian = "
                f"{np.median(np.abs(median_preview['dy'])):.2f} px  |  "
                "magenta : référence  ·  vert : point choisi (initial + A-scan)"
            ),
            use_container_width=True,
        )
