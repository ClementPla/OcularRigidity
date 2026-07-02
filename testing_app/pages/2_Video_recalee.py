"""Page : generation de la/les video(s) recalee(s) de toute la condition + apercu.

Recale TOUTES les images de la condition (ordre des horodatages du XML) avec les
parametres regles sur les pages precedentes (« Recalage initial » + « Correction
A-scan »), repris depuis ``st.session_state``, et enregistre la sortie dans
``RawImages/registered/``.

Deux variantes, enregistrees sous des noms distincts (elles ne s'ecrasent pas) :
  - SANS correction A-scan : ``registered_video.mp4`` ;
  - AVEC correction A-scan  : ``registered_video_ascan.mp4`` (recalage de chaque
    A-scan sur la mediane, avec les parametres reglés dans l'app).

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
    read_median_params,
    load_ascan_params,
    ascan_params_path,
    build_reg_cfg,
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
    # 2) Parametres repris des pages precedentes (session_state)
    # ------------------------------------------------------------------- #
    x_method = st.session_state.get("w_x_method", "fullframe")
    y_enabled = bool(st.session_state.get("w_y_enabled", True))
    flatten = bool(st.session_state.get("w_flatten", True))
    subpixel = bool(st.session_state.get("w_subpixel", True))

    # Parametres A-scan : CHARGES depuis le fichier enregistre sur la page
    # « Correction A-scan » (bouton « Enregistrer les paramètres de correction
    # A-scan »). Fallback sur la session si rien n'est encore enregistre.
    saved_ascan = load_ascan_params(ctx.data_dir)
    if saved_ascan:
        median_params = saved_ascan
        st.success(
            f"Paramètres A-scan chargés depuis {ascan_params_path(ctx.data_dir).name} "
            f"(enregistré {saved_ascan.get('saved', '?')})."
        )
    else:
        median_params = read_median_params()
        st.warning(
            "Aucun paramètre A-scan enregistré pour cette condition — valeurs de "
            "session utilisées. Enregistrez-les sur « Correction A-scan »."
        )

    # Deux configs : la base (sans A-scan) et la variante A-scan (median force ON,
    # avec les parametres A-scan enregistres/charges ci-dessus).
    reg_cfg_base = build_reg_cfg(
        x_method, y_enabled, flatten, subpixel, False, median_params
    )
    reg_cfg_ascan = build_reg_cfg(
        x_method, y_enabled, flatten, subpixel, True, median_params
    )

    prep = (
        " + ".join(
            [
                t
                for t, on in (
                    ("compensation", median_params["use_shadow"]),
                    ("LoG", median_params["use_log"]),
                )
                if on
            ]
        )
        or "aucun"
    )

    st.header("Parametres de recalage")
    st.caption(
        "Repris de « Recalage initial » et « Correction A-scan » (partagés via la "
        "session ; modifiez-les sur ces pages)."
    )
    st.write(
        f"- Horizontal (X) : **{x_method}**\n"
        f"- Vertical (Y) : **{'oui' if y_enabled else 'non'}"
        f"{' (flatten)' if (y_enabled and flatten) else ''}**\n"
        f"- Sous-pixel : **{'oui' if subpixel else 'non'}**\n"
        f"- Correction A-scan : max_vshift = **{median_params['max_vshift']} px** · "
        f"prétraitement = **{prep}**"
    )
    with st.expander("RegistrationConfig (base / A-scan)"):
        st.caption("Sans A-scan :")
        st.json(dataclasses.asdict(reg_cfg_base), expanded=False)
        st.caption("Avec A-scan :")
        st.json(dataclasses.asdict(reg_cfg_ascan), expanded=False)

    # ------------------------------------------------------------------- #
    # 3) Generation : deux boutons (sans / avec correction A-scan)
    # ------------------------------------------------------------------- #
    st.header("Génération")
    st.caption(
        f"Le skip/drop de la config ({reg_cfg_base.skip_first_n_frames}/"
        f"{reg_cfg_base.drop_last_n_frames}) est appliqué. Sorties dans "
        f"`{raw_dir.name}/registered/`."
    )
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**Sans A-scan** → `{BASE_VIDEO}`")
        if st.button("Enregistrer (sans A-scan)"):
            _run_export(raw_dir, reg_cfg_base, "", ctx)
    with b2:
        st.markdown(f"**Avec A-scan** → `{ASCAN_VIDEO}`")
        if st.button("Enregistrer (A-scan)"):
            _run_export(raw_dir, reg_cfg_ascan, ASCAN_SUFFIX, ctx)

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
