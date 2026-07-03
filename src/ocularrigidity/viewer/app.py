"""Single-video Gradio UI for the cardiac-rigidity pipeline.

Drag in a raw OCT acquisition (a grayscale video file) plus its ``timestamp.txt``
and the app runs the full pipeline on that one video:

    raw frames ──segment──► masks ──register──► folded one-cycle
                                                     │
                          ┌──────────────────────────┤
                          ▼                           ▼
                   segment one-cycle           ONE-CYCLE video
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        MASK (overlay)        DISPLACEMENT (quiver gif)

The mask + displacement are computed on the *folded one-cycle* (matching the
cohort pipeline), so timestamps are required. Every hyperparameter is exposed
and defaults are pulled straight from :mod:`ocularrigidity.pipeline_config`, so
the UI and the batch scripts stay in sync.

Launch with::

    python -m ocularrigidity.viewer.app
"""

from __future__ import annotations

import dataclasses
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import numpy as np

from ocularrigidity.data.compression import cube_to_mp4_fastest, read_gray
from ocularrigidity.data.io import save_mask
from ocularrigidity.motion.pulsation import (
    NCycleConfig,
    PulseExtractionConfig,
    run_cardiac_pipeline,
)
from ocularrigidity.pipeline_config import PULSATION, REGISTRATION, SEGMENTATION, DELTA_Y
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.viewer.gif import render_mask_quiver

# Output labels used by the CheckboxGroup and the dispatch below.
OUT_MASK = "Mask (overlay)"
OUT_ONE_CYCLE = "One-cycle video"
OUT_DISPLACEMENT = "Displacement (quiver gif)"


@lru_cache(maxsize=1)
def _model():
    """Choroid segmentation model (loaded once, kept on GPU)."""
    return get_choroid_segmentation_model().cuda()


# --- small rendering helpers -------------------------------------------------


def _to_uint8(cube: np.ndarray) -> np.ndarray:
    """Clip + cast a (possibly float) frame stack to display-ready uint8."""
    if cube.dtype == np.uint8:
        return cube
    return np.clip(np.nan_to_num(cube), 0, 255).astype(np.uint8)


def _write_mp4(
    frames: np.ndarray, path: str, fps: int, *, quality: int = 8, preset: str | None = None
) -> str:
    """Write a (T, H, W) gray or (T, H, W, 3) RGB stack to a browser-playable mp4.

    ``preset`` is forwarded to libx264 (e.g. ``"ultrafast"`` for fast, larger
    files when browsing); ``quality`` trades size for fidelity (0–10).
    """
    frames = _to_uint8(frames)
    if frames.ndim == 3:
        # gray (T, H, W) -> RGB: repeat the new trailing channel axis, NOT width.
        frames = np.repeat(frames[..., None], 3, axis=-1)
    imageio.mimwrite(
        path,
        list(frames),
        fps=fps,
        codec="libx264",
        quality=quality,
        macro_block_size=16,  # pad odd dims so libx264 doesn't choke
        output_params=["-preset", preset] if preset else None,
    )
    return path


def _overlay_video(
    frames: np.ndarray, masks: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Composite a red mask overlay onto grayscale frames -> (T, H, W, 3) uint8."""
    frames = _to_uint8(frames)
    out = np.repeat(frames[..., None], 3, axis=3).astype(np.float32)
    color = np.array([255, 85, 85], dtype=np.float32)
    m = masks.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _segment(
    cube: np.ndarray,
    batch_size: int,
    use_graphcut: bool,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Run the choroid segmentation model on a (T, H, W) uint8 cube -> bool masks.

    ``scale_factor`` downsamples (or upsamples) each frame before inference and
    the model output is interpolated back to the original size; useful to trade
    accuracy for speed/VRAM on large frames.
    """
    return infer(
        _model(),
        cube,
        batch_size=int(batch_size),
        scale_factor=float(scale_factor),
        return_logit=False,
        use_graphcut=use_graphcut,
        graphcut_kwargs=DELTA_Y.graphcut_kwargs if use_graphcut else None,
        use_amp=True,
        verbose=True,
    )


# --- main pipeline handler ---------------------------------------------------


def run_pipeline(
    video_path_text,
    ts_path_text,
    video_file,
    timestamps_file,
    outputs,
    max_frames,
    # registration
    skip_first,
    drop_last,
    flatten_rpe,
    correct_transversal,
    lateral_method,
    subpixel,
    reg_batch,
    # cardiac / pulsation
    ica_or_pca,
    phase_method,
    sigma_col,
    n_bins,
    n_cycle,
    fold_method,
    expected_bpm,
    bpm_band_frac,
    col_start,
    col_end,
    # segmentation
    seg_batch,
    use_graphcut,
    seg_scale_raw,
    seg_scale_oc,
    # displacement quiver
    q_stride,
    q_reference,
    q_arrow_scale,
    q_min_mag,
    q_cmap,
    q_lk_window,
    q_lk_levels,
    q_smooth_window,
    q_cyclic,
    q_only_y,
    q_side_by_side,
    q_fps,
    # output
    out_fps,
    progress=gr.Progress(),
):
    log: list[str] = []

    def say(msg: str, frac: float | None = None):
        log.append(msg)
        print(f"[ui] {msg}", flush=True)  # mirror to the launching terminal
        if frac is not None:
            progress(frac, desc=msg)
        return "\n".join(log)

    def _resolve(path_text, uploaded, what):
        """Prefer a typed server-side path; fall back to a browser upload."""
        if path_text and str(path_text).strip():
            p = str(path_text).strip()
            if not Path(p).exists():
                raise FileNotFoundError(f"{what} path does not exist: {p}")
            return p
        if uploaded is not None:
            return uploaded.name if hasattr(uploaded, "name") else uploaded
        raise ValueError(
            f"No {what} provided. Type a server-side path (recommended for "
            f"large files) or drag a file in."
        )

    try:
        want = set(outputs or [])
        if not want:
            return None, None, None, "Select at least one output to generate."

        video_path = _resolve(video_path_text, video_file, "video")
        ts_path = _resolve(ts_path_text, timestamps_file, "timestamps")
        say(f"Video: {video_path}")
        say(f"Timestamps: {ts_path}")

        workdir = Path(tempfile.mkdtemp(prefix="ocular_ui_"))
        say(f"Working dir: {workdir}", 0.02)

        # 1. Read the raw acquisition into a (T, H, W) uint8 cube.
        say("Reading video…", 0.05)
        cube = read_gray(video_path)
        if cube.ndim != 3:
            return None, None, None, f"Expected a grayscale video; got shape {cube.shape}."
        say(f"Loaded raw cube: {cube.shape} ({cube.dtype})")

        # Optional truncation for quick tests — keep timestamps in lockstep.
        ts_lines = Path(ts_path).read_text().splitlines()
        if max_frames and int(max_frames) > 0:
            n = int(max_frames)
            cube = cube[:n]
            ts_lines = ts_lines[:n]
            say(f"Truncated to first {len(cube)} frames for this run.")
        ts_staged = workdir / "timestamp.txt"
        ts_staged.write_text("\n".join(ts_lines) + "\n")

        # 2. Segment the raw frames -> masks needed for registration.
        say("Segmenting raw frames…", 0.10)
        raw_masks = _segment(cube, seg_batch, use_graphcut, seg_scale_raw)
        say(f"Raw masks: {raw_masks.shape}")

        # 3. Stage into the layout RegisteredVideo / run_cardiac_pipeline expect:
        #    <data>/sample/cube.mp4  and  <masks>/sample/mask.npz
        say("Staging + encoding for registration…", 0.30)
        data_root = workdir / "data"
        masks_root = workdir / "masks"
        (data_root / "sample").mkdir(parents=True, exist_ok=True)
        (masks_root / "sample").mkdir(parents=True, exist_ok=True)
        cube_to_mp4_fastest(cube, str(data_root / "sample" / "cube.mp4"), fps=int(out_fps), cq=18)
        save_mask(raw_masks, masks_root / "sample" / "mask.npz")

        # 4. Register + fold into the one-cycle video.
        say(f"Registering + folding ({ica_or_pca}/{phase_method})…", 0.45)
        bpm = float(expected_bpm) if expected_bpm and float(expected_bpm) > 0 else None
        col_slice = slice(int(col_start), int(col_end)) if int(col_end) > 0 else None
        result = run_cardiac_pipeline(
            video_relpath="sample",
            root_masks=str(masks_root),
            root_data=str(data_root),
            timestamps_path=str(ts_staged),
            config=PulseExtractionConfig(
                sigma_col=float(sigma_col),
                col_slice=col_slice,
                expected_bpm=bpm,
                expected_bpm_band_frac=float(bpm_band_frac),
                ICA_or_PCA=ica_or_pca,
                verbose=True,
            ),
            fold_config=NCycleConfig(
                n_bins=int(n_bins),
                n_cycle=int(n_cycle),
                fold_method=fold_method,
                phase_method=phase_method,
                verbose=True,
            ),
            registration_config=dataclasses.replace(
                REGISTRATION,
                skip_first_n_frames=int(skip_first),
                drop_last_n_frames=int(drop_last),
                flatten_rpe=bool(flatten_rpe),
                correct_transversal=bool(correct_transversal),
                lateral_method=lateral_method,
                subpixel=bool(subpixel),
                use_encoded_video=True,
                batch_size=int(reg_batch),
            ),
            compute_n_cycle_video=True,
            cache_dir=None,
            verbose=True,
        )
        one_cycle = _to_uint8(result.cycles)
        say(
            f"One-cycle: {one_cycle.shape}  |  "
            f"BPM≈{result.cardiac_freq * 60:.1f} (conf={result.confidence})"
        )

        mask_out = one_cycle_out = disp_out = None

        # 5a. One-cycle video output.
        if OUT_ONE_CYCLE in want:
            say("Writing one-cycle video…", 0.70)
            one_cycle_out = _write_mp4(
                one_cycle, str(workdir / "one_cycle.mp4"), fps=int(out_fps)
            )

        # 5b/c. Mask + displacement both run on the folded one-cycle.
        if want & {OUT_MASK, OUT_DISPLACEMENT}:
            say("Segmenting the one-cycle…", 0.78)
            oc_masks = _segment(one_cycle, seg_batch, use_graphcut, seg_scale_oc)

            if OUT_MASK in want:
                say("Rendering mask overlay…", 0.85)
                mask_out = _write_mp4(
                    _overlay_video(one_cycle, oc_masks),
                    str(workdir / "mask_overlay.mp4"),
                    fps=int(out_fps),
                )

            if OUT_DISPLACEMENT in want:
                say("Rendering displacement quiver…", 0.92)
                disp_out = str(workdir / "displacement.gif")
                render_mask_quiver(
                    one_cycle,
                    oc_masks.astype(np.uint8),
                    disp_out,
                    side_by_side=bool(q_side_by_side),
                    fps=int(q_fps),
                    stride=int(q_stride),
                    reference=int(q_reference),
                    arrow_scale=float(q_arrow_scale),
                    min_magnitude=float(q_min_mag),
                    arrow_cmap=q_cmap,
                    lk_window=int(q_lk_window),
                    lk_levels=int(q_lk_levels),
                    smooth_window=int(q_smooth_window),
                    cyclic=bool(q_cyclic),
                    only_y=bool(q_only_y),
                )

        say("Done.", 1.0)
        return one_cycle_out, mask_out, disp_out, "\n".join(log)

    except Exception as e:  # surface the full traceback in the UI log + terminal
        tb = traceback.format_exc()
        print(tb, flush=True)
        return None, None, None, "\n".join(log) + f"\n\nERROR: {e}\n\n{tb}"


# --- UI ----------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Ocular Rigidity — single video") as demo:
        gr.Markdown(
            "## Ocular Rigidity — single-video pipeline\n"
            "Drop a **raw OCT acquisition** and its **timestamp.txt**, pick the "
            "output(s), then **Run**. Mask & displacement are computed on the "
            "folded one-cycle. Defaults come from `pipeline_config`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "**Recommended for large files:** paste a server-side path "
                    "(e.g. on the mounted share) — no browser upload. "
                    "Drag-and-drop below only works well for small clips."
                )
                video_path_text = gr.Textbox(
                    label="Raw video path (server-side)",
                    placeholder="/mnt/smb/.../cube.mp4",
                )
                ts_path_text = gr.Textbox(
                    label="Timestamps path (server-side)",
                    placeholder="/mnt/smb/.../timestamp.txt",
                )
                video_file = gr.File(
                    label="…or drag a video (small clips only)",
                    file_types=["video", ".mp4", ".mkv", ".avi", ".mov"],
                )
                timestamps_file = gr.File(
                    label="…or drag timestamps (.txt)", file_types=[".txt"]
                )
                outputs = gr.CheckboxGroup(
                    [OUT_MASK, OUT_ONE_CYCLE, OUT_DISPLACEMENT],
                    value=[OUT_ONE_CYCLE, OUT_DISPLACEMENT],
                    label="Generate",
                )
                max_frames = gr.Number(
                    value=0,
                    precision=0,
                    label="Max frames (0 = all; truncate for a quick test)",
                )
                run_btn = gr.Button("Run", variant="primary")

                with gr.Accordion("Registration", open=False):
                    skip_first = gr.Number(value=REGISTRATION.skip_first_n_frames, precision=0, label="skip_first_n_frames")
                    drop_last = gr.Number(value=REGISTRATION.drop_last_n_frames, precision=0, label="drop_last_n_frames")
                    lateral_method = gr.Dropdown(["fullframe", "xcorr", "both"], value=REGISTRATION.lateral_method, label="lateral_method")
                    flatten_rpe = gr.Checkbox(value=REGISTRATION.flatten_rpe, label="flatten_rpe")
                    correct_transversal = gr.Checkbox(value=REGISTRATION.correct_transversal, label="correct_transversal")
                    subpixel = gr.Checkbox(value=REGISTRATION.subpixel, label="subpixel")
                    reg_batch = gr.Number(value=REGISTRATION.batch_size, precision=0, label="batch_size")

                with gr.Accordion("Cardiac / pulsation", open=False):
                    ica_or_pca = gr.Dropdown(["pca", "ica"], value=PULSATION.methods[0], label="ICA_or_PCA")
                    phase_method = gr.Dropdown(["peak_locked", "iq"], value=PULSATION.phase_methods[0], label="phase_method_for_fold")
                    fold_method = gr.Dropdown(["median", "mean"], value=PULSATION.one_cycle_fold_method, label="one_cycle_fold_method")
                    n_cycle = gr.Number(value=PULSATION.n_cycle, precision=0, label="n_cycle")
                    n_bins = gr.Number(value=PULSATION.n_bins, precision=0, label="n_bins")
                    sigma_col = gr.Number(value=PULSATION.sigma_col, label="sigma_col")
                    expected_bpm = gr.Number(value=0, label="expected_bpm (HR; 0 = auto)")
                    bpm_band_frac = gr.Number(value=PULSATION.expected_bpm_band_frac, label="expected_bpm_band_frac")
                    col_start = gr.Number(value=PULSATION.col_slice.start, precision=0, label="col_slice start")
                    col_end = gr.Number(value=PULSATION.col_slice.stop, precision=0, label="col_slice stop (0 = none)")

                with gr.Accordion("Segmentation", open=False):
                    seg_batch = gr.Number(value=SEGMENTATION.batch_size, precision=0, label="batch_size")
                    use_graphcut = gr.Checkbox(value=True, label="use_graphcut")
                    seg_scale_raw = gr.Number(value=1.0, label="scale_factor (raw video)")
                    seg_scale_oc = gr.Number(value=1.0, label="scale_factor (one-cycle)")

                with gr.Accordion("Displacement (quiver)", open=False):
                    q_stride = gr.Number(value=8, precision=0, label="stride")
                    q_reference = gr.Number(value=0, precision=0, label="reference frame")
                    q_arrow_scale = gr.Number(value=20.0, label="arrow_scale")
                    q_min_mag = gr.Number(value=0.05, label="min_magnitude (px)")
                    q_cmap = gr.Dropdown(["viridis", "plasma", "magma", "jet"], value="viridis", label="arrow_cmap")
                    q_lk_window = gr.Number(value=21, precision=0, label="lk_window")
                    q_lk_levels = gr.Number(value=3, precision=0, label="lk_levels")
                    q_smooth_window = gr.Number(value=4, precision=0, label="smooth_window")
                    q_fps = gr.Number(value=10, precision=0, label="gif fps")
                    q_cyclic = gr.Checkbox(value=True, label="cyclic")
                    q_only_y = gr.Checkbox(value=False, label="only_y (vertical only)")
                    q_side_by_side = gr.Checkbox(value=False, label="side_by_side")

                with gr.Accordion("Output", open=False):
                    out_fps = gr.Number(value=PULSATION.output_fps, precision=0, label="video fps")

            with gr.Column(scale=2):
                one_cycle_video = gr.Video(label="One-cycle video")
                mask_video = gr.Video(label="Mask overlay")
                disp_image = gr.Image(label="Displacement (quiver gif)")
                log_box = gr.Textbox(label="Log", lines=14, max_lines=30)

        run_btn.click(
            run_pipeline,
            inputs=[
                video_path_text, ts_path_text,
                video_file, timestamps_file, outputs, max_frames,
                skip_first, drop_last, flatten_rpe, correct_transversal,
                lateral_method, subpixel, reg_batch,
                ica_or_pca, phase_method, sigma_col, n_bins, n_cycle,
                fold_method, expected_bpm, bpm_band_frac, col_start, col_end,
                seg_batch, use_graphcut, seg_scale_raw, seg_scale_oc,
                q_stride, q_reference, q_arrow_scale, q_min_mag, q_cmap,
                q_lk_window, q_lk_levels, q_smooth_window, q_cyclic,
                q_only_y, q_side_by_side, q_fps,
                out_fps,
            ],
            outputs=[one_cycle_video, mask_video, disp_image, log_box],
        )

    return demo


def main():
    # show_error surfaces exceptions in the browser; allowed_paths lets Gradio
    # serve the mp4/gif we write under the system temp dir.
    build_demo().queue().launch(
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )


if __name__ == "__main__":
    main()
