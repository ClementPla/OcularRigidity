import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import numpy as np
from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM
from ocularrigidity.pipeline_config import FRIEDENWALD, N_CYCLES
from ocularrigidity.data.io import load_mask_frames
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.friedenwald import K_from_deltaCT_mm, deltaCT_to_deltaV_uL
from ocularrigidity.motion.pipeline_results import (
    CardiacPipelineResults,
    peek_cardiac_freq,
)
from ocularrigidity.segmentation.closing_structures import trim_choroid
from ocularrigidity.thickness.delta import cycle_rates, measure_delta_ct_from_disp

#: Columns trimmed off each side of the mask before ΔCT is measured: the edges
#: of the B-scan are unreliable. Matches the notebooks and the report.
TRIM = 100


def load_pulsation_results(
    root_cardiac_pipeline: Path, overwrite=False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-case and per-cycle pulsation measures, cached.

    Reads the ``measures`` / ``deltaY.pkl`` outputs of the cardiac pipeline.
    Returns ``(cases, cycles)``: one row per video, and one row per
    (video, cardiac cycle).

    The first call walks every mask and re-measures ΔCT, which takes minutes; the
    result is pickled next to the pipeline outputs and reused. Pass
    ``overwrite=True`` after re-running the pipeline.
    """
    formatted_results_path = root_cardiac_pipeline / "formatted_results.pkl"
    formatted_cycles_path_cycles = (
        root_cardiac_pipeline / "formatted_results_cycles.pkl"
    )
    if (
        formatted_results_path.exists()
        and formatted_cycles_path_cycles.exists()
        and not overwrite
    ):
        return pd.read_pickle(formatted_results_path), pd.read_pickle(
            formatted_cycles_path_cycles
        )
    df_from_mask = pd.read_pickle(root_cardiac_pipeline / "deltaY.pkl")
    mask_groups = {
        video: g.groupby("cycle")["deltaY"].median().to_dict()
        for video, g in df_from_mask.groupby("video")
    }
    root_measures = root_cardiac_pipeline / "measures"
    videos = df_from_mask["video"].unique()
    case_rows = []
    cycle_rows = []  # long, per-cycle, for the repeatability plot
    biomeasures = load_measurements(
        include_HR=True, include_IOP=True, include_OPA=True, include_axial_length=True
    )
    for video in tqdm(videos, desc="Loading pulsation results"):
        input_path = root_measures / f"{video}"
        mask_file = input_path / "segmented_cycles.npz"
        pkl_file = input_path / "deltaA_per_cycle.pkl"
        measure_file = input_path / "measure.pkl"
        mask_file = input_path / "segmented_cycles.npz"
        if not (pkl_file.exists() and mask_file.exists() and measure_file.exists()):
            continue
        cardiac_freq = peek_cardiac_freq(measure_file)
        if cardiac_freq is None:
            cardiac_freq = CardiacPipelineResults.load(measure_file).cardiac_freq
        data = pickle.load(open(pkl_file, "rb"))
        disp_per_cycle = data["displacement_per_cycle"]
        ref_per_cycle = data["reference_coordinates_per_cycle"]
        T = int(np.load(mask_file)["shape"][0])
        frame_per_cycle = T // N_CYCLES
        deltaY_by_cycle = mask_groups.get(video, {})
        row = biomeasures[biomeasures["MeasureValue"] == video].iloc[0]
        if pd.isna(row.AxialLength) or pd.isna(row.IOP) or pd.isna(row.OPA):
            continue
        hr = cardiac_freq * 60.0
        period_s = 60.0 / hr if pd.notna(hr) and hr > 0 else np.nan

        cts, min_cts, mask_cts = [], [], []
        ups, downs, asyms = [], [], []
        ref_masks = trim_choroid(
            load_mask_frames(mask_file, [i * frame_per_cycle for i in range(N_CYCLES)]),
            TRIM,
        )
        for i in range(N_CYCLES):
            ref_mask = ref_masks[i : i + 1]  # was masks[i*fpc : i*fpc+1]
            try:
                res = measure_delta_ct_from_disp(
                    disp_per_cycle[i], ref_per_cycle[i], ref_mask, reference_frame_idx=0
                )
            except Exception:
                continue
            rates = cycle_rates(res.ct_series_mm, period_s, n_harm=7)
            ups.append(rates.thickening_um_s)
            downs.append(rates.thinning_um_s)
            asyms.append(rates.asymmetry)
            ct_mask = deltaY_by_cycle.get(i, np.nan) * AXIAL_PIXEL_SIZE_MM
            cts.append(res.deltaCT_mm)
            min_cts.append(res.min_ct_mm)
            mask_cts.append(ct_mask)
            cycle_rows.append(
                {
                    "video": video,
                    "cycle": i,
                    "deltaCT": res.deltaCT_mm,  # mm, displacement-based
                    "deltaCT_Mask": ct_mask,  # mm, mask-based
                    "minCT": res.min_ct_mm,
                    "RelativeGrowth": res.deltaCT_mm / res.min_ct_mm
                    if res.min_ct_mm != 0
                    else np.nan,
                    "thickening_um_s": rates.thickening_um_s,
                    "thinning_um_s": rates.thinning_um_s,
                    "rate_asymmetry": rates.asymmetry,
                    "thickening_fraction": rates.thickening_fraction,
                }
            )
        if not cts:
            continue

        median_ct = np.nanmedian(cts)  # mm
        choroid_thickness = np.nanmedian(min_cts)  # mm
        median_ct_mask = (
            np.nanmedian(mask_cts) if np.any(np.isfinite(mask_cts)) else np.nan
        )

        case_rows.append(
            {
                "case_id": row.Id,
                "deltaCT": median_ct,
                "deltaCT_Mask": median_ct_mask,
                "minCT": choroid_thickness,
                "K": K_from_deltaCT_mm(
                    median_ct,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
                    choroidal_thickness_mm=choroid_thickness,
                ),
                "dV": deltaCT_to_deltaV_uL(
                    np.asarray(median_ct, dtype=float) * 1000.0,
                    row.AxialLength,
                    choroid_thickness,
                    cfg=FRIEDENWALD,
                ),
                "K_Mask": K_from_deltaCT_mm(
                    median_ct_mask,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
                    choroidal_thickness_mm=choroid_thickness,
                ),
                "thickening_um_s": np.nanmedian(ups)
                if np.any(np.isfinite(ups))
                else np.nan,
                "thinning_um_s": np.nanmedian(downs)
                if np.any(np.isfinite(downs))
                else np.nan,
                "rate_asymmetry": np.nanmedian(asyms)
                if np.any(np.isfinite(asyms))
                else np.nan,
                "video": video,
                "caseId": f"{row.PatientId}/{row.Date}/{row.Eye}",
                "RelativeGrowth": median_ct / choroid_thickness,
                "predicted_HR": cardiac_freq * 60.0,
                "HR": row.HR,
            }
        )
    if not case_rows:
        raise RuntimeError(
            f"No case produced a measurement under {root_cardiac_pipeline}. "
            f"Of the {len(videos)} videos in deltaY.pkl, none had the "
            "measures/<video>/{deltaA_per_cycle.pkl,segmented_cycles.npz,"
            "measure.pkl} trio with usable biomeasures - most likely the "
            "deltaA stage has not been run for this pipeline root."
        )
    df_cycles = pd.DataFrame(cycle_rows)
    df = pd.DataFrame(case_rows).set_index("case_id")
    df.to_pickle(formatted_results_path)
    df_cycles.to_pickle(formatted_cycles_path_cycles)
    return df, df_cycles
