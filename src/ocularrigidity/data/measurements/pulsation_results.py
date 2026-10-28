import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import warnings
import numpy as np
from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM, CHOROID_TRIM_PX
from ocularrigidity.rigidity.friedenwald import FRIEDENWALD
from ocularrigidity.pipeline_config import N_CYCLES
from ocularrigidity.data.io import load_mask
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.rigidity.friedenwald import K_from_deltaCT_mm, PV_ratio_deltaCT_mm, deltaCT_to_deltaV_uL
from ocularrigidity.motion.pipeline_results import (
    CardiacPipelineResults,
    peek_cardiac_freq,
)
from ocularrigidity.segmentation.closing_structures import trim_choroid
from ocularrigidity.thickness.delta import (
    cycle_rates,
    mask_ct_series_mm,
    measure_delta_ct_from_disp,
)

TRIM = CHOROID_TRIM_PX


def _resample_frames(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resample a (T, ...) array to `n_frames` along the time axis."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == n_frames:
        return arr
    pos = np.linspace(0, arr.shape[0] - 1, n_frames)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, arr.shape[0] - 1)
    w = (pos - lo).reshape(-1, *([1] * (arr.ndim - 1)))
    return np.where(w == 0, arr[lo], arr[lo] * (1 - w) + arr[hi] * w)


def load_pulsation_results(
    root_cardiac_pipeline: Path, overwrite=False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-case and per-cycle pulsation measures, cached."""
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

        cts, min_cts, mask_cts, mask_thick_cts = [], [], [], []
        ups, downs, asyms = [], [], []
        strain_ups, strain_downs, stretch_series = [], [], []
        half_times = []
        masks = trim_choroid(load_mask(mask_file), TRIM)
        mask_ct_series = mask_ct_series_mm(masks)  # (T,) mm
        delta_ct_maps = []
        for i in range(N_CYCLES):
            cycle = slice(i * frame_per_cycle, (i + 1) * frame_per_cycle)
            ref_mask = masks[cycle.start : cycle.start + 1]
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
            # Same rates as a strain rate: µm/s over the thinnest CT of the cycle.
            pct_per_um = 0.1 / res.min_ct_mm if res.min_ct_mm else np.nan
            strain_ups.append(rates.thickening_um_s * pct_per_um)
            strain_downs.append(rates.thinning_um_s * pct_per_um)
            stretch_series.append(res.stretch_series)  # (T,)
            half_times.append(rates.thinning_half_time_s)
            ct_mask = deltaY_by_cycle.get(i, np.nan) * AXIAL_PIXEL_SIZE_MM
            cycle_series = mask_ct_series[cycle]
            ct_mask_thick = (
                float(np.nanmax(cycle_series) - np.nanmin(cycle_series))
                if np.any(np.isfinite(cycle_series))
                else np.nan
            )
            cts.append(res.deltaCT_mm)
            min_cts.append(res.min_ct_mm)
            mask_cts.append(ct_mask)
            mask_thick_cts.append(ct_mask_thick)
            delta_ct_maps.append(res.delta_ct_spatial_mm)  # (T, W) mm
            cycle_rows.append(
                {
                    "video": video,
                    "cycle": i,
                    "deltaCT": res.deltaCT_mm,  # mm, displacement-based
                    "deltaCT_Mask": ct_mask,  # mm, mask-based
                    # mm, peak-to-peak of the per-frame mean mask thickness
                    "deltaCT_MaskThickness": ct_mask_thick,
                    "minCT": res.min_ct_mm,
                    "RelativeGrowth": res.deltaCT_mm / res.min_ct_mm
                    if res.min_ct_mm != 0
                    else np.nan,
                    "thickening_um_s": rates.thickening_um_s,
                    "thinning_um_s": rates.thinning_um_s,
                    "rate_asymmetry": rates.asymmetry,
                    "thickening_fraction": rates.thickening_fraction,
                    "thickening_strain_pct_s": strain_ups[-1],
                    "thinning_strain_pct_s": strain_downs[-1],
                    "thinning_half_time_s": rates.thinning_half_time_s,
                    # same, as a fraction of the cardiac period
                    "thinning_half_fraction": rates.thinning_half_time_s / period_s,
                }
            )
        if not cts:
            continue
        
        
        
        median_ct = np.nanmedian(cts)  # mm
        choroid_thickness = np.nanmedian(min_cts)  # mm
        median_ct_mask = (
            np.nanmedian(mask_cts) if np.any(np.isfinite(mask_cts)) else np.nan
        )
        median_ct_mask_thick = (
            np.nanmedian(mask_thick_cts)
            if np.any(np.isfinite(mask_thick_cts))
            else np.nan
        )
        
        # Resample the deltaCT maps to a common shape across cycles and average them to get a spatial map of deltaCT across the choroid.
        min_frames = min(map(lambda x: x.shape[0], delta_ct_maps))
        delta_ct_maps_resampled = [
            _resample_frames(delta_ct_map, min_frames)
            for delta_ct_map in delta_ct_maps
        ]
        with warnings.catch_warnings():
            # Columns without any CSI anchor are all-NaN by construction.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_delta_ct_map = np.nanmedian(delta_ct_maps_resampled, axis=0)
            median_stretch_series = np.nanmedian(
                [_resample_frames(s, min_frames) for s in stretch_series], axis=0
            )

        case_rows.append(
            {
                "case_id": row.Id,
                "deltaCT": median_ct,
                "deltaCT_Mask": median_ct_mask,
                "deltaCT_MaskThickness": median_ct_mask_thick,
                "minCT": choroid_thickness,
                "K_with_choroid": K_from_deltaCT_mm(
                    median_ct,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
                    choroidal_thickness_mm=choroid_thickness,
                ),
                "deltaCT_spatial_mm": median_delta_ct_map,  # (T, W) mm
                # (T,) (ΔCT(t) + CT) / CT over the folded cycle, 1 at the thinnest
                "stretch_series": median_stretch_series,
                "K": K_from_deltaCT_mm(
                    median_ct,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,),
                "dV": deltaCT_to_deltaV_uL(
                    np.asarray(median_ct, dtype=float) * 1000.0,
                    row.AxialLength,
                    choroid_thickness,
                    cfg=FRIEDENWALD,
                ),
                "K_Mask_with_choroid": K_from_deltaCT_mm(
                    median_ct_mask,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
                    choroidal_thickness_mm=choroid_thickness,
                ),
                "P/V": PV_ratio_deltaCT_mm(
                    median_ct,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
                    cfg=FRIEDENWALD,
                ),
                "K_Mask": K_from_deltaCT_mm(
                    median_ct_mask,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,),
                "K_MaskThickness_with_choroid": K_from_deltaCT_mm(
                    median_ct_mask_thick,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
                    choroidal_thickness_mm=choroid_thickness,
                ),
                "K_MaskThickness": K_from_deltaCT_mm(
                    median_ct_mask_thick,
                    row.AxialLength,
                    row.IOP,
                    row.OPA,
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
                "thickening_strain_pct_s": np.nanmedian(strain_ups)
                if np.any(np.isfinite(strain_ups))
                else np.nan,
                "thinning_strain_pct_s": np.nanmedian(strain_downs)
                if np.any(np.isfinite(strain_downs))
                else np.nan,
                "thinning_half_time_s": np.nanmedian(half_times)
                if np.any(np.isfinite(half_times))
                else np.nan,
                "thinning_half_fraction": np.nanmedian(half_times) / period_s
                if np.any(np.isfinite(half_times))
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
