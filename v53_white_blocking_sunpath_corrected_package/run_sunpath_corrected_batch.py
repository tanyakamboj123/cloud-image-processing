
"""
Batch test for V5.3 white-blocking + sun-path correction.

Simple usage:
    python run_sunpath_corrected_batch.py

Default folder layout:
    test_images/
        WhatsApp Image 2026-09-18 at 7.27.55 AM.jpeg
        ...
    actual_ghi.csv   # optional

Or edit IMAGE_DIR and ACTUAL_CSV below.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sky_prediction_v5_3_white_blocking_sunpath_corrected as core
from okta_correction_only_white_blocking import corrected_visual_okta


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
LAT = 27.54225
LON = 72.471
TZ = 5.5

IMAGE_DIR = Path("test_images")
ACTUAL_CSV = Path("actual_ghi.csv")
OUTPUT_DIR = Path("outputs_sunpath_corrected")


def parse_timestamp_from_name(path: Path) -> pd.Timestamp:
    name = path.name

    m = re.search(r"(\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2}) (AM|PM)", name, re.I)
    if m:
        date, hh, mm, ss, ap = m.groups()
        hh = int(hh)
        ap = ap.upper()
        if ap == "PM" and hh != 12:
            hh += 12
        if ap == "AM" and hh == 12:
            hh = 0
        return pd.Timestamp(f"{date} {hh:02d}:{int(mm):02d}:{int(ss):02d}")

    # fallback patterns
    patterns = [
        r"(?P<y>20\d{2})[-_ ](?P<m>\d{1,2})[-_ ](?P<d>\d{1,2})[T_ -]+(?P<H>\d{1,2})[-_: ](?P<M>\d{2})(?:[-_: ](?P<S>\d{2}))?",
        r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[T_ -]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})?",
    ]
    for pat in patterns:
        m = re.search(pat, path.stem)
        if m:
            g = m.groupdict()
            return pd.Timestamp(
                year=int(g["y"]), month=int(g["m"]), day=int(g["d"]),
                hour=int(g["H"]), minute=int(g["M"]), second=int(g.get("S") or 0)
            )

    return pd.NaT


def load_actual_ghi(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}

    ts_col = None
    for c in ["timestamp_ist", "datetime", "date_time", "time_stamp", "timestamp", "date"]:
        if c in lower:
            ts_col = lower[c]
            break

    ghi_col = None
    candidates = [
        "actual_ghi_wm2", "actual_ghi", "plant_ghi_wm2", "ghi_wm2",
        "plant_ghi", "ghi", "ghi_kw_m2",
        "gspl | avg-wms - live ghi power (kw/m^2)",
    ]
    for c in candidates:
        if c.lower() in lower:
            ghi_col = lower[c.lower()]
            break

    if ts_col is None or ghi_col is None:
        print("Actual CSV found, but columns not recognized. Use timestamp_ist, actual_ghi_wm2.")
        print("Columns:", list(df.columns))
        return None

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
        "actual_raw": pd.to_numeric(df[ghi_col], errors="coerce"),
    }).dropna()

    if len(out) == 0:
        return None

    # If data is in kW/m2, convert to W/m2.
    if out["actual_raw"].quantile(0.95) < 5:
        out["actual_ghi_Wm2"] = out["actual_raw"] * 1000.0
    else:
        out["actual_ghi_Wm2"] = out["actual_raw"]

    return out[["timestamp", "actual_ghi_Wm2"]].sort_values("timestamp")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        images.extend(IMAGE_DIR.glob(ext))
    images = sorted(images)

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR.resolve()}")

    rows = []
    for p in images:
        ts = parse_timestamp_from_name(p)
        if pd.isna(ts):
            print("Skipping, timestamp not found:", p.name)
            continue

        metrics, img_rgb, masks = core.analyse_image_versions(str(p), ts, LAT, LON, TZ)
        cosz = core.solar_cos_zenith_noaa(ts, LAT, LON, TZ)
        clear = core.clear_sky_ghi_haurwitz(cosz)
        okta_info = corrected_visual_okta(str(p), metrics=metrics, masks=masks)

        rows.append({
            "timestamp": ts,
            "image_file": p.name,
            "clear_sky_ghi_Wm2": clear,
            "solar_elevation_deg": metrics.get("solar_elevation_deg"),
            "okta_corrected": okta_info.get("okta_corrected"),
            "predicted_csi_before_sunpath": metrics.get("predicted_csi_v5_before_sunpath_correction"),
            "predicted_csi_after_sunpath": metrics.get("predicted_csi_v5"),
            "predicted_ghi_after_sunpath_Wm2": metrics.get("predicted_csi_v5") * clear,
            "sunpath_correction_flag": metrics.get("sunpath_correction_flag"),
            "morning_low_sun_floor_flag": metrics.get("morning_low_sun_floor_flag"),
            "morning_low_sun_csi_floor": metrics.get("morning_low_sun_csi_floor"),
            "afternoon_sunpath_cap_flag": metrics.get("afternoon_sunpath_cap_flag"),
            "afternoon_sunpath_csi_cap": metrics.get("afternoon_sunpath_csi_cap"),
            "sun_edge_flag": metrics.get("sun_edge_flag"),
            "cloud_nature": metrics.get("v6_cloud_nature"),
            "sunpath_correction_reason": metrics.get("sunpath_correction_reason"),
            "true_clear_blue_fraction": metrics.get("v8_true_clear_blue_sky_fraction"),
            "display_cloud_fraction": metrics.get("v8_display_cloud_fraction"),
            "display_dark_cloud_fraction": metrics.get("v8_display_dark_cloud_fraction"),
            "non_sun_light_grey_cloud_fraction": metrics.get("non_sun_light_grey_cloud_fraction"),
            "direct_sun_score": metrics.get("direct_sun_score"),
            "direct_sun_visible_flag": metrics.get("direct_sun_visible_flag"),
            "thin_veil_cloud_flag": metrics.get("thin_veil_cloud_flag"),
            "thin_veil_score": metrics.get("thin_veil_score"),
        })

    result = pd.DataFrame(rows).sort_values("timestamp")

    actual = load_actual_ghi(ACTUAL_CSV)
    if actual is not None:
        result = pd.merge_asof(
            result.sort_values("timestamp"),
            actual.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=10),
        )
        result["actual_csi"] = np.where(
            result["clear_sky_ghi_Wm2"] > 20,
            result["actual_ghi_Wm2"] / result["clear_sky_ghi_Wm2"],
            np.nan,
        )
        result["error_actual_minus_predicted_Wm2"] = (
            result["actual_ghi_Wm2"] - result["predicted_ghi_after_sunpath_Wm2"]
        )
        result["absolute_error_Wm2"] = result["error_actual_minus_predicted_Wm2"].abs()
        eval_rows = result.dropna(subset=["actual_ghi_Wm2", "predicted_ghi_after_sunpath_Wm2"])
        if len(eval_rows):
            mae = eval_rows["absolute_error_Wm2"].mean()
            rmse = np.sqrt((eval_rows["error_actual_minus_predicted_Wm2"] ** 2).mean())
            bias = eval_rows["error_actual_minus_predicted_Wm2"].mean()
            print(f"Rows: {len(eval_rows)}")
            print(f"MAE:  {mae:.2f} W/m2")
            print(f"RMSE: {rmse:.2f} W/m2")
            print(f"Bias Actual-Predicted: {bias:.2f} W/m2")

    out_csv = OUTPUT_DIR / "sunpath_corrected_predictions.csv"
    result.to_csv(out_csv, index=False)
    print("Saved:", out_csv.resolve())

    preview_cols = [
        "timestamp", "actual_ghi_Wm2", "predicted_ghi_after_sunpath_Wm2",
        "error_actual_minus_predicted_Wm2", "okta_corrected",
        "predicted_csi_before_sunpath", "predicted_csi_after_sunpath",
        "morning_low_sun_floor_flag", "afternoon_sunpath_cap_flag", "cloud_nature"
    ]
    preview_cols = [c for c in preview_cols if c in result.columns]
    print(result[preview_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
