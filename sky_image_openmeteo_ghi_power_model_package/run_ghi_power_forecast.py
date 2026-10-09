
"""
Sky-image + Open-Meteo GHI and DC-power prediction model.

Run:
    python run_ghi_power_forecast.py

Folder inputs:
    inputs/images/  -> sky images, any filename if date/time is present in filename
    inputs/ghi/     -> actual GHI CSV/XLSX, any filename, optional
    inputs/power/   -> actual DC voltage/current/power CSV/XLSX, any filename, optional

Outputs:
    outputs/ghi_comparison_table.csv
    outputs/power_comparison_table.csv
    outputs/combined_ghi_power_table.csv
    outputs/image_feature_table.csv

What it does:
    1) Analyse every uploaded image.
    2) Predict GHI and DC power at each image time.
    3) Predict t+15 min from every image.
    4) From the latest image, predict next 2 hours at 15-min interval.
    5) Compare with actual GHI and actual DC power when available.
"""

from __future__ import annotations

import json
import math
import re
import sys
import uuid
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import pvlib
except Exception:  # pragma: no cover
    pvlib = None

WORK_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORK_DIR / "inputs"
IMAGE_DIR = INPUT_DIR / "images"
GHI_DIR = INPUT_DIR / "ghi"
POWER_DIR = INPUT_DIR / "power"
OPENMETEO_DIR = INPUT_DIR / "openmeteo"
MODEL_DIR = INPUT_DIR / "models"
OUTPUT_DIR = WORK_DIR / "outputs"
CLOUD_MODEL_DIR = WORK_DIR / "cloud_model"
CONFIG_PATH = WORK_DIR / "config.json"
_RUN_LOCK = threading.RLock()

if str(CLOUD_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CLOUD_MODEL_DIR))

import sky_prediction_v5_3_white_blocking_sunpath_corrected as sky_core
from okta_correction_only_white_blocking import corrected_visual_okta


def load_config() -> dict[str, Any]:
    default = {
        "site_latitude": 27.54225,
        "site_longitude": 72.471,
        "timezone": "Asia/Kolkata",
        "utc_offset_hours": 5.5,
        "surface_tilt_deg": 25.0,
        "surface_azimuth_deg": 180.0,
        "ground_albedo": 0.20,
        "dc_capacity_kw": None,
        "temperature_coefficient_per_C": -0.004,
        "default_performance_ratio": 0.82,
        "openmeteo_enabled": True,
        "forecast_horizon_minutes": 120,
        "forecast_step_minutes": 15,
        "ghi_units_auto_detect": True,
        "power_units_auto_detect": True,
        "power_unit": "auto",
        "timestamp_dayfirst": True,
        "actual_match_tolerance_minutes": 8,
        "video_frame_interval_seconds": 60,
        "video_start_times": {},
        "image_timestamps": {},
        "power_calibration_enabled": True,
    }
    if CONFIG_PATH.exists():
        try:
            user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            default.update(user_cfg)
        except Exception as exc:
            raise ValueError(f"Invalid config.json: {exc}") from exc
    return default

CFG = load_config()
LAT = float(CFG["site_latitude"])
LON = float(CFG["site_longitude"])
TZ_NAME = str(CFG["timezone"])
UTC_OFFSET = float(CFG["utc_offset_hours"])
SURFACE_TILT = float(CFG["surface_tilt_deg"])
SURFACE_AZIMUTH = float(CFG["surface_azimuth_deg"])
ALBEDO = float(CFG["ground_albedo"])
DEFAULT_PR = float(CFG["default_performance_ratio"])
GAMMA = float(CFG["temperature_coefficient_per_C"])
DC_CAPACITY_KW = CFG.get("dc_capacity_kw", None)
HORIZON_MIN = int(CFG.get("forecast_horizon_minutes", 120))
STEP_MIN = int(CFG.get("forecast_step_minutes", 15))


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_" + uuid.uuid4().hex[:10] + path.suffix)
        df.to_csv(alt, index=False)
        print(f"File was open/locked. Saved as: {alt}")
        return alt


def clip(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return lo
        return float(np.clip(v, lo, hi))
    except Exception:
        return lo


def clean_name(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def parse_timestamp_from_name(path: Path) -> pd.Timestamp:
    name = path.name

    # WhatsApp Image 2026-09-21 at 12.30.13 PM.jpeg
    m = re.search(r"(20\d{2}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2}) (AM|PM)", name, re.I)
    if m:
        date, hh, mm, ss, ap = m.groups()
        hh = int(hh); mm = int(mm); ss = int(ss); ap = ap.upper()
        if ap == "PM" and hh != 12:
            hh += 12
        if ap == "AM" and hh == 12:
            hh = 0
        return pd.Timestamp(f"{date} {hh:02d}:{mm:02d}:{ss:02d}")

    # 2026-09-17_06-15.png / 2026_09_17 06-15-00.jpg
    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})[_ T-]+(\d{1,2})[-_:](\d{2})(?:[-_:](\d{2}))?", name)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return pd.Timestamp(year=int(y), month=int(mo), day=int(d), hour=int(h), minute=int(mi), second=int(s or 0))

    # 27-09-2026_11-00.jpg
    m = re.search(r"(\d{1,2})[-_](\d{1,2})[-_](20\d{2})[_ T-]+(\d{1,2})[-_:](\d{2})(?:[-_:](\d{2}))?", name)
    if m:
        d, mo, y, h, mi, s = m.groups()
        return pd.Timestamp(year=int(y), month=int(mo), day=int(d), hour=int(h), minute=int(mi), second=int(s or 0))

    return pd.NaT


def local_solar_features(ts: pd.Timestamp) -> dict[str, float]:
    cosz = sky_core.solar_cos_zenith_noaa(pd.Timestamp(ts), LAT, LON, UTC_OFFSET)
    clear = sky_core.clear_sky_ghi_haurwitz(cosz)
    elev = math.degrees(math.asin(max(-1.0, min(1.0, cosz)))) if cosz > 0 else 0.0
    return {
        "clear_sky_ghi_wm2": float(clear),
        "solar_elevation_deg": float(elev),
        "solar_zenith_deg": float(90.0 - elev),
    }


def list_input_files(folder: Path) -> list[Path]:
    return sorted(p for p in user_input_files(folder) if p.is_file()
                  and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
                  and not p.name.startswith("~$"))


def user_input_files(folder: Path):
    """Exclude dashboard snapshots from the ordinary CLI input scan."""
    for path in folder.rglob("*"):
        parents = path.relative_to(folder).parts[:-1]
        if any(part == "_dashboard" or re.fullmatch(r"[a-f0-9]{32}", part) for part in parents):
            continue
        if path.is_file():
            yield path


def read_table_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file: {path}")


def find_timestamp_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    priority = ["timestamp_ist", "datetime", "date_time", "timestamp", "date time", "date", "time", "datatime"]
    lower_map = {clean_name(c): c for c in cols}
    for key in priority:
        if key in lower_map:
            return lower_map[key]
    # Try any column with date/time in name
    for c in cols:
        n = clean_name(c)
        if "date" in n or "time" in n:
            return c
    return cols[0]


def parse_datetime_series(s: pd.Series) -> pd.Series:
    # Normalize offset-aware values to the site timezone BEFORE dropping the zone.
    # ISO dates stay year-first; ambiguous DD/MM dates follow configuration.
    def parse(value):
        if pd.isna(value):
            return pd.NaT
        try:
            if isinstance(value, (int, float, np.number)):
                ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))
            else:
                iso = bool(re.match(r"^\d{4}-\d{2}-\d{2}", str(value)))
                ts = pd.to_datetime(value, dayfirst=not iso and CFG.get("timestamp_dayfirst", True))
            if ts.tzinfo is not None:
                ts = ts.tz_convert(TZ_NAME).tz_localize(None)
            return ts
        except (ValueError, TypeError, OverflowError):
            return pd.NaT
    return pd.to_datetime(s.map(parse), errors="coerce").astype("datetime64[ns]")


def table_timestamps(df: pd.DataFrame) -> pd.Series:
    names = {clean_name(c): c for c in df.columns}
    if "date" in names and "time" in names:
        dates = parse_datetime_series(df[names["date"]]).dt.strftime("%Y-%m-%d")
        return parse_datetime_series(dates + " " + df[names["time"]].astype(str))
    return parse_datetime_series(df[find_timestamp_col(df)])


def infer_unit_to_wm2(series: pd.Series, colname: str = "") -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    name = clean_name(colname)
    if "kw/m" in name or "kwm" in name or "kw m" in name:
        return x * 1000.0
    if "w/m" in name or "wm2" in name or "w m" in name:
        return x
    if x.dropna().empty:
        return x
    # Plant WMS often stores 0.725 for 725 W/m2.
    if CFG.get("ghi_units_auto_detect", True) and x.quantile(0.95) < 5:
        print(f"Warning: assuming kW/m2 for unlabelled GHI column {colname}; multiplying by 1000.")
        return x * 1000.0
    return x


# ---------------------------------------------------------------------
# Actual GHI and power loading
# ---------------------------------------------------------------------
def load_actual_ghi() -> pd.DataFrame | None:
    files = list_input_files(GHI_DIR)
    if not files:
        return None

    frames = []
    for f in files:
        try:
            df = read_table_any(f)
        except Exception as exc:
            print(f"Skipping GHI file {f.name}: {exc}")
            continue
        if df.empty:
            continue

        ts_col = find_timestamp_col(df)
        lower_map = {clean_name(c): c for c in df.columns}
        ghi_col = None
        preferred = [
            "actual_ghi_wm2", "actual ghi wm2", "actual ghi", "ghi_wm2", "ghi",
            "gspl | avg-wms - live ghi power (kw/m^2)",
        ]
        for key in preferred:
            if key in lower_map:
                ghi_col = lower_map[key]
                break
        if ghi_col is None:
            for c in df.columns:
                n = clean_name(c)
                if c == ts_col:
                    continue
                if not any(word in n for word in ["poa", "gti", "dni", "dhi"]) and ("ghi" in n or "irradiance" in n):
                    ghi_col = c
                    break
        if ghi_col is None:
            print(f"GHI column not found in {f.name}")
            continue

        tmp = pd.DataFrame({
            "timestamp": table_timestamps(df),
            "actual_ghi_wm2": infer_unit_to_wm2(df[ghi_col], str(ghi_col)),
            "source_ghi_file": f.name,
        }).dropna(subset=["timestamp", "actual_ghi_wm2"])
        frames.append(tmp)

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def infer_power_kw(series: pd.Series, colname: str = "") -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    name = clean_name(colname).replace("_", " ")
    unit = str(CFG.get("power_unit", "auto")).lower()
    if unit == "auto":
        match = re.search(r"(?<![a-z])(mw|kw|w)(?![a-z])", name)
        if match:
            unit = match.group(1)
        else:
            q = x.abs().quantile(0.95)
            unit = ("mw" if q < 100 else "w" if q > 200000 else "kw") if CFG.get("power_units_auto_detect", True) else "kw"
            print(f"Warning: {colname}: unit inferred as {unit} from magnitude; set power_unit to override.")
    if unit not in {"mw", "kw", "w"}:
        raise ValueError("power_unit must be auto, MW, kW or W")
    factor = {"mw": 1000., "kw": 1., "w": .001}[unit]
    print(f"Power conversion: {colname}: {unit} -> kW, multiplier {factor:g}")
    return x * factor


def vi_pairs(columns) -> list[tuple[str, str]]:
    # Pair by device/string identifier, never by independent column sorting.
    groups = {"v": {}, "i": {}}
    for c in columns:
        n = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()
        if re.search(r"\bac\b", n):
            continue
        kind = "v" if re.search(r"voltage|vdc|\bv\b", n) else "i" if re.search(r"current|idc|\bamps?\b|\bi\b", n) else None
        if kind is None:
            continue
        key = re.sub(r"voltage|current|vdc|idc|\bamps?\b|\b(?:dc|input|v|i|a|volts)\b", " ", n)
        key = re.sub(r"\s+", " ", key).strip()
        if key in groups[kind]:
            raise ValueError(f"Ambiguous voltage/current identifier: {key!r}")
        groups[kind][key] = c
    if set(groups["v"]) != set(groups["i"]):
        raise ValueError("Unmatched voltage/current identifiers; rename columns to matching device IDs")
    return [(groups["v"][key], groups["i"][key]) for key in sorted(groups["v"])]


def load_actual_power() -> pd.DataFrame | None:
    frames = []
    for f in list_input_files(POWER_DIR):
        try:
            df = read_table_any(f)
            if df.empty:
                continue
            out = pd.DataFrame({"timestamp": table_timestamps(df)})
            power_cols = []
            for c in df.columns:
                n = re.sub(r"[^a-z0-9]+", " ", str(c).lower())
                if re.search(r"\bpdc\b", n) or (re.search(r"\bdc\b", n) and "power" in n):
                    power_cols.append(c)
            total = [c for c in power_cols if "total" in clean_name(c)]
            if len(total) > 1:
                raise ValueError("Multiple total DC power columns; select one total in the input")
            selected = total or power_cols
            try:
                pairs = vi_pairs(df.columns)
            except ValueError:
                if not selected:
                    raise
                pairs = []
            if selected:
                values = [infer_power_kw(df[c], str(c)) for c in selected]
                out["actual_dc_power_kw"] = pd.concat(values, axis=1).sum(axis=1, min_count=len(values))
                method = "power_columns:" + ";".join(map(str, selected))
            elif pairs:
                values = [pd.to_numeric(df[v], errors="coerce") * pd.to_numeric(df[i], errors="coerce") / 1000 for v, i in pairs]
                out["actual_dc_power_kw"] = pd.concat(values, axis=1).sum(axis=1, min_count=len(values))
                method = "sum_vdc_times_idc:" + ";".join(f"{v} * {i} / 1000" for v, i in pairs)
            else:
                raise ValueError("No explicit DC power or matched V/I columns (AC power is not DC power)")
            # A fleet has no single voltage. Leave these blank for multiple pairs.
            if len(pairs) == 1:
                out["actual_dc_voltage_v"] = pd.to_numeric(df[pairs[0][0]], errors="coerce")
                out["actual_dc_current_a"] = pd.to_numeric(df[pairs[0][1]], errors="coerce")
            out["power_source_method"] = method
            out["source_power_file"] = f.name
            out = out.dropna(subset=["timestamp", "actual_dc_power_kw"])
            print(f"Power file {f.name}: {len(out)} valid rows; {method}")
            frames.append(out)
        except Exception as exc:
            print(f"Skipping power file {f.name}: {exc}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def nearest_actual(times: pd.Series, actual: pd.DataFrame | None, value_col: str, tolerance_min: int | None = None) -> pd.Series:
    result = pd.Series(np.nan, index=times.index, dtype=float)
    if actual is None or actual.empty or value_col not in actual.columns:
        return result
    tolerance_min = CFG.get("actual_match_tolerance_minutes", 8) if tolerance_min is None else tolerance_min
    temp = pd.DataFrame({"timestamp": pd.to_datetime(times).to_numpy(), "_position": np.arange(len(times))})
    temp["timestamp"] = temp["timestamp"].astype("datetime64[ns]")
    right = actual[["timestamp", value_col]].dropna().sort_values("timestamp").copy()
    right["timestamp"] = right["timestamp"].astype("datetime64[ns]")
    merged = pd.merge_asof(temp.dropna().sort_values("timestamp"), right,
                           on="timestamp", direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_min))
    result.iloc[merged["_position"].to_numpy()] = merged[value_col].to_numpy()
    return result


# ---------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------
OM_NAMES = {
    "shortwave_radiation": "om_ghi_wm2", "ghi": "om_ghi_wm2",
    "direct_radiation": "om_direct_horizontal_wm2",
    "diffuse_radiation": "om_dhi_wm2", "dhi": "om_dhi_wm2",
    "direct_normal_irradiance": "om_dni_wm2", "dni": "om_dni_wm2",
    "temperature_2m": "om_temp_c", "wind_speed_10m": "om_wind_ms",
    "relative_humidity_2m": "om_relative_humidity_pct",
    "cloud_cover": "om_cloud_cover_pct", "cloud_cover_low": "om_cloud_low_pct",
    "cloud_cover_mid": "om_cloud_mid_pct", "cloud_cover_high": "om_cloud_high_pct",
}


def normalize_openmeteo(df: pd.DataFrame, units=None, timezone=None) -> pd.DataFrame:
    units = units or {}
    ts = table_timestamps(df)
    # JSON timezone describes naive API times; explicit offsets are already normalized.
    if timezone and timezone != TZ_NAME:
        raw = df[find_timestamp_col(df)].astype(str)
        naive = ~raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        ts.loc[naive] = ts.loc[naive].dt.tz_localize(timezone).dt.tz_convert(TZ_NAME).dt.tz_localize(None)
    out = pd.DataFrame({"timestamp": ts})
    for c in df.columns:
        n = clean_name(c)
        key = re.split(r"\s*\(", n)[0].strip().replace(" ", "_")
        target = OM_NAMES.get(key, key if key in OM_NAMES.values() else None)
        if target is None:
            continue
        values = pd.to_numeric(df[c], errors="coerce")
        unit = str(units.get(c, n)).lower()
        if target == "om_wind_ms":
            if "km/h" in unit or "kmh" in unit:
                values /= 3.6
            elif "mph" in unit:
                values *= .44704
            elif "kn" == unit or "knot" in unit:
                values *= .514444
            elif not any(u in unit for u in ["m/s", "ms", "m s"]):
                print("Warning: unlabelled manual wind speed assumed m/s.")
        if target == "om_temp_c" and ("°f" in unit or "fahrenheit" in unit):
            values = (values - 32) * 5 / 9
        if target.endswith("wm2") and "kw" in unit:
            values *= 1000
        out[target] = values
    if len(out.columns) == 1:
        raise ValueError("No recognized Open-Meteo weather columns")
    return out.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False).mean(numeric_only=True).sort_values("timestamp")


def load_manual_openmeteo(start, end) -> pd.DataFrame | None:
    frames = []
    for path in sorted(user_input_files(OPENMETEO_DIR)):
        if path.suffix.lower() not in {".json", ".csv"}:
            continue
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict) and "hourly" in data:
                    frame = normalize_openmeteo(pd.DataFrame(data["hourly"]), data.get("hourly_units"), data.get("timezone"))
                else:
                    frame = normalize_openmeteo(pd.DataFrame(data))
            else:
                # Open-Meteo CSV exports can have a metadata block before the time header.
                lines = path.read_text(encoding="utf-8-sig").splitlines()
                skip = next((i for i, line in enumerate(lines) if re.match(r'"?(time|timestamp|datetime)"?[,;]', line, re.I)), 0)
                table = pd.read_csv(path, skiprows=skip, sep=None, engine="python", encoding="utf-8-sig")
                timezone = None
                if skip >= 2 and "timezone" in lines[0].lower():
                    import csv
                    metadata = dict(zip(next(csv.reader([lines[0]])), next(csv.reader([lines[1]]))))
                    timezone = metadata.get("timezone")
                frame = normalize_openmeteo(table, timezone=timezone)
            frame = frame[frame["timestamp"].between(pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h"))]
            if not frame.empty and frame.drop(columns="timestamp").notna().any().any():
                frames.append(frame)
                print(f"Manual Open-Meteo: {path.name}, {len(frame)} rows")
        except Exception as exc:
            print(f"Skipping manual Open-Meteo {path.name}: {exc}")
    if not frames:
        return None
    return pd.concat(frames).groupby("timestamp", as_index=False).mean(numeric_only=True).sort_values("timestamp")


def fetch_openmeteo(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    if CFG.get("openmeteo_enabled", True) and requests is not None:
        today = pd.Timestamp.now(tz=TZ_NAME).normalize().tz_localize(None)
        # Reanalysis has a publication delay; recent past dates use forecast past data.
        archive = pd.Timestamp(end).normalize() < today - pd.Timedelta(days=5)
        url = "https://archive-api.open-meteo.com/v1/archive" if archive else "https://api.open-meteo.com/v1/forecast"
        params = dict(latitude=LAT, longitude=LON, start_date=pd.Timestamp(start).date().isoformat(),
                      end_date=pd.Timestamp(end).date().isoformat(), timezone=TZ_NAME,
                      wind_speed_unit="ms", hourly=",".join(k for k in OM_NAMES if k not in {"ghi", "dni", "dhi"}))
        try:
            response = requests.get(url, params=params, timeout=25)
            response.raise_for_status()
            data = response.json()
            frame = normalize_openmeteo(pd.DataFrame(data["hourly"]), data.get("hourly_units"), data.get("timezone"))
            if frame.empty or not frame.drop(columns="timestamp").notna().any().any():
                raise ValueError("Empty weather response")
            print(f"Live Open-Meteo OK: {len(frame)} rows from {url}")
            print("Note: hourly radiation is a preceding-hour mean; interpolating it to image times is approximate.")
            return frame
        except Exception as exc:
            print(f"Warning: Open-Meteo download failed: {exc}")
    else:
        print("Live Open-Meteo disabled/unavailable; checking manual files.")
    manual = load_manual_openmeteo(start, end) if CFG.get("use_manual_openmeteo", True) else None
    if manual is not None:
        return manual
    print("Warning: no usable Open-Meteo data; using sky images + clear-sky fallback.")
    return None


def interp_openmeteo(times: pd.Series, om: pd.DataFrame | None) -> pd.DataFrame:
    times = pd.to_datetime(times).reset_index(drop=True)
    out = pd.DataFrame({"prediction_time": times})
    for c in set(OM_NAMES.values()):
        out[c] = np.nan
    if om is None or om.empty:
        return out
    base = om.copy().sort_values("timestamp")
    x = base["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
    xi = times.astype("datetime64[ns]").astype("int64").to_numpy()
    for c in set(OM_NAMES.values()).intersection(base.columns):
        y = pd.to_numeric(base[c], errors="coerce").to_numpy()
        valid = np.isfinite(y)
        if valid.sum():
            # Never extend stale weather beyond its coverage or across large gaps.
            xv = x[valid]; yv = y[valid]
            values = np.interp(xi, xv, yv, left=np.nan, right=np.nan)
            right = np.searchsorted(xv, xi).clip(0, len(xv)-1)
            left = (right-1).clip(0)
            gap = (xv[right] - xv[left]) > pd.Timedelta(hours=3).value
            values[gap & (xi != xv[right])] = np.nan
            out[c] = values
    return out


# ---------------------------------------------------------------------
# Image model
# ---------------------------------------------------------------------
def collect_image_inputs() -> list[tuple[Path, pd.Timestamp, str]]:
    import cv2
    import hashlib
    items = []
    for path in sorted(user_input_files(IMAGE_DIR)):
        relative = path.relative_to(IMAGE_DIR).as_posix()
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            override = CFG.get("image_timestamps", {}).get(relative)
            ts = parse_datetime_series(pd.Series([override])).iloc[0] if override else parse_timestamp_from_name(path)
            items.append((path, ts, relative))
        elif suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            override = CFG.get("video_start_times", {}).get(relative)
            start = parse_datetime_series(pd.Series([override])).iloc[0] if override else parse_timestamp_from_name(path)
            if pd.isna(start):
                print(f"Skipping video {relative}: set video_start_times in config.json")
                continue
            if not override:
                print(f"Warning: {relative}: assuming filename timestamp is recording start, not share time.")
            interval = float(CFG.get("video_frame_interval_seconds", 60))
            if interval <= 0:
                raise ValueError("video_frame_interval_seconds must be positive")
            folder = OUTPUT_DIR / "video_frames" / hashlib.sha256(relative.encode()).hexdigest()[:12]
            folder.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(path))
            count = 0
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
                    raise ValueError("Video could not be opened or has invalid FPS")
                step = max(1, round(fps * interval))
                frame_index = 0
                while True:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # Approximation: constant-frame-rate recording, one real second per video second.
                    elapsed = frame_index / fps
                    ts = start + pd.Timedelta(seconds=elapsed)
                    output = folder / f"frame_{frame_index:09d}.jpg"
                    if not cv2.imwrite(str(output), frame):
                        raise OSError(f"Could not write {output}")
                    items.append((output, ts, f"{relative}#frame={frame_index}"))
                    count += 1
                    frame_index += step
                print(f"Video {relative}: extracted {count} frames at approximately {interval:g}-second intervals")
            except Exception as exc:
                print(f"Video extraction failed for {relative}: {exc}")
            finally:
                cap.release()
    return items


def analyse_images() -> pd.DataFrame:
    image_files = collect_image_inputs()
    if not image_files:
        raise FileNotFoundError(f"No usable images/videos in {IMAGE_DIR}. Add inputs and run again.")

    rows = []
    for p, ts, source in image_files:
        if pd.isna(ts):
            print(f"Skipping image because timestamp not found in filename: {p.name}")
            continue
        try:
            offset = pd.Timestamp(ts).tz_localize(TZ_NAME).utcoffset().total_seconds() / 3600
            metrics, img_rgb, masks = sky_core.analyse_image_versions(str(p), ts, LAT, LON, offset)
            okta = corrected_visual_okta(str(p), metrics=metrics, masks=masks)
        except Exception as exc:
            print(f"Image analysis failed for {p.name}: {exc}")
            continue

        solar = local_solar_features(ts)
        csi = clip(metrics.get("predicted_csi_v5", np.nan), 0.0, 1.15)
        rows.append({
            "image_time": ts,
            "image_file": source,
            "image_current_csi": csi,
            "image_current_ghi_wm2": csi * solar["clear_sky_ghi_wm2"],
            "clear_sky_ghi_at_image_wm2": solar["clear_sky_ghi_wm2"],
            "solar_elevation_at_image_deg": solar["solar_elevation_deg"],
            "okta_corrected": okta.get("okta_corrected", np.nan),
            "cloud_nature": metrics.get("v6_cloud_nature", ""),
            "cloudiness_index": metrics.get("v6_cloudiness_index_0_to_1", np.nan),
            "cloud_impact_index": metrics.get("v6_cloud_impact_index_0_to_1", np.nan),
            "ramp_risk_index": metrics.get("v6_ramp_risk_index_0_to_1", np.nan),
            "direct_sun_visible_flag": metrics.get("direct_sun_visible_flag", 0),
            "direct_sun_score": metrics.get("direct_sun_score", np.nan),
            "white_blocking_cloud_score": metrics.get("white_blocking_cloud_score", np.nan),
            "white_blocking_cloud_flag": metrics.get("white_blocking_cloud_flag", 0),
            "morning_low_sun_floor_flag": metrics.get("morning_low_sun_floor_flag", 0),
            "midday_scattered_cloud_floor_flag": metrics.get("midday_scattered_cloud_floor_flag", 0),
            "midday_direct_sun_floor_flag": metrics.get("midday_direct_sun_floor_flag", 0),
            "midday_cumulus_edge_floor_flag": metrics.get("midday_cumulus_edge_floor_flag", 0),
            "afternoon_cumulus_blocking_cap_flag": metrics.get("afternoon_cumulus_blocking_cap_flag", 0),
            "afternoon_direct_sun_blocked_cap_flag": metrics.get("afternoon_direct_sun_blocked_cap_flag", 0),
            "afternoon_sunpath_cap_flag": metrics.get("afternoon_sunpath_cap_flag", 0),
            "v8_true_clear_blue_sky_fraction": metrics.get("v8_true_clear_blue_sky_fraction", np.nan),
            "v8_display_cloud_fraction": metrics.get("v8_display_cloud_fraction", np.nan),
            "v8_display_bright_cloud_fraction": metrics.get("v8_display_bright_cloud_fraction", np.nan),
            "v8_display_dark_cloud_fraction": metrics.get("v8_display_dark_cloud_fraction", np.nan),
            "non_sun_light_grey_cloud_fraction": metrics.get("non_sun_light_grey_cloud_fraction", np.nan),
            "sunpath_correction_reason": metrics.get("sunpath_correction_reason", ""),
        })

    if not rows:
        raise RuntimeError("No valid image rows created. Check image filenames contain date/time.")

    df = pd.DataFrame(rows).sort_values("image_time").reset_index(drop=True)
    dt_min = df["image_time"].diff().dt.total_seconds() / 60.0
    df["image_csi_delta_prev"] = df["image_current_csi"].diff()
    df["image_csi_trend_per_15min"] = (df["image_csi_delta_prev"] / dt_min * 15.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["image_csi_trend_per_15min"] = df["image_csi_trend_per_15min"].clip(-0.18, 0.18)
    safe_to_csv(df, OUTPUT_DIR / "image_feature_table.csv")
    return df


# ---------------------------------------------------------------------
# Forecast GHI model
# ---------------------------------------------------------------------
def image_future_csi_rule(row: pd.Series, horizon_min: int) -> tuple[float, str, str]:
    csi0 = clip(row.get("image_current_csi", 0.7), 0.0, 1.15)
    trend15 = clip(row.get("image_csi_trend_per_15min", 0.0), -0.18, 0.18)

    direct = int(row.get("direct_sun_visible_flag", 0) or 0)
    direct_score = clip(row.get("direct_sun_score", 0.0), 0.0, 1.0)
    true_blue = clip(row.get("v8_true_clear_blue_sky_fraction", 0.0), 0.0, 1.0)
    cloud_frac = clip(row.get("v8_display_cloud_fraction", 0.0), 0.0, 1.0)
    dark = clip(row.get("v8_display_dark_cloud_fraction", 0.0), 0.0, 1.0)
    grey = clip(row.get("non_sun_light_grey_cloud_fraction", 0.0), 0.0, 1.0)
    ramp = clip(row.get("ramp_risk_index", 0.0), 0.0, 1.0)
    white_score = clip(row.get("white_blocking_cloud_score", 0.0), 0.0, 1.0)

    midday_edge = int(row.get("midday_cumulus_edge_floor_flag", 0) or 0)
    midday_scattered = int(row.get("midday_scattered_cloud_floor_flag", 0) or 0)
    aft_cumulus_block = int(row.get("afternoon_cumulus_blocking_cap_flag", 0) or 0)
    aft_direct_block = int(row.get("afternoon_direct_sun_blocked_cap_flag", 0) or 0)

    lead = max(0.0, horizon_min / 15.0)
    trend_weight = max(0.10, 0.75 * math.exp(-0.35 * max(0.0, lead - 1.0)))
    persistence_weight = max(0.45, 0.90 * math.exp(-0.18 * max(0.0, lead - 1.0)))
    csi = csi0 + trend_weight * trend15 * min(lead, 2.0)

    condition = "image_persistence"
    confidence = "current" if horizon_min == 0 else ("high" if horizon_min <= 30 else ("medium" if horizon_min <= 75 else "low"))

    clear_direct = clip(0.90 + 0.10 * direct_score + 0.05 * true_blue - 0.06 * dark - 0.04 * grey, 0.86, 1.07)
    scattered = clip(0.70 + 0.18 * true_blue + 0.06 * direct - 0.08 * dark - 0.05 * grey, 0.62, 0.88)
    blocked = clip(0.38 + 0.10 * true_blue + 0.08 * direct - 0.12 * dark - 0.08 * grey, 0.25, 0.58)
    overcast = clip(0.30 + 0.12 * true_blue - 0.10 * dark - 0.08 * grey, 0.18, 0.50)

    if direct and direct_score > 0.45 and dark < 0.28 and white_score < 0.80:
        baseline = clear_direct
        condition = "direct_sun_high_transmission"
    elif midday_edge:
        baseline = clip(0.78 + 0.08 * true_blue - 0.04 * dark, 0.72, 0.90)
        condition = "direct_sun_near_cumulus_edge"
    elif aft_cumulus_block or aft_direct_block or (white_score > 0.75 and true_blue < 0.30):
        baseline = blocked
        condition = "sun_path_blocking_or_cumulus"
    elif cloud_frac > 0.85 and true_blue < 0.12:
        baseline = overcast
        condition = "dense_cloud_or_overcast"
    elif cloud_frac > 0.35 or midday_scattered:
        baseline = scattered
        condition = "scattered_cloud_clear_gap"
    else:
        baseline = clip(0.82 + 0.10 * true_blue, 0.70, 0.95)
        condition = "mostly_clear_or_hazy"

    if horizon_min == 0:
        csi = csi0
    else:
        csi = persistence_weight * csi + (1.0 - persistence_weight) * baseline

    if condition == "direct_sun_high_transmission" and horizon_min > 0:
        csi = max(csi, clear_direct - 0.03 * lead)
    elif condition == "direct_sun_near_cumulus_edge" and horizon_min > 0:
        csi = clip(csi, 0.68, 0.95)
        if horizon_min > 45:
            confidence = "low"
    elif "blocking" in condition and horizon_min > 0:
        csi = min(csi, blocked + 0.08 * max(0.0, 1.0 - lead / 8.0))
        if horizon_min > 45:
            confidence = "low"
    elif ramp > 0.55 and horizon_min > 30:
        confidence = "low"

    return clip(csi, 0.0, 1.12), condition, confidence


def make_prediction_rows(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    latest_time = features["image_time"].max()

    # A. prediction at every image time, horizon 0.
    for _, r in features.iterrows():
        rows.append({
            "section": "image_time_current",
            "origin_image_time": r["image_time"],
            "prediction_time": r["image_time"],
            "horizon_min": 0,
            "image_file": r["image_file"],
        })

    # B. t+15 rolling forecast from every image.
    for _, r in features.iterrows():
        rows.append({
            "section": "rolling_image_t_plus_15",
            "origin_image_time": r["image_time"],
            "prediction_time": r["image_time"] + pd.Timedelta(minutes=15),
            "horizon_min": 15,
            "image_file": r["image_file"],
        })

    # C. next 2 hours from latest image.
    latest_row = features.sort_values("image_time").iloc[-1]
    for h in range(STEP_MIN, HORIZON_MIN + 1, STEP_MIN):
        rows.append({
            "section": "latest_image_2hr_future",
            "origin_image_time": latest_time,
            "prediction_time": latest_time + pd.Timedelta(minutes=h),
            "horizon_min": h,
            "image_file": latest_row["image_file"],
        })

    pred = pd.DataFrame(rows).drop_duplicates(subset=["section", "origin_image_time", "prediction_time", "horizon_min", "image_file"])
    return pred.sort_values(["prediction_time", "section", "horizon_min"]).reset_index(drop=True)


def attach_predictions(pred: pd.DataFrame, features: pd.DataFrame, om: pd.DataFrame | None) -> pd.DataFrame:
    feat = features.copy()
    df = pred.merge(feat, left_on=["origin_image_time", "image_file"], right_on=["image_time", "image_file"], how="left")

    solar_rows = [local_solar_features(t) for t in df["prediction_time"]]
    solar_df = pd.DataFrame(solar_rows)
    df = pd.concat([df.reset_index(drop=True), solar_df.reset_index(drop=True)], axis=1)

    image_csi_list = []
    cond_list = []
    conf_list = []
    for _, r in df.iterrows():
        csi, cond, conf = image_future_csi_rule(r, int(r["horizon_min"]))
        image_csi_list.append(csi)
        cond_list.append(cond)
        conf_list.append(conf)
    df["image_forecast_csi"] = image_csi_list
    df["image_forecast_condition"] = cond_list
    df["forecast_confidence"] = conf_list

    om_i = interp_openmeteo(df["prediction_time"], om)
    df = pd.concat([df.reset_index(drop=True), om_i.drop(columns=["prediction_time"], errors="ignore").reset_index(drop=True)], axis=1)

    clear = pd.to_numeric(df["clear_sky_ghi_wm2"], errors="coerce").replace(0, np.nan)
    df["openmeteo_csi"] = pd.to_numeric(df.get("om_ghi_wm2", np.nan), errors="coerce") / clear
    df["openmeteo_csi"] = df["openmeteo_csi"].clip(0, 1.15)

    # Image weight decreases with horizon; Open-Meteo becomes more important farther out.
    h = pd.to_numeric(df["horizon_min"], errors="coerce").fillna(0)
    img_w = np.where(h <= 0, 0.85, np.maximum(0.35, 0.72 - 0.0032 * h))
    om_available = df["openmeteo_csi"].notna()
    df["fusion_image_weight"] = np.where(om_available, img_w, 1.0)
    df["fusion_openmeteo_weight"] = np.where(om_available, 1.0 - df["fusion_image_weight"], 0.0)

    df["predicted_csi"] = np.where(
        om_available,
        df["fusion_image_weight"] * df["image_forecast_csi"] + df["fusion_openmeteo_weight"] * df["openmeteo_csi"],
        df["image_forecast_csi"],
    )
    df["predicted_csi"] = df["predicted_csi"].clip(0, 1.12)
    df["predicted_ghi_wm2"] = (df["predicted_csi"] * df["clear_sky_ghi_wm2"]).clip(0, 1200)
    return df


# ---------------------------------------------------------------------
# GHI -> GTI/POA conversion
# ---------------------------------------------------------------------
def estimate_cell_temperature(poa_wm2: pd.Series, temp_c: pd.Series | float | None, wind_ms: pd.Series | float | None) -> pd.Series:
    poa = pd.to_numeric(poa_wm2, errors="coerce").fillna(0)
    if temp_c is None:
        temp = pd.Series(25.0, index=poa.index)
    else:
        temp = pd.to_numeric(pd.Series(temp_c, index=poa.index), errors="coerce").fillna(25.0)
    if wind_ms is None:
        wind = pd.Series(1.0, index=poa.index)
    else:
        wind = pd.to_numeric(pd.Series(wind_ms, index=poa.index), errors="coerce").fillna(1.0).clip(lower=0)
    # Simple NOCT-like approximation with wind cooling term.
    tcell = temp + (poa / 800.0) * 25.0 - 1.5 * np.log1p(wind)
    return tcell


def decompose_ghi_simple(ghi: pd.Series, zenith_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Erbs-like diffuse fraction approximation.
    ghi = pd.to_numeric(ghi, errors="coerce").fillna(0).clip(lower=0)
    zen = pd.to_numeric(zenith_deg, errors="coerce").fillna(90).clip(0, 89.9)
    cosz = np.cos(np.radians(zen)).clip(0.01, 1.0)
    doy_clear = ghi.copy()
    # Use Haurwitz clear-sky as rough extraterrestrial-normalized proxy.
    kt = (ghi / pd.Series([sky_core.clear_sky_ghi_haurwitz(math.cos(math.radians(z))) for z in zen], index=ghi.index).replace(0, np.nan)).clip(0, 1.2).fillna(0)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.80, 0.9511 - 0.1604*kt + 4.388*kt**2 - 16.638*kt**3 + 12.336*kt**4,
                           0.165))
    kd = pd.Series(kd, index=ghi.index).clip(0.05, 1.0)
    dhi = (kd * ghi).clip(lower=0)
    dni = ((ghi - dhi) / cosz).clip(lower=0, upper=1200)
    return dni, dhi


def compute_gti_poa(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    times = pd.to_datetime(out["prediction_time"])
    ghi = pd.to_numeric(out["predicted_ghi_wm2"], errors="coerce").fillna(0).clip(lower=0)

    # Scale Open-Meteo DNI/DHI to our predicted GHI when available.
    om_dni = pd.to_numeric(out.get("om_dni_wm2", pd.Series(np.nan, index=out.index)), errors="coerce")
    om_dhi = pd.to_numeric(out.get("om_dhi_wm2", pd.Series(np.nan, index=out.index)), errors="coerce")

    if pvlib is not None:
        localized = pd.DatetimeIndex(times).tz_localize(TZ_NAME, nonexistent="shift_forward", ambiguous="NaT")
        solpos = pvlib.solarposition.get_solarposition(localized, LAT, LON)
        zenith = pd.Series(solpos["apparent_zenith"].values, index=out.index).clip(0, 90)
        azimuth = pd.Series(solpos["azimuth"].values, index=out.index)
        cosz = np.cos(np.radians(zenith)).clip(lower=0)
        horizontal = pd.to_numeric(out.get("om_direct_horizontal_wm2", pd.Series(np.nan, index=out.index)), errors="coerce")
        om_dni = om_dni.fillna(horizontal / cosz.where(cosz > .065))
        # Preserve GHI = DNI*cos(zenith) + DHI after scaling weather components.
        denominator = om_dni * cosz + om_dhi
        usable = (om_dni >= 0) & (om_dhi >= 0) & (denominator > 0)
        scale = ghi / denominator.where(usable)
        dni_from_om = (om_dni * scale).where(usable)
        dhi_from_om = (om_dhi * scale).where(usable)

        dni_calc = pd.Series(index=out.index, dtype=float)
        dhi_calc = pd.Series(index=out.index, dtype=float)
        missing = dni_from_om.isna() | dhi_from_om.isna()
        if missing.any():
            try:
                erbs = pvlib.irradiance.erbs(ghi[missing], zenith[missing], localized[missing].dayofyear)
                dni_calc.loc[missing] = erbs["dni"].values
                dhi_calc.loc[missing] = erbs["dhi"].values
            except Exception:
                d1, d2 = decompose_ghi_simple(ghi[missing], zenith[missing])
                dni_calc.loc[missing] = d1
                dhi_calc.loc[missing] = d2
        dni_calc.loc[~missing] = dni_from_om.loc[~missing]
        dhi_calc.loc[~missing] = dhi_from_om.loc[~missing]
        dni_calc = dni_calc.fillna(0).clip(0, 1200).where((ghi > 0) & (zenith < 87), 0)
        dni_calc = np.minimum(dni_calc, ghi / cosz.clip(lower=.001))
        dhi_calc = (ghi - dni_calc * cosz).clip(lower=0)

        # Perez transposition for GTI/POA.
        try:
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=SURFACE_TILT,
                surface_azimuth=SURFACE_AZIMUTH,
                dni=dni_calc.clip(0, 1200),
                ghi=ghi,
                dhi=dhi_calc.clip(0, 1200),
                solar_zenith=zenith,
                solar_azimuth=azimuth,
                albedo=ALBEDO,
                model="perez",
                dni_extra=pd.Series(pvlib.irradiance.get_extra_radiation(localized).to_numpy(), index=out.index),
                airmass=pvlib.atmosphere.get_relative_airmass(zenith),
            )
            out["predicted_gti_poa_wm2"] = pd.Series(poa["poa_global"].values, index=out.index).clip(0, 1400)
            method = pd.Series(np.where(missing, "Perez with estimated DNI/DHI", "Perez with Open-Meteo DNI/DHI"), index=out.index)
        except Exception as exc:
            print(f"Perez transposition failed; using isotropic fallback: {exc}")
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=SURFACE_TILT,
                surface_azimuth=SURFACE_AZIMUTH,
                dni=dni_calc.clip(0, 1200),
                ghi=ghi,
                dhi=dhi_calc.clip(0, 1200),
                solar_zenith=zenith,
                solar_azimuth=azimuth,
                albedo=ALBEDO,
                model="isotropic",
            )
            out["predicted_gti_poa_wm2"] = pd.Series(poa["poa_global"].values, index=out.index).clip(0, 1400)
            method = "isotropic fallback via pvlib"

        out["predicted_gti_poa_wm2"] = out["predicted_gti_poa_wm2"].where(ghi > 0, 0).fillna(0)

        out["solar_zenith_deg"] = zenith.values
        out["solar_azimuth_deg"] = azimuth.values
        out["predicted_dni_wm2"] = dni_calc.clip(0, 1200).values
        out["predicted_dhi_wm2"] = dhi_calc.clip(0, 1200).values
        out["gti_method"] = method
        return out

    # No pvlib fallback: this is approximate and cannot implement Perez geometry.
    print("Warning: pvlib unavailable; POA uses a crude tilt approximation, not Perez.")
    zenith = pd.to_numeric(out.get("solar_zenith_deg", np.nan), errors="coerce").fillna(90).clip(0, 90)
    dni, dhi = decompose_ghi_simple(ghi, zenith)
    tilt = math.radians(SURFACE_TILT)
    # Crude fixed-tilt approximation if pvlib unavailable.
    tilt_gain = 0.95 + 0.12 * math.cos(tilt)
    poa = (ghi * tilt_gain).clip(0, 1300)
    out["predicted_dni_wm2"] = dni
    out["predicted_dhi_wm2"] = dhi
    out["predicted_gti_poa_wm2"] = poa
    out["gti_method"] = "simple fallback; install pvlib for Perez"
    return out


# ---------------------------------------------------------------------
# Power prediction
# ---------------------------------------------------------------------
def fit_power_from_actual(df: pd.DataFrame, actual_power: pd.DataFrame | None) -> tuple[float | None, str]:
    if actual_power is None or actual_power.empty or df.empty:
        return None, "insufficient_prior_actual_power"
    left = df[["prediction_time", "predicted_gti_poa_wm2"]].drop_duplicates("prediction_time").sort_values("prediction_time").copy()
    right = actual_power[["timestamp", "actual_dc_power_kw"]].sort_values("timestamp").copy()
    left["prediction_time"] = left["prediction_time"].astype("datetime64[ns]")
    right["timestamp"] = right["timestamp"].astype("datetime64[ns]")
    valid = pd.merge_asof(left, right, left_on="prediction_time", right_on="timestamp", direction="nearest",
                          tolerance=pd.Timedelta(minutes=CFG.get("actual_match_tolerance_minutes", 8)))
    valid = valid.dropna().drop_duplicates("timestamp")
    valid = valid[(valid["predicted_gti_poa_wm2"] > 100) & (valid["actual_dc_power_kw"] > 0)]
    if len(valid) >= 3:
        ratio = valid["actual_dc_power_kw"] / valid["predicted_gti_poa_wm2"]
        return float(ratio.median()), "calibrated_median_from_prior_current_rows"
    return None, "insufficient_prior_actual_power"


def predict_power(df: pd.DataFrame, actual_power: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    poa = pd.to_numeric(out["predicted_gti_poa_wm2"], errors="coerce").fillna(0).clip(lower=0)
    out["estimated_cell_temperature_c"] = estimate_cell_temperature(poa, out.get("om_temp_c"), out.get("om_wind_ms"))
    out["predicted_dc_power_kw"] = np.nan
    out["power_model_method"] = "unavailable"
    out["used_dc_capacity_kw"] = np.nan
    out["power_calibration_kw_per_wm2"] = np.nan
    warned = False
    for origin, indices in out.groupby("origin_image_time").groups.items():
        # Strictly earlier training rows/measurements: evaluating a forecast must not fit its targets.
        prior = out[(out["section"] == "image_time_current") & (out["prediction_time"] < origin)]
        history = actual_power[actual_power["timestamp"] < origin] if actual_power is not None else None
        k, method = fit_power_from_actual(prior, history) if CFG.get("power_calibration_enabled", True) else (None, "disabled")
        if k is not None:
            out.loc[indices, "predicted_dc_power_kw"] = k * poa.loc[indices]
            out.loc[indices, "power_model_method"] = method
            out.loc[indices, "power_calibration_kw_per_wm2"] = k
            continue
        if DC_CAPACITY_KW is not None:
            cap = float(DC_CAPACITY_KW)
            method = "PVWatts_like_configured_capacity"
        elif history is not None and not history.empty and history["actual_dc_power_kw"].max() > 0:
            cap = float(history["actual_dc_power_kw"].quantile(.98) / max(DEFAULT_PR, .1))
            method = "PVWatts_like_capacity_estimated_from_prior_power"
        else:
            cap = 1000.0
            method = "PVWatts_like_PLACEHOLDER_1000kW"
            if not warned:
                print("Warning: no prior calibration/capacity; using PLACEHOLDER 1000 kW DC. Set dc_capacity_kw.")
                warned = True
        correction = (1 + GAMMA * (out.loc[indices, "estimated_cell_temperature_c"] - 25)).clip(.70, 1.10)
        out.loc[indices, "predicted_dc_power_kw"] = (cap * poa.loc[indices] / 1000 * correction * DEFAULT_PR).clip(0, cap * 1.05)
        out.loc[indices, "power_model_method"] = method
        out.loc[indices, "used_dc_capacity_kw"] = cap
    return out


# ---------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------
def attach_actuals(df: pd.DataFrame, actual_ghi: pd.DataFrame | None, actual_power: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    out["actual_ghi_wm2"] = nearest_actual(out["prediction_time"], actual_ghi, "actual_ghi_wm2")
    out["actual_dc_power_kw"] = nearest_actual(out["prediction_time"], actual_power, "actual_dc_power_kw")
    if actual_power is not None and not actual_power.empty:
        for col in ["actual_dc_voltage_v", "actual_dc_current_a"]:
            if col in actual_power.columns:
                out[col] = nearest_actual(out["prediction_time"], actual_power, col)

    out["ghi_error_actual_minus_predicted_wm2"] = out["actual_ghi_wm2"] - out["predicted_ghi_wm2"]
    out["ghi_absolute_error_wm2"] = out["ghi_error_actual_minus_predicted_wm2"].abs()
    out["ghi_accuracy_percent"] = np.where(
        out["actual_ghi_wm2"] > 50,
        np.clip(100.0 * (1.0 - out["ghi_absolute_error_wm2"] / out["actual_ghi_wm2"]), 0.0, 100.0),
        np.nan,
    )

    out["power_error_actual_minus_predicted_kw"] = out["actual_dc_power_kw"] - out["predicted_dc_power_kw"]
    out["power_absolute_error_kw"] = out["power_error_actual_minus_predicted_kw"].abs()
    out["power_accuracy_percent"] = np.where(
        out["actual_dc_power_kw"] > 1,
        np.clip(100.0 * (1.0 - out["power_absolute_error_kw"] / out["actual_dc_power_kw"]), 0.0, 100.0),
        np.nan,
    )
    return out


def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(3)
    for c in ["origin_image_time", "prediction_time", "image_time", "forecast_time"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def make_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = df.copy()
    aliases = {"origin_image_time": "image_time", "prediction_time": "forecast_time",
               "image_file": "source_image_file", "cloud_nature": "cloud_class", "okta_corrected": "okta",
               "om_ghi_wm2": "openmeteo_ghi_wm2", "om_temp_c": "openmeteo_temperature_c", "om_wind_ms": "openmeteo_wind_speed_ms"}
    for source, target in aliases.items():
        table[target] = table[source] if source in table else np.nan
    common = ["section", "image_time", "forecast_time", "horizon_min"]
    ghi_cols = common + ["predicted_ghi_wm2", "actual_ghi_wm2", "ghi_error_actual_minus_predicted_wm2",
        "ghi_absolute_error_wm2", "ghi_accuracy_percent", "predicted_csi", "clear_sky_ghi_wm2", "cloud_class",
        "okta", "forecast_confidence", "source_image_file"]
    power_cols = common + ["predicted_ghi_wm2", "predicted_gti_poa_wm2", "predicted_dc_power_kw", "actual_dc_voltage_v",
        "actual_dc_current_a", "actual_dc_power_kw", "power_error_actual_minus_predicted_kw", "power_absolute_error_kw",
        "power_accuracy_percent", "power_model_method", "forecast_confidence", "source_image_file", "gti_method"]
    combined_cols = common + ["predicted_ghi_wm2", "actual_ghi_wm2", "predicted_gti_poa_wm2", "predicted_dc_power_kw",
        "actual_dc_power_kw", "ghi_error_actual_minus_predicted_wm2", "power_error_actual_minus_predicted_kw",
        "ghi_accuracy_percent", "power_accuracy_percent", "openmeteo_ghi_wm2", "openmeteo_temperature_c",
        "openmeteo_wind_speed_ms", "cloud_class", "okta", "forecast_confidence", "source_image_file", "gti_method"]
    return tuple(round_numeric(table.reindex(columns=cols)) for cols in (ghi_cols, power_cols, combined_cols))


def summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for section, group in df.groupby("section", sort=False):
        for target, unit, error_col in [("ghi", "W/m2", "ghi_error_actual_minus_predicted_wm2"),
                                         ("power", "kW", "power_error_actual_minus_predicted_kw")]:
            error = group[error_col].replace([np.inf, -np.inf], np.nan).dropna()
            accuracy = group[f"{target}_accuracy_percent"].dropna()
            rows.append(dict(section=section, target=target, unit=unit, matched_count=len(error),
                             mae=error.abs().mean(), rmse=np.sqrt((error**2).mean()), bias_actual_minus_predicted=error.mean(),
                             average_accuracy_percent=accuracy.mean(), accuracy_count=len(accuracy)))
    return pd.DataFrame(rows)


def print_summary(combined: pd.DataFrame) -> None:
    print("\nCreated rows by section:")
    print(combined["section"].value_counts(dropna=False).to_string())
    print("\nPreview:")
    preview_cols = ["section", "forecast_time", "predicted_ghi_wm2", "predicted_dc_power_kw", "forecast_confidence"]
    print(combined[preview_cols].head(20).to_string(index=False))


def main() -> None:
    for folder in [IMAGE_DIR, GHI_DIR, POWER_DIR, OPENMETEO_DIR, MODEL_DIR, OUTPUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    if STEP_MIN <= 0 or HORIZON_MIN < STEP_MIN:
        raise ValueError("Forecast step/horizon must be positive and horizon >= step")
    if DC_CAPACITY_KW is not None and float(DC_CAPACITY_KW) <= 0:
        raise ValueError("dc_capacity_kw must be positive or null")
    print(f"Site: {LAT}, {LON}, {TZ_NAME}; tilt={SURFACE_TILT}, azimuth={SURFACE_AZIMUTH}, albedo={ALBEDO}")
    print(f"Capacity={DC_CAPACITY_KW} kW; loss factor={DEFAULT_PR}; temperature coefficient={GAMMA}/C")
    model_files = [p for p in MODEL_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if model_files:
        print(f"Found {len(model_files)} files in inputs/models; reserved for a future trained-model interface. Using existing adaptive image rules.")
    features = analyse_images()
    start = features["image_time"].min() - pd.Timedelta(hours=2)
    end = features["image_time"].max() + pd.Timedelta(minutes=HORIZON_MIN + 60)

    actual_ghi = load_actual_ghi()
    actual_power = load_actual_power()
    print(f"Loaded actual GHI: {0 if actual_ghi is None else len(actual_ghi)} rows")
    print(f"Loaded actual DC power: {0 if actual_power is None else len(actual_power)} rows")
    if actual_ghi is None or actual_ghi.empty:
        print("Warning: no usable actual GHI measurements; comparison metrics will be blank.")
    if actual_power is None or actual_power.empty:
        print("Warning: no usable actual DC power measurements; comparison metrics will be blank.")
    om = fetch_openmeteo(start, end)
    if om is not None:
        safe_to_csv(om, OUTPUT_DIR / "openmeteo_interpolated_source_hourly.csv")

    pred_rows = make_prediction_rows(features)
    pred = attach_predictions(pred_rows, features, om)
    pred = compute_gti_poa(pred)
    pred = predict_power(pred, actual_power)
    pred = attach_actuals(pred, actual_ghi, actual_power)
    for col in ["actual_ghi_wm2", "actual_dc_power_kw"]:
        if pred[col].isna().all():
            print(f"Warning: no matching {col} values within timestamp tolerance.")

    metrics = summary_metrics(pred)
    safe_to_csv(round_numeric(metrics), OUTPUT_DIR / "summary_metrics.csv")
    for actual, filename in [(actual_ghi, "actual_ghi_normalized.csv"), (actual_power, "actual_power_normalized.csv")]:
        if actual is not None:
            safe_to_csv(actual, OUTPUT_DIR / filename)
    ghi_table, power_table, combined_table = make_tables(pred)
    safe_to_csv(ghi_table, OUTPUT_DIR / "ghi_comparison_table.csv")
    safe_to_csv(power_table, OUTPUT_DIR / "power_comparison_table.csv")
    safe_to_csv(combined_table, OUTPUT_DIR / "combined_ghi_power_table.csv")
    safe_to_csv(round_numeric(pred), OUTPUT_DIR / "all_internal_predictions_with_features.csv")

    print("\nDone.")
    print("Outputs:")
    print(f" - {OUTPUT_DIR / 'ghi_comparison_table.csv'}")
    print(f" - {OUTPUT_DIR / 'power_comparison_table.csv'}")
    print(f" - {OUTPUT_DIR / 'combined_ghi_power_table.csv'}")
    print(f" - {OUTPUT_DIR / 'image_feature_table.csv'}")
    print(f" - {OUTPUT_DIR / 'summary_metrics.csv'}")
    print_summary(combined_table)
    print("\nMetrics (blank means no matched actual measurements):")
    print(metrics.round(3).to_string(index=False))


def run_forecast(lat=LAT, lon=LON, timezone=TZ_NAME, panel_tilt_deg=SURFACE_TILT,
                 panel_azimuth_deg=SURFACE_AZIMUTH, albedo=ALBEDO,
                 plant_dc_capacity_kw=DC_CAPACITY_KW, temp_coeff_per_c=GAMMA,
                 loss_factor=DEFAULT_PR, forecast_horizon_hours=2,
                 forecast_interval_min=15, use_openmeteo=True,
                 use_manual_openmeteo=True, input_dirs=None, output_dir=None,
                 video_frame_interval_seconds=60, video_start_times=None,
                 image_timestamps=None):
    """Run the same CLI pipeline with isolated settings; return paths and status.

    A fresh module namespace prevents configuration from leaking between sessions.
    Calls are serialized because stdout capture is process-wide. Only paths written
    by this run are returned, including unique filenames used for Excel-locked CSVs.
    """
    import contextlib
    import importlib.util
    import io
    from zoneinfo import ZoneInfo
    log = io.StringIO()
    paths = {}
    error = None
    with _RUN_LOCK, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            zone = ZoneInfo(timezone)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Latitude or longitude is outside its valid range.")
            if not (0 <= panel_tilt_deg <= 90 and 0 <= panel_azimuth_deg <= 360 and 0 <= albedo <= 1):
                raise ValueError("Check panel tilt, azimuth and albedo.")
            if plant_dc_capacity_kw is None or plant_dc_capacity_kw <= 0:
                raise ValueError("Enter a positive DC capacity in kW for the system being compared.")
            if not 0 < loss_factor <= 1:
                raise ValueError("Loss factor must be greater than zero and at most one.")
            if forecast_interval_min <= 0 or forecast_horizon_hours * 60 < forecast_interval_min:
                raise ValueError("Forecast horizon must be at least one forecast interval.")
            spec = importlib.util.spec_from_file_location("_isolated_forecast_" + uuid.uuid4().hex, __file__)
            engine = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine)
            engine.LAT, engine.LON, engine.TZ_NAME = lat, lon, timezone
            engine.SURFACE_TILT, engine.SURFACE_AZIMUTH, engine.ALBEDO = panel_tilt_deg, panel_azimuth_deg, albedo
            engine.DC_CAPACITY_KW, engine.GAMMA, engine.DEFAULT_PR = plant_dc_capacity_kw, temp_coeff_per_c, loss_factor
            engine.HORIZON_MIN, engine.STEP_MIN = round(forecast_horizon_hours * 60), int(forecast_interval_min)
            engine.CFG.update(openmeteo_enabled=use_openmeteo, use_manual_openmeteo=use_manual_openmeteo,
                              power_calibration_enabled=False, video_frame_interval_seconds=video_frame_interval_seconds,
                              video_start_times=video_start_times or {}, image_timestamps=image_timestamps or {})
            # Honor historical timezone offsets for each image/forecast timestamp.
            original_solar = engine.local_solar_features
            def solar(ts):
                engine.UTC_OFFSET = pd.Timestamp(ts).tz_localize(zone).utcoffset().total_seconds() / 3600
                return original_solar(ts)
            engine.local_solar_features = solar
            engine.UTC_OFFSET = pd.Timestamp.now(tz=zone).utcoffset().total_seconds() / 3600
            for key, attr in [("images", "IMAGE_DIR"), ("ghi", "GHI_DIR"), ("power", "POWER_DIR"), ("openmeteo", "OPENMETEO_DIR"), ("models", "MODEL_DIR")]:
                if input_dirs and key in input_dirs:
                    setattr(engine, attr, Path(input_dirs[key]).resolve())
            engine.OUTPUT_DIR = Path(output_dir or OUTPUT_DIR).resolve()
            original_save = engine.safe_to_csv
            def save(frame, path):
                actual = original_save(frame, path)
                paths[path.name] = str(actual)
                return actual
            engine.safe_to_csv = save
            engine.main()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(error)
    lines = log.getvalue().splitlines()
    notices = [line for line in lines if any(term in line.lower() for term in ["warning", "failed", "skipping", "could not", "fallback", "unavailable", "no matching", "locked"])]
    return {"paths": paths, "status": {"success": error is None, "error": error,
            "warnings": notices, "log": log.getvalue(),
            "openmeteo": "Live" if any("Live Open-Meteo OK" in s for s in lines) else "Manual" if any("Manual Open-Meteo:" in s for s in lines) else "Unavailable",
            "power_detection": [s for s in lines if s.startswith("Power file") or s.startswith("Power conversion")]}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true", help="Skip live weather; use manual weather or image/clear-sky fallback")
    args = parser.parse_args()
    INPUT_DIR = args.input_dir.resolve()
    IMAGE_DIR, GHI_DIR, POWER_DIR, OPENMETEO_DIR, MODEL_DIR = [INPUT_DIR / name for name in ["images", "ghi", "power", "openmeteo", "models"]]
    OUTPUT_DIR = args.output_dir.resolve()
    if args.offline:
        CFG["openmeteo_enabled"] = False
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Cannot run forecast: {exc}", file=sys.stderr)
        sys.exit(1)
