from pathlib import Path
import math, zipfile, shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from skimage.filters import threshold_otsu
except Exception:
    threshold_otsu = None

#ROOT = Path('/mnt/data')
#OUT = ROOT / 'sky_prediction_v7_1km_cloud_field'
#(OUT/'outputs'/'overlays').mkdir(parents=True, exist_ok=True)

DEFAULT_LAT = 27.54225
DEFAULT_LON = 72.471
DEFAULT_TZ = 5.5


def solar_cos_zenith_noaa(dt_local, lat_deg: float, lon_deg: float, timezone_offset: float) -> float:
    dt_local = pd.to_datetime(dt_local)
    n = int(dt_local.dayofyear)
    hour = dt_local.hour + dt_local.minute / 60.0 + dt_local.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (n - 1 + (hour - 12.0) / 24.0)
    eq_time = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    time_offset = eq_time + 4.0 * lon_deg - 60.0 * timezone_offset
    true_solar_time_min = hour * 60.0 + time_offset
    hour_angle_deg = true_solar_time_min / 4.0 - 180.0
    hour_angle_rad = math.radians(hour_angle_deg)
    lat_rad = math.radians(lat_deg)
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle_rad)
    return max(0.0, min(1.0, cos_zenith))


def clear_sky_ghi_haurwitz(cos_zenith: float) -> float:
    if cos_zenith <= 0.0:
        return 0.0
    return 1098.0 * cos_zenith * math.exp(-0.059 / cos_zenith)


def local_texture(gray_float: np.ndarray, kernel: int = 21) -> np.ndarray:
    local_mean = cv2.GaussianBlur(gray_float, (kernel, kernel), 0)
    local_mean_sq = cv2.GaussianBlur(gray_float * gray_float, (kernel, kernel), 0)
    return np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0.0))


def soft_clip(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def select_ghi_columns(df):
    avg_cols = [c for c in df.columns if 'AVG-WMS' in c and 'GHI' in c]
    sensor_cols = [c for c in df.columns if 'WMS-' in c and 'GHI' in c]
    return (avg_cols[0] if avg_cols else None), sensor_cols


def interpolate_series_at_time(s, target):
    s = s.sort_index()
    target = pd.to_datetime(target)
    if target in s.index:
        val = s.loc[target]
        if hasattr(val, 'iloc'):
            val = val.iloc[0]
        if pd.notna(val):
            return float(val), 'exact'
    # interpolate only if nearest valid samples are reasonably close
    valid = s.dropna()
    before = valid[valid.index <= target]
    after = valid[valid.index >= target]
    if len(before) and len(after):
        t0, t1 = before.index[-1], after.index[0]
        if (target - t0) <= pd.Timedelta(minutes=10) and (t1 - target) <= pd.Timedelta(minutes=10):
            if t0 == t1:
                return float(valid.loc[t0]), 'exact_valid'
            y0, y1 = float(valid.loc[t0]), float(valid.loc[t1])
            frac = (target - t0) / (t1 - t0)
            return y0 + (y1 - y0) * frac, f'interpolated_{t0.strftime("%H:%M")}_{t1.strftime("%H:%M")}'
    return np.nan, 'not_available'


def read_actual_ghi_from_plant(plant_csv: str, timestamp_local):
    df = pd.read_csv(plant_csv)
    if 'DATETIME' not in df.columns:
        raise ValueError('DATETIME column not found.')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    avg_col, sensor_cols = select_ghi_columns(df)
    target = pd.to_datetime(timestamp_local)

    # Prefer AVG-WMS when present and locally available.
    if avg_col:
        s_avg = pd.to_numeric(df[avg_col], errors='coerce') * 1000.0
        s_avg.index = df['DATETIME']
        avg_val, avg_method = interpolate_series_at_time(s_avg, target)
        if pd.notna(avg_val):
            return float(avg_val), 'AVG-WMS', avg_method

    # Fallback only when AVG-WMS is unavailable near timestamp: median of individual WMS sensors.
    if sensor_cols:
        sensor_df = df.set_index('DATETIME')[sensor_cols].apply(pd.to_numeric, errors='coerce') * 1000.0
        # Interpolate each sensor locally and take median of available values.
        vals = []
        methods = []
        for c in sensor_cols:
            val, method = interpolate_series_at_time(sensor_df[c], target)
            if pd.notna(val):
                vals.append(val)
                methods.append(method)
        if vals:
            return float(np.nanmedian(vals)), 'SENSOR-MEDIAN-FALLBACK', '+'.join(sorted(set(methods)))
    return np.nan, 'NO-GHI-AVAILABLE', 'not_available'



def cloudiness_level(index: float) -> str:
    """Human-readable class for 0-1 cloudiness / attenuation index."""
    if index < 0.25:
        return 'low'
    if index < 0.50:
        return 'moderate'
    if index < 0.75:
        return 'high'
    return 'very_high'


def describe_forecast_risk(risk_index: float) -> str:
    if risk_index < 0.25:
        return 'low_ramp_risk'
    if risk_index < 0.50:
        return 'moderate_ramp_risk'
    if risk_index < 0.75:
        return 'high_ramp_risk'
    return 'very_high_ramp_risk'


def estimate_cloud_nature(
    f_cloud_combined: float,
    blue_total: float,
    f_light_grey: float,
    f_textured_light_grey: float,
    f_smooth_light_grey: float,
    f_dark_v4: float,
    mean_texture: float,
    mean_saturation: float,
    glare_score: float,
    uniform_overcast_score: float,
    conflict_score: float,
    thick_light_grey_score: float,
    thin_light_grey_score: float,
    extreme_dense_overcast_flag: bool,
    sun_block_conflict_flag: bool,
    hazy_blue_context_flag: bool,
) -> tuple[str, str, str]:
    """Return cloud_nature, cloud_depth_proxy, and forecast_usefulness text.

    These are descriptive labels derived only from image features. They are not
    meteorological ceilometer labels; they are intended for forecasting features.
    """
    if extreme_dense_overcast_flag or (thick_light_grey_score > 0.35 and f_light_grey > 0.45):
        return (
            'thick_smooth_light_grey_overcast',
            'thick_low_transmission',
            'useful_for_low_GHI_and_low_short_term_recovery_flag'
        )
    if uniform_overcast_score > 0.55:
        return (
            'uniform_layered_overcast',
            'moderate_to_thick_layer',
            'useful_for_persistent_low_to_medium_GHI_flag'
        )
    if sun_block_conflict_flag or conflict_score > 0.35:
        return (
            'patchy_cloud_with_probable_sun_blocking',
            'spatially_variable_sun_path_blockage',
            'useful_for_ramp_risk_and_sun_region_weighting'
        )
    if f_dark_v4 > 0.25:
        return (
            'dark_convective_or_rain_cloud',
            'thick_dark_cloud',
            'useful_for_strong_GHI_drop_warning'
        )
    if hazy_blue_context_flag or (glare_score > 0.35 and blue_total > 0.25):
        return (
            'hazy_blue_sun_glare_transmissive_cloud',
            'thin_to_moderate_transmissive_cloud',
            'useful_for_high_diffuse_or_cloud_edge_enhancement_flag'
        )
    if f_textured_light_grey > 0.20 or thin_light_grey_score > 0.20:
        return (
            'textured_light_grey_broken_cloud',
            'thin_to_moderate_broken_cloud',
            'useful_for_cloud_motion_and_short_term_ramp_tracking'
        )
    if f_cloud_combined < 0.35 and blue_total > 0.35:
        return (
            'scattered_shallow_cumulus_or_mostly_open_sky',
            'shallow_scattered_cloud',
            'useful_for_high_CSI_but_local_cloud_shadow_tracking'
        )
    if f_cloud_combined < 0.70:
        return (
            'broken_patchy_mixed_cloud',
            'moderate_patchy_cloud',
            'useful_for_cloud_motion_vector_forecasting'
        )
    if f_smooth_light_grey > 0.25 or (f_light_grey > 0.35 and mean_saturation < 0.12):
        return (
            'light_grey_overcast_or_multilayer_cloud',
            'ambiguous_thin_vs_thick_light_grey_cloud',
            'low_confidence_without_sequence_or_plant_feedback'
        )

    return (
        'mixed_cloud_field',
        'ambiguous_mixed_depth',
        'use_as_general_cloudiness_feature'
    )


def describe_1km_cloud_field(
    cloud_nature: str,
    cloud_depth_proxy: str,
    f_cloud_combined: float,
    blue_total: float,
    f_light_grey: float,
    f_textured_light_grey: float,
    f_smooth_light_grey: float,
    f_dark_v4: float,
    f_bright_v4: float,
    glare_score: float,
    uniform_overcast_score: float,
    conflict_score: float,
    cloudiness_index: float,
    ramp_risk_index: float,
    predicted_csi: float,
    true_clear_blue_fraction: float | None = None,
    blue_tinted_cloud_fraction: float | None = None,
    display_bright_cloud_fraction: float | None = None,
    display_dark_cloud_fraction: float | None = None,
    display_cloud_fraction: float | None = None,
) -> tuple[str, str, str, str, str]:
    """Convert image factors into wording for an assumed 1 km × 1 km sky cell.

    This is not a direct satellite 1 km retrieval. It assumes the camera view is
    representative of the local 1 km × 1 km area around the plant/camera.
    """
    cloud_pct = 100.0 * float(np.clip(f_cloud_combined, 0, 1))
    # If display-friendly fractions are supplied, prefer them for wording.
    open_pct = 100.0 * float(np.clip(true_clear_blue_fraction if true_clear_blue_fraction is not None else blue_total, 0, 1))
    blue_tinted_pct = 100.0 * float(np.clip(blue_tinted_cloud_fraction if blue_tinted_cloud_fraction is not None else 0.0, 0, 1))
    grey_pct = 100.0 * float(np.clip(f_light_grey, 0, 1))
    smooth_grey_pct = 100.0 * float(np.clip(f_smooth_light_grey, 0, 1))
    textured_grey_pct = 100.0 * float(np.clip(f_textured_light_grey, 0, 1))
    dark_pct = 100.0 * float(np.clip(display_dark_cloud_fraction if display_dark_cloud_fraction is not None else f_dark_v4, 0, 1))
    bright_pct = 100.0 * float(np.clip(display_bright_cloud_fraction if display_bright_cloud_fraction is not None else f_bright_v4, 0, 1))
    visual_cloud_pct = 100.0 * float(np.clip(display_cloud_fraction if display_cloud_fraction is not None else max(f_cloud_combined, f_bright_v4 + f_dark_v4), 0, 1))

    distribution = (
        f"Approximate 1 km cell composition: about {visual_cloud_pct:.0f}% of the scene looks cloud-affected and about "
        f"{open_pct:.0f}% looks like true clear-blue sky. Within the cloud-affected part, around "
        f"{blue_tinted_pct:.0f}% of the total scene appears as blue-tinted cloud, around "
        f"{grey_pct:.0f}% appears as light-grey cloud (smooth {smooth_grey_pct:.0f}%, textured {textured_grey_pct:.0f}%), "
        f"around {bright_pct:.0f}% appears as bright/white cloud, and around {dark_pct:.0f}% appears as dark cloud. "
        f"In simple words, much of the scene may look bluish to the camera, but only the true clear-blue part should be treated as open sky."
    )

    assumption = (
        "1 km × 1 km wording assumes this ground sky image represents the local "
        "sky cell around the camera/plant; it is a descriptive cloud-field label, "
        "not a direct satellite 1 km product."
    )

    # Choose the dominant field type and wording.
    if 'thick_smooth_light_grey' in cloud_nature or (f_smooth_light_grey > 0.55 and predicted_csi < 0.25):
        field_type = 'dominant thick smooth light-grey overcast layer'
        impact = (
            "Most of the 1 km cell is likely covered by a smooth, optically thick, "
            "light-grey stratiform cloud deck. Direct sunlight is strongly blocked; "
            "GHI is expected to remain low and spatially uniform unless a cloud edge enters the cell."
        )
        forecast_note = (
            "Useful for low-GHI persistence forecasting. Ramp risk is usually low inside the cell, "
            "but can become high at the boundary of this overcast deck."
        )
    elif 'uniform_layered_overcast' in cloud_nature:
        field_type = 'continuous layered overcast field'
        impact = (
            "The 1 km cell is likely dominated by a layered overcast sheet. The cloud is not necessarily dark, "
            "but the continuous cover reduces the direct beam and keeps CSI in a low-to-medium range."
        )
        forecast_note = (
            "Useful for persistence and slow-recovery forecasting; cloud motion is less important than layer thinning/thickening."
        )
    elif 'dark_convective' in cloud_nature:
        field_type = 'dark convective or rain-bearing cloud patch'
        impact = (
            "A dense dark cloud portion is likely present inside the 1 km cell. It can produce strong local shading, "
            "rapid GHI drops, and possible rain-cloud attenuation."
        )
        forecast_note = (
            "Use this as a high-priority ramp-down warning feature; track cloud edge movement frame-to-frame."
        )
    elif 'sun_blocking' in cloud_nature or conflict_score > 0.35:
        field_type = 'broken patchy cloud with probable sun-path blocking'
        impact = (
            "The 1 km cell likely contains mixed blue/open regions and cloud patches, but the sun path is probably obstructed. "
            "This creates large spatial variability: some parts may receive sun while the plant sensor can remain shaded."
        )
        forecast_note = (
            "High value for short-term ramp forecasting; use image sequence/cloud-motion vectors to see if the sun-blocking patch moves away."
        )
    elif 'hazy_blue' in cloud_nature:
        field_type = 'hazy-blue transmissive cloud or thin veil cloud field'
        impact = (
            "The 1 km cell likely has thin/hazy cloud or cloud-edge glare with significant transmission. "
            "Diffuse irradiance may be high, and CSI can remain moderate to high despite visible cloud."
        )
        forecast_note = (
            "Watch for cloud-edge enhancement and fast changes near the sun region; confidence improves with fixed exposure and image sequence."
        )
    elif 'textured_light_grey' in cloud_nature or f_textured_light_grey > 0.20:
        field_type = 'textured broken light-grey cloud field'
        impact = (
            "The 1 km cell likely contains broken stratocumulus/altocumulus-like light-grey cloud elements. "
            "Transmission is mixed: cloud gaps and thin patches can allow moderate GHI, while thicker patches create local dips."
        )
        forecast_note = (
            "Good candidate for cloud-motion-vector forecasting because edges and texture can be tracked between frames."
        )
    elif 'scattered_shallow' in cloud_nature or (f_cloud_combined < 0.35 and blue_total > 0.35):
        field_type = 'scattered shallow cumulus with mostly open sky'
        impact = (
            "The 1 km cell is mostly open sky with small shallow cloud patches. GHI should generally be high, "
            "but brief local dips are possible if a small cloud crosses the sun."
        )
        forecast_note = (
            "Use for high-CSI forecasting with localized shadow tracking; ramp risk depends on cloud speed and sun-region crossing."
        )
    elif f_light_grey > 0.35:
        field_type = 'ambiguous light-grey multilayer cloud field'
        impact = (
            "The 1 km cell is dominated by light-grey cloud. From RGB alone, this can be thin transmissive cloud or thick low-transmission cloud, "
            "so the irradiance impact is uncertain without image sequence or plant feedback."
        )
        forecast_note = (
            "Mark as low-confidence for CSI; future model should learn this class from plant-derived CSI targets."
        )
    else:
        field_type = 'mixed cloud field'
        impact = (
            "The 1 km cell appears to contain a mixed cloud field with no single dominant class. "
            "GHI impact depends strongly on whether the sun-region portion is covered."
        )
        forecast_note = (
            "Use all image features plus previous GHI trend for forecasting; single-image confidence is limited."
        )

    # Add a compact severity phrase.
    if cloudiness_index >= 0.75:
        severity = 'Overall cloudiness impact is high.'
    elif cloudiness_index >= 0.50:
        severity = 'Overall cloudiness impact is moderate.'
    else:
        severity = 'Overall cloudiness impact is low to moderate.'
    if ramp_risk_index >= 0.60:
        severity += ' Short-term ramp risk is high because the scene is patchy/textured or sun-region-sensitive.'
    elif ramp_risk_index >= 0.35:
        severity += ' Short-term ramp risk is moderate.'
    else:
        severity += ' Short-term ramp risk is low for the current frame.'

    impact = impact + ' ' + severity
    return field_type, distribution, impact, forecast_note, assumption


def analyse_image_versions(image_path: str, timestamp_local=None, lat=DEFAULT_LAT, lon=DEFAULT_LON, tz=DEFAULT_TZ):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0

    # Remove screenshot/video progress bar and very dark non-sky objects.
    analysis_area = np.ones((h, w), dtype=bool)
    analysis_area[int(0.955 * h):, :] = False
    valid_sky = (V > 0.08) & analysis_area
    n_sky = int(valid_sky.sum())
    if n_sky == 0:
        raise ValueError('No valid sky pixels detected')

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tex = local_texture(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    rbr = (R + 1.0) / (B + 1.0)
    brr = (B + 1.0) / (R + 1.0)
    nbd = (B - R) / (B + R + 1.0)
    nbg = (B - G) / (B + G + 1.0)
    norm_blue = B / (R + G + B + 1.0)
    grayness = np.std(np.stack([R, G, B], axis=-1), axis=-1) / (np.mean(np.stack([R, G, B], axis=-1), axis=-1) + 1.0)

    mean_brightness = float(V[valid_sky].mean())
    mean_saturation = float(S[valid_sky].mean())
    mean_texture = float(tex[valid_sky].mean())
    lap_var = float(np.var(lap[valid_sky]))
    mean_rbr = float(rbr[valid_sky].mean())
    mean_brr = float(brr[valid_sky].mean())
    mean_norm_blue = float(norm_blue[valid_sky].mean())
    brightness_p10 = float(np.percentile(V[valid_sky], 10))
    brightness_p90 = float(np.percentile(V[valid_sky], 90))
    brightness_spread = brightness_p90 - brightness_p10

    hue_blue_conf = np.clip(1.0 - np.abs(H - 105.0) / 35.0, 0.0, 1.0)

    # V1: original hard RGB/HSV rule.
    clear_v1 = valid_sky & (B > 1.05 * R) & (B > 1.02 * G) & (S > 0.18) & (V > 0.25)
    dark_v1 = valid_sky & (~clear_v1) & (V < 0.50)
    bright_v1 = valid_sky & (~clear_v1) & (~dark_v1)
    f_clear_v1 = float(clear_v1.sum() / n_sky)
    f_dark_v1 = float(dark_v1.sum() / n_sky)
    f_bright_v1 = float(bright_v1.sum() / n_sky)
    csi_v1 = float(np.clip(1.00 * f_clear_v1 + 0.60 * f_bright_v1 + 0.15 * f_dark_v1, 0.0, 1.05))

    # V2: soft RGB/HSV with uniform-overcast cap.
    blue_conf = (
        soft_clip(nbd, 0.025, 0.145)
        * soft_clip(nbg, 0.010, 0.090)
        * soft_clip(S, 0.080, 0.300)
        * hue_blue_conf
        * soft_clip(V, 0.180, 0.630)
    )
    blue_conf = np.where(valid_sky, blue_conf, 0.0)
    dark_score = np.where(valid_sky, soft_clip(0.58 - V, 0.0, 0.42) * (1.0 - blue_conf), 0.0)
    f_clear_v2 = float(blue_conf[valid_sky].mean())
    f_dark_v2 = float(dark_score[valid_sky].mean())
    f_bright_v2 = float(max(0.0, 1.0 - f_clear_v2 - f_dark_v2))
    cloud_fraction_v2 = 1.0 - f_clear_v2
    raw_csi_v2 = float(np.clip(0.98 * f_clear_v2 + 0.55 * f_bright_v2 + 0.13 * f_dark_v2, 0.0, 1.05))
    low_sat_v2 = float(np.clip((0.13 - mean_saturation) / 0.10, 0.0, 1.0))
    low_tex_v2 = float(np.clip((0.018 - mean_texture) / 0.014, 0.0, 1.0))
    no_blue_v2 = float(np.clip((0.080 - f_clear_v2) / 0.080, 0.0, 1.0))
    full_cloud_v2 = float(np.clip((cloud_fraction_v2 - 0.85) / 0.15, 0.0, 1.0))
    overcast_strength_v2 = low_sat_v2 * low_tex_v2 * no_blue_v2 * full_cloud_v2
    csi_v2 = raw_csi_v2
    if overcast_strength_v2 > 0.20:
        overcast_cap_v2 = 0.24 + 0.12 * np.clip(mean_texture / 0.018, 0.0, 1.0) + 0.08 * np.clip((mean_brightness - 0.50) / 0.25, 0.0, 1.0)
        csi_v2 = min(csi_v2, float(overcast_cap_v2))
    csi_v2 = float(np.clip(csi_v2, 0.02, 1.05))

    # V3: adaptive RBR/Otsu + dense-overcast and sun-blocked caps.
    if threshold_otsu is not None:
        try:
            rbr_thr = float(threshold_otsu(rbr[valid_sky]))
            cloud_otsu = valid_sky & (rbr > rbr_thr)
            f_cloud_otsu = float(cloud_otsu.sum() / n_sky)
        except Exception:
            rbr_thr = np.nan
            cloud_otsu = valid_sky & (~clear_v1)
            f_cloud_otsu = float(cloud_otsu.sum() / n_sky)
    else:
        rbr_thr = np.nan
        cloud_otsu = valid_sky & (~clear_v1)
        f_cloud_otsu = float(cloud_otsu.sum() / n_sky)

    f_clear_soft = f_clear_v2
    f_cloud_combined = float(np.clip(0.55 * cloud_fraction_v2 + 0.45 * f_cloud_otsu, 0.0, 1.0))

    # Pseudo sun / bright-region analysis when camera geometry is not calibrated.
    V_blur = cv2.GaussianBlur(V, (31, 31), 0)
    V_blur_masked = np.where(valid_sky, V_blur, -1.0)
    sun_y, sun_x = np.unravel_index(np.argmax(V_blur_masked), V_blur_masked.shape)
    radius = max(8, int(0.14 * min(h, w)))
    yy, xx = np.ogrid[:h, :w]
    sun_region = ((yy - sun_y) ** 2 + (xx - sun_x) ** 2 <= radius ** 2) & valid_sky
    if sun_region.sum() < 20:
        sun_region = valid_sky
    sun_brightness = float(V[sun_region].mean())
    sun_saturation = float(S[sun_region].mean())
    sun_texture = float(tex[sun_region].mean())
    sun_blue_conf = float(blue_conf[sun_region].mean())
    sun_dark_score = float(dark_score[sun_region].mean())
    top2_v = float(np.percentile(V[valid_sky], 98))
    top_mask = (V >= top2_v) & valid_sky
    top2_s = float(S[top_mask].mean()) if top_mask.sum() else mean_saturation
    glare_score = float(
        np.clip((top2_v - 0.72) / 0.16, 0.0, 1.0)
        * np.clip((0.16 - top2_s) / 0.14, 0.0, 1.0)
        * np.clip((mean_texture - 0.006) / 0.020, 0.0, 1.0)
    )

    no_blue = float(np.clip((0.12 - f_clear_soft) / 0.12, 0.0, 1.0))
    low_sat = float(np.clip((0.10 - mean_saturation) / 0.08, 0.0, 1.0))
    low_tex = float(np.clip((0.010 - mean_texture) / 0.008, 0.0, 1.0))
    full_cloud = float(np.clip((f_cloud_combined - 0.78) / 0.22, 0.0, 1.0))
    uniform_overcast_score = float(no_blue * low_sat * low_tex * full_cloud)

    conflict_score = float(
        np.clip((f_clear_soft - 0.25) / 0.45, 0.0, 1.0)
        * np.clip((f_cloud_otsu - 0.55) / 0.35, 0.0, 1.0)
        * np.clip((0.18 - top2_s) / 0.12, 0.0, 1.0)
    )

    csi_v3 = csi_v2
    sky_class_v3 = 'base_v2_patchy_or_mixed_cloud'
    dense_overcast_flag = (
        (f_clear_soft < 0.030)
        and (mean_saturation < 0.075)
        and (mean_texture < 0.0058)
        and (f_cloud_combined > 0.75)
    )
    if dense_overcast_flag:
        sky_class_v3 = 'dense_uniform_overcast_cap'
        csi_v3 = min(csi_v3, 0.10)

    sun_block_conflict_flag = (
        (f_clear_soft > 0.35)
        and (f_cloud_otsu > 0.72)
        and (top2_s > 0.10)
        and (glare_score < 0.45)
    )
    if sun_block_conflict_flag:
        sky_class_v3 = 'sun_blocked_conflicting_blue_cloud_cap'
        csi_v3 = min(csi_v3, 0.43)

    if (not dense_overcast_flag) and (not sun_block_conflict_flag) and (glare_score > 0.50) and (mean_texture > 0.012):
        sky_class_v3 = 'patchy_cloud_with_glare_keep_base'
        csi_v3 = max(csi_v3, min(0.62, csi_v2))
    csi_v3 = float(np.clip(csi_v3, 0.02, 1.00))

    # V4: sky-class-specific correction.
    # Adds a hazy-blue/sun-glare class so low-saturation bluish sky is not counted as cloud.
    strong_blue_mask = valid_sky & (hue_blue_conf > 0.45) & (norm_blue > 0.365) & (nbd > 0.075) & (S > 0.12) & (V > 0.20)
    hazy_blue_raw_mask = valid_sky & (~strong_blue_mask) & (hue_blue_conf > 0.45) & (norm_blue > 0.342) & (nbd > 0.020) & (S < 0.16) & (V > 0.20)
    dark_v4_mask = valid_sky & (dark_score > 0.45)

    f_strong_blue = float(strong_blue_mask.sum() / n_sky)
    f_hazy_blue_raw = float(hazy_blue_raw_mask.sum() / n_sky)
    f_dark_v4 = float(dark_v4_mask.sum() / n_sky)

    # This context gate prevents uniform grey overcast from being treated as hazy-blue sky.
    hazy_blue_context_flag = (
        (mean_saturation >= 0.080)
        and (mean_saturation <= 0.160)
        and (mean_texture > 0.012)
        and (brightness_spread > 0.300)
        and (mean_norm_blue > 0.346)
        and (glare_score > 0.250)
        and (uniform_overcast_score < 0.20)
        and (not dense_overcast_flag)
    )

    if hazy_blue_context_flag:
        f_hazy_blue = f_hazy_blue_raw
    else:
        # In non-hazy-blue cases we do not let low-saturation blue-ish pixels boost CSI.
        f_hazy_blue = 0.0
    f_bright_v4 = float(max(0.0, 1.0 - f_strong_blue - f_hazy_blue - f_dark_v4))

    csi_v4 = csi_v3
    sky_class_v4 = sky_class_v3

    # More conservative cap for extremely smooth/low-saturation overcast.
    extreme_dense_overcast_flag = (
        (f_clear_soft < 0.025)
        and (mean_saturation < 0.066)
        and (mean_texture < 0.0065)
        and (brightness_spread < 0.280)
        and (f_cloud_combined > 0.80)
        and (overcast_strength_v2 > 0.65)
    )
    if extreme_dense_overcast_flag:
        sky_class_v4 = 'extreme_dense_uniform_overcast_cap'
        # Slightly allow more transmission if the sky is bright, but keep it very low.
        extreme_cap = 0.045 + 0.020 * np.clip((mean_brightness - 0.50) / 0.25, 0.0, 1.0)
        csi_v4 = min(csi_v4, float(extreme_cap))

    # More conservative cap for blue-looking image with strong adaptive-cloud conflict and no strong glare.
    if sun_block_conflict_flag:
        sky_class_v4 = 'strong_blue_but_sun_blocked_conflict_cap'
        csi_v4 = min(csi_v4, 0.36)

    # Hazy-blue / circumsolar-glare boost. This specifically fixes cases where the strict rule counts
    # low-saturation bluish sky as cloud even though it is transmitting high irradiance.
    if hazy_blue_context_flag:
        sky_class_v4 = 'hazy_blue_sun_glare_transmissive_boost'
        hazy_csi = 1.00 * f_strong_blue + 0.82 * f_hazy_blue + 0.55 * f_bright_v4 + 0.12 * f_dark_v4
        glare_bonus = 0.04 * np.clip((glare_score - 0.25) / 0.45, 0.0, 1.0)
        csi_v4 = max(csi_v4, float(hazy_csi + glare_bonus))
        csi_v4 = min(csi_v4, 0.86)

    csi_v4 = float(np.clip(csi_v4, 0.02, 1.00))

    correction_strength_v4_pre = abs(csi_v4 - csi_v3)
    confidence_v4_pre = float(np.clip(
        1.0 - 0.55 * correction_strength_v4_pre - 0.25 * uniform_overcast_score - 0.20 * conflict_score + 0.10 * glare_score,
        0.1, 1.0
    ))
    hazy_mask_for_grey = hazy_blue_raw_mask if hazy_blue_context_flag else np.zeros_like(valid_sky, dtype=bool)

    # V5: dedicated light-grey cloud module.
    # Purpose: split bright/light-grey cloud into thick low-transmission and textured/glare-rich transmissive cases.
    rgb_mean_map = (R + G + B) / 3.0
    rgb_std_map = np.sqrt(((R - rgb_mean_map) ** 2 + (G - rgb_mean_map) ** 2 + (B - rgb_mean_map) ** 2) / 3.0)
    # High value means R, G, B are close: a grey/white cloud-like pixel.
    grey_similarity_map = 1.0 - np.clip(rgb_std_map / (rgb_mean_map + 1.0), 0.0, 1.0)

    light_grey_mask = (
        valid_sky
        & (grey_similarity_map > 0.92)
        & (S < 0.145)
        & (V > 0.34)
        & (V < 0.88)
        & (~strong_blue_mask)
        & (~hazy_mask_for_grey)
        & (~dark_v4_mask)
    )
    smooth_light_grey_mask = light_grey_mask & (tex < 0.010)
    textured_light_grey_mask = light_grey_mask & (tex >= 0.018)

    f_light_grey = float(light_grey_mask.sum() / n_sky)
    f_smooth_light_grey = float(smooth_light_grey_mask.sum() / n_sky)
    f_textured_light_grey = float(textured_light_grey_mask.sum() / n_sky)
    mean_light_grey_brightness = float(V[light_grey_mask].mean()) if light_grey_mask.sum() else np.nan
    mean_light_grey_saturation = float(S[light_grey_mask].mean()) if light_grey_mask.sum() else np.nan
    mean_light_grey_texture = float(tex[light_grey_mask].mean()) if light_grey_mask.sum() else np.nan

    blue_total_v4 = f_strong_blue + f_hazy_blue
    grey_amount_score = float(np.clip((f_light_grey - 0.35) / 0.55, 0.0, 1.0))
    grey_low_sat_score = float(np.clip((0.125 - mean_saturation) / 0.090, 0.0, 1.0))
    grey_no_blue_score = float(np.clip((0.14 - blue_total_v4) / 0.14, 0.0, 1.0))
    grey_texture_score = float(np.clip((mean_texture - 0.006) / 0.022, 0.0, 1.0))
    grey_low_texture_score = float(np.clip((0.014 - mean_texture) / 0.014, 0.0, 1.0))
    grey_spread_score = float(np.clip((brightness_spread - 0.22) / 0.20, 0.0, 1.0))
    grey_low_glare_score = float(np.clip((0.24 - glare_score) / 0.24, 0.0, 1.0))

    light_grey_score = grey_amount_score * grey_low_sat_score * max(grey_no_blue_score, 0.35)
    thick_light_grey_score = light_grey_score * max(grey_low_texture_score, 0.35) * max(grey_low_glare_score, 0.35)
    thin_light_grey_score = light_grey_score * grey_texture_score * np.clip((glare_score - 0.12) / 0.40, 0.0, 1.0)

    csi_v5 = csi_v4
    sky_class_v5 = sky_class_v4

    # Keep the already-successful V4 extreme dense-overcast cap. The grey module only confirms it.
    if extreme_dense_overcast_flag:
        sky_class_v5 = 'extreme_dense_overcast_grey_confirmed'
        csi_v5 = csi_v4

    # Thick light-grey cloud cap: light-grey pixels dominate, blue/hazy sky is low,
    # and texture is not high enough to suggest bright cloud edges or transmission.
    elif (light_grey_score > 0.18) and (mean_texture < 0.020) and (not hazy_blue_context_flag):
        light_grey_cap = (
            0.31
            + 0.13 * grey_texture_score
            + 0.06 * glare_score
            + 0.05 * grey_spread_score
            + 0.03 * f_strong_blue
        )
        csi_v5 = min(csi_v5, float(light_grey_cap))
        sky_class_v5 = 'thick_light_grey_cloud_cap'

    # Thin / textured light-grey cloud: grey cloud exists but has texture and glare/edge structure,
    # so it is allowed to transmit more than the conservative cap.
    elif (light_grey_score > 0.18) and (mean_texture >= 0.020) and (glare_score > 0.15) and (not hazy_blue_context_flag):
        thin_light_grey_floor = (
            0.48
            + 0.12 * np.clip((mean_texture - 0.018) / 0.025, 0.0, 1.0)
            + 0.04 * glare_score
            + 0.05 * grey_spread_score
            + 0.03 * f_strong_blue
        )
        csi_v5 = max(csi_v5, min(float(thin_light_grey_floor), 0.68))
        sky_class_v5 = 'thin_textured_light_grey_cloud_boost'

    # Small tightening for the existing blue-looking but sun-blocked conflict case.
    # This is not a grey-cloud rule, but it prevents optimistic CSI when adaptive cloud evidence disagrees with blue fraction.
    if sun_block_conflict_flag and (not hazy_blue_context_flag):
        csi_v5 = min(csi_v5, float(0.335 + 0.05 * glare_score))
        sky_class_v5 = 'blue_sky_conflict_refined_cap'

    csi_v5 = float(np.clip(csi_v5, 0.02, 1.00))
    correction_strength_v5 = abs(csi_v5 - csi_v4)
    confidence_v5 = float(np.clip(
        confidence_v4_pre - 0.18 * correction_strength_v5 - 0.12 * thick_light_grey_score + 0.08 * thin_light_grey_score,
        0.1, 1.0
    ))

    # Low-sun-angle correction for morning/evening images.
    # Reason: when the sun is close to the horizon, upper-sky blue pixels can look clear,
    # but plant GHI may remain low because of horizon haze, dust, low cloud, or local obstruction.
    # This correction is only applied when CSI is unrealistically high at low solar elevation
    # and no strong glare/sun evidence is present in the image.
    csi_v5_before_low_sun = float(csi_v5)
    low_sun_correction_flag = 0
    low_sun_csi_cap = np.nan
    low_sun_reason = ''
    if timestamp_local is not None:
        _cosz_low = solar_cos_zenith_noaa(pd.to_datetime(timestamp_local), lat, lon, tz)
        _solar_elev_low = math.degrees(math.asin(_cosz_low)) if _cosz_low > 0 else 0.0
        _clear_low = clear_sky_ghi_haurwitz(_cosz_low)
        low_sun_angle_flag = int(_solar_elev_low < 12.0)
        if (
            (_solar_elev_low < 12.0)
            and (_clear_low > 60.0)
            and (csi_v5 > 0.70)
            and (glare_score < 0.20)
            and (sun_brightness < 0.82)
        ):
            low_sun_csi_cap = float(
                0.18
                + 0.020 * _solar_elev_low
                + 0.08 * glare_score
                + 0.04 * brightness_spread
            )
            low_sun_csi_cap = float(np.clip(low_sun_csi_cap, 0.25, 0.48))
            csi_v5 = min(csi_v5, low_sun_csi_cap)
            low_sun_correction_flag = 1
            sky_class_v5 = sky_class_v5 + '_low_sun_angle_cap'
            low_sun_reason = (
                'Low sun angle with high image-only CSI but weak glare/sun evidence; '
                'CSI capped because morning/evening GHI can be limited by horizon haze, '
                'low cloud, dust, or local obstruction not visible in the upper-sky image.'
            )
        else:
            low_sun_reason = 'Low-sun cap not applied; conditions did not meet high-risk morning/evening overprediction rule.'
    else:
        low_sun_angle_flag = 0
        _solar_elev_low = np.nan
        low_sun_reason = 'Timestamp not supplied; low-sun correction not evaluated.'

    correction_strength_v5 = abs(csi_v5 - csi_v4)
    confidence_v5 = float(np.clip(
        confidence_v5 - 0.25 * low_sun_correction_flag,
        0.1, 1.0
    ))

    correction_strength_v4 = abs(csi_v4 - csi_v3)
    confidence_v4 = float(np.clip(
        1.0 - 0.55 * correction_strength_v4 - 0.25 * uniform_overcast_score - 0.20 * conflict_score + 0.10 * glare_score,
        0.1, 1.0
    ))


    # V6: cloud-factor descriptors useful for forecasting.
    # These are derived only from image + time metadata, not from plant GHI.
    blue_total_v5 = float(f_strong_blue + f_hazy_blue)
    cloud_cover_index = float(np.clip(f_cloud_combined, 0.0, 1.0))
    cloud_impact_index = float(np.clip(1.0 - csi_v5, 0.0, 1.0))
    cloudiness_index = float(np.clip(0.50 * cloud_cover_index + 0.50 * cloud_impact_index, 0.0, 1.0))
    ramp_risk_index = float(np.clip(
        0.45 * np.clip(mean_texture / 0.025, 0.0, 1.0)
        + 0.30 * np.clip(brightness_spread / 0.45, 0.0, 1.0)
        + 0.20 * conflict_score
        + 0.10 * np.clip(glare_score, 0.0, 1.0)
        - 0.20 * uniform_overcast_score,
        0.0, 1.0
    ))
    cloud_nature, cloud_depth_proxy, forecasting_value = estimate_cloud_nature(
        f_cloud_combined=f_cloud_combined,
        blue_total=blue_total_v5,
        f_light_grey=f_light_grey,
        f_textured_light_grey=f_textured_light_grey,
        f_smooth_light_grey=f_smooth_light_grey,
        f_dark_v4=f_dark_v4,
        mean_texture=mean_texture,
        mean_saturation=mean_saturation,
        glare_score=glare_score,
        uniform_overcast_score=uniform_overcast_score,
        conflict_score=conflict_score,
        thick_light_grey_score=thick_light_grey_score,
        thin_light_grey_score=thin_light_grey_score,
        extreme_dense_overcast_flag=extreme_dense_overcast_flag,
        sun_block_conflict_flag=sun_block_conflict_flag,
        hazy_blue_context_flag=hazy_blue_context_flag,
    )

    one_km_cloud_field_type, one_km_cloud_distribution, one_km_cloud_impact_description, one_km_forecasting_note, one_km_assumption = describe_1km_cloud_field(
        cloud_nature=cloud_nature,
        cloud_depth_proxy=cloud_depth_proxy,
        f_cloud_combined=f_cloud_combined,
        blue_total=blue_total_v5,
        f_light_grey=f_light_grey,
        f_textured_light_grey=f_textured_light_grey,
        f_smooth_light_grey=f_smooth_light_grey,
        f_dark_v4=f_dark_v4,
        f_bright_v4=f_bright_v4,
        glare_score=glare_score,
        uniform_overcast_score=uniform_overcast_score,
        conflict_score=conflict_score,
        cloudiness_index=cloudiness_index,
        ramp_risk_index=ramp_risk_index,
        predicted_csi=csi_v5,
    )

    # V8 display-friendly classes.
    # Earlier versions used strong_blue_mask as a technical blue-dominance feature.
    # For dashboard display, separate true clear sky from blue-tinted cloud.
    # This prevents a white/bluish cloud field from being reported as mostly clear sky.
    blue_dominant_mask_v8 = strong_blue_mask
    hazy_mask_v4 = hazy_blue_raw_mask if hazy_blue_context_flag else np.zeros_like(valid_sky, dtype=bool)

    # True clear blue = blue-dominant, not adaptive-cloud, not light-grey, not dark,
    # not too grey, and sufficiently saturated. This is stricter and more intuitive.
    true_clear_blue_mask_v8 = (
        blue_dominant_mask_v8
        & (~cloud_otsu)
        & (~light_grey_mask)
        & (~dark_v4_mask)
        & (grey_similarity_map < 0.935)
        & (S > 0.145)
        & (norm_blue > 0.372)
    )

    # Blue-tinted cloud = pixels that look blue to RGB/HSV but are probably cloud
    # because Otsu/grey/light-grey evidence says they are not clean open sky.
    blue_tinted_cloud_mask_v8 = blue_dominant_mask_v8 & (~true_clear_blue_mask_v8)

    clear_mask_v4 = true_clear_blue_mask_v8
    hazy_mask_v4 = hazy_mask_v4

    # Display bright cloud should include normal bright clouds + light-grey cloud
    # + blue-tinted cloud, excluding true clear and dark.
    bright_mask_v4 = (
        (valid_sky & (~true_clear_blue_mask_v8) & (~hazy_mask_v4) & (~dark_v4_mask))
        | blue_tinted_cloud_mask_v8
        | light_grey_mask
    ) & valid_sky & (~dark_v4_mask) & (~true_clear_blue_mask_v8) & (~hazy_mask_v4)

    # Interpretable dashboard fractions. These are for explanation/display.
    # The prediction logic above remains unchanged so V5/V7 accuracy is not affected.
    v8_blue_dominant_pixel_fraction = float(blue_dominant_mask_v8.sum() / n_sky)
    v8_true_clear_blue_sky_fraction = float(true_clear_blue_mask_v8.sum() / n_sky)
    v8_blue_tinted_cloud_fraction = float(blue_tinted_cloud_mask_v8.sum() / n_sky)
    v8_display_hazy_blue_fraction = float(hazy_mask_v4.sum() / n_sky)
    v8_display_bright_cloud_fraction = float(bright_mask_v4.sum() / n_sky)
    v8_display_dark_cloud_fraction = float(dark_v4_mask.sum() / n_sky)
    v8_display_cloud_fraction = float(np.clip(
        v8_display_bright_cloud_fraction + v8_display_dark_cloud_fraction,
        0.0, 1.0
    ))
    v8_display_okta = int(np.clip(np.rint(8.0 * v8_display_cloud_fraction), 0, 8))
    v8_display_note = (
        'Blue-dominant pixels are separated into true clear-blue sky and blue-tinted cloud. '
        'This makes the displayed cloud factors closer to visual interpretation.'
    )

    # Mid/high-sun scattered-clear-sky correction.
    # When the sky is mostly true clear-blue with only scattered bright clouds,
    # the earlier cloud penalty can underpredict GHI. This boost is not applied
    # to low-sun morning/evening cases, dense overcast, dark cloud, or conflict cases.
    scattered_clear_boost_flag = 0
    scattered_clear_csi_floor = np.nan
    scattered_clear_reason = ''
    if timestamp_local is not None:
        _cosz_sc = solar_cos_zenith_noaa(pd.to_datetime(timestamp_local), lat, lon, tz)
        _solar_elev_sc = math.degrees(math.asin(_cosz_sc)) if _cosz_sc > 0 else 0.0
        if (
            (_solar_elev_sc >= 20.0)
            and (v8_true_clear_blue_sky_fraction > 0.55)
            and (v8_display_cloud_fraction < 0.45)
            and (v8_display_dark_cloud_fraction < 0.04)
            and (f_light_grey < 0.28)
            and (conflict_score < 0.12)
            and (uniform_overcast_score < 0.15)
            and (low_sun_correction_flag == 0)
        ):
            _clear_score = np.clip((v8_true_clear_blue_sky_fraction - 0.55) / 0.35, 0.0, 1.0)
            _open_score = np.clip((0.45 - v8_display_cloud_fraction) / 0.45, 0.0, 1.0)
            scattered_clear_csi_floor = float(0.88 + 0.10 * _clear_score + 0.03 * _open_score)
            scattered_clear_csi_floor = float(np.clip(scattered_clear_csi_floor, 0.88, 0.96))
            if csi_v5 < scattered_clear_csi_floor:
                csi_v5 = scattered_clear_csi_floor
                scattered_clear_boost_flag = 1
                sky_class_v5 = sky_class_v5 + '_scattered_clear_sky_boost'
                scattered_clear_reason = (
                    'High sun angle with mostly true clear-blue sky and scattered bright clouds; '
                    'CSI boosted because clouds are unlikely to block the sun path strongly.'
                )
            else:
                scattered_clear_reason = 'Scattered-clear condition detected, but CSI was already high enough.'
        else:
            scattered_clear_reason = 'Scattered-clear boost not applied; image did not satisfy mostly clear mid/high-sun condition.'
    else:
        scattered_clear_reason = 'Timestamp not supplied; scattered-clear correction not evaluated.'

    # Recompute dependent indices after the possible scattered-clear-sky boost.
    cloud_impact_index = float(np.clip(1.0 - csi_v5, 0.0, 1.0))
    cloudiness_index = float(np.clip(0.50 * cloud_cover_index + 0.50 * cloud_impact_index, 0.0, 1.0))
    cloud_nature, cloud_depth_proxy, forecasting_value = estimate_cloud_nature(
        f_cloud_combined=f_cloud_combined,
        blue_total=blue_total_v5,
        f_light_grey=f_light_grey,
        f_textured_light_grey=f_textured_light_grey,
        f_smooth_light_grey=f_smooth_light_grey,
        f_dark_v4=f_dark_v4,
        mean_texture=mean_texture,
        mean_saturation=mean_saturation,
        glare_score=glare_score,
        uniform_overcast_score=uniform_overcast_score,
        conflict_score=conflict_score,
        thick_light_grey_score=thick_light_grey_score,
        thin_light_grey_score=thin_light_grey_score,
        extreme_dense_overcast_flag=extreme_dense_overcast_flag,
        sun_block_conflict_flag=sun_block_conflict_flag,
        hazy_blue_context_flag=hazy_blue_context_flag,
    )

    # Rebuild the 1 km wording using display-friendly fractions so the wording is intuitive.
    one_km_cloud_field_type, one_km_cloud_distribution, one_km_cloud_impact_description, one_km_forecasting_note, one_km_assumption = describe_1km_cloud_field(
        cloud_nature=cloud_nature,
        cloud_depth_proxy=cloud_depth_proxy,
        f_cloud_combined=f_cloud_combined,
        blue_total=blue_total_v5,
        f_light_grey=f_light_grey,
        f_textured_light_grey=f_textured_light_grey,
        f_smooth_light_grey=f_smooth_light_grey,
        f_dark_v4=f_dark_v4,
        f_bright_v4=f_bright_v4,
        glare_score=glare_score,
        uniform_overcast_score=uniform_overcast_score,
        conflict_score=conflict_score,
        cloudiness_index=cloudiness_index,
        ramp_risk_index=ramp_risk_index,
        predicted_csi=csi_v5,
        true_clear_blue_fraction=v8_true_clear_blue_sky_fraction,
        blue_tinted_cloud_fraction=v8_blue_tinted_cloud_fraction,
        display_bright_cloud_fraction=v8_display_bright_cloud_fraction,
        display_dark_cloud_fraction=v8_display_dark_cloud_fraction,
        display_cloud_fraction=v8_display_cloud_fraction,
    )

    if timestamp_local is not None:
        cosz = solar_cos_zenith_noaa(pd.to_datetime(timestamp_local), lat, lon, tz)
        solar_elev = math.degrees(math.asin(cosz)) if cosz > 0 else 0.0
    else:
        solar_elev = np.nan

    metrics = {
        'valid_sky_pixel_count': n_sky,
        'mean_brightness_0_to_1': mean_brightness,
        'mean_saturation_0_to_1': mean_saturation,
        'mean_texture_0_to_1': mean_texture,
        'laplacian_variance': lap_var,
        'brightness_p10': brightness_p10,
        'brightness_p90': brightness_p90,
        'brightness_spread': brightness_spread,
        'mean_red_blue_ratio': mean_rbr,
        'mean_blue_red_ratio': mean_brr,
        'mean_normalized_blue': mean_norm_blue,
        'v1_clear_fraction': f_clear_v1,
        'v1_bright_cloud_fraction': f_bright_v1,
        'v1_dark_cloud_fraction': f_dark_v1,
        'v1_cloud_fraction': 1.0 - f_clear_v1,
        'predicted_csi_v1': csi_v1,
        'v2_soft_clear_fraction': f_clear_v2,
        'v2_soft_bright_cloud_fraction': f_bright_v2,
        'v2_soft_dark_cloud_fraction': f_dark_v2,
        'v2_cloud_fraction': cloud_fraction_v2,
        'v2_overcast_strength': overcast_strength_v2,
        'predicted_csi_v2': csi_v2,
        'v3_otsu_cloud_fraction': f_cloud_otsu,
        'v3_combined_cloud_fraction': f_cloud_combined,
        'v3_sun_brightness': sun_brightness,
        'v3_sun_saturation': sun_saturation,
        'v3_sun_texture': sun_texture,
        'v3_sun_blue_confidence': sun_blue_conf,
        'v3_sun_dark_score': sun_dark_score,
        'v3_glare_score': glare_score,
        'v3_uniform_overcast_score': uniform_overcast_score,
        'v3_conflict_score': conflict_score,
        'v3_dense_overcast_flag': int(dense_overcast_flag),
        'v3_sun_block_conflict_flag': int(sun_block_conflict_flag),
        'v3_sky_class': sky_class_v3,
        'predicted_csi_v3': csi_v3,
        'v4_strong_blue_fraction': f_strong_blue,
        'v4_hazy_blue_raw_fraction': f_hazy_blue_raw,
        'v4_hazy_blue_used_fraction': f_hazy_blue,
        'v4_bright_cloud_fraction': f_bright_v4,
        'v4_dark_cloud_fraction': f_dark_v4,
        'v4_cloud_fraction_for_report': max(0.0, 1.0 - f_strong_blue - f_hazy_blue),
        'v4_hazy_blue_context_flag': int(hazy_blue_context_flag),
        'v4_extreme_dense_overcast_flag': int(extreme_dense_overcast_flag),
        'v4_sky_class': sky_class_v4,
        'v4_confidence_0_to_1': confidence_v4,
        'predicted_csi_v4': csi_v4,
        'v5_light_grey_fraction': f_light_grey,
        'v5_smooth_light_grey_fraction': f_smooth_light_grey,
        'v5_textured_light_grey_fraction': f_textured_light_grey,
        'v5_light_grey_mean_brightness': mean_light_grey_brightness,
        'v5_light_grey_mean_saturation': mean_light_grey_saturation,
        'v5_light_grey_mean_texture': mean_light_grey_texture,
        'v5_light_grey_score': float(light_grey_score),
        'v5_thick_light_grey_score': float(thick_light_grey_score),
        'v5_thin_light_grey_score': float(thin_light_grey_score),
        'v5_sky_class': sky_class_v5,
        'v5_confidence_0_to_1': confidence_v5,
        'predicted_csi_v5': csi_v5,
        'predicted_csi_v5_before_low_sun': csi_v5_before_low_sun,
        'low_sun_angle_flag': int(low_sun_angle_flag),
        'low_sun_correction_flag': int(low_sun_correction_flag),
        'low_sun_csi_cap': low_sun_csi_cap,
        'low_sun_reason': low_sun_reason,
        'scattered_clear_boost_flag': int(scattered_clear_boost_flag),
        'scattered_clear_csi_floor': scattered_clear_csi_floor,
        'scattered_clear_reason': scattered_clear_reason,
        'v8_blue_dominant_pixel_fraction': v8_blue_dominant_pixel_fraction,
        'v8_true_clear_blue_sky_fraction': v8_true_clear_blue_sky_fraction,
        'v8_blue_tinted_cloud_fraction': v8_blue_tinted_cloud_fraction,
        'v8_display_hazy_blue_fraction': v8_display_hazy_blue_fraction,
        'v8_display_bright_cloud_fraction': v8_display_bright_cloud_fraction,
        'v8_display_dark_cloud_fraction': v8_display_dark_cloud_fraction,
        'v8_display_cloud_fraction': v8_display_cloud_fraction,
        'v8_display_okta': v8_display_okta,
        'v8_display_note': v8_display_note,
        'v6_cloud_cover_index_0_to_1': cloud_cover_index,
        'v6_cloud_impact_index_0_to_1': cloud_impact_index,
        'v6_cloudiness_index_0_to_1': cloudiness_index,
        'v6_cloudiness_level': cloudiness_level(cloudiness_index),
        'v6_ramp_risk_index_0_to_1': ramp_risk_index,
        'v6_ramp_risk_level': describe_forecast_risk(ramp_risk_index),
        'v6_cloud_nature': cloud_nature,
        'v6_cloud_depth_proxy': cloud_depth_proxy,
        'v6_forecasting_value': forecasting_value,
        'v7_1km_cloud_field_type': one_km_cloud_field_type,
        'v7_1km_cloud_distribution': one_km_cloud_distribution,
        'v7_1km_cloud_impact_description': one_km_cloud_impact_description,
        'v7_1km_forecasting_note': one_km_forecasting_note,
        'v7_1km_assumption': one_km_assumption,
        'solar_elevation_deg': solar_elev,
    }
    masks = {
        'strong_blue': clear_mask_v4,
        'hazy_blue': hazy_mask_v4,
        'bright': bright_mask_v4,
        'dark': dark_v4_mask,
        'sun_region': sun_region,
        'light_grey': light_grey_mask,
        'smooth_light_grey': smooth_light_grey_mask,
        'textured_light_grey': textured_light_grey_mask,
        'cloud_otsu': cloud_otsu,
    }
    return metrics, img_rgb, masks


def save_overlay(img_rgb, masks, out_path, title):
    overlay = img_rgb.copy()
    strong_color = np.array([0, 120, 255], dtype=np.uint8)
    hazy_color = np.array([120, 220, 255], dtype=np.uint8)
    bright_color = np.array([255, 255, 255], dtype=np.uint8)
    dark_color = np.array([130, 60, 160], dtype=np.uint8)
    grey_color = np.array([210, 210, 210], dtype=np.uint8)
    overlay[masks['bright']] = (0.78 * overlay[masks['bright']] + 0.22 * bright_color).astype(np.uint8)
    if 'light_grey' in masks:
        overlay[masks['light_grey']] = (0.70 * overlay[masks['light_grey']] + 0.30 * grey_color).astype(np.uint8)
    overlay[masks['strong_blue']] = (0.55 * overlay[masks['strong_blue']] + 0.45 * strong_color).astype(np.uint8)
    overlay[masks['hazy_blue']] = (0.58 * overlay[masks['hazy_blue']] + 0.42 * hazy_color).astype(np.uint8)
    overlay[masks['dark']] = (0.55 * overlay[masks['dark']] + 0.45 * dark_color).astype(np.uint8)
    ys, xs = np.where(masks['sun_region'])
    if len(xs) > 0:
        cx, cy = int(xs.mean()), int(ys.mean())
        rr = int(np.sqrt(len(xs) / np.pi))
        cv2.circle(overlay, (cx, cy), rr, (255, 0, 0), 2)

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title('Display overlay: true blue / bright cloud / dark cloud')
    ax2.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)



def save_cloud_factor_dashboard(img_rgb, masks, row, out_path):
    """Save a compact visual report for one image with descriptive cloud factors."""
    overlay = img_rgb.copy()
    colors = {
        'bright': np.array([255, 255, 255], dtype=np.uint8),
        'light_grey': np.array([210, 210, 210], dtype=np.uint8),
        'strong_blue': np.array([0, 120, 255], dtype=np.uint8),
        'hazy_blue': np.array([120, 220, 255], dtype=np.uint8),
        'dark': np.array([130, 60, 160], dtype=np.uint8),
    }
    for name, alpha in [('bright', 0.20), ('light_grey', 0.35), ('strong_blue', 0.45), ('hazy_blue', 0.42), ('dark', 0.50)]:
        if name in masks:
            m = masks[name]
            overlay[m] = ((1-alpha) * overlay[m] + alpha * colors[name]).astype(np.uint8)
    ys, xs = np.where(masks.get('sun_region', np.zeros(img_rgb.shape[:2], dtype=bool)))
    if len(xs) > 0:
        cx, cy = int(xs.mean()), int(ys.mean())
        rr = int(np.sqrt(len(xs) / np.pi))
        cv2.circle(overlay, (cx, cy), rr, (255, 0, 0), 2)

    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.2], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img_rgb); ax.set_title('Original sky image'); ax.axis('off')
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(overlay); ax.set_title('Class overlay + pseudo-sun region'); ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    factors = [
        ('strong blue', row.get('v4_strong_blue_fraction', np.nan)),
        ('hazy blue', row.get('v4_hazy_blue_used_fraction', np.nan)),
        ('bright cloud', row.get('v4_bright_cloud_fraction', np.nan)),
        ('dark cloud', row.get('v4_dark_cloud_fraction', np.nan)),
        ('light grey', row.get('v5_light_grey_fraction', np.nan)),
        ('combined cloud', row.get('v3_combined_cloud_fraction', np.nan)),
    ]
    names = [f[0] for f in factors]
    vals = [100*float(f[1]) if pd.notna(f[1]) else 0.0 for f in factors]
    ax.barh(names, vals)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Fraction of valid sky (%)')
    ax.set_title('Cloud/sky fractions')
    ax.grid(axis='x', alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    # create visual masks: Otsu cloud, light grey, dark cloud, strong+hazy blue
    mask_img = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8) + 245
    if 'cloud_otsu' in masks: mask_img[masks['cloud_otsu']] = [220, 220, 220]
    if 'light_grey' in masks: mask_img[masks['light_grey']] = [150, 150, 150]
    if 'dark' in masks: mask_img[masks['dark']] = [90, 50, 140]
    if 'strong_blue' in masks: mask_img[masks['strong_blue']] = [0, 110, 255]
    if 'hazy_blue' in masks: mask_img[masks['hazy_blue']] = [110, 215, 255]
    ax.imshow(mask_img)
    ax.set_title('Factor masks\nblue=hazy/clear, grey=cloud, purple=dark')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    gauge_names = ['Cloudiness', 'Cloud impact', 'Ramp risk', 'Grey score', 'Glare']
    gauge_vals = [
        row.get('v6_cloudiness_index_0_to_1', np.nan),
        row.get('v6_cloud_impact_index_0_to_1', np.nan),
        row.get('v6_ramp_risk_index_0_to_1', np.nan),
        row.get('v5_light_grey_score', np.nan),
        row.get('v3_glare_score', np.nan),
    ]
    ax.bar(gauge_names, [float(v) if pd.notna(v) else 0.0 for v in gauge_vals])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Index (0–1)')
    ax.set_title('Forecast-useful indices')
    ax.tick_params(axis='x', rotation=35)
    ax.grid(axis='y', alpha=0.3)

    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    text = (
        f"Label: {row.get('label','')}\n"
        f"Time: {row.get('timestamp_local','')}\n"
        f"Nature: {row.get('v6_cloud_nature','')}\n"
        f"Depth proxy: {row.get('v6_cloud_depth_proxy','')}\n"
        f"Cloudiness index: {row.get('v6_cloudiness_index_0_to_1',np.nan):.2f} ({row.get('v6_cloudiness_level','')})\n"
        f"Ramp risk: {row.get('v6_ramp_risk_index_0_to_1',np.nan):.2f} ({row.get('v6_ramp_risk_level','')})\n"
        f"Pred. CSI V5: {row.get('predicted_csi_v5',np.nan):.3f}\n"
        f"Pred. GHI V5: {row.get('predicted_ghi_v5_Wm2',np.nan):.1f} W/m²\n"
        f"1 km field: {row.get('v7_1km_cloud_field_type','')}\n"
        f"1 km wording: {row.get('v7_1km_cloud_impact_description','')}\n"
        f"Forecast note: {row.get('v7_1km_forecasting_note','')}"
    )
    ax.text(0.02, 0.98, text, va='top', fontsize=10.5, bbox=dict(facecolor='white', edgecolor='navy', alpha=0.95, boxstyle='round,pad=0.6'))

    fig.suptitle('V7 cloud-field summary: cloud type wording for assumed 1 km × 1 km area', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_factor_summary_plots(result: pd.DataFrame, out_dir: Path):
    labels = result['label'].tolist()
    x = np.arange(len(result))
    # Forecast factors heatmap
    factor_cols = [
        'v6_cloudiness_index_0_to_1', 'v6_cloud_impact_index_0_to_1', 'v6_ramp_risk_index_0_to_1',
        'v5_light_grey_fraction', 'v5_thick_light_grey_score', 'v5_thin_light_grey_score',
        'v3_glare_score', 'v3_uniform_overcast_score', 'v3_conflict_score'
    ]
    factor_cols = [c for c in factor_cols if c in result.columns]
    data = result[factor_cols].astype(float).T.values
    plt.figure(figsize=(14, 6.8))
    im = plt.imshow(data, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Index / fraction (0–1)')
    plt.yticks(np.arange(len(factor_cols)), [c.replace('v6_', '').replace('v5_', '').replace('v3_', '').replace('_0_to_1','').replace('_',' ') for c in factor_cols])
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.title('Cloud factors useful for future forecasting')
    plt.tight_layout(); plt.savefig(out_dir/'cloud_factor_heatmap.png', dpi=180); plt.close()

    # 1km impact plot
    plt.figure(figsize=(14, 5.5))
    plt.bar(x - 0.18, result['predicted_solar_MW_over_1km2'], width=0.36, label='Predicted incoming solar irradiance over 1 km²')
    plt.bar(x + 0.18, result['cloud_loss_MW_over_1km2'], width=0.36, label='Cloud-induced irradiance loss over 1 km²')
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel('MW equivalent over 1 km²')
    plt.title('Predicted cloud impact at 1 km × 1 km resolution')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'one_km_cloud_impact_summary.png', dpi=180); plt.close()

    # Cloud nature count
    plt.figure(figsize=(12, 5.5))
    counts = result['v6_cloud_nature'].value_counts()
    plt.barh(counts.index, counts.values)
    plt.xlabel('Number of images')
    plt.title('Detected cloud nature classes')
    plt.tight_layout(); plt.savefig(out_dir/'cloud_nature_class_summary.png', dpi=180); plt.close()


def run_from_manifest(manifest_csv, output_dir, lat=DEFAULT_LAT, lon=DEFAULT_LON, tz=DEFAULT_TZ):
    out_dir = Path(output_dir)
    (out_dir / 'overlays').mkdir(parents=True, exist_ok=True)
    (out_dir / 'factor_dashboards').mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_csv)
    rows = []
    for _, item in manifest.iterrows():
        label = str(item['label'])
        ts = pd.to_datetime(item['timestamp_local'])
        image_path = str(item['image_path'])
        metrics, img_rgb, masks = analyse_image_versions(image_path, ts, lat, lon, tz)
        cosz = solar_cos_zenith_noaa(ts, lat, lon, tz)
        solar_elev = math.degrees(math.asin(cosz)) if cosz > 0 else 0.0
        clear = clear_sky_ghi_haurwitz(cosz)
        row = {
            'label': label,
            'timestamp_local': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': image_path,
            'solar_elevation_deg': solar_elev,
            'clear_sky_ghi_Wm2': clear,
            **metrics,
            'predicted_ghi_v1_Wm2': metrics['predicted_csi_v1'] * clear,
            'predicted_ghi_v2_Wm2': metrics['predicted_csi_v2'] * clear,
            'predicted_ghi_v3_Wm2': metrics['predicted_csi_v3'] * clear,
            'predicted_ghi_v4_Wm2': metrics['predicted_csi_v4'] * clear,
            'predicted_ghi_v5_Wm2': metrics['predicted_csi_v5'] * clear,
        }
        # 1 km × 1 km impact: 1 W/m² over 1 km² equals 1 MW of incoming solar irradiance.
        row['predicted_solar_MW_over_1km2'] = row['predicted_ghi_v5_Wm2']
        row['clear_sky_solar_MW_over_1km2'] = clear
        row['cloud_loss_MW_over_1km2'] = max(0.0, clear - row['predicted_ghi_v5_Wm2'])
        row['pv_equiv_loss_MW_1km2_20pct'] = 0.20 * row['cloud_loss_MW_over_1km2']
        plant_csv = item.get('plant_csv', np.nan)
        if isinstance(plant_csv, str) and plant_csv.strip() and plant_csv.strip().lower() != 'nan':
            actual, source, method = read_actual_ghi_from_plant(plant_csv, ts)
            row['plant_csv'] = plant_csv
            row['actual_ghi_source'] = source
            row['actual_ghi_method'] = method
            row['actual_ghi_Wm2'] = actual
            row['actual_csi'] = actual / clear if clear > 0 and pd.notna(actual) else np.nan
            for ver in ['v1', 'v2', 'v3', 'v4', 'v5']:
                pred_col = f'predicted_ghi_{ver}_Wm2'
                row[f'error_{ver}_Wm2'] = row[pred_col] - actual
                row[f'abs_error_{ver}_Wm2'] = abs(row[pred_col] - actual)
        save_overlay(img_rgb, masks, out_dir/'overlays'/f'{label}_v6_cloud_overlay.png', f'{label} | {metrics["v5_sky_class"]} | V5 CSI {metrics["predicted_csi_v5"]:.3f}')
        save_cloud_factor_dashboard(img_rgb, masks, row, out_dir/'factor_dashboards'/f'{label}_cloud_factor_dashboard.png')
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir/'v7_1km_cloud_field_predictions_full.csv', index=False)

    rounded_cols = [
        'label','timestamp_local','actual_ghi_source','actual_ghi_method','v4_sky_class','v4_confidence_0_to_1','v5_sky_class','v5_confidence_0_to_1',
        'v4_strong_blue_fraction','v4_hazy_blue_used_fraction','v4_bright_cloud_fraction','v4_dark_cloud_fraction',
        'v4_cloud_fraction_for_report','v4_hazy_blue_context_flag','v4_extreme_dense_overcast_flag','v5_light_grey_fraction','v5_smooth_light_grey_fraction','v5_textured_light_grey_fraction','v5_light_grey_score','v5_thick_light_grey_score','v5_thin_light_grey_score',
        'predicted_csi_v1','predicted_csi_v2','predicted_csi_v3','predicted_csi_v4','predicted_csi_v5',
        'v6_cloud_nature','v6_cloud_depth_proxy','v6_cloudiness_index_0_to_1','v6_cloudiness_level','v6_cloud_impact_index_0_to_1','v6_ramp_risk_index_0_to_1','v6_ramp_risk_level','v6_forecasting_value', 'v7_1km_cloud_field_type','v7_1km_cloud_distribution','v7_1km_cloud_impact_description','v7_1km_forecasting_note','v7_1km_assumption',
        'clear_sky_ghi_Wm2','predicted_ghi_v1_Wm2','predicted_ghi_v2_Wm2','predicted_ghi_v3_Wm2','predicted_ghi_v4_Wm2','predicted_ghi_v5_Wm2',
        'actual_ghi_Wm2','actual_csi','error_v1_Wm2','error_v2_Wm2','error_v3_Wm2','error_v4_Wm2','error_v5_Wm2',
        'predicted_solar_MW_over_1km2','clear_sky_solar_MW_over_1km2','cloud_loss_MW_over_1km2','pv_equiv_loss_MW_1km2_20pct'
    ]
    rounded_cols = [c for c in rounded_cols if c in result.columns]
    result[rounded_cols].round(3).to_csv(out_dir/'v7_1km_cloud_field_predictions_rounded.csv', index=False)

    if 'actual_ghi_Wm2' in result.columns:
        summary_rows = []
        for ver, name in [('v1','V1 original'),('v2','V2 improved'),('v3','V3 scientific'),('v4','V4 sky-class + hazy-blue'),('v5','V5 grey-cloud module')]:
            err = result[f'error_{ver}_Wm2'].astype(float)
            summary_rows.append({
                'method': name,
                'n_images': int(err.notna().sum()),
                'MAE_Wm2': float(err.abs().mean()),
                'RMSE_Wm2': float(np.sqrt(np.mean(err * err))),
                'MBE_Wm2': float(err.mean())
            })
        summary = pd.DataFrame(summary_rows)
        summary.round(3).to_csv(out_dir/'v3_v4_v5_v6_error_summary.csv', index=False)

        x = np.arange(len(result)); labels = result['label'].tolist()
        plt.figure(figsize=(14,5.2))
        plt.plot(x, result['actual_ghi_Wm2'], marker='o', label='Actual GHI')
        plt.plot(x, result['predicted_ghi_v1_Wm2'], marker='o', label='V1')
        plt.plot(x, result['predicted_ghi_v2_Wm2'], marker='o', label='V2')
        plt.plot(x, result['predicted_ghi_v3_Wm2'], marker='o', label='V3')
        plt.plot(x, result['predicted_ghi_v4_Wm2'], marker='o', label='V4')
        plt.plot(x, result['predicted_ghi_v5_Wm2'], marker='o', label='V5 grey')
        plt.xticks(x, labels, rotation=35, ha='right')
        plt.ylabel('GHI (W/m²)')
        plt.title('Predicted vs actual GHI: V7 cloud-field wording model')
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'v3_v4_v5_predicted_vs_actual_ghi.png', dpi=180); plt.close()

        plt.figure(figsize=(14,5.2))
        plt.plot(x, result['actual_csi'], marker='o', label='Actual CSI')
        plt.plot(x, result['predicted_csi_v1'], marker='o', label='V1')
        plt.plot(x, result['predicted_csi_v2'], marker='o', label='V2')
        plt.plot(x, result['predicted_csi_v3'], marker='o', label='V3')
        plt.plot(x, result['predicted_csi_v4'], marker='o', label='V4')
        plt.plot(x, result['predicted_csi_v5'], marker='o', label='V5 grey')
        plt.xticks(x, labels, rotation=35, ha='right')
        plt.ylabel('CSI')
        plt.title('Predicted vs actual CSI: V7 cloud-field wording model')
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'v3_v4_v5_predicted_vs_actual_csi.png', dpi=180); plt.close()

        errdf = result[['label','error_v1_Wm2','error_v2_Wm2','error_v3_Wm2','error_v4_Wm2','error_v5_Wm2']].set_index('label')
        ax = errdf.plot(kind='bar', figsize=(14,5.2))
        ax.axhline(0, linewidth=1)
        ax.set_ylabel('Error (W/m²)')
        ax.set_title('Prediction error comparison: V7 cloud-field wording model')
        plt.tight_layout(); plt.savefig(out_dir/'v3_v4_v5_error_bar.png', dpi=180); plt.close()
    # Major factor table for forecasting work.
    major_cols = [
        'label','timestamp_local','v6_cloud_nature','v6_cloud_depth_proxy','v7_1km_cloud_field_type','v7_1km_cloud_distribution','v7_1km_cloud_impact_description','v7_1km_forecasting_note','v6_cloudiness_index_0_to_1','v6_cloudiness_level',
        'v6_cloud_impact_index_0_to_1','v6_ramp_risk_index_0_to_1','v6_ramp_risk_level','v6_forecasting_value',
        'v4_strong_blue_fraction','v4_hazy_blue_used_fraction','v4_bright_cloud_fraction','v4_dark_cloud_fraction',
        'v5_light_grey_fraction','v5_smooth_light_grey_fraction','v5_textured_light_grey_fraction','v5_thick_light_grey_score','v5_thin_light_grey_score',
        'v3_combined_cloud_fraction','v3_glare_score','v3_uniform_overcast_score','v3_conflict_score','solar_elevation_deg',
        'clear_sky_ghi_Wm2','predicted_csi_v5','predicted_ghi_v5_Wm2','predicted_solar_MW_over_1km2','cloud_loss_MW_over_1km2','pv_equiv_loss_MW_1km2_20pct',
        'actual_ghi_Wm2','actual_csi','error_v5_Wm2'
    ]
    major_cols = [c for c in major_cols if c in result.columns]
    result[major_cols].round(3).to_csv(out_dir/'major_cloud_factors_and_1km_cloud_type.csv', index=False)
    save_factor_summary_plots(result, out_dir)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V7 1km cloud-field wording + image-only CSI/GHI prediction.')
    parser.add_argument('--manifest', required=True, help='CSV with label,image_path,timestamp_local,plant_csv')
    parser.add_argument('--lat', type=float, default=DEFAULT_LAT)
    parser.add_argument('--lon', type=float, default=DEFAULT_LON)
    parser.add_argument('--timezone-offset', type=float, default=DEFAULT_TZ)
    parser.add_argument('--output-dir', default='outputs_v7_1km_cloud_field')
    args = parser.parse_args()
    res = run_from_manifest(args.manifest, args.output_dir, args.lat, args.lon, args.timezone_offset)
    print(res.round(3).to_string(index=False))


# Create a ready-to-run manifest for this conversation when executed directly by create_v4_analysis.py
if False:
    pass
