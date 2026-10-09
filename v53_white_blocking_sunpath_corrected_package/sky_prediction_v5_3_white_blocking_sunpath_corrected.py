
"""
V5.3 white-blocking + sun-path correction.

This file keeps the original narrow white-blocking model unchanged and adds
one post-correction layer on top of `predicted_csi_v5`.

Base model:
    sky_prediction_v5_3_white_blocking_corrected.py

New correction:
1. Morning low-sun exposure floor:
   fixes early-morning underprediction when the camera makes sky/clouds look dark
   even though the sun/glow is present.

2. Afternoon hazy sun-path cap:
   fixes afternoon overprediction when the sun/glare is visible but the actual
   sun path is attenuated by haze, light-grey cloud, or edge/halo effects.

This is not the "general white-cloud" model. It imports and uses the narrow
`v53_white_blocking_cloud_model_package` core.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pandas as pd

import sky_prediction_v5_3_white_blocking_corrected as base
from sky_prediction_v5_3_white_blocking_corrected import *  # re-export all helper functions/constants


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return default


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _image_edge_brightness_features(img_rgb: np.ndarray) -> dict[str, float]:
    """
    Detect whether strong bright sun/glare is close to an image boundary.

    This is useful because afternoon images often show only part of the sun
    or its halo near the edge. In that situation, the direct-sun/thin-veil
    boost is less reliable.
    """
    try:
        arr = img_rgb
        if arr.dtype != np.uint8:
            arr8 = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr8 = arr

        hsv = cv2.cvtColor(arr8, cv2.COLOR_RGB2HSV)
        S = hsv[:, :, 1].astype(float) / 255.0
        V = hsv[:, :, 2].astype(float) / 255.0

        valid = V > 0.08
        n_valid = max(int(valid.sum()), 1)

        # Not too strict because many phone images compress the sun/halo.
        bright = valid & (V > 0.88) & (S < 0.55)

        h, w = V.shape
        border_px = max(12, int(0.12 * min(h, w)))
        border = np.zeros_like(bright, dtype=bool)
        border[:border_px, :] = True
        border[-border_px:, :] = True
        border[:, :border_px] = True
        border[:, -border_px:] = True

        bright_fraction = float(bright.sum() / n_valid)
        edge_bright_fraction = float((bright & border).sum() / n_valid)

        if bright.any():
            ys, xs = np.where(bright)
            cx = float(xs.mean() / max(w - 1, 1))
            cy = float(ys.mean() / max(h - 1, 1))
            distance_to_edge = float(min(cx, 1.0 - cx, cy, 1.0 - cy))
        else:
            cx = cy = distance_to_edge = np.nan

        edge_flag = int(
            (edge_bright_fraction > 0.0015)
            or (bright_fraction > 0.004 and np.isfinite(distance_to_edge) and distance_to_edge < 0.16)
        )

        return {
            "sunpath_bright_fraction": bright_fraction,
            "sunpath_edge_bright_fraction": edge_bright_fraction,
            "sunpath_bright_centroid_x": cx,
            "sunpath_bright_centroid_y": cy,
            "sunpath_bright_distance_to_edge": distance_to_edge,
            "sun_edge_flag": edge_flag,
        }
    except Exception:
        return {
            "sunpath_bright_fraction": 0.0,
            "sunpath_edge_bright_fraction": 0.0,
            "sunpath_bright_centroid_x": np.nan,
            "sunpath_bright_centroid_y": np.nan,
            "sunpath_bright_distance_to_edge": np.nan,
            "sun_edge_flag": 0,
        }


def apply_sunpath_post_correction(
    metrics: dict[str, Any],
    img_rgb: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Apply the new CSI correction and return an updated metrics dictionary.

    The correction is intentionally post-processing only:
    - original V5.3 white-blocking outputs remain available
    - final corrected CSI is written back to `predicted_csi_v5`
    - the raw value is saved as `predicted_csi_v5_before_sunpath_correction`
    """
    m = dict(metrics)

    csi0 = _safe_float(m.get("predicted_csi_v5"), np.nan)
    if not np.isfinite(csi0):
        return m

    csi = float(csi0)
    elev = _safe_float(m.get("solar_elevation_deg"), np.nan)

    # Common image/model features
    mean_brightness = _safe_float(m.get("mean_brightness_0_to_1"))
    mean_saturation = _safe_float(m.get("mean_saturation_0_to_1"))
    true_blue = _safe_float(m.get("v8_true_clear_blue_sky_fraction"))
    blue_tinted = _safe_float(m.get("v8_blue_tinted_cloud_fraction"))
    display_cloud = _safe_float(m.get("v8_display_cloud_fraction"))
    bright_cloud = _safe_float(m.get("v8_display_bright_cloud_fraction"))
    dark_cloud = _safe_float(m.get("v8_display_dark_cloud_fraction"))
    light_grey = _safe_float(m.get("v5_light_grey_fraction"))
    non_sun_light_grey = _safe_float(m.get("non_sun_light_grey_cloud_fraction"), light_grey)
    direct_sun_score = _safe_float(m.get("direct_sun_score"))
    direct_sun_visible = int(_safe_float(m.get("direct_sun_visible_flag"), 0))
    thin_veil_flag = int(_safe_float(m.get("thin_veil_cloud_flag"), 0))
    thin_veil_score = _safe_float(m.get("thin_veil_score"))
    white_score = _safe_float(m.get("white_blocking_cloud_score"))
    uniform_overcast = _safe_float(m.get("v3_uniform_overcast_score"))
    extreme_dense = int(_safe_float(m.get("v4_extreme_dense_overcast_flag"), 0))
    dense_overcast = int(_safe_float(m.get("v3_dense_overcast_flag"), 0))
    glare_score = _safe_float(m.get("v3_glare_score"))
    sun_halo = _safe_float(m.get("sun_glare_or_halo_fraction"))

    if img_rgb is not None:
        edge = _image_edge_brightness_features(img_rgb)
    else:
        edge = {
            "sunpath_bright_fraction": 0.0,
            "sunpath_edge_bright_fraction": 0.0,
            "sunpath_bright_centroid_x": np.nan,
            "sunpath_bright_centroid_y": np.nan,
            "sunpath_bright_distance_to_edge": np.nan,
            "sun_edge_flag": 0,
        }

    sun_edge_flag = int(edge.get("sun_edge_flag", 0))

    # Save original value and diagnostic flags.
    m["predicted_csi_v5_before_sunpath_correction"] = float(csi0)
    m["sunpath_correction_flag"] = 0
    m["morning_low_sun_floor_flag"] = 0
    m["morning_low_sun_csi_floor"] = np.nan
    m["afternoon_sunpath_cap_flag"] = 0
    m["afternoon_sunpath_csi_cap"] = np.nan
    m["sunpath_correction_reason"] = "No sun-path post-correction applied."

    for k, v in edge.items():
        m[k] = v

    # ------------------------------------------------------------------
    # 1) Morning low-sun exposure floor
    # ------------------------------------------------------------------
    # Problem observed:
    # before about 8 AM, phone exposure can make the image look dark/grey.
    # Original model may classify it as dark/rain cloud and underestimate.
    #
    # Do NOT apply if there is strong evidence of real dense/dark overcast.
    # ------------------------------------------------------------------
    real_dense_cloud = (
        (extreme_dense == 1)
        or (dense_overcast == 1 and mean_brightness < 0.40)
        or (uniform_overcast > 0.45)
        or ((dark_cloud > 0.58) and (mean_brightness < 0.43) and (direct_sun_score < 0.25))
    )

    morning_visible_glow = (
        (direct_sun_score > 0.18)
        or (glare_score > 0.18)
        or (sun_halo > 0.018)
        or (edge.get("sunpath_bright_fraction", 0.0) > 0.002)
        or (mean_brightness > 0.44)
    )

    if (
        np.isfinite(elev)
        and (4.0 <= elev < 18.0)
        and (csi < 0.62)
        and (not real_dense_cloud)
        and morning_visible_glow
    ):
        morning_floor = (
            0.50
            + 0.010 * elev
            + 0.080 * min(direct_sun_score, 1.0)
            + 0.040 * mean_brightness
            + 0.030 * glare_score
            + 0.040 * min(sun_halo / 0.06, 1.0)
            - 0.090 * dark_cloud
            - 0.035 * max(display_cloud - 0.65, 0.0)
        )
        morning_floor = _clip(morning_floor, 0.50, 0.72)

        if csi < morning_floor:
            csi = morning_floor
            m["morning_low_sun_floor_flag"] = 1
            m["morning_low_sun_csi_floor"] = float(morning_floor)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Morning low-sun exposure floor applied: low solar elevation with "
                "visible glow/direct-sun evidence. Original V5.3 likely treated "
                "camera-exposure darkening as thick cloud."
            )

    # ------------------------------------------------------------------
    # 2) Afternoon hazy sun-path cap
    # ------------------------------------------------------------------
    # Problem observed:
    # after ~3 PM, visible sun/glare or thin-veil class can overestimate GHI.
    # The sun is visible, but haze/light-grey cloud attenuates the sun path.
    #
    # Do NOT cap mostly clear direct-sun cases with high true-blue fraction,
    # low non-sun grey cloud, and moderate cloud cover.
    # ------------------------------------------------------------------
    mostly_clear_direct_sun = (
        (direct_sun_visible == 1)
        and (true_blue > 0.45)
        and (non_sun_light_grey < 0.18)
        and (display_cloud < 0.55)
        and (dark_cloud < 0.30)
    )

    thin_hazy_path = (
        (thin_veil_flag == 1)
        and (thin_veil_score > 0.55)
        and (non_sun_light_grey > 0.18)
        and ((display_cloud > 0.34) or (true_blue < 0.62) or (mean_saturation < 0.25))
    )

    direct_hazy_path = (
        (direct_sun_visible == 1)
        and (
            (display_cloud > 0.70)
            or (non_sun_light_grey > 0.30)
            or (dark_cloud > 0.30)
            or ((mean_saturation < 0.18) and (true_blue < 0.25))
        )
    )

    white_hazy_path = (
        (white_score > 0.55)
        and (display_cloud > 0.55)
        and (true_blue < 0.45)
    )

    edge_or_halo_uncertain_path = (
        (sun_edge_flag == 1)
        and (display_cloud > 0.30)
        and (non_sun_light_grey > 0.12)
    )

    afternoon_hazy_sunpath = (
        thin_hazy_path
        or direct_hazy_path
        or white_hazy_path
        or edge_or_halo_uncertain_path
    )

    if (
        np.isfinite(elev)
        and (30.0 <= elev <= 52.0)
        and (csi > 0.70)
        and afternoon_hazy_sunpath
        and (not mostly_clear_direct_sun)
    ):
        afternoon_cap = (
            0.54
            + 0.12 * true_blue
            + 0.05 * min(direct_sun_score, 1.0)
            + 0.04 * direct_sun_visible
            - 0.12 * non_sun_light_grey
            - 0.10 * dark_cloud
            - 0.08 * max(display_cloud - 0.35, 0.0)
            - 0.04 * sun_edge_flag
        )
        afternoon_cap = _clip(afternoon_cap, 0.48, 0.70)

        if csi > afternoon_cap:
            csi = afternoon_cap
            m["afternoon_sunpath_cap_flag"] = 1
            m["afternoon_sunpath_csi_cap"] = float(afternoon_cap)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Afternoon hazy sun-path cap applied: CSI was high, but the image "
                "shows thin veil/light-grey haze/cloud along the solar path. "
                "Visible sun/glare is not treated as fully clear-sky transmission."
            )

    # Clip final CSI.
    csi = _clip(csi, 0.0, 1.15)

    # If changed, update dependent metrics.
    if abs(csi - csi0) > 1e-9:
        m["predicted_csi_v5"] = float(csi)
        cloud_impact_index = _clip(1.0 - csi, 0.0, 1.0)
        cloud_cover_index = _safe_float(m.get("v6_cloud_cover_index_0_to_1"))
        m["v6_cloud_impact_index_0_to_1"] = cloud_impact_index
        m["v6_cloudiness_index_0_to_1"] = _clip(0.50 * cloud_cover_index + 0.50 * cloud_impact_index, 0.0, 1.0)

        # Keep old class text, but add clear diagnostic class where useful.
        if m["morning_low_sun_floor_flag"] == 1:
            m["v6_cloud_nature"] = "low_sun_exposure_corrected_bright_haze_or_glare"
            m["v6_cloud_depth_proxy"] = "camera_exposure_darkening_not_thick_cloud"
            m["v6_forecasting_value"] = (
                "Low-sun image was dark/grey but sun/glow evidence is present. "
                "CSI floor applied to avoid early-morning underprediction."
            )
        elif m["afternoon_sunpath_cap_flag"] == 1:
            m["v6_cloud_nature"] = "afternoon_hazy_sunpath_attenuation"
            m["v6_cloud_depth_proxy"] = "hazy_or_light_grey_sunpath_attenuation"
            m["v6_forecasting_value"] = (
                "Afternoon sun/glare is visible, but haze/light-grey cloud along "
                "the solar path can attenuate plant GHI. CSI cap applied."
            )

        old_class = str(m.get("v5_sky_class", ""))
        if m["morning_low_sun_floor_flag"] == 1 and "morning_sunpath_floor" not in old_class:
            m["v5_sky_class"] = old_class + "_morning_sunpath_floor"
        if m["afternoon_sunpath_cap_flag"] == 1 and "afternoon_sunpath_cap" not in old_class:
            m["v5_sky_class"] = old_class + "_afternoon_sunpath_cap"

    return m


def analyse_image_versions(image_path: str, timestamp_local=None, lat=DEFAULT_LAT, lon=DEFAULT_LON, tz=DEFAULT_TZ):
    """
    Same API as the original V5.3 white-blocking core.

    Returns:
        metrics, img_rgb, masks
    """
    metrics, img_rgb, masks = base.analyse_image_versions(image_path, timestamp_local, lat, lon, tz)
    metrics = apply_sunpath_post_correction(metrics, img_rgb=img_rgb)
    return metrics, img_rgb, masks
