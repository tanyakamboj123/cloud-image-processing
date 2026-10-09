
"""
V5.3 white-blocking + sun-path correction.

This file keeps the original narrow white-blocking model unchanged and adds
one post-correction layer on top of `predicted_csi_v5`.

Base model:
    sky_prediction_v5_3_white_blocking_corrected.py

Final adaptive correction layers:
1. Morning low-sun exposure floor:
   fixes early-morning underprediction when the camera makes sky/clouds look dark
   even though the sun/glow is present.

2. Midday scattered-cloud / clear-gap floor:
   avoids too-low GHI when clouds are present but blue gaps/direct sun evidence
   indicate that the sun path is not fully blocked.

3. Midday direct-sun high-transmission floor:
   keeps near-noon direct-sun images close to clear-sky/high-transmission GHI.

4. Midday direct-sun cumulus-edge floor:
   handles sun-visible cloud-edge images so they do not collapse to low GHI.

5. Afternoon cumulus sun-blocking cap:
   reduces high predictions when direct sun is not visible and large cumulus
   clouds likely block the solar path.

6. Strict afternoon haze cap:
   only for strong overprediction-risk haze/edge cases; normal evening is protected.

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
    timestamp_local: Any | None = None,
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

    # Time guard:
    # 9:00 AM and 3:00 PM can have a similar solar elevation.
    # The first sun-path correction used elevation only, so the afternoon cap
    # could disturb already-correct 9:00/9:15 AM predictions.
    local_hour = np.nan
    try:
        if timestamp_local is not None:
            ts = pd.Timestamp(timestamp_local)
            local_hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    except Exception:
        local_hour = np.nan
    m["sunpath_local_hour"] = local_hour

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
    m["midday_scattered_cloud_floor_flag"] = 0
    m["midday_scattered_cloud_csi_floor"] = np.nan
    m["midday_direct_sun_floor_flag"] = 0
    m["midday_direct_sun_csi_floor"] = np.nan
    m["midday_cumulus_edge_floor_flag"] = 0
    m["midday_cumulus_edge_csi_floor"] = np.nan
    m["afternoon_cumulus_blocking_cap_flag"] = 0
    m["afternoon_cumulus_blocking_csi_cap"] = np.nan
    m["afternoon_direct_sun_blocked_cap_flag"] = 0
    m["afternoon_direct_sun_blocked_csi_cap"] = np.nan
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
        and (np.isfinite(local_hour) and local_hour <= 8.15)
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
    # 2) Midday scattered-cloud / clear-gap floor
    # ------------------------------------------------------------------
    # Problem observed:
    # From about 9 AM to 2 PM, the base V5.3 white-blocking model can predict
    # low CSI when it sees many bright white/scattered clouds, even if the sun
    # path is clear or there are strong blue gaps. At this time, clear-sky GHI
    # is high, so a small CSI underprediction creates a large GHI error.
    #
    # This correction is intentionally conservative:
    # - it only works between 08:15 and 14:00 local time
    # - it does not apply for real dense/overcast or strong white-blocking cases
    # - it requires blue gaps, direct-sun evidence, or a scattered-cloud pattern
    # ------------------------------------------------------------------

    strong_real_white_blocking = (
        (white_score > 0.82)
        and (display_cloud > 0.68)
        and (true_blue < 0.20)
        and (direct_sun_score < 0.25)
    )

    clear_gap_present = (
        (true_blue > 0.22)
        or ((true_blue > 0.15) and (direct_sun_score > 0.25))
        or ((direct_sun_visible == 1) and (display_cloud < 0.72))
    )

    scattered_cloud_pattern = (
        (display_cloud > 0.22)
        and (display_cloud < 0.88)
        and (true_blue > 0.12)
        and (dark_cloud < 0.45)
        and (uniform_overcast < 0.40)
    )

    midday_safeguard_condition = (
        np.isfinite(local_hour)
        and (8.15 <= local_hour <= 14.00)
        and np.isfinite(elev)
        and (elev >= 18.0)
        and (csi < 0.72)
        and (not real_dense_cloud)
        and (not strong_real_white_blocking)
        and (clear_gap_present or scattered_cloud_pattern)
    )

    if midday_safeguard_condition:
        midday_floor = (
            0.60
            + 0.18 * true_blue
            + 0.06 * direct_sun_visible
            + 0.06 * min(direct_sun_score, 1.0)
            + 0.04 * max(0.0, 0.60 - display_cloud)
            - 0.08 * dark_cloud
            - 0.06 * non_sun_light_grey
            - 0.04 * max(display_cloud - 0.75, 0.0)
        )

        # If the sun is clearly visible and the sky is not very cloudy, allow a
        # higher floor. This protects normal clear/scattered-cloud midday cases.
        if (direct_sun_visible == 1) and (true_blue > 0.25) and (display_cloud < 0.65):
            midday_floor += 0.06

        midday_floor = _clip(midday_floor, 0.62, 0.84)

        if csi < midday_floor:
            csi = midday_floor
            m["midday_scattered_cloud_floor_flag"] = 1
            m["midday_scattered_cloud_csi_floor"] = float(midday_floor)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Midday scattered-cloud/clear-gap floor applied: image has blue "
                "gaps or direct-sun/scattered-cloud evidence, so the low CSI from "
                "the white-blocking model was treated as too pessimistic."
            )


    # ------------------------------------------------------------------
    # 3) Midday direct-sun high-transmission floor
    # ------------------------------------------------------------------
    # Problem observed:
    # Around noon, images with a clear visible sun and glare streak were still
    # underpredicted because the model treated the glare/haze as cloud impact.
    # For these images actual CSI can be close to 1.0 or slightly above due to
    # cloud-edge enhancement / simple clear-sky-model underestimation.
    # ------------------------------------------------------------------
    midday_direct_sun_clear = (
        np.isfinite(local_hour)
        and (10.50 <= local_hour <= 13.50)
        and np.isfinite(elev)
        and (elev >= 50.0)
        and (direct_sun_visible == 1)
        and (direct_sun_score > 0.35)
        and (not real_dense_cloud)
        and (dark_cloud < 0.33)
        and (uniform_overcast < 0.42)
        and (not strong_real_white_blocking)
        # Do not treat strongly cloud-edge/nearby-cumulus images as full-clear.
        and not ((display_cloud > 0.58) and (dark_cloud > 0.22))
    )

    if midday_direct_sun_clear:
        direct_floor = (
            0.945
            + 0.030 * min(direct_sun_score, 1.0)
            + 0.015 * min(glare_score, 1.0)
            + 0.030 * true_blue
            - 0.030 * dark_cloud
            - 0.020 * non_sun_light_grey
            - 0.015 * max(display_cloud - 0.40, 0.0)
        )
        # If small scattered clouds are away from the sun, allow slight enhancement.
        if (display_cloud < 0.50) and (true_blue > 0.35) and (dark_cloud < 0.20) and (white_score < 0.20):
            direct_floor += 0.030

        direct_floor = _clip(direct_floor, 0.94, 1.03)

        if csi < direct_floor:
            csi = direct_floor
            m["midday_direct_sun_floor_flag"] = 1
            m["midday_direct_sun_csi_floor"] = float(direct_floor)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Midday direct-sun high-transmission floor applied: sun disk/glare "
                "is clearly visible with no dense blocking cloud. Prediction kept "
                "near clear-sky/high-transmission level."
            )

    # ------------------------------------------------------------------
    # 4) Midday direct-sun cumulus-edge floor
    # ------------------------------------------------------------------
    # Problem observed:
    # Images where the sun is visible but thick cumulus cloud is close to the
    # sun path should not be treated as fully clear, but they also should not
    # collapse to very low GHI. This floor keeps them in the moderate-high range.
    # ------------------------------------------------------------------
    cumulus_edge_visible_sun = (
        np.isfinite(local_hour)
        and (10.25 <= local_hour <= 13.75)
        and np.isfinite(elev)
        and (elev >= 48.0)
        and (direct_sun_visible == 1)
        and (direct_sun_score > 0.25)
        and (not real_dense_cloud)
        and (display_cloud > 0.42)
        and ((dark_cloud > 0.18) or (white_score > 0.50) or (bright_cloud > 0.30))
        and (true_blue > 0.08)
        and (uniform_overcast < 0.55)
    )

    if cumulus_edge_visible_sun and (m.get("midday_direct_sun_floor_flag", 0) == 0):
        edge_floor = (
            0.76
            + 0.10 * min(direct_sun_score, 1.0)
            + 0.07 * true_blue
            + 0.03 * min(glare_score, 1.0)
            - 0.10 * dark_cloud
            - 0.05 * max(display_cloud - 0.55, 0.0)
            - 0.04 * non_sun_light_grey
        )
        edge_floor = _clip(edge_floor, 0.72, 0.94)

        if csi < edge_floor:
            csi = edge_floor
            m["midday_cumulus_edge_floor_flag"] = 1
            m["midday_cumulus_edge_csi_floor"] = float(edge_floor)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Midday cumulus-edge floor applied: sun remains visible but thick "
                "nearby cloud exists. Prediction kept moderate-high rather than low."
            )


    # ------------------------------------------------------------------
    # 5) Afternoon hazy sun-path cap
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


    # ------------------------------------------------------------------
    # 6) Afternoon cumulus sun-blocking cap
    # ------------------------------------------------------------------
    # Problem observed:
    # After 14:00, some images have blue gaps and bright cumulus, but the sun
    # itself is not directly visible. The old/safe-evening model preserved a
    # high CSI (~0.8), while actual CSI was ~0.48-0.55. This cap only applies
    # when direct sun is missing and cloud evidence is strong.
    # ------------------------------------------------------------------
    afternoon_cumulus_blocking = (
        np.isfinite(local_hour)
        and (14.00 <= local_hour <= 16.25)
        and np.isfinite(elev)
        and (28.0 <= elev <= 55.0)
        and (direct_sun_visible == 0)
        and (direct_sun_score < 0.28)
        and (csi > 0.64)
        and (not real_dense_cloud)  # dense cloud is already handled by base model
        and (display_cloud > 0.32)
        and ((white_score > 0.20) or (bright_cloud > 0.28) or (non_sun_light_grey > 0.20) or (thin_veil_score > 0.65))
        and (true_blue > 0.04)
        and (true_blue < 0.72)
        and (uniform_overcast < 0.65)
    )

    if afternoon_cumulus_blocking:
        cumulus_cap = (
            0.50
            + 0.10 * true_blue
            + 0.02 * max(0.0, 0.65 - display_cloud)
            - 0.07 * dark_cloud
            - 0.04 * non_sun_light_grey
            - 0.04 * max(display_cloud - 0.65, 0.0)
        )
        cumulus_cap = _clip(cumulus_cap, 0.45, 0.64)

        if csi > cumulus_cap:
            csi = cumulus_cap
            m["afternoon_cumulus_blocking_cap_flag"] = 1
            m["afternoon_cumulus_blocking_csi_cap"] = float(cumulus_cap)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Afternoon cumulus sun-blocking cap applied: direct sun is not "
                "visible and large/bright cloud likely blocks the solar path."
            )

    # ------------------------------------------------------------------
    # 7) Afternoon visible-sun but cloud-thickened path cap
    # ------------------------------------------------------------------
    # If direct sun/glare is visible but nearly the whole image is cloudy/grey
    # and blue sky is almost absent, the visible glare should not be trusted as
    # full transmission. This handles cases such as 15:23 where old V53 stayed high.
    afternoon_direct_sun_blocked = (
        np.isfinite(local_hour)
        and (14.00 <= local_hour <= 16.25)
        and np.isfinite(elev)
        and (28.0 <= elev <= 55.0)
        and (direct_sun_visible == 1)
        and (direct_sun_score > 0.35)
        and (csi > 0.64)
        and (display_cloud > 0.78)
        and (true_blue < 0.12)
        and ((non_sun_light_grey > 0.30) or (dark_cloud > 0.25) or (white_score > 0.50))
        and (not mostly_clear_direct_sun)
    )

    if afternoon_direct_sun_blocked:
        blocked_cap = (
            0.56
            + 0.04 * min(direct_sun_score, 1.0)
            - 0.07 * dark_cloud
            - 0.05 * non_sun_light_grey
            - 0.04 * max(display_cloud - 0.82, 0.0)
        )
        blocked_cap = _clip(blocked_cap, 0.48, 0.64)

        if csi > blocked_cap:
            csi = blocked_cap
            m["afternoon_direct_sun_blocked_cap_flag"] = 1
            m["afternoon_direct_sun_blocked_csi_cap"] = float(blocked_cap)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Afternoon visible-sun blocked-path cap applied: glare/direct sun is visible, "
                "but most of the image is cloudy/grey and blue sky is nearly absent."
            )


    # Safe-evening guard:
    # The first afternoon cap sometimes reduced already-correct evening predictions.
    # Therefore the cap is now allowed only for strong overprediction-risk cases:
    #  - limited time window after 14:00, not late evening
    #  - original CSI must be very high
    #  - blue sky fraction must be low
    #  - clear haze/dark-cloud/sun-edge evidence must exist
    if (
        np.isfinite(elev)
        and (30.0 <= elev <= 52.0)
        and (np.isfinite(local_hour) and (14.00 <= local_hour <= 16.10))
        and (csi > 0.82)
        and afternoon_hazy_sunpath
        and (not mostly_clear_direct_sun)
        and (true_blue < 0.35)
        and ((non_sun_light_grey > 0.25) or (dark_cloud > 0.25) or (sun_edge_flag == 1))
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
        afternoon_cap = _clip(afternoon_cap, 0.55, 0.72)

        if csi > afternoon_cap:
            csi = afternoon_cap
            m["afternoon_sunpath_cap_flag"] = 1
            m["afternoon_sunpath_csi_cap"] = float(afternoon_cap)
            m["sunpath_correction_flag"] = 1
            m["sunpath_correction_reason"] = (
                "Strict afternoon hazy sun-path cap applied: CSI was very high, blue fraction was low, and the image "
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
        elif m["midday_direct_sun_floor_flag"] == 1:
            m["v6_cloud_nature"] = "midday_direct_sun_high_transmission"
            m["v6_cloud_depth_proxy"] = "clear_sun_high_transmission_or_cloud_edge_enhancement"
            m["v6_forecasting_value"] = (
                "Direct midday sun is clearly visible and no dense blocking cloud is present. "
                "CSI floor raised toward clear-sky/high-transmission level."
            )
        elif m["midday_cumulus_edge_floor_flag"] == 1:
            m["v6_cloud_nature"] = "midday_direct_sun_cumulus_edge_corrected"
            m["v6_cloud_depth_proxy"] = "sun_visible_with_nearby_cumulus_edge"
            m["v6_forecasting_value"] = (
                "Sun is visible but cumulus is near the sun path. CSI floor kept the "
                "prediction moderate-high rather than too low."
            )
        elif m["midday_scattered_cloud_floor_flag"] == 1:
            m["v6_cloud_nature"] = "midday_scattered_cloud_clear_gap_corrected"
            m["v6_cloud_depth_proxy"] = "scattered_clouds_with_clear_sky_gaps"
            m["v6_forecasting_value"] = (
                "Midday cloud cover is present, but blue gaps/direct-sun evidence "
                "suggest the original white-blocking CSI was too pessimistic. "
                "CSI floor applied."
            )
        elif m["afternoon_cumulus_blocking_cap_flag"] == 1:
            m["v6_cloud_nature"] = "afternoon_cumulus_sunpath_blocking"
            m["v6_cloud_depth_proxy"] = "direct_sun_hidden_by_cumulus"
            m["v6_forecasting_value"] = (
                "Direct sun is not visible and large/bright cumulus likely blocks the solar path. "
                "CSI cap applied to avoid overprediction."
            )
        elif m["afternoon_direct_sun_blocked_cap_flag"] == 1:
            m["v6_cloud_nature"] = "afternoon_visible_sun_but_blocked_path"
            m["v6_cloud_depth_proxy"] = "visible_glare_with_cloudy_grey_sunpath"
            m["v6_forecasting_value"] = (
                "Direct sun/glare is visible, but the sky is cloud-filled and blue fraction is very low. "
                "CSI cap applied to avoid overprediction."
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
        if m["midday_direct_sun_floor_flag"] == 1 and "midday_direct_sun_floor" not in old_class:
            m["v5_sky_class"] = old_class + "_midday_direct_sun_floor"
        if m["midday_cumulus_edge_floor_flag"] == 1 and "midday_cumulus_edge_floor" not in old_class:
            m["v5_sky_class"] = old_class + "_midday_cumulus_edge_floor"
        if m["midday_scattered_cloud_floor_flag"] == 1 and "midday_scattered_cloud_floor" not in old_class:
            m["v5_sky_class"] = old_class + "_midday_scattered_cloud_floor"
        if m["afternoon_cumulus_blocking_cap_flag"] == 1 and "afternoon_cumulus_blocking_cap" not in old_class:
            m["v5_sky_class"] = old_class + "_afternoon_cumulus_blocking_cap"
        if m["afternoon_direct_sun_blocked_cap_flag"] == 1 and "afternoon_direct_sun_blocked_cap" not in old_class:
            m["v5_sky_class"] = old_class + "_afternoon_direct_sun_blocked_cap"
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
    metrics = apply_sunpath_post_correction(metrics, img_rgb=img_rgb, timestamp_local=timestamp_local)
    return metrics, img_rgb, masks
