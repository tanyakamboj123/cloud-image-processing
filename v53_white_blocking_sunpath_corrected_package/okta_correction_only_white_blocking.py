
import cv2
import numpy as np


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        val = float(x)
        if np.isnan(val):
            return default
        return val
    except Exception:
        return default


def _safe_mask(mask, shape):
    if mask is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(mask).astype(bool)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def corrected_visual_okta(image_path, metrics=None, masks=None):
    metrics = metrics or {}
    masks = masks or {}

    img_bgr = cv2.imread(str(image_path))
    old_cf = _safe_float(metrics.get("v8_display_cloud_fraction", metrics.get("v3_combined_cloud_fraction", 0.0)))
    old_okta = int(np.clip(np.rint(8.0 * old_cf), 0, 8))

    if img_bgr is None:
        return {"okta_correction_status": "failed_image_read", "okta_old": old_okta, "okta_corrected": old_okta}

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    B, G, R = [x.astype(np.float32) for x in cv2.split(img_bgr)]

    yy, xx = np.mgrid[0:h, 0:w]
    valid_sky = V > 0.08
    valid_sky[int(h * 0.955):, :] = False
    n_sky = int(valid_sky.sum())
    if n_sky <= 0:
        return {"okta_correction_status": "failed_no_valid_sky", "okta_old": old_okta, "okta_corrected": old_okta}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray, (0, 0), 3)
    mean2 = cv2.GaussianBlur(gray * gray, (0, 0), 3)
    tex = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

    v_blur = cv2.GaussianBlur(V, (0, 0), 11)
    v_blur_masked = v_blur.copy()
    v_blur_masked[~valid_sky] = -1.0
    _, _, _, max_loc = cv2.minMaxLoc(v_blur_masked.astype(np.float32))
    sun_x, sun_y = max_loc

    sun_core_mask = valid_sky & (V > 0.965) & (S < 0.55)
    sun_radius = max(20, int(0.20 * min(h, w)))
    sun_geom = ((yy - sun_y) ** 2 + (xx - sun_x) ** 2 <= sun_radius ** 2) & valid_sky
    sun_halo_mask = sun_geom & (((V > 0.58) & (S < 0.34)) | (V > 0.88))
    flare_mask = valid_sky & (V > 0.72) & (S < 0.26) & (tex < 0.018)
    glare_exclusion_mask = sun_core_mask | sun_halo_mask | flare_mask

    border_margin = max(6, int(0.04 * min(h, w)))
    border_zone = np.zeros((h, w), dtype=bool)
    border_zone[:border_margin, :] = True
    border_zone[-border_margin:, :] = True
    border_zone[:, :border_margin] = True
    border_zone[:, -border_margin:] = True
    dark_object_mask = (V < 0.18) & ((S < 0.35) | (B < 70)) & (border_zone | (tex > 0.025))

    cloud_otsu = _safe_mask(masks.get("cloud_otsu"), (h, w))
    light_grey = _safe_mask(masks.get("light_grey"), (h, w))
    dark_cloud = _safe_mask(masks.get("dark"), (h, w))
    bright_cloud = _safe_mask(masks.get("bright"), (h, w))
    white_blocking = _safe_mask(masks.get("white_blocking_cloud"), (h, w))

    norm_blue = B / (R + G + B + 1.0)
    blue_clear = valid_sky & (B > 1.04 * R) & (B > 1.015 * G) & (S > 0.10) & (V > 0.18) & (norm_blue > 0.34)

    white_or_veil_cloud = valid_sky & (S < 0.23) & (V > 0.38) & (tex > 0.0035)
    textured_cloud = valid_sky & (tex > 0.012) & (S < 0.45) & (V > 0.28)

    real_cloud_mask = (
        (cloud_otsu | light_grey | dark_cloud | bright_cloud | white_blocking | white_or_veil_cloud | textured_cloud)
        & valid_sky
        & (~blue_clear)
        & (~glare_exclusion_mask)
        & (~dark_object_mask)
    )

    real_cloud_fraction = float(real_cloud_mask.sum() / n_sky)
    sun_core_fraction = float(sun_core_mask.sum() / n_sky)
    sun_halo_fraction = float(sun_halo_mask.sum() / n_sky)
    glare_fraction = float(glare_exclusion_mask.sum() / n_sky)
    foreground_fraction = float(dark_object_mask.sum() / n_sky)

    direct_sun_score = float(np.clip(
        0.60 * np.clip((sun_core_fraction - 0.0003) / 0.010, 0.0, 1.0)
        + 0.40 * np.clip((sun_halo_fraction - 0.020) / 0.160, 0.0, 1.0),
        0.0, 1.0
    ))

    uniform_overcast_score = _safe_float(metrics.get("v3_uniform_overcast_score", 0.0))
    extreme_overcast = bool(metrics.get("v4_extreme_dense_overcast_flag", False))
    conflict_score = _safe_float(metrics.get("v3_conflict_score", 0.0))
    sun_block_conflict = bool(metrics.get("v3_sun_block_conflict_flag", False))

    if direct_sun_score > 0.35 and old_cf > 0.60:
        corrected_cf = 0.72 * real_cloud_fraction + 0.18 * old_cf + 0.10 * max(0.0, 1.0 - _safe_float(metrics.get("v8_true_clear_blue_sky_fraction")) - glare_fraction)
        reason = "Direct sun/glare detected; sun core/halo/flare excluded from visual cloud cover."
    elif direct_sun_score > 0.35 and old_cf <= 0.60:
        corrected_cf = old_cf
        reason = "Direct sun detected, but original okta was not high; original okta kept."
    else:
        corrected_cf = 0.70 * old_cf + 0.30 * real_cloud_fraction
        reason = "No strong direct-sun glare; original okta mildly corrected using real-cloud evidence."

    if uniform_overcast_score > 0.55 or extreme_overcast:
        corrected_cf = max(corrected_cf, old_cf)
        reason += " Dense/overcast safeguard active."

    if sun_block_conflict or conflict_score > 0.14:
        corrected_cf = max(corrected_cf, min(0.80, old_cf))
        reason += " Sun-path/cloud-conflict safeguard active."

    if direct_sun_score > 0.65 and real_cloud_fraction < 0.35 and old_cf > 0.60:
        corrected_cf = min(corrected_cf, 0.38)
        reason += " Strong direct-sun low-real-cloud cap active."

    corrected_cf = float(np.clip(corrected_cf, 0.0, 1.0))
    corrected_okta = int(np.clip(np.rint(8.0 * corrected_cf), 0, 8))

    if corrected_okta <= 1:
        category = "clear_to_few_clouds"
    elif corrected_okta <= 3:
        category = "scattered_clouds"
    elif corrected_okta <= 5:
        category = "partly_cloudy"
    elif corrected_okta <= 7:
        category = "broken_cloud"
    else:
        category = "overcast"

    return {
        "okta_correction_status": "ok",
        "okta_old": old_okta,
        "okta_old_cloud_fraction": old_cf,
        "okta_corrected": corrected_okta,
        "okta_corrected_cloud_fraction": corrected_cf,
        "okta_category": category,
        "okta_correction_reason": reason,
        "okta_real_cloud_fraction_raw": real_cloud_fraction,
        "okta_glare_fraction": glare_fraction,
        "okta_sun_core_fraction": sun_core_fraction,
        "okta_sun_halo_fraction": sun_halo_fraction,
        "okta_foreground_fraction": foreground_fraction,
        "okta_direct_sun_score": direct_sun_score,
    }
