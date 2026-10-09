
# V5.3 White-Blocking + Sun-Path Corrected Model

This package is based on your **narrow V5.3 white-blocking model**, not the general white-cloud model.

## What changed

The original file is kept unchanged:

```text
sky_prediction_v5_3_white_blocking_corrected.py
```

The new model is a wrapper:

```text
sky_prediction_v5_3_white_blocking_sunpath_corrected.py
```

It adds one post-correction layer after the original V5.3 CSI prediction.

### 1. Morning low-sun correction

Problem observed:

```text
Before 8 AM, the image becomes dark/grey due to camera exposure.
The model thinks it is thick cloud and underpredicts GHI.
```

Correction:

```text
If solar elevation is low and sun/glow evidence exists,
apply a CSI floor.
```

### 2. Afternoon sun-path correction

Problem observed:

```text
After around 3 PM, sun/glare can still be visible,
but haze/light-grey cloud attenuates the solar path.
The model overpredicts GHI.
```

Correction:

```text
If afternoon CSI is high but thin-veil/light-grey/hazy sun-path evidence exists,
apply a CSI cap.
```

### 3. Sun-edge / glare reliability flag

The model checks whether bright sun/glare is near the image edge. If yes, the direct-sun/thin-veil boost is treated as less reliable.

## Run Streamlit app

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_v5_3_white_blocking_sunpath_dashboard.py
```

## Run batch test

Put images in:

```text
test_images/
```

Optional actual file:

```text
actual_ghi.csv
```

Then run:

```bash
python run_sunpath_corrected_batch.py
```

Output:

```text
outputs_sunpath_corrected/sunpath_corrected_predictions.csv
```

## Important diagnostic columns

```text
predicted_csi_before_sunpath_correction
predicted_csi_v5
sunpath_correction_flag
morning_low_sun_floor_flag
morning_low_sun_csi_floor
afternoon_sunpath_cap_flag
afternoon_sunpath_csi_cap
sun_edge_flag
sunpath_correction_reason
```

## Files in this package

```text
sky_prediction_v5_3_white_blocking_corrected.py          # original narrow model
sky_prediction_v5_3_white_blocking_sunpath_corrected.py  # new corrected model
okta_correction_only_white_blocking.py
streamlit_v5_3_white_blocking_sunpath_dashboard.py
run_sunpath_corrected_batch.py
requirements.txt
```


## V2 time-guarded correction

This updated version adds time guards so the correction does not disturb the
9:00 AM / 9:15 AM region when the old V5.3 model was already correct.

```text
Morning low-sun floor:
    only before 08:15 local time
    and solar elevation below 18 degrees

Afternoon sun-path cap:
    only after 14:00 local time
    and solar elevation between 30 and 52 degrees
```

Reason:

```text
9 AM and 3 PM can have similar solar elevation.
The previous correction used solar elevation only, so the afternoon cap could
activate around 9 AM. This version separates morning and afternoon using
local clock time.
```


## V3 midday scattered-cloud floor

This version adds a third correction:

```text
08:15 to 14:00 local time:
    If base V53 predicts low CSI,
    but the image has blue gaps / direct-sun evidence / scattered-cloud pattern,
    apply a conservative CSI floor.
```

Purpose:

```text
Normal days after 11 AM and near 2 PM were sometimes underpredicted because
the narrow white-blocking model treated bright scattered cloud cover as
sun-path blocking. This correction prevents too-low GHI when the sky still
has clear gaps or direct-sun evidence.
```

It does NOT apply for:

```text
dense overcast
extreme overcast
very high white-blocking score with very low blue-sky fraction
dark-cloud-dominant images
```

New output columns:

```text
midday_scattered_cloud_floor_flag
midday_scattered_cloud_csi_floor
```
