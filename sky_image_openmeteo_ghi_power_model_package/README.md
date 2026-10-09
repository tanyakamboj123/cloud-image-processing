# Streamlit dashboard

The seven-tab dashboard is in `app.py`. It starts with the existing 7 October
plant inputs and the confirmed 435,000 kW DC capacity.

```powershell
cd C:\codex\sky_image_openmeteo_ghi_power_model_package
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501. Click **Run forecast** in Run Forecast, or **Run model**
in the sidebar. To upload new files, turn off **Use existing project input files**.
The existing-file mode uses files directly in the input folders; CLI also supports
nested ordinary input folders. Uploaded media with arbitrary names need a recording
time under **Recording times & arbitrary filenames**.

Inputs are saved under `inputs/<category>/_dashboard/<session>/<dataset>/` to avoid
mixing datasets and browser sessions. These snapshots are excluded from ordinary
CLI scans. Each dashboard run writes the requested CSV filenames beneath
`outputs/dashboard/<session>/<run>/`. Direct CLI runs continue writing directly
to `outputs/`. Clear previous outputs affects only the current dashboard session;
original plant reports and input files are preserved. Dashboard settings are saved
with each result and do not overwrite config.json.

GHI and power result tabs contain Plotly comparison/error charts and section-specific
metrics. Combined Results provides section/time filters. Power Results explains
reported power versus V/I totals, and includes POA and per-row Perez decomposition
method. Downloads includes every generated CSV and a ZIP with logs, configuration,
and extracted frames. The detected power columns and warnings are in Run Forecast.
Results are marked stale when settings or inputs change; run again to update them.

The dashboard calls `run_ghi_power_forecast.run_forecast(...)` directly. It returns
`{"paths": {...}, "status": {"success": ..., "error": ..., "warnings": ..., "log": ...}}`.
Per-run backend settings are isolated and execution is serialized for log capture.
The dashboard always uses its configured capacity and physical temperature/loss
model; it does not silently replace that capacity with fitted early-morning power.

```powershell
.\.venv\Scripts\python.exe run_ghi_power_forecast.py
.\.venv\Scripts\python.exe -m unittest test_forecast test_dashboard -v
```

Validated with Streamlit 1.65.0 and Plotly 7.1.0. The local server binds only to
127.0.0.1. This is a local dashboard, not an authenticated public deployment.

# Sky images + Open-Meteo + GHI + DC power

Run from this package directory with Python 3.10+ (validated here with Python 3.14.5).

```powershell
cd C:\codex\sky_image_openmeteo_ghi_power_model_package
.\.venv\Scripts\python.exe run_ghi_power_forecast.py
```

For a new installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_ghi_power_forecast.py
```

All default paths resolve relative to the script, so launching it from another directory also works. Input/output directories are created automatically. With no usable images, execution stops with an actionable message; measurements and weather are optional.

## Inputs

Files may have any names. The following directories are scanned recursively:

- `inputs/images/`: JPG/JPEG/PNG/BMP/WebP images and MP4/MOV/AVI/MKV/WebM/M4V videos.
- `inputs/ghi/`: CSV, XLSX or XLS containing timestamps and GHI (not POA/GTI).
- `inputs/power/`: CSV, XLSX or XLS containing timestamps and explicit DC power or voltage/current pairs.
- `inputs/openmeteo/`: manually downloaded Open-Meteo hourly CSV/JSON.
- `inputs/models/`: scanned and reported, but reserved for a future trained-model interface. Arbitrary model files are not loaded; the bundled adaptive image rules remain active.

CSV supports automatic delimiter detection. Excel uses openpyxl for XLSX and xlrd for XLS. Temporary Excel files beginning with `~$` are ignored.

Timestamps are local Asia/Kolkata by default. Offset-aware timestamps are converted to site time. Ambiguous numeric date strings use day-first parsing; ISO dates remain year-first. Separate `Date` and `Time` columns are supported. Matching uses the nearest measurement within 8 minutes, configurable by `actual_match_tolerance_minutes`; unmatched values stay blank.

Recognized media names include `WhatsApp Image 2026-09-29 at 10.44.02 AM.jpeg`, `WhatsApp Video 2026-10-07 at 7.18.44 AM.mp4`, and `2026-10-07_10-30-00.jpg`.

For arbitrary media names, configure timestamps using paths relative to `inputs/images/`:

```json
"image_timestamps": {"sky.jpeg": "2026-10-07 10:30:00"},
"video_start_times": {"sky.mp4": "2026-10-07 07:18:44"}
```

A WhatsApp filename may represent sharing time. Confirm its recording time or override it. Videos default to one frame every 60 seconds, including the first frame. Frames use recording start + frame number/FPS. This assumes ordinary constant-frame-rate footage, not a time-lapse. Extracted frames go to `outputs/video_frames/`. Configure `video_frame_interval_seconds` for a different interval.

## Plant configuration

Edit `config.json`; the corresponding runtime constants are near the top of `run_ghi_power_forecast.py`.

| Setting | Current default | Meaning |
|---|---:|---|
| site_latitude / site_longitude | 27.54225 / 72.471 | Site coordinates |
| timezone / utc_offset_hours | Asia/Kolkata / 5.5 | Site timezone and image-core solar offset; keep consistent |
| surface_tilt_deg | 25 | Assumed fixed panel tilt |
| surface_azimuth_deg | 180 | pvlib convention: north 0, east 90, south 180 |
| ground_albedo | 0.2 | Ground reflectance |
| dc_capacity_kw | 435000 | Set confirmed plant DC nameplate capacity |
| temperature_coefficient_per_C | -0.004 | Relative power change per degree C |
| default_performance_ratio | 0.82 | Multiplicative default loss factor |
| power_unit | auto | Override unlabelled power with W, kW or MW |
| power_calibration_enabled | true | Use at least three distinct earlier power observations when available |

Explicit unit labels take precedence in automatic mode. Unlabelled power uses the original magnitude heuristic: 95th percentile below 100 implies MW, above 200000 implies W, otherwise kW. This is ambiguous for small plants; conversions are printed and `power_unit` can override them. Unlabelled GHI below 5 at its 95th percentile is assumed kW/m2. Explicit W/m2 labels prevent this heuristic from rescaling low-light values.

Accepted DC names include DC Power, Power DC, Pdc, Total DC Power and Active Power DC, with units. An explicit total is preferred; otherwise device DC power columns are summed. AC/export power is not accepted as DC. Supported V/I labels include Vdc/Idc, DC Voltage/DC Current, Voltage DC/Current DC and Input or String Voltage/Current. Device identifiers must match, for example `String 1 DC Voltage` and `String 1 DC Current`. All pairs must be complete; missing values do not become zero. Multiple-pair totals leave the single voltage/current output fields blank because no single fleet voltage exists.

## Weather and prediction methods

The default run first requests live Open-Meteo data. Recent dates use the forecast endpoint; older dates use the archive endpoint. Wind is requested in m/s. Humidity and cloud-cover layers are retained as features alongside radiation, temperature and wind.

If live download fails, matching manual CSV/JSON files are loaded. API JSON timezone and wind-unit metadata and common CSV unit-labelled columns are recognized. CSV metadata blocks are skipped. Flat files without timezone information are assumed local site time; unlabelled wind is assumed m/s. Out-of-range weather is not extrapolated, and gaps over three hours are not interpolated. If no data is usable, a warning identifies the image + clear-sky fallback.

Open-Meteo radiation is a preceding-hour mean. Interpolation to image times is an approximation. Historical archive/retrieved weather is not necessarily the forecast that was available at the image time, so retrospective metrics are not a strict operational backtest. See the [Open-Meteo documentation](https://open-meteo.com/en/docs).

The three cloud-model source files remain unchanged. Current image CSI, cloud classes, visual okta and sun/blocking features come from the existing adaptive rules. The wrapper preserves current CSI before fusion; forecast-specific blocking caps apply only at future horizons. GHI fusion still uses the existing horizon-dependent image/Open-Meteo CSI weighting. Confidence labels are heuristic, not calibrated probabilities. Cloud layers and humidity are exported as features; no new trained fusion model was introduced.

Predicted GHI is decomposed with pvlib Erbs when weather DNI/DHI are missing. Otherwise weather component proportions are scaled to match predicted GHI. Direct horizontal radiation is converted to DNI when necessary. Perez transposition uses solar position, tilt, azimuth, albedo, extraterrestrial irradiance and relative airmass. Nighttime output is zero. See [pvlib Perez inputs](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html). If pvlib fails or is unavailable, explicitly labelled approximate fallbacks remain available.

Power calibration now uses only current-image predictions and actual measurements strictly earlier than each forecast origin, with at least three distinct actual timestamps. It never fits future target measurements. This corrects the original same-run target leakage. The calibrated median kW/(W/m2) method remains approximate and does not separately model temperature. Without enough calibration data, power uses configured capacity, or a clearly labelled estimate from prior power, or a **1000 kW placeholder** if neither exists. The capacity model uses POA, an approximate cell-temperature model, coefficient and loss factor. Confirm capacity before interpreting power accuracy.

## Outputs

- `outputs/ghi_comparison_table.csv`
- `outputs/power_comparison_table.csv`
- `outputs/combined_ghi_power_table.csv`
- `outputs/image_feature_table.csv`
- `outputs/summary_metrics.csv`
- `outputs/all_internal_predictions_with_features.csv`
- Normalized actual GHI/power and downloaded weather CSVs when available.

The three comparison tables use the requested `image_time`, `forecast_time`, `source_image_file`, `cloud_class` and `okta` names. Internal diagnostics retain method, weather fields, capacity and calibration factors. Sections are `image_time_current`, `rolling_image_t_plus_15` and `latest_image_2hr_future`; defaults generate eight future rows at +15 through +120 minutes.

Errors and bias mean actual minus predicted. MAE/RMSE include all matched pairs. Accuracy is clipped to 0-100 using `100 * (1 - abs(error)/actual)` and is blank for GHI <=50 W/m2 or power <=1 kW. Summary rows include matched and accuracy sample counts. Blank metrics mean no eligible measurements, not zero error.

If Excel locks a CSV, the writer uses a unique suffixed filename and prints that path.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest test_forecast -v
.\.venv\Scripts\python.exe -m pip check
```

`validation/synthetic_case/` contains artificial video/measurement fixtures and example output tables, clearly labelled as synthetic. Their metrics only test software mechanics; they do not establish real-world forecast accuracy. The specified 7 October plant files were absent when this package was reviewed.

For explicit offline runs, or a separate input/output location:

```powershell
.\.venv\Scripts\python.exe run_ghi_power_forecast.py --offline
.\.venv\Scripts\python.exe run_ghi_power_forecast.py --input-dir validation/synthetic_case/inputs --output-dir validation/cli_outputs --offline
```

The saved synthetic validation uses one-second video sampling; the CLI uses the configured default of 60 seconds, so its short fixture video yields only one image row unless that setting is changed.
