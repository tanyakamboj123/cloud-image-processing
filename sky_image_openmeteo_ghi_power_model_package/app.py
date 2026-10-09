"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_support import ROOT, stage_inputs, archive_outputs, clear_session_outputs
from run_ghi_power_forecast import load_config, parse_timestamp_from_name, run_forecast

st.set_page_config(page_title="Sky Image + Open-Meteo GHI and PV Power Forecasting Dashboard",
                   page_icon=":material/solar_power:", layout="wide")
for key, value in {"session_id": uuid.uuid4().hex, "result": None, "last_config": None, "last_version": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value
cfg = load_config()
SECTIONS = {"image_time_current": "Image-time current", "rolling_image_t_plus_15": "Rolling +15 minutes",
            "latest_image_2hr_future": "Latest-image future forecast"}


@st.cache_data(ttl=600, max_entries=80)
def read_output(path, modified):
    return pd.read_csv(path)


@st.cache_data(ttl=600, max_entries=12)
def zip_download(output_dir):
    return archive_outputs({}, output_dir)


def table(result, name):
    path = result["paths"].get(name)
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return read_output(path, Path(path).stat().st_mtime_ns)


def show_results(result, kind):
    data = table(result, f"{kind}_comparison_table.csv")
    if data.empty:
        st.info("Run the model to generate this comparison.")
        return
    choices = data.section.unique().tolist()
    selected = st.selectbox("Forecast section", choices, format_func=lambda x: SECTIONS.get(x, x),
                            index=choices.index("latest_image_2hr_future") if "latest_image_2hr_future" in choices else 0,
                            key=f"{kind}_section")
    data = data[data.section == selected].copy()
    data["forecast_time"] = pd.to_datetime(data.forecast_time)
    data = data.sort_values("forecast_time")
    unit = "W/m²" if kind == "ghi" else "kW"
    predicted, actual, error = (("predicted_ghi_wm2", "actual_ghi_wm2", "ghi_error_actual_minus_predicted_wm2") if kind == "ghi" else
                               ("predicted_dc_power_kw", "actual_dc_power_kw", "power_error_actual_minus_predicted_kw"))
    errors = data[error].dropna()
    scores = [errors.abs().mean(), np.sqrt((errors**2).mean()), errors.mean(), data[f"{kind}_accuracy_percent"].mean()]
    with st.container(horizontal=True):
        for label, number in zip([f"MAE · {unit}", f"RMSE · {unit}", f"Bias · {unit}", "Average accuracy · %"], scores):
            st.metric(label, "—" if pd.isna(number) else f"{number:,.2f}", border=True)
    st.caption(f"{len(errors)} matched comparisons · Bias = actual − predicted. Accuracy is clipped to 0–100%; low-output samples are excluded.")
    if errors.empty:
        st.warning("No actual measurements match this section. Check file columns, dates, timezone and coverage.")
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.forecast_time, y=data[predicted], name="Predicted", mode="lines+markers", line=dict(color="#d18a16", width=3)))
    figure.add_trace(go.Scatter(x=data.forecast_time, y=data[actual], name="Actual / measured", mode="lines+markers", line=dict(color="#137e85", width=3, dash="dot")))
    figure.update_layout(height=370, xaxis_title="Forecast time (site local)", yaxis_title=unit,
                         margin=dict(l=15, r=15, t=30, b=15), legend=dict(orientation="h", y=1.12), hovermode="x unified")
    st.plotly_chart(figure, key=f"{kind}_comparison_chart", alt=f"Actual versus predicted {kind} by forecast time")
    error_chart = go.Figure(go.Bar(x=data.forecast_time, y=data[error], marker_color="#137e85", name="Actual − predicted"))
    error_chart.update_layout(height=240, yaxis_title=f"Error ({unit})", xaxis_title="Forecast time (site local)", margin=dict(l=15, r=15, t=10, b=15))
    error_chart.add_hline(y=0, line_color="#777777")
    st.plotly_chart(error_chart, key=f"{kind}_error_chart", alt=f"Actual minus predicted {kind} error")
    st.dataframe(data, hide_index=True, alt=f"{kind} comparison results")


with st.sidebar:
    st.markdown("### :material/solar_power: Solar forecast")
    st.caption("Site & system configuration")
    with st.expander("Location", expanded=True):
        lat = st.number_input("Site latitude", -90., 90., float(cfg["site_latitude"]), format="%.5f")
        lon = st.number_input("Site longitude", -180., 180., float(cfg["site_longitude"]), format="%.5f")
        timezone = st.text_input("Timezone", cfg["timezone"])
    with st.expander("PV system", expanded=True):
        capacity = st.number_input("Plant DC capacity (kW)", min_value=1., value=float(cfg.get("dc_capacity_kw") or 435000), step=1000.)
        st.caption(f"{capacity / 1000:,.3f} MW DC · Match the capacity to the measured system.")
        tilt = st.number_input("Panel tilt (degrees)", 0., 90., float(cfg["surface_tilt_deg"]))
        azimuth = st.number_input("Panel azimuth (degrees)", 0., 360., float(cfg["surface_azimuth_deg"]), help="North 0°, east 90°, south 180°.")
        albedo = st.number_input("Albedo", 0., 1., float(cfg["ground_albedo"]), step=.01)
        gamma = st.number_input("Temperature coefficient (/°C)", -.02, .01, float(cfg["temperature_coefficient_per_C"]), step=.001, format="%.4f")
        loss = st.number_input("Loss factor", .01, 1., float(cfg["default_performance_ratio"]), step=.01)
    with st.expander("Forecast & weather", expanded=True):
        horizon = st.number_input("Forecast horizon (hours)", .25, 24., 2., step=.25)
        interval = st.selectbox("Forecast interval (minutes)", [5, 10, 15, 30, 60], index=2)
        live = st.toggle("Enable live Open-Meteo", value=True, key="live_weather")
        manual = st.toggle("Use manual Open-Meteo if available", value=True)
        frame_interval = st.number_input("Video frame interval (seconds)", 1., 3600., 60., step=1.)
    sidebar_run = st.button("Run model", type="primary", icon=":material/play_arrow:", width="stretch", key="run_sidebar")
    if st.button("Clear previous outputs", icon=":material/delete:", width="stretch", help="Clears only this browser session’s dashboard outputs; original CLI/plant results and inputs are preserved.", key="clear_outputs"):
        try:
            clear_session_outputs(st.session_state.session_id)
            st.session_state.result = None
            st.session_state.last_config = None
            st.success("This session’s outputs cleared.")
        except OSError as exc:
            st.error(f"Could not clear outputs; close any open files and retry. {exc}")

st.title("Sky Image + Open-Meteo GHI and PV Power Forecasting Dashboard")
st.caption("SKY OBSERVATIONS  /  WEATHER FUSION  /  PLANT PERFORMANCE")
with st.container(horizontal=True):
    st.metric("DC capacity", f"{capacity / 1000:,.1f} MW", border=True)
    st.metric("Forecast window", f"{horizon:g} hours", border=True)
    st.metric("Forecast spacing", f"{interval} min", border=True)
    st.metric("Weather mode", "Live + fallback" if live else "Manual / clear sky", border=True)
tabs = st.tabs(["Upload Data", "Run Forecast", "GHI Results", "Power Results", "Combined Results", "Image Features", "Downloads"])
uploads = {name: [] for name in ["images", "ghi", "power", "openmeteo", "models"]}
with tabs[0]:
    st.subheader("Bring your sky and plant data together")
    use_existing = st.toggle("Use existing project input files", value=True, key="use_existing", help="Uses files directly in the project input folders. Turn off to run only your uploads.")
    if use_existing:
        st.info("Existing project files selected. The supplied 7 October data is ready to run.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### :material/cloud_upload: Sky observations")
            uploads["images"] = st.file_uploader("Sky images", ["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True, disabled=use_existing, key="sky_uploads")
            video = st.file_uploader("Sky video", ["mp4", "mov", "avi", "mkv", "webm", "m4v"], disabled=use_existing, key="video_upload")
            if video:
                uploads["images"] = uploads["images"] + [video]
            st.caption("Video time is recording start + elapsed seconds. Filename time may be WhatsApp sharing time.")
    with right:
        with st.container(border=True):
            st.markdown("#### :material/table_chart: Measurements & weather")
            for kind, label, types in [("ghi", "Actual GHI", ["csv", "xlsx", "xls"]), ("power", "Actual plant DC power or V/I", ["csv", "xlsx", "xls"]), ("openmeteo", "Manual Open-Meteo (optional)", ["csv", "json"])]:
                item = st.file_uploader(label, types, disabled=use_existing, key=f"{kind}_upload")
                uploads[kind] = [item] if item else []
    dirs, inventory, version = None, [], None
    try:
        dirs, inventory, version = stage_inputs(st.session_state.session_id, uploads, use_existing)
        if inventory:
            st.dataframe(pd.DataFrame(inventory), hide_index=True, alt="Selected input files saved for this session")
            st.caption("Files are saved under inputs/images, inputs/ghi, inputs/power and inputs/openmeteo in an isolated session folder.")
    except (OSError, ValueError) as exc:
        st.error(f"Could not save input files: {exc}")
    video_times, image_times = {}, {}
    with st.expander("Recording times & arbitrary filenames"):
        st.caption("Recognized filename timestamps are used automatically. Enter local YYYY-MM-DD HH:MM:SS to override, or supply a time for an arbitrary filename.")
        if dirs:
            for media in sorted(dirs["images"].iterdir()):
                parsed = parse_timestamp_from_name(media)
                value = st.text_input(f"Recording time · {media.name}", value="" if pd.isna(parsed) else str(parsed), key=f"time_{version}_{media.name}")
                if value.strip():
                    target = video_times if media.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"} else image_times
                    target[media.name] = value.strip()

settings = dict(lat=lat, lon=lon, timezone=timezone, panel_tilt_deg=tilt, panel_azimuth_deg=azimuth, albedo=albedo,
                plant_dc_capacity_kw=capacity, temp_coeff_per_c=gamma, loss_factor=loss,
                forecast_horizon_hours=horizon, forecast_interval_min=interval, use_openmeteo=live,
                use_manual_openmeteo=manual, video_frame_interval_seconds=frame_interval,
                video_start_times=video_times, image_timestamps=image_times)
with tabs[1]:
    st.subheader("Review and run")
    image_count = sum(row["Type"] == "images" and Path(row["File"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for row in inventory)
    video_count = sum(row["Type"] == "images" for row in inventory) - image_count
    st.markdown(f"**Location:** {lat:.5f}, {lon:.5f} · {timezone}  \n**Panels:** tilt {tilt:g}° / azimuth {azimuth:g}°  \n**Capacity:** {capacity:,.0f} kW DC  \n**Forecast:** {horizon:g} hours at {interval} minutes, plus rolling +15 minutes  \n**Sky input:** {image_count} images and {video_count} videos")
    for kind, label in [("ghi", "Actual GHI"), ("power", "Actual DC power")]:
        if not any(row["Type"] == kind for row in inventory):
            st.warning(f"{label} file missing. Predictions will run; corresponding actuals and accuracy will be blank.")
    st.caption("Power uses your configured DC capacity, temperature correction and loss factor. Weather availability is checked during the run.")
    tab_run = st.button("Run forecast", type="primary", icon=":material/play_arrow:", key="run_main")
    if sidebar_run or tab_run:
        st.session_state.result = None
        if not dirs or not any(row["Type"] == "images" for row in inventory):
            st.error("Add at least one sky image or video in Upload Data before running.")
        else:
            output = ROOT / "outputs" / "dashboard" / st.session_state.session_id / uuid.uuid4().hex
            with st.spinner("Analysing sky, retrieving weather and calculating forecasts…"):
                result = run_forecast(**settings, input_dirs=dirs, output_dir=output)
            result["output_dir"] = str(output)
            result["image_dir"] = str(dirs["images"])
            st.session_state.result = result
            st.session_state.last_config = settings.copy()
            st.session_state.last_version = version
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.log").write_text(result["status"]["log"], encoding="utf-8")
            (output / "run_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    result = st.session_state.result
    if result:
        status = result["status"]
        if status["success"]:
            st.success(f"Forecast complete · Open-Meteo: {status['openmeteo']} · Results are ready in the tabs.")
        else:
            st.error(f"Forecast could not complete. {status['error']}")
        for warning in status["warnings"]:
            st.warning(warning)
        with st.expander("Detected DC power columns and conversions"):
            st.text("\n".join(status["power_detection"]) or "No usable power columns were detected.")
        with st.expander("Backend run log"):
            st.code(status["log"], language="text")

result = st.session_state.result
if result and (settings != st.session_state.last_config or version != st.session_state.last_version):
    st.warning("Settings or inputs changed. Displayed results belong to the last run; run again to apply changes.")
ready = result and result["status"]["success"]
with tabs[2]:
    st.subheader("GHI · sunlight on a horizontal surface")
    if ready:
        show_results(result, "ghi")
    else:
        st.info("Run a forecast to view GHI predictions and comparison metrics.")
with tabs[3]:
    st.subheader("DC power · predicted versus measured")
    st.caption("Actual DC uses a reported DC power column when present; otherwise it is calculated as Σ(V × I)/1,000. Incomplete inverter totals are excluded.")
    if ready:
        show_results(result, "power")
        with st.expander("Power calculation and irradiance method"):
            internal = table(result, "all_internal_predictions_with_features.csv")
            st.dataframe(internal.reindex(columns=["prediction_time", "predicted_gti_poa_wm2", "gti_method", "estimated_cell_temperature_c", "used_dc_capacity_kw", "power_model_method"]), hide_index=True, alt="Power conversion method and assumptions")
    else:
        st.info("Run a forecast to view DC power predictions.")
with tabs[4]:
    st.subheader("The complete forecast")
    if ready:
        combined = table(result, "combined_ghi_power_table.csv")
        chosen = st.multiselect("Sections", list(combined.section.unique()), default=list(combined.section.unique()), format_func=lambda x: SECTIONS.get(x, x))
        combined["forecast_time"] = pd.to_datetime(combined.forecast_time)
        start, end = combined.forecast_time.min().to_pydatetime(), combined.forecast_time.max().to_pydatetime()
        if start < end:
            start, end = st.slider("Forecast time range (site local)", min_value=start, max_value=end, value=(start, end), format="MM/DD HH:mm")
        filtered = combined[combined.section.isin(chosen) & combined.forecast_time.between(start, end)]
        st.caption("GHI and GTI/POA: W/m² · DC power: kW · Confidence labels are heuristic, not probabilities.")
        st.dataframe(filtered, hide_index=True, alt="Combined GHI, plane-of-array irradiance, power and cloud features")
    else:
        st.info("Run a forecast to explore the combined results.")
with tabs[5]:
    st.subheader("What the image model sees")
    if ready:
        features = table(result, "image_feature_table.csv")
        st.dataframe(features, hide_index=True, alt="Image CSI, GHI, okta, cloud class and blocking flags")
        previews = list(Path(result["output_dir"]).glob("video_frames/*/*.jpg"))[:6]
        if not previews:
            previews = [p for p in Path(result["image_dir"]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}][:6]
        if previews:
            st.image([str(p) for p in previews], width=220, caption=[p.name for p in previews], alt="Sky images or extracted video frames used by the image model")
    else:
        st.info("Image features and thumbnails will appear after a run.")
with tabs[6]:
    st.subheader("Take your results with you")
    if ready:
        for name, path in result["paths"].items():
            st.download_button(name, Path(path).read_bytes(), file_name=name, mime="text/csv", key=f"download_{name}", icon=":material/download:")
        st.download_button("Download complete outputs ZIP", zip_download(result["output_dir"]), file_name="solar_forecast_outputs.zip", mime="application/zip", type="primary", key="download_zip")
        st.caption(f"Saved locally: {result['output_dir']}")
    else:
        st.info("Run a forecast to enable CSV and ZIP downloads.")
st.divider()
st.caption("Model assumptions: fixed panel orientation, approximate cell temperature and hourly weather interpolation. Validate across multiple days before operational use.")
