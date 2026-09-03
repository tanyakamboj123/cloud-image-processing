
from pathlib import Path
import tempfile
import math
import io
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Keep this file in the same folder as sky_prediction_low_sun_scattered_clear_corrected.py
import sky_prediction_low_sun_scattered_clear_corrected as core

st.set_page_config(page_title="Sky Cloud Analysis Dashboard", layout="wide")

def safe_round(x, n=2):
    try:
        if pd.isna(x):
            return np.nan
        return round(float(x), n)
    except Exception:
        return x

def okta_from_fraction(cloud_fraction):
    try:
        return int(np.clip(np.rint(8.0 * float(cloud_fraction)), 0, 8))
    except Exception:
        return 0

def make_mask_preview(mask, title):
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf

def build_results_row(image_path, timestamp_local, lat, lon, tz, plant_csv_path, label):
    metrics, img_rgb, masks = core.analyse_image_versions(str(image_path), timestamp_local, lat, lon, tz)
    cosz = core.solar_cos_zenith_noaa(pd.to_datetime(timestamp_local), lat, lon, tz)
    clear = core.clear_sky_ghi_haurwitz(cosz)
    solar_elev = math.degrees(math.asin(cosz)) if cosz > 0 else 0.0

    row = {
        "label": label,
        "timestamp_local": str(pd.to_datetime(timestamp_local)),
        "clear_sky_ghi_Wm2": clear,
        "solar_elevation_deg": solar_elev,
        **metrics,
        "predicted_ghi_v1_Wm2": metrics["predicted_csi_v1"] * clear,
        "predicted_ghi_v2_Wm2": metrics["predicted_csi_v2"] * clear,
        "predicted_ghi_v3_Wm2": metrics["predicted_csi_v3"] * clear,
        "predicted_ghi_v4_Wm2": metrics["predicted_csi_v4"] * clear,
        "predicted_ghi_v5_Wm2": metrics["predicted_csi_v5"] * clear,
    }

    # Use the new display-friendly cloud fraction for dashboard okta.
    row["okta_display"] = okta_from_fraction(row.get("v8_display_cloud_fraction", row.get("v3_combined_cloud_fraction", 0.0)))
    row["predicted_solar_MW_over_1km2"] = row["predicted_ghi_v5_Wm2"]
    row["clear_sky_solar_MW_over_1km2"] = clear
    row["cloud_loss_MW_over_1km2"] = max(0.0, clear - row["predicted_ghi_v5_Wm2"])
    row["pv_equiv_loss_MW_1km2_20pct"] = 0.20 * row["cloud_loss_MW_over_1km2"]

    if plant_csv_path:
        actual, source, method = core.read_actual_ghi_from_plant(plant_csv_path, timestamp_local)
        row["actual_ghi_source"] = source
        row["actual_ghi_method"] = method
        row["actual_ghi_Wm2"] = actual
        row["actual_csi"] = actual / clear if clear > 0 and pd.notna(actual) else np.nan
        for ver in ["v1", "v2", "v3", "v4", "v5"]:
            row[f"error_{ver}_Wm2"] = row[f"predicted_ghi_{ver}_Wm2"] - actual if pd.notna(actual) else np.nan
    return row, img_rgb, masks

def factor_table(row):
    factors = [
        ("Blue-dominant pixel fraction (technical)", row.get("v8_blue_dominant_pixel_fraction")),
        ("True clear-blue sky fraction (display)", row.get("v8_true_clear_blue_sky_fraction")),
        ("Blue-tinted cloud fraction", row.get("v8_blue_tinted_cloud_fraction")),
        ("Hazy blue fraction", row.get("v8_display_hazy_blue_fraction")),
        ("Corrected bright/white cloud fraction", row.get("v8_display_bright_cloud_fraction")),
        ("Dark cloud fraction", row.get("v8_display_dark_cloud_fraction")),
        ("Corrected visual cloud fraction", row.get("v8_display_cloud_fraction")),
        ("Okta number", row.get("okta_display")),
        ("Combined cloud fraction used internally", row.get("v3_combined_cloud_fraction")),
        ("Cloudiness index", row.get("v6_cloudiness_index_0_to_1")),
        ("Cloud impact index", row.get("v6_cloud_impact_index_0_to_1")),
        ("Ramp risk index", row.get("v6_ramp_risk_index_0_to_1")),
        ("Glare score", row.get("v3_glare_score")),
        ("Uniform overcast score", row.get("v3_uniform_overcast_score")),
        ("Conflict score", row.get("v3_conflict_score")),
        ("Light-grey fraction", row.get("v5_light_grey_fraction")),
        ("Smooth light-grey fraction", row.get("v5_smooth_light_grey_fraction")),
        ("Textured light-grey fraction", row.get("v5_textured_light_grey_fraction")),
        ("Solar elevation (deg)", row.get("solar_elevation_deg")),
        ("Low-sun angle flag", row.get("low_sun_angle_flag")),
        ("Low-sun correction applied", row.get("low_sun_correction_flag")),
        ("Low-sun CSI cap", row.get("low_sun_csi_cap")),
        ("Scattered-clear boost applied", row.get("scattered_clear_boost_flag")),
        ("Scattered-clear CSI floor", row.get("scattered_clear_csi_floor")),
        ("Clear-sky GHI (W/m²)", row.get("clear_sky_ghi_Wm2")),
        ("Predicted CSI V5", row.get("predicted_csi_v5")),
        ("Predicted GHI V5 (W/m²)", row.get("predicted_ghi_v5_Wm2")),
    ]
    df = pd.DataFrame(factors, columns=["Factor", "Value"])
    df["Value"] = df["Value"].apply(lambda x: safe_round(x, 2) if isinstance(x, (float, int, np.floating, np.integer)) else x)
    return df

def compare_table(row):
    df = pd.DataFrame([
        {"Model": "RGB-HSV Threshold Cloud Classification Model", "Predicted CSI": row["predicted_csi_v1"], "Predicted GHI (W/m²)": row["predicted_ghi_v1_Wm2"]},
        {"Model": "Soft Confidence Model", "Predicted CSI": row["predicted_csi_v2"], "Predicted GHI (W/m²)": row["predicted_ghi_v2_Wm2"]},
        {"Model": "Adaptive Otsu Model", "Predicted CSI": row["predicted_csi_v3"], "Predicted GHI (W/m²)": row["predicted_ghi_v3_Wm2"]},
        {"Model": "Hazy-Overcast Model", "Predicted CSI": row["predicted_csi_v4"], "Predicted GHI (W/m²)": row["predicted_ghi_v4_Wm2"]},
        {"Model": "Grey-Cloud Enhanced Model", "Predicted CSI": row["predicted_csi_v5"], "Predicted GHI (W/m²)": row["predicted_ghi_v5_Wm2"]},
    ])
    if "actual_ghi_Wm2" in row and pd.notna(row["actual_ghi_Wm2"]):
        df["Actual GHI (W/m²)"] = row["actual_ghi_Wm2"]
        df["Actual CSI"] = row["actual_csi"]
        df["Error (W/m²)"] = [
            row.get("error_v1_Wm2", np.nan),
            row.get("error_v2_Wm2", np.nan),
            row.get("error_v3_Wm2", np.nan),
            row.get("error_v4_Wm2", np.nan),
            row.get("error_v5_Wm2", np.nan),
        ]
    return df.round(2)

st.title("Sky Cloud Analysis Dashboard")
st.caption("Interactive sky-image pipeline for cloud factors, CSI/GHI prediction, and optional plant-data comparison.")

with st.expander("Why Streamlit is used here", expanded=False):
    st.markdown("""
**Streamlit** is a Python library used to build an interactive web app directly from Python code.

In this project it is useful because:
- you can upload any sky image from the browser
- you can enter date, time, latitude, longitude, and timezone without editing code
- you can optionally upload plant CSV for actual GHI comparison
- the app shows predicted CSI, predicted GHI, cloudiness index, okta number, cloud type, masks, and 1 km cloud-field description
- it gives downloadable CSV outputs for reports and future forecasting work
""")

with st.sidebar:
    st.header("Inputs")
    image_file = st.file_uploader("Upload sky image", type=["png", "jpg", "jpeg", "webp"])
    timestamp_local = st.text_input("Image timestamp (local)", value="2026-08-09 08:20")
    latitude = st.number_input("Latitude", value=27.54225, format="%.6f")
    longitude = st.number_input("Longitude", value=72.471000, format="%.6f")
    timezone_offset = st.number_input("Timezone offset from UTC (hours)", value=5.5, step=0.5)
    label = st.text_input("Run label", value="uploaded_image")
    plant_csv = st.file_uploader("Optional plant CSV", type=["csv"])
    run_button = st.button("Analyze image", type="primary", use_container_width=True)

if "run_data_v8" not in st.session_state:
    st.session_state.run_data_v8 = None

if run_button:
    if image_file is None:
        st.error("Please upload an image first.")
        st.stop()
    try:
        pd.to_datetime(timestamp_local)
    except Exception:
        st.error("Timestamp format is invalid. Use YYYY-MM-DD HH:MM")
        st.stop()

    tmp_dir = Path(tempfile.mkdtemp())
    image_path = tmp_dir / image_file.name
    image_path.write_bytes(image_file.read())

    plant_path = None
    if plant_csv is not None:
        plant_path = tmp_dir / plant_csv.name
        plant_path.write_bytes(plant_csv.read())

    with st.spinner("Running analysis..."):
        row, img_rgb, masks = build_results_row(image_path, timestamp_local, float(latitude), float(longitude), float(timezone_offset), str(plant_path) if plant_path else None, label)
        overlay_path = tmp_dir / "overlay.png"
        dashboard_path = tmp_dir / "dashboard.png"
        core.save_overlay(img_rgb, masks, overlay_path, f"{label} | {row['v5_sky_class']}")
        core.save_cloud_factor_dashboard(img_rgb, masks, row, dashboard_path)
        result_df = pd.DataFrame([row])
        st.session_state.run_data_v8 = {
            "row": row,
            "result_df": result_df,
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "dashboard_path": str(dashboard_path),
            "masks": masks,
        }

if st.session_state.run_data_v8 is None:
    st.info("Upload an image and click **Analyze image**.")
else:
    data = st.session_state.run_data_v8
    row = data["row"]
    masks = data["masks"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Quick Summary", "Cloud Factors", "1 km Description", "Model Comparison", "Images & Masks", "Download Results"
    ])

    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predicted CSI (V5)", safe_round(row.get("predicted_csi_v5"), 2))
        c2.metric("Predicted GHI (W/m²)", safe_round(row.get("predicted_ghi_v5_Wm2"), 1))
        c3.metric("Cloudiness index", safe_round(row.get("v6_cloudiness_index_0_to_1"), 2))
        c4.metric("Okta number", int(row.get("okta_display", 0)))
        c5.metric("Clear-sky GHI (W/m²)", safe_round(row.get("clear_sky_ghi_Wm2"), 1))

        st.subheader("Clearer sky/cloud interpretation")
        display_df = pd.DataFrame([{
            "Blue-dominant pixels": safe_round(row.get("v8_blue_dominant_pixel_fraction"), 2),
            "True clear-blue sky": safe_round(row.get("v8_true_clear_blue_sky_fraction"), 2),
            "Blue-tinted cloud": safe_round(row.get("v8_blue_tinted_cloud_fraction"), 2),
            "Corrected bright/white cloud": safe_round(row.get("v8_display_bright_cloud_fraction"), 2),
            "Corrected visual cloud": safe_round(row.get("v8_display_cloud_fraction"), 2),
            "Cloud nature": row.get("v6_cloud_nature", ""),
            "1 km field type": row.get("v7_1km_cloud_field_type", ""),
        }])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.info(row.get("v8_display_note", ""))
        if int(row.get("low_sun_correction_flag", 0)) == 1:
            st.warning(row.get("low_sun_reason", "Low-sun correction was applied."))
        elif int(row.get("low_sun_angle_flag", 0)) == 1:
            st.caption(row.get("low_sun_reason", ""))
        if int(row.get("scattered_clear_boost_flag", 0)) == 1:
            st.success(row.get("scattered_clear_reason", "Scattered-clear-sky boost was applied."))

    with tab2:
        st.subheader("Major cloud factors for forecasting")
        st.dataframe(factor_table(row), use_container_width=True, hide_index=True)
        chart_df = pd.DataFrame({
            "Factor": ["True clear blue", "Blue-tinted cloud", "Bright/white cloud", "Dark cloud", "Light grey", "Visual cloud", "Cloudiness index", "Cloud impact index", "Ramp risk"],
            "Value": [
                row.get("v8_true_clear_blue_sky_fraction", 0),
                row.get("v8_blue_tinted_cloud_fraction", 0),
                row.get("v8_display_bright_cloud_fraction", 0),
                row.get("v8_display_dark_cloud_fraction", 0),
                row.get("v5_light_grey_fraction", 0),
                row.get("v8_display_cloud_fraction", 0),
                row.get("v6_cloudiness_index_0_to_1", 0),
                row.get("v6_cloud_impact_index_0_to_1", 0),
                row.get("v6_ramp_risk_index_0_to_1", 0),
            ],
        })
        st.bar_chart(chart_df.set_index("Factor"))

    with tab3:
        st.subheader("1 km × 1 km cloud-field interpretation")
        st.markdown(f"**Cloud field type:** {row.get('v7_1km_cloud_field_type', '')}")
        st.write(row.get("v7_1km_cloud_distribution", ""))
        st.write(row.get("v7_1km_cloud_impact_description", ""))
        st.write(row.get("v7_1km_forecasting_note", ""))
        st.caption(row.get("v7_1km_assumption", ""))

        impact_df = pd.DataFrame([
            {"Measure": "Predicted incoming solar power over 1 km² (MW)", "Value": safe_round(row.get("predicted_solar_MW_over_1km2"), 2)},
            {"Measure": "Clear-sky solar power over 1 km² (MW)", "Value": safe_round(row.get("clear_sky_solar_MW_over_1km2"), 2)},
            {"Measure": "Cloud loss over 1 km² (MW)", "Value": safe_round(row.get("cloud_loss_MW_over_1km2"), 2)},
            {"Measure": "PV-equivalent loss over 1 km² at 20% efficiency (MW)", "Value": safe_round(row.get("pv_equiv_loss_MW_1km2_20pct"), 2)},
        ])
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Model-wise prediction comparison")
        st.dataframe(compare_table(row), use_container_width=True, hide_index=True)
        if "actual_ghi_Wm2" in row and pd.notna(row["actual_ghi_Wm2"]):
            st.info(f"Actual GHI source: {row.get('actual_ghi_source', '')} | method: {row.get('actual_ghi_method', '')}")

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original uploaded image")
            st.image(Image.open(data["image_path"]), use_container_width=True)
            st.subheader("Detailed cloud-factor dashboard")
            st.image(Image.open(data["dashboard_path"]), use_container_width=True)
        with c2:
            st.subheader("Overlay image")
            st.image(Image.open(data["overlay_path"]), use_container_width=True)
            st.subheader("Useful masks")
            m1, m2 = st.columns(2)
            with m1:
                st.image(make_mask_preview(masks["strong_blue"], "True clear-blue mask"))
                st.image(make_mask_preview(masks["light_grey"], "Light-grey cloud mask"))
            with m2:
                st.image(make_mask_preview(masks["bright"], "Corrected bright/cloud mask"))
                st.image(make_mask_preview(masks["cloud_otsu"], "Adaptive cloud mask"))

    with tab6:
        result_df = data["result_df"]
        st.download_button(
            "Download full result CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{row.get('label','result')}_full_results_low_sun.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download factor table CSV",
            factor_table(row).to_csv(index=False).encode("utf-8"),
            file_name=f"{row.get('label','result')}_factor_table_low_sun.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download model comparison CSV",
            compare_table(row).to_csv(index=False).encode("utf-8"),
            file_name=f"{row.get('label','result')}_model_comparison_low_sun.csv",
            mime="text/csv",
        )

        st.code("streamlit run streamlit_sky_prediction_dashboard_low_sun_corrected.py")
        st.markdown("Keep this app file and `sky_prediction_low_sun_scattered_clear_corrected.py` in the same folder.")
