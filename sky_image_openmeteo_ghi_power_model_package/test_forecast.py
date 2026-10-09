"""Regression checks using synthetic data only; no plant accuracy claims."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import run_ghi_power_forecast as m


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=m.WORK_DIR)
        self.root = Path(self.temp.name)
        self.patchers = []
        for name, folder in [("IMAGE_DIR", "images"), ("GHI_DIR", "ghi"), ("POWER_DIR", "power"),
                             ("OPENMETEO_DIR", "openmeteo"), ("MODEL_DIR", "models"), ("OUTPUT_DIR", "outputs")]:
            path = self.root / folder
            path.mkdir()
            self.patchers.append(patch.object(m, name, path))
        self.patchers.append(patch.object(m, "CFG", dict(m.CFG)))
        for p in self.patchers:
            p.start()
        m.CFG.update(video_frame_interval_seconds=1, power_unit="auto", openmeteo_enabled=False, power_calibration_enabled=True)

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.temp.cleanup()

    def test_whatsapp_and_timezone(self):
        self.assertEqual(m.parse_timestamp_from_name(Path("WhatsApp Image 2026-09-29 at 10.44.02 AM.jpeg")), pd.Timestamp("2026-09-29 10:44:02"))
        self.assertEqual(m.parse_timestamp_from_name(Path("WhatsApp Video 2026-10-07 at 7.18.44 AM.mp4")), pd.Timestamp("2026-10-07 07:18:44"))
        times = m.parse_datetime_series(pd.Series(["2026-10-07T01:48:44Z", "07/10/2026 07:18:44", "2026-10-07 07:18:44"]))
        self.assertTrue((times == pd.Timestamp("2026-10-07 07:18:44")).all())

    def test_nearest_order_index_tolerance(self):
        actual = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-07 10:00", "2026-10-07 11:00"]), "value": [10., 20.]})
        times = pd.Series(pd.to_datetime(["2026-10-07 11:01", "2026-10-07 10:00", "2026-10-07 13:00", "2026-10-07 11:01"]), index=[9, 3, 4, 8])
        result = m.nearest_actual(times, actual, "value")
        self.assertEqual(result.index.tolist(), [9, 3, 4, 8])
        np.testing.assert_allclose(result, [20, 10, np.nan, 20], equal_nan=True)

    def test_power_units_and_pairs(self):
        np.testing.assert_allclose(m.infer_power_kw(pd.Series([5000]), "DC Power (W)"), [5])
        np.testing.assert_allclose(m.infer_power_kw(pd.Series([5]), "DC Power (kW)"), [5])
        np.testing.assert_allclose(m.infer_power_kw(pd.Series([.005]), "Power DC (MW)"), [5])
        columns = ["String 2 DC Current", "String 1 DC Voltage", "String 2 DC Voltage", "String 1 DC Current"]
        self.assertEqual(m.vi_pairs(columns), [(columns[1], columns[3]), (columns[2], columns[0])])
        with self.assertRaises(ValueError):
            m.vi_pairs(["String 1 DC Voltage", "String 2 DC Current"])

    def test_power_loading_excel_csv_and_ac_rejection(self):
        pd.DataFrame({"timestamp": ["2026-10-07 10:00"], "String 2 DC Current": [3], "String 1 DC Voltage": [100],
                      "String 2 DC Voltage": [200], "String 1 DC Current": [2]}).to_excel(m.POWER_DIR / "arbitrary.xlsx", index=False)
        result = m.load_actual_power()
        self.assertAlmostEqual(result.actual_dc_power_kw.iloc[0], .8)
        self.assertNotIn("actual_dc_voltage_v", result)
        pd.DataFrame({"timestamp": ["2026-10-07 11:00"], "Total DC Power (kW)": [5], "DC Voltage": [500], "DC Current": [9]}).to_csv(m.POWER_DIR / "other.CSV", index=False)
        result = m.load_actual_power()
        self.assertEqual(result.actual_dc_power_kw.tolist(), [.8, 5])
        self.assertEqual(result.actual_dc_voltage_v.iloc[1], 500)
        pd.DataFrame({"timestamp": ["2026-10-07 12:00"], "AC Power (kW)": [100]}).to_csv(m.POWER_DIR / "ac.csv", index=False)
        self.assertEqual(len(m.load_actual_power()), 2)

    def test_ghi_excel_units_and_split_timestamp(self):
        pd.DataFrame({"Date": ["07/10/2026"], "Time": ["10:00:00"], "GHI (kW/m2)": [.8]}).to_excel(m.GHI_DIR / "any.xlsx", index=False)
        result = m.load_actual_ghi()
        self.assertEqual(result.actual_ghi_wm2.iloc[0], 800)
        self.assertEqual(result.timestamp.iloc[0], pd.Timestamp("2026-10-07 10:00"))
        self.assertEqual(m.infer_unit_to_wm2(pd.Series([2]), "GHI W/m2").iloc[0], 2)

    def test_manual_weather_json_csv_fallback(self):
        data = {"timezone": "UTC", "hourly_units": {"wind_speed_10m": "km/h"},
                "hourly": {"time": ["2026-10-07T04:30", "2026-10-07T05:30"], "shortwave_radiation": [500, 600], "wind_speed_10m": [36, 18]}}
        (m.OPENMETEO_DIR / "anything.json").write_text(json.dumps(data))
        m.CFG["openmeteo_enabled"] = True
        with patch.object(m.requests, "get", side_effect=ConnectionError("synthetic offline test")) as get:
            result = m.fetch_openmeteo(pd.Timestamp("2026-10-07 09:00"), pd.Timestamp("2026-10-07 12:00"))
            get.assert_called_once()
        self.assertEqual(result.timestamp.iloc[0], pd.Timestamp("2026-10-07 10:00"))
        self.assertEqual(result.om_wind_ms.tolist(), [10, 5])
        (m.OPENMETEO_DIR / "anything.json").unlink()
        (m.OPENMETEO_DIR / "download.csv").write_text("latitude,longitude,timezone\n27.5,72.4,Asia/Kolkata\n\ntime,shortwave_radiation (W/m2),wind_speed_10m (km/h)\n2026-10-07T10:00,500,36\n2026-10-07T11:00,600,18\n")
        result = m.load_manual_openmeteo(pd.Timestamp("2026-10-07"), pd.Timestamp("2026-10-08"))
        self.assertEqual(result.om_ghi_wm2.tolist(), [500, 600])
        self.assertEqual(result.om_wind_ms.tolist(), [10, 5])

    def test_weather_interpolation_does_not_extrapolate(self):
        weather = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-07 10:00", "2026-10-07 11:00"]), "om_ghi_wm2": [500., 700.]})
        times = pd.Series(pd.to_datetime(["2026-10-07 10:30", "2026-10-07 12:00"]))
        result = m.interp_openmeteo(times, weather)
        self.assertEqual(result.om_ghi_wm2.iloc[0], 600)
        self.assertTrue(pd.isna(result.om_ghi_wm2.iloc[1]))

    def test_perez_no_weather_and_night(self):
        frame = pd.DataFrame({"prediction_time": pd.to_datetime(["2026-10-07 12:00", "2026-10-07 00:00"]), "predicted_ghi_wm2": [800., 0.]})
        result = m.compute_gti_poa(frame)
        self.assertTrue(result.gti_method.str.startswith("Perez").all())
        self.assertGreater(result.predicted_gti_poa_wm2.iloc[0], 0)
        self.assertEqual(result.predicted_gti_poa_wm2.iloc[1], 0)
        np.testing.assert_allclose(result.predicted_dni_wm2 * np.cos(np.radians(result.solar_zenith_deg)) + result.predicted_dhi_wm2, frame.predicted_ghi_wm2)

    def test_perez_weather_components_preserve_ghi(self):
        frame = pd.DataFrame({"prediction_time": pd.to_datetime(["2026-10-07 12:00"]),
                              "predicted_ghi_wm2": [200.], "om_dni_wm2": [900.], "om_dhi_wm2": [200.]})
        result = m.compute_gti_poa(frame)
        reconstructed = result.predicted_dni_wm2 * np.cos(np.radians(result.solar_zenith_deg)) + result.predicted_dhi_wm2
        np.testing.assert_allclose(reconstructed, [200])
        self.assertTrue(result.gti_method.str.startswith("Perez").all())

    def test_same_timestamp_images_are_both_predicted(self):
        frame = pd.DataFrame({"image_time": pd.to_datetime(["2026-10-07 12:00"]*2), "image_file": ["a.jpg", "b.jpg"]})
        result = m.make_prediction_rows(frame)
        self.assertEqual(len(result), 12)
        self.assertEqual((result.section == "image_time_current").sum(), 2)

    def test_current_csi_preserved_for_blocking_sky(self):
        row = pd.Series({"image_current_csi": .9, "white_blocking_cloud_score": .9, "v8_true_clear_blue_sky_fraction": .1})
        csi, _, _ = m.image_future_csi_rule(row, 0)
        self.assertEqual(csi, .9)

    def test_missing_input_folders_created(self):
        for name in ["IMAGE_DIR", "GHI_DIR", "POWER_DIR", "OPENMETEO_DIR", "MODEL_DIR", "OUTPUT_DIR"]:
            getattr(m, name).rmdir()
        with self.assertRaises(FileNotFoundError):
            m.main()
        self.assertTrue(all(getattr(m, name).is_dir() for name in ["IMAGE_DIR", "GHI_DIR", "POWER_DIR", "OPENMETEO_DIR", "MODEL_DIR", "OUTPUT_DIR"]))

    def test_no_future_power_leakage(self):
        times = pd.date_range("2026-10-07 10:00", periods=4, freq="15min")
        frame = pd.DataFrame({"prediction_time": times, "origin_image_time": times, "section": "image_time_current", "predicted_gti_poa_wm2": [500.]*4})
        actual = pd.DataFrame({"timestamp": times, "actual_dc_power_kw": [100., 100., 100., 99999.]})
        first = m.predict_power(frame, actual)
        actual.loc[3, "actual_dc_power_kw"] = 1
        second = m.predict_power(frame, actual)
        np.testing.assert_allclose(first.predicted_dc_power_kw, second.predicted_dc_power_kw)
        self.assertEqual(first.predicted_dc_power_kw.iloc[3], 100)

    def test_metrics_and_missing_values(self):
        frame = pd.DataFrame({"prediction_time": pd.to_datetime(["2026-10-07 10:00", "2026-10-07 11:00"]),
                              "predicted_ghi_wm2": [80, 120], "predicted_dc_power_kw": [8, 12], "section": ["image_time_current"]*2})
        actual = pd.DataFrame({"timestamp": frame.prediction_time, "actual_ghi_wm2": [100, 100], "actual_dc_power_kw": [10, 10]})
        scored = m.attach_actuals(frame, actual, actual)
        result = m.summary_metrics(scored)
        self.assertEqual(result.mae.tolist(), [20, 2])
        self.assertEqual(result.rmse.tolist(), [20, 2])
        self.assertEqual(result.bias_actual_minus_predicted.tolist(), [0, 0])
        self.assertEqual(result.average_accuracy_percent.tolist(), [80, 80])

    def test_locked_csv_returns_unique_path(self):
        original = pd.DataFrame.to_csv
        target = self.root / "locked.csv"
        def locked(df, path, *args, **kwargs):
            if Path(path) == target:
                raise PermissionError("simulated Excel lock")
            return original(df, path, *args, **kwargs)
        with patch.object(pd.DataFrame, "to_csv", locked):
            first = m.safe_to_csv(pd.DataFrame({"value": [1]}), target)
            second = m.safe_to_csv(pd.DataFrame({"value": [2]}), target)
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists() and second.exists())

    def test_video_and_end_to_end_without_weather(self):
        path = m.IMAGE_DIR / "WhatsApp Video 2026-10-07 at 7.18.44 AM.mp4"
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2., (128, 96))
        self.assertTrue(video.isOpened())
        for _ in range(6):
            frame = np.full((96, 128, 3), (220, 150, 80), dtype=np.uint8)
            cv2.circle(frame, (90, 30), 12, (250, 250, 250), -1)
            video.write(frame)
        video.release()
        times = pd.date_range("2026-10-07 07:18:44", periods=150, freq="min")
        pd.DataFrame({"timestamp": times, "GHI W/m2": 200.}).to_csv(m.GHI_DIR / "measured.csv", index=False)
        pd.DataFrame({"timestamp": times, "Vdc": 500., "Idc": 10.}).to_excel(m.POWER_DIR / "measured.xlsx", index=False)
        m.main()
        ghi = pd.read_csv(m.OUTPUT_DIR / "ghi_comparison_table.csv")
        power = pd.read_csv(m.OUTPUT_DIR / "power_comparison_table.csv")
        features = pd.read_csv(m.OUTPUT_DIR / "image_feature_table.csv")
        self.assertEqual(len(features), 3)
        self.assertEqual(ghi.section.value_counts().to_dict(), {"latest_image_2hr_future": 8, "image_time_current": 3, "rolling_image_t_plus_15": 3})
        self.assertTrue(power.actual_dc_power_kw.eq(5).all())
        self.assertTrue(ghi.actual_ghi_wm2.eq(200).all())
        self.assertTrue(power.predicted_dc_power_kw.notna().all())
        self.assertEqual(ghi[ghi.section == "latest_image_2hr_future"].horizon_min.tolist(), list(range(15, 121, 15)))
        self.assertEqual(len(pd.read_csv(m.OUTPUT_DIR / "summary_metrics.csv")), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
