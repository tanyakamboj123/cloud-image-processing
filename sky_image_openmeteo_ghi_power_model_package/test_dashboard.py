import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid
import zipfile

from streamlit.testing.v1 import AppTest
from dashboard_support import stage_inputs, safe_name, archive_outputs, ROOT
import run_ghi_power_forecast as model


class Upload(io.BytesIO):
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name


class DashboardTests(unittest.TestCase):
    def test_upload_save_isolation_and_archive(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            sid = uuid.uuid4().hex
            data = b'timestamp,GHI\n2026-10-07 10:00,500\n'
            dirs, inventory, _ = stage_inputs(sid, {"ghi": [Upload("../../meter.csv", data)]}, root=root)
            self.assertEqual((dirs["ghi"] / "meter.csv").read_bytes(), data)
            self.assertEqual(model.list_input_files(root / "inputs" / "ghi"), [])
            self.assertEqual(len(model.list_input_files(dirs["ghi"])), 1)
            empty, _, _ = stage_inputs(sid, {}, root=root)
            self.assertEqual(list(empty["ghi"].iterdir()), [])
            zipped = archive_outputs({}, dirs["ghi"])
            with zipfile.ZipFile(io.BytesIO(zipped)) as z:
                self.assertEqual(z.read("meter.csv"), data)
            with self.assertRaises(ValueError):
                safe_name("CON.csv")

    def test_api_validation_and_no_global_mutation(self):
        capacity = model.DC_CAPACITY_KW
        original = dict(model.CFG)
        bad = model.run_forecast(timezone="invalid/zone")
        self.assertFalse(bad["status"]["success"])
        self.assertEqual(bad["paths"], {})
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            dirs = {key: Path(temp) / key for key in ["images", "ghi", "power", "openmeteo", "models"]}
            result = model.run_forecast(input_dirs=dirs, output_dir=Path(temp)/"out", plant_dc_capacity_kw=123, use_openmeteo=False)
            self.assertFalse(result["status"]["success"])
            self.assertIn("No usable images", result["status"]["error"])
        self.assertEqual(model.DC_CAPACITY_KW, capacity)
        self.assertEqual(model.CFG, original)

    def test_dashboard_run_charts_filters_downloads_and_clear(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 7)
        app.toggle(key="live_weather").set_value(False)
        app.button(key="run_main").click().run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        result = app.session_state.result
        self.assertTrue(result["status"]["success"], result["status"])
        self.assertEqual(len(app.get("plotly_chart")), 4)
        self.assertGreaterEqual(len(app.get("download_button")), 6)
        self.assertTrue(any("no usable Open-Meteo" in w.value for w in app.warning))
        paths = result["paths"]
        self.assertTrue(all(Path(path).exists() for path in paths.values()))
        app.selectbox(key="ghi_section").select("image_time_current").run()
        self.assertFalse(app.exception)
        app.toggle(key="use_existing").set_value(False).run()
        self.assertTrue(any("Settings or inputs changed" in w.value for w in app.warning))
        app.button(key="run_main").click().run()
        self.assertTrue(any("Add at least one" in e.value for e in app.error))
        self.assertIsNone(app.session_state.result)
        app.button(key="clear_outputs").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(Path(result["output_dir"]).exists())
        self.assertTrue((ROOT / "outputs" / "plant_435mw" / "REPORT.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
