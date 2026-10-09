"""Offscreen file workflow tests; no production model is replaced in application code."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication, QPushButton

from app.domain import CellResult, FrameResult, InferenceSettings
from app.ui import MainWindow


APP = QApplication.instance() or QApplication([])


def pump_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(.005)
    APP.processEvents()
    if not predicate():
        raise AssertionError("Qt UI did not reach the expected state before timeout")


class FakeEngine:
    device = "cpu"
    loaded = True

    def __init__(self, block=False, failure=False):
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.failure = failure
        self.settings = []

    def analyze(self, image, settings):
        self.settings.append(settings)
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("test engine timeout")
        if self.failure:
            raise RuntimeError("test analysis failure")
        return FrameResult([CellResult(1, (4, 4, 20, 20), .8, .60, True)], device="cpu")

    def get_model_metadata(self):
        return {"device": "cpu", "loaded": True}

    def close(self):
        self.loaded = False


class UserFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.image = self.directory / "input.png"
        Image.new("RGB", (80, 60), (90, 100, 110)).save(self.image)
        self.window = MainWindow(config=self.directory / "absent.json", autoload=False)

    def tearDown(self):
        worker = self.window.worker
        if worker is not None:
            worker.cancel()
            if hasattr(worker.engine, "release"):
                worker.engine.release.set()
            self.assertTrue(worker.wait(3000))
            pump_until(lambda: self.window.worker is None)
        if self.window.loader is not None:
            self.assertTrue(self.window.loader.wait(3000))
            pump_until(lambda: self.window.loader is None)
        self.window.close()
        APP.processEvents()
        self.temporary.cleanup()

    def prepare(self, engine=None):
        self.window.engine = engine or FakeEngine()
        self.window.open_source(self.image)

    def make_video(self):
        import cv2
        import numpy as np

        path = self.directory / "input.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
        self.assertTrue(writer.isOpened())
        for intensity in (30, 90, 150):
            writer.write(np.full((48, 64, 3), intensity, dtype=np.uint8))
        writer.release()
        return path

    def test_unloaded_and_unanalysed_states_have_no_invented_counts(self):
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())
        self.window.open_source(self.image)
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.table.rowCount(), 0)
        self.assertIn("—", self.window.summary.text())
        self.assertFalse(self.window.start_button.isEnabled())

    def test_classification_ui_uses_normalized_units_and_run_threshold_in_card(self):
        self.assertEqual(self.window.class_threshold.maximum(), 1.)
        self.assertEqual(self.window.class_threshold.decimals(), 2)
        self.assertEqual(self.window.class_threshold.text(), "0.50")
        self.assertEqual(self.window.det_threshold.text(), "0.50")
        self.prepare()
        self.window.run_settings = InferenceSettings(classification_threshold=.44)
        self.window.receive_result(0, self.window.original_image,
                                   FrameResult([CellResult(1, (4, 4, 20, 20), .8, .60, True)]))
        self.window.class_threshold.setValue(.50)
        self.assertIn("0〜1", self.window.candidate_note.text())
        self.assertIn("0.44", self.window.candidate_note.text())
        self.assertIn("44%", self.window.candidate_note.text())

    def test_ds1000_improvement_is_opt_in_and_resets_when_machine_changes(self):
        engine = FakeEngine()
        engine.calibration = {"machines": {"DS1000": {"detection_threshold": .5, "classification_threshold": 60.}}}
        self.prepare(engine)
        self.assertEqual(self.window.current_settings().classification_threshold, .5)
        self.assertFalse(self.window.calibration_button.isEnabled())
        self.window.machine.setCurrentText("DS1000")
        self.assertTrue(self.window.calibration_button.isEnabled())
        self.window.calibration_button.click()
        self.assertTrue(self.window.current_settings().use_calibration)
        self.assertEqual(self.window.current_settings().classification_threshold, .60)
        self.window.machine.setCurrentText("DS10")
        self.assertFalse(self.window.current_settings().use_calibration)
        self.assertEqual(self.window.current_settings().classification_threshold, .5)

    def test_image_run_snapshots_settings_and_disables_mutating_controls(self):
        engine = FakeEngine(block=True)
        self.prepare(engine)
        self.window.start_analysis()
        self.assertTrue(engine.started.wait(1))
        for control in (self.window.source_button, self.window.model_button, self.window.detector_path,
                        self.window.classifier_path, self.window.mode, self.window.machine, self.window.stride):
            self.assertFalse(control.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())
        model_selectors = [button for button in self.window.findChildren(QPushButton) if button.text() == "選択"]
        self.assertEqual(len(model_selectors), 2)
        self.assertTrue(all(not button.isEnabled() for button in model_selectors))
        self.window.class_threshold.setValue(.70)  # Programmatic change cannot alter the captured run.
        engine.release.set()
        pump_until(lambda: self.window.worker is None)
        self.assertEqual(engine.settings[0].classification_threshold, .5)
        self.assertEqual(self.window.run_settings.classification_threshold, .5)
        self.assertEqual(self.window.result.candidate_count, 1)
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertTrue(self.window.source_button.isEnabled())

    def test_seek_to_unanalysed_video_frame_invalidates_visible_result(self):
        self.window.engine = FakeEngine()
        self.window.open_source(self.make_video())
        self.window.run_settings = InferenceSettings()
        result = FrameResult([CellResult(1, (4, 4, 20, 20), .8, .60, True)])
        self.window.receive_result(0, self.window.original_image, result)
        self.window.set_busy(False)
        self.assertTrue(self.window.export_button.isEnabled())
        self.window.seek.setValue(1)
        self.assertEqual(self.window.frame_index, 1)
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.table.rowCount(), 0)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertIn("未解析", self.window.summary.text())

    def test_nullable_classification_and_detection_columns_are_rendered(self):
        self.prepare()
        result = FrameResult([CellResult(1, (0, 0, 80, 60), None, .60, True),
                              CellResult(2, (4, 4, 20, 20), .8, None, False)])
        self.window.receive_result(0, self.window.original_image, result)
        self.assertEqual(self.window.table.item(0, 1).text(), "—")
        self.assertEqual(self.window.table.item(1, 2).text(), "—")
        self.assertEqual(self.window.table.item(0, 3).text(), "超過")

    def test_every_offered_machine_option_produces_valid_settings(self):
        for index in range(self.window.machine.count()):
            self.window.machine.setCurrentIndex(index)
            settings = self.window.current_settings()
            self.assertEqual(settings.machine, self.window.machine.currentText())

    def test_manual_model_edit_requires_reload_and_invalidates_export(self):
        self.prepare()
        self.window.run_settings = InferenceSettings()
        result = FrameResult([CellResult(1, (4, 4, 20, 20), .8, .60, True)])
        self.window.receive_result(0, self.window.original_image, result)
        self.window.set_busy(False)
        self.window.detector_path.setText("different_detector.pt")
        self.assertIsNone(self.window.engine)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_failed_analysis_reports_failure_and_restores_retry_controls(self):
        self.prepare(FakeEngine(failure=True))
        self.window.start_analysis()
        pump_until(lambda: self.window.worker is None)
        self.assertIn("test analysis failure", self.window.log.toPlainText())
        self.assertIn("失敗", self.window.statusBar().currentMessage())
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertTrue(self.window.start_button.isEnabled())

    def test_failed_model_load_does_not_enable_inference(self):
        self.window.open_source(self.image)

        class FailureEngine:
            def __init__(self, *args, **kwargs):
                pass

            def load(self):
                raise RuntimeError("test load failure")

        with patch("app.inference.Engine", FailureEngine):
            self.window.load_models()
            pump_until(lambda: self.window.loader is None)
        self.assertIsNone(self.window.engine)
        self.assertIn("読み込み失敗", self.window.model_status.text())
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.source_button.isEnabled())

    def test_close_during_analysis_requests_cancel_and_waits_for_thread(self):
        engine = FakeEngine(block=True)
        self.prepare(engine)
        self.window.show()
        self.window.start_analysis()
        self.assertTrue(engine.started.wait(1))
        self.window.close()
        self.assertTrue(self.window.close_pending)
        self.assertIsNotNone(self.window.worker)
        self.assertTrue(self.window.worker.cancel_event.is_set())
        engine.release.set()
        pump_until(lambda: self.window.worker is None)
        self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()
