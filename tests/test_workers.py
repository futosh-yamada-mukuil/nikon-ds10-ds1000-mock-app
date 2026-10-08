"""Thread lifecycle tests use real temporary media and explicit test-only engines."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication

from app.domain import FrameResult, InferenceSettings
from app.media import MediaError, MediaSource
from app.workers import AnalysisWorker, ModelWorker


APP = QApplication.instance() or QApplication([])


def pump_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(.005)
    APP.processEvents()
    if not predicate():
        raise AssertionError("Qt worker did not reach the expected state before timeout")


class TestEngine:
    def __init__(self, failure=False, block=False):
        self.failure = failure
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.calls = []

    def analyze(self, image, settings):
        self.calls.append((image.size, settings))
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Test engine release timed out")
        if self.failure:
            raise RuntimeError("test inference failure")
        return FrameResult([], device="cpu")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.image = Path(self.temporary.name) / "image.png"
        Image.new("RGB", (60, 40), (20, 30, 40)).save(self.image)
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            if isinstance(worker, AnalysisWorker):
                worker.cancel()
                if hasattr(worker.engine, "release"):
                    worker.engine.release.set()
            self.assertTrue(worker.wait(3000))
        APP.processEvents()
        self.temporary.cleanup()

    def watch(self, worker):
        self.workers.append(worker)
        frames, progress, failed, completed = [], [], [], []
        worker.frame_ready.connect(lambda index, image, result: frames.append((index, image, result)))
        worker.progress.connect(lambda current, total: progress.append((current, total)))
        worker.failed.connect(failed.append)
        worker.completed.connect(completed.append)
        return frames, progress, failed, completed

    def test_image_analysis_emits_original_index_progress_and_completion(self):
        settings = InferenceSettings()
        engine = TestEngine()
        worker = AnalysisWorker(engine, self.image, settings)
        frames, progress, failed, completed = self.watch(worker)
        worker.start()
        pump_until(lambda: bool(completed))
        self.assertTrue(worker.wait(1000))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0], 0)
        self.assertEqual(frames[0][1].mode, "RGB")
        self.assertEqual(progress, [(1, 1)])
        self.assertEqual(completed, [False])
        self.assertEqual(failed, [])
        self.assertIs(engine.calls[0][1], settings)

    def test_pause_before_start_resume_and_cancel_while_paused_finish_safely(self):
        engine = TestEngine()
        worker = AnalysisWorker(engine, self.image, InferenceSettings())
        frames, _, failed, completed = self.watch(worker)
        worker.pause()
        worker.start()
        self.assertFalse(engine.started.wait(.05))
        worker.resume()
        pump_until(lambda: bool(completed))
        self.assertEqual(len(frames), 1)
        self.assertEqual(failed, [])
        self.assertTrue(worker.wait(1000))
        second = AnalysisWorker(TestEngine(), self.image, InferenceSettings())
        frames, _, failed, completed = self.watch(second)
        second.pause()
        second.start()
        second.cancel()
        pump_until(lambda: bool(completed))
        self.assertTrue(second.wait(1000))
        self.assertEqual(completed, [True])
        self.assertEqual(frames, [])
        self.assertEqual(failed, [])

    def test_cancel_during_frame_omits_cancelled_frame_and_closes_media(self):
        engine = TestEngine(block=True)
        media_instances = []

        def open_media(path):
            media = MediaSource(path)
            media_instances.append(media)
            return media

        with patch("app.media.MediaSource", side_effect=open_media):
            worker = AnalysisWorker(engine, self.image, InferenceSettings())
            frames, progress, failed, completed = self.watch(worker)
            worker.start()
            self.assertTrue(engine.started.wait(1))
            worker.cancel()
            engine.release.set()
            pump_until(lambda: bool(completed))
            self.assertTrue(worker.wait(1000))
        self.assertEqual(frames, [])
        self.assertEqual(progress, [])
        self.assertEqual(completed, [True])
        self.assertEqual(failed, [])
        with self.assertRaises(MediaError):
            media_instances[0].read()

    def test_video_stride_processes_correct_frames(self):
        import cv2
        import numpy as np

        path = Path(self.temporary.name) / "input.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(6):
            writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
        writer.release()
        worker = AnalysisWorker(TestEngine(), path, InferenceSettings(), start=1, stride=2)
        frames, progress, failed, completed = self.watch(worker)
        worker.start()
        pump_until(lambda: bool(completed))
        self.assertTrue(worker.wait(1000))
        self.assertEqual([row[0] for row in frames], [1, 3, 5])
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(failed, [])

    def test_failure_is_explicit_and_media_is_closed_without_success_completion(self):
        engine = TestEngine(failure=True)
        worker = AnalysisWorker(engine, self.image, InferenceSettings())
        frames, progress, failed, completed = self.watch(worker)
        worker.start()
        pump_until(lambda: bool(failed))
        self.assertTrue(worker.wait(1000))
        self.assertIn("test inference failure", failed[0])
        self.assertEqual((frames, progress, completed), ([], [], []))

    def test_model_worker_emits_real_engine_success_or_explicit_load_failure(self):
        class FakeModelEngine:
            def __init__(self, *args, device, log, calibration_path=None):
                self.loaded = False
                self.log = log

            def load(self):
                self.loaded = True
                self.log("test model loaded")

        with patch("app.inference.Engine", FakeModelEngine):
            worker = ModelWorker(Path("det.pt"), Path("class.pth"), "cpu")
            self.workers.append(worker)
            ready, failed, messages = [], [], []
            worker.ready.connect(ready.append)
            worker.failed.connect(failed.append)
            worker.message.connect(messages.append)
            worker.start()
            pump_until(lambda: bool(ready))
            self.assertTrue(worker.wait(1000))
        self.assertTrue(ready[0].loaded)
        self.assertEqual(failed, [])
        self.assertEqual(messages, ["test model loaded"])

        class FailingModelEngine(FakeModelEngine):
            def load(self):
                raise RuntimeError("test load failure")

        with patch("app.inference.Engine", FailingModelEngine):
            worker = ModelWorker(Path("det.pt"), Path("class.pth"), "cpu")
            self.workers.append(worker)
            ready, failed = [], []
            worker.ready.connect(ready.append)
            worker.failed.connect(failed.append)
            worker.start()
            pump_until(lambda: bool(failed))
            self.assertTrue(worker.wait(1000))
        self.assertEqual(ready, [])
        self.assertIn("test load failure", failed[0])


if __name__ == "__main__":
    unittest.main()
