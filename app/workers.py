"""Keep model loading and file inference off the GUI thread."""

from pathlib import Path
import threading
import traceback

from PySide6.QtCore import QThread, Signal


class ModelWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, detector, classifier, device, parent=None):
        super().__init__(parent)
        self.paths = detector, classifier
        self.device = device

    def run(self):
        try:
            from .inference import Engine

            engine = Engine(*self.paths, device=self.device, log=self.message.emit)
            engine.load()
            self.ready.emit(engine)
        except Exception:
            self.failed.emit(traceback.format_exc())


class AnalysisWorker(QThread):
    frame_ready = Signal(int, object, object)
    progress = Signal(int, int)
    failed = Signal(str)
    completed = Signal(bool)

    def __init__(self, engine, source, settings, start=0, stride=10, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.source = Path(source)
        self.settings = settings
        self.start_frame = start
        self.stride = stride
        self.cancel_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()

    def pause(self):
        self.resume_event.clear()

    def resume(self):
        self.resume_event.set()

    def cancel(self):
        self.cancel_event.set()
        self.resume_event.set()

    def run(self):
        from .media import MediaSource

        media = None
        try:
            media = MediaSource(self.source)
            indices = range(self.start_frame, media.frame_count, self.stride) if media.is_video else [0]
            total = len(indices)
            for position, index in enumerate(indices):
                self.resume_event.wait()
                if self.cancel_event.is_set():
                    break
                image = media.read(index)
                result = self.engine.analyze(image, self.settings)
                if self.cancel_event.is_set():
                    break
                self.frame_ready.emit(index, image, result)
                self.progress.emit(position + 1, total)
            self.completed.emit(self.cancel_event.is_set())
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if media is not None:
                media.close()
