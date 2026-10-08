import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.calibration import load_calibration, score_features
from app.domain import InferenceSettings
from app.inference import Engine
from scripts.train_cell_calibration import fit_head


def fixture():
    head = {"mean": [0.] * 128, "std": [1.] * 128, "weight": [0.] * 128,
            "bias": 1., "eligible_for_app": True, "classification_threshold": 60., "detection_threshold": .5}
    return {"schema_version": 1, "feature_size": 128, "detector_sha256": "det", "classifier_sha256": "cls", "machines": {"DS1000": head}}


class CalibrationTests(unittest.TestCase):
    def test_changed_artifact_wrong_base_model_and_invalid_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.json"
            data = fixture()
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_calibration(path, "det", "cls")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("app.calibration.CALIBRATION_SHA256", digest):
                self.assertEqual(set(load_calibration(path, "det", "cls")["machines"]), {"DS1000"})
                with self.assertRaisesRegex(ValueError, "組み合わせ"):
                    load_calibration(path, "other", "cls")
            data["machines"]["DS1000"]["std"][0] = 0
            path.write_text(json.dumps(data))
            with patch("app.calibration.CALIBRATION_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "条件"):
                    load_calibration(path, "det", "cls")

    def test_ds10_and_whole_image_classification_cannot_use_ds1000_head(self):
        for kwargs in ({"machine": "DS10"}, {"machine": "DS1000", "mode": "classification_only"}):
            with self.assertRaisesRegex(ValueError, "DS1000"):
                InferenceSettings(use_calibration=True, **kwargs)

    def test_calibrated_forward_uses_frozen_features_and_removes_hook(self):
        import torch

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(5)], torch.nn.Linear(128, 2))
            def forward(self, tensor):
                return self.fc(torch.zeros((len(tensor), 128)))

        engine = Engine(Path("det"), Path("cls"))
        engine._classifier = TinyClassifier().eval()
        before = engine._classifier.fc[5].weight.clone()
        engine._transform = lambda image: torch.zeros((3, 224, 224))
        engine.calibration = fixture()
        score = engine._classify_calibrated(Image.new("RGB", (224, 224)), "DS1000")
        self.assertAlmostEqual(score, float(score_features(np.zeros((1, 128)), fixture()["machines"]["DS1000"])[0]))
        self.assertEqual(len(engine._classifier.fc[5]._forward_pre_hooks), 0)
        self.assertTrue(torch.equal(before, engine._classifier.fc[5].weight))
        engine.loaded = True
        engine._detect = MagicMock(return_value=[((0, 0, 8, 8), .5)])
        result = engine.analyze(Image.new("RGB", (12, 12)), InferenceSettings(machine="DS1000", use_calibration=True, classification_threshold=60.))
        self.assertTrue(result.cells[0].candidate)

    def test_local_fit_requires_both_labels_and_leaves_features_unchanged(self):
        x = np.array([[-2.], [-1.], [1.], [2.]], dtype=np.float32)
        original = x.copy()
        head = fit_head(x, np.array([False, False, True, True]))
        values = score_features(x, head)
        self.assertLess(values[0], 50.)
        self.assertGreater(values[-1], 50.)
        np.testing.assert_array_equal(x, original)
        with self.assertRaisesRegex(ValueError, "Both"):
            fit_head(x, np.array([True] * 4))
