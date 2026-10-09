"""Real contract tests; model predictions are isolated from the test fixtures."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app.domain import CellResult, FrameResult, InferenceSettings, classification_score_from_percent
from app.inference import (
    CLASSIFIER_SHA256, DETECTOR_SHA256, Engine, ModelError, build_classifier,
    choose_device, crop_and_resize, file_sha256, strip_state_dict_prefix, verify_model,
)


class ModelContractTests(unittest.TestCase):
    def test_only_expected_model_contents_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"not a checkpoint")
            digest = file_sha256(path)
            self.assertEqual(verify_model(path, digest), digest)
            with self.assertRaisesRegex(ModelError, "内容が異なります"):
                verify_model(path, CLASSIFIER_SHA256)
            with self.assertRaisesRegex(ModelError, "ありません"):
                verify_model(path.with_name("missing.pth"), digest)

    def test_model_load_refuses_identity_before_checkpoint_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.pt"
            path.write_bytes(b"invalid")
            engine = Engine(path, path)
            with patch("torch.load") as load:
                with self.assertRaises(ModelError):
                    engine.load()
                load.assert_not_called()
            self.assertFalse(engine.loaded)

    def test_common_prefix_only_is_removed(self):
        value = object()
        self.assertEqual(strip_state_dict_prefix({"backbone.fc.weight": value}), {"fc.weight": value})
        mixed = {"backbone.fc.weight": value, "layer1.weight": value}
        self.assertEqual(strip_state_dict_prefix(mixed), mixed)

    def test_classifier_reconstructs_head_and_loads_strictly(self):
        state = {"backbone.fc.1.weight": SimpleNamespace(shape=(128, 2048)),
                 "backbone.fc.5.weight": SimpleNamespace(shape=(2, 128))}
        model = MagicMock()
        with patch("torchvision.models.resnet101", return_value=model) as constructor:
            result = build_classifier(state)
        constructor.assert_called_once_with(weights=None)
        self.assertIs(result, model)
        self.assertEqual(model.fc[1].in_features, 2048)
        self.assertEqual(model.fc[1].out_features, 128)
        self.assertEqual(model.fc[5].out_features, 2)
        self.assertTrue(model.load_state_dict.call_args.kwargs["strict"])
        model.eval.assert_called_once()
        with self.assertRaises(ModelError):
            build_classifier({"fc.1.weight": SimpleNamespace(shape=(128, 2048)),
                              "fc.5.weight": SimpleNamespace(shape=(3, 128))})

    def test_load_restores_detector_head_resolution_and_uses_safe_reads(self):
        detector_state = {"class_embed.bias": SimpleNamespace(shape=(2,))}
        checkpoint = {"model": detector_state, "args": SimpleNamespace(resolution=784, num_classes=2)}
        classifier = MagicMock()
        detector = MagicMock()
        engine = Engine(Path("det.pt"), Path("class.pth"), "cpu")
        with patch("app.inference.verify_model", side_effect=[DETECTOR_SHA256, CLASSIFIER_SHA256]), \
                patch("torch.load", side_effect=[checkpoint, {}]) as safe_load, \
                patch("rfdetr.RFDETRBase", return_value=detector) as constructor, \
                patch("app.inference.build_classifier", return_value=classifier), \
                patch("app.inference.choose_device", return_value="cpu"):
            engine.load()
        self.assertTrue(engine.loaded)
        self.assertEqual(constructor.call_args.kwargs["num_classes"], 1)
        self.assertEqual(constructor.call_args.kwargs["resolution"], 784)
        self.assertEqual(constructor.call_args.kwargs["pretrain_weights"], str(engine.detector_path))
        self.assertTrue(all(call.kwargs["weights_only"] for call in safe_load.call_args_list))
        self.assertFalse(engine.get_model_metadata()["classifier"]["class_mapping_verified"])
        engine.close()
        self.assertFalse(engine.loaded)

    def test_force_cpu_overrides_gpu_request(self):
        with patch.dict("os.environ", {"FORCE_CPU": "true"}):
            self.assertEqual(choose_device("cuda"), "cpu")


class InferenceBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 80), (120, 130, 140))
        self.engine = Engine(Path("det.pt"), Path("class.pth"))
        self.engine.loaded = True

    def test_threshold_equality_is_negative_and_original_precision_is_used(self):
        self.engine._detect = MagicMock(return_value=[((10, 10, 20, 20), 0.29), ((20, 20, 30, 30), 0.5)])
        self.engine._classify = MagicMock(side_effect=[.44, .440000000001])
        result = self.engine.analyze(self.image, InferenceSettings(detection_threshold=.29, classification_threshold=.44))
        self.assertEqual([cell.candidate for cell in result.cells], [False, True])
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.detection_count, 2)

    def test_invalid_crop_is_skipped_without_a_fake_classification(self):
        self.engine._detect = MagicMock(return_value=[((1, 1, 3, 3), 0.9), ((10, 10, 20, 20), 0.8)])
        self.engine._classify = MagicMock(return_value=.60)
        result = self.engine.analyze(self.image, InferenceSettings())
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.detection_count, 2)
        self.assertEqual(result.cells[0].id, 2)
        self.engine._classify.assert_called_once()

    def test_detection_only_does_not_invent_classification(self):
        self.engine._detect = MagicMock(return_value=[((10, 10, 20, 20), 0.8)])
        self.engine._classify = MagicMock()
        result = self.engine.analyze(self.image, InferenceSettings(mode="detection_only"))
        self.assertIsNone(result.cells[0].score)
        self.assertFalse(result.cells[0].candidate)
        self.engine._classify.assert_not_called()

    def test_classification_only_uses_whole_image_and_no_detection_confidence(self):
        self.engine._detect = MagicMock()
        self.engine._classify = MagicMock(return_value=.50)
        result = self.engine.analyze(self.image, InferenceSettings(mode="classification_only"))
        self.assertEqual(result.cells[0].box, (0, 0, 100, 80))
        self.assertIsNone(result.cells[0].confidence)
        self.assertEqual(self.engine._classify.call_args.args[0].size, (224, 224))
        self.engine._detect.assert_not_called()

    def test_unloaded_and_failed_models_do_not_return_successful_empty_results(self):
        self.engine.loaded = False
        with self.assertRaises(ModelError):
            self.engine.analyze(self.image, InferenceSettings())
        self.engine.loaded = True
        self.engine._detect = MagicMock(side_effect=RuntimeError("backend failed"))
        with self.assertRaisesRegex(ModelError, "backend failed"):
            self.engine.analyze(self.image, InferenceSettings())

    def test_probability_scale_and_candidate_boundary(self):
        import math
        import torch

        self.engine._transform = lambda image: torch.zeros((3, 224, 224))
        for probability in (.30, .44, .60):
            self.engine._classifier = lambda tensor: torch.tensor(
                [[math.log(1 - probability), math.log(probability)]], dtype=torch.float64)
            with self.subTest(probability=probability):
                self.assertAlmostEqual(self.engine._classify(self.image), probability, places=14)
        self.engine._detect = MagicMock(return_value=[((10, 10, 20, 20), .5)] * 3)
        self.engine._classify = MagicMock(side_effect=[.30, .44, .60])
        result = self.engine.analyze(self.image, InferenceSettings(classification_threshold=.44))
        self.assertEqual([cell.candidate for cell in result.cells], [False, False, True])

    def test_nonfinite_model_score_is_rejected(self):
        import torch

        self.engine._transform = lambda image: torch.zeros((3, 224, 224))
        self.engine._classifier = lambda tensor: torch.tensor([[float("nan"), 0.]])
        with self.assertRaisesRegex(ModelError, "不正なスコア"):
            self.engine._classify(self.image)

    def test_explicit_percent_conversion_and_unit_interval_limits(self):
        self.assertEqual(classification_score_from_percent(60.), .60)
        # Small legacy values still mean percentages, never inferred probabilities.
        self.assertEqual(classification_score_from_percent(.5), .005)
        for value in (0., 1.):
            self.assertEqual(InferenceSettings(classification_threshold=value).classification_threshold, value)
        for value in (-.01, 1.01, 44., float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                InferenceSettings(classification_threshold=value)
        for value in (-1., 101., float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                classification_score_from_percent(value)

    def test_crop_uses_exclusive_image_edges_and_clips_outside_boxes(self):
        crop, box = crop_and_resize(self.image, (110, 90, -10, -10))
        self.assertEqual(box, (0, 0, 100, 80))
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(crop_and_resize(self.image, (-20, -20, 4, 4))[1], (0, 0, 4, 4))
        self.assertEqual(crop_and_resize(self.image, (96, 76, 120, 100))[1], (96, 76, 100, 80))
        self.assertEqual(crop_and_resize(self.image, (0, 0, 3, 20)), (None, None))

    def test_crop_accepts_exact_four_pixel_edges_and_rejects_smaller(self):
        crop, box = crop_and_resize(self.image, (0, 0, 4, 4))
        self.assertEqual(box, (0, 0, 4, 4))
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(crop_and_resize(self.image, (0, 0, 4, 3)), (None, None))

    def test_crop_preserves_all_four_image_corners(self):
        image = Image.new("RGB", (12, 10))
        for box in ((0, 0, 4, 4), (8, 0, 12, 4), (0, 6, 4, 10), (8, 6, 12, 10)):
            crop, clipped = crop_and_resize(image, box)
            self.assertEqual(clipped, box)
            self.assertEqual(crop.size, (224, 224))

    def test_detection_rescales_coordinates_filters_equality_and_sanitizes_nan(self):
        import numpy as np

        detected = SimpleNamespace(xyxy=np.array([[64, 32, 128, 64], [10, 10, 20, 20]]),
                                   confidence=np.array([0.29, float("nan")]))
        self.engine._detector = MagicMock()
        self.engine._detector.predict.return_value = detected
        result = self.engine._detect(Image.new("RGB", (1280, 720)), 0.29)
        self.assertEqual(result, [((128, 64, 256, 128), 0.29)])
        self.assertEqual(self.engine._detector.predict.call_args.args[0].size, (640, 360))
        self.assertEqual(self.engine._detector.predict.call_args.kwargs["threshold"], 0.0)

    def test_detection_restores_coordinates_using_rounded_non_square_dimensions(self):
        import numpy as np

        # 1001x667 is resized to 640x426: its two realized axis scales differ.
        self.engine._detector = MagicMock()
        self.engine._detector.predict.return_value = SimpleNamespace(
            xyxy=np.array([[639, 425, 640, 426], [100, 100, 200, 200]]),
            confidence=np.array([0.9, 0.9]),
        )
        result = self.engine._detect(Image.new("RGB", (1001, 667)), 0.5)
        self.assertEqual(self.engine._detector.predict.call_args.args[0].size, (640, 426))
        self.assertEqual(result, [((999, 665, 1001, 667), 0.9),
                                 ((156, 156, 312, 313), 0.9)])

    def test_settings_reject_unknown_modes_and_nonfinite_thresholds(self):
        for values in ({"mode": "unknown"}, {"machine": "unknown"},
                       {"classification_threshold": float("nan")}, {"detection_threshold": 1.01}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                InferenceSettings(**values)


if __name__ == "__main__":
    unittest.main()
