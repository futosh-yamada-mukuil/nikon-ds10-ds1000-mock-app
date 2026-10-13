"""Synthetic pixels and fake predictors only; no supplied checkpoint is opened."""
import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app.inference import Engine, IMAGENET_MEAN, IMAGENET_STD, crop_and_resize
from app.tiled_detection import detect_tiled, suppress_tile_duplicates, tile_boxes
from scripts import compare_detection as detection
from scripts import compare_preprocessing as preprocessing
from scripts.measurement_io import write_text_outputs


class GeometryAndTilesTests(unittest.TestCase):
    def test_portrait_landscape_and_no_resize_restore_exclusive_edges(self):
        import numpy as np

        for size, resized in (((1001, 667), (640, 426)), ((667, 1001), (426, 640)),
                              ((640, 319), (640, 319)), ((99, 80), (99, 80))):
            with self.subTest(size=size):
                engine = Engine(Path("synthetic-detector"), Path("synthetic-classifier"))
                engine._detector = MagicMock()
                engine._detector.predict.return_value = SimpleNamespace(
                    xyxy=np.array([[0, 0, *resized]], dtype=float), confidence=np.array([0.5]))
                with patch.dict("sys.modules", {"torch": SimpleNamespace(no_grad=nullcontext)}):
                    result = engine._detect(Image.new("RGB", size), 0.5)
                self.assertEqual(result, [((0, 0, *size), 0.5)])
                self.assertEqual(engine._detector.predict.call_args.args[0].size, resized)

    def test_four_corner_pixels_and_outside_boxes(self):
        image = Image.new("RGB", (12, 10))
        boxes = ((0, 0, 4, 4), (8, 0, 12, 4), (0, 6, 4, 10), (8, 6, 12, 10))
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 120, 140))
        for box, color in zip(boxes, colors):
            image.paste(color, box)
        for box, color in zip(boxes, colors):
            crop, clipped = crop_and_resize(image, box)
            self.assertEqual(clipped, box)
            self.assertEqual(set(crop.getdata()), {color})
        for box in ((12, 0, 20, 10), (-20, 0, -1, 10), (0, 10, 12, 20), (0, -10, 12, -1),
                    (9, 6, 15, 20), (8, 7, 20, 20)):
            self.assertEqual(crop_and_resize(image, box), (None, None))

    def test_tiles_cover_far_edges_once_and_restore_global_coordinates(self):
        tiles = list(tile_boxes(15, 11, 8, 0.25))
        self.assertEqual(tiles[-1], (7, 3, 15, 11))
        self.assertEqual(len(tiles), len(set(tiles)))
        covered = {(x, y) for left, top, right, bottom in tiles
                   for x in range(left, right) for y in range(top, bottom)}
        self.assertEqual(len(covered), 15 * 11)
        engine = MagicMock()
        engine._detect.side_effect = ([[]] * (len(tiles) - 1) + [[((4, 4, 9, 10), .9)]])
        result = detect_tiled(engine, Image.new("RGB", (15, 11)), tile_size=8)
        self.assertEqual(result, [((11, 7, 15, 11), .9)])
        self.assertTrue(all(call.args[0].size == (8, 8) for call in engine._detect.call_args_list))

    def test_small_image_and_outside_tile_detection(self):
        self.assertEqual(list(tile_boxes(5, 4)), [(0, 0, 5, 4)])
        engine = MagicMock()
        engine._detect.return_value = [((-2, -3, 5, 4), .5), ((-9, 0, -1, 3), .9)]
        self.assertEqual(detect_tiled(engine, Image.new("RGB", (5, 4))), [((0, 0, 5, 4), .5)])

    def test_nms_only_suppresses_other_tiles_and_keeps_iou_equality(self):
        a = ((0, 0, 6, 4), .7, 0)
        b = ((2, 0, 8, 4), .8, 1)  # IoU exactly .5
        c = ((0, 0, 6, 4), .9, 2)
        self.assertEqual(suppress_tile_duplicates([a, b], .5), [b, a])
        self.assertEqual(suppress_tile_duplicates([a, b, c], .5), [c, b])
        same_tile = ((0, 0, 6, 4), .9, 0)
        self.assertEqual(suppress_tile_duplicates([a, same_tile], .5), [same_tile, a])
        self.assertEqual(suppress_tile_duplicates([a, (a[0], .7, 1)], .5), [a])

    def test_invalid_tiling_settings_fail_before_prediction(self):
        engine = MagicMock()
        for kwargs in ({"tile_size": 3}, {"overlap": 1}, {"overlap": float("nan")},
                       {"nms_iou": -1}, {"threshold": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_tiled(engine, Image.new("RGB", (10, 10)), **kwargs)
        engine._detect.assert_not_called()


class MeasurementCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "synthetic.png"
        Image.new("RGB", (12, 10), (180, 40, 90)).save(self.image)
        self.csv = self.root / "rois.csv"
        self.csv.write_text("image,cell_id,x1,y1,x2,y2\nsynthetic.png,c1,-2,0,12,10\n")
        self.model = self.root / "synthetic-model-not-loaded.pth"
        self.model.write_bytes(b"synthetic fixture, never loaded")
        self.out_csv, self.out_json = self.root / "result.csv", self.root / "result.json"
        self.args = ["--input-csv", str(self.csv), "--image-root", str(self.root),
                     "--classifier", str(self.model), "--jpeg-quality", "83",
                     "--output-csv", str(self.out_csv), "--output-json", str(self.out_json)]

    def test_invalid_later_roi_is_rejected_before_model_load(self):
        for invalid in ("synthetic.png,c2,0,0,3,10", "missing.png,c2,0,0,4,4",
                        "./synthetic.png,c1,0,0,4,4", "synthetic.png,c2,NaN,0,4,4"):
            with self.subTest(invalid=invalid):
                self.csv.write_text("image,cell_id,x1,y1,x2,y2\nsynthetic.png,c1,0,0,4,4\n" + invalid + "\n")
                with patch.object(preprocessing, "load_classifier_for_comparison") as load, \
                        redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    preprocessing.main(self.args)
                load.assert_not_called()
                self.assertFalse(self.out_csv.exists())

    def test_cli_requires_quality_and_finite_thresholds_before_load(self):
        cases = [self.args[:6] + self.args[8:], self.args + ["--classification-threshold", "nan"],
                 self.args + ["--jpeg-quality", "96"]]
        for args in cases:
            with patch.object(preprocessing, "load_classifier_for_comparison") as load, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                preprocessing.main(args)
            load.assert_not_called()

    def test_cli_help_does_not_load_models(self):
        with patch.object(preprocessing, "load_classifier_for_comparison") as load, \
                redirect_stdout(StringIO()) as output, self.assertRaises(SystemExit) as stopped:
            preprocessing.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("50%", output.getvalue())
        load.assert_not_called()

    def test_cli_uses_same_engine_records_hashes_coords_and_strict_equality(self):
        def fake_load(engine):
            engine._hashes["classifier"] = "a" * 64
            engine.loaded = True
            engine._classify = MagicMock(side_effect=[.5, .500000001])

        original = self.image.read_bytes()
        with patch.object(preprocessing, "load_classifier_for_comparison", side_effect=fake_load):
            self.assertEqual(preprocessing.main(self.args), 0)
        payload = json.loads(self.out_json.read_text())
        row = payload["results"][0]
        self.assertEqual(row["original_x1"], -2)
        self.assertEqual(row["x1"], 0)
        self.assertEqual(row["x2"], 12)
        self.assertFalse(row["direct_candidate"])
        self.assertTrue(row["comparison_candidate"])
        self.assertEqual(payload["metadata"]["classifier_sha256"], "a" * 64)
        self.assertEqual(payload["metadata"]["device_requested"], "auto")
        self.assertEqual(payload["metadata"]["training_time_jpeg_quality"], "unverified")
        self.assertEqual(self.image.read_bytes(), original)

    def test_classifier_loader_forwards_auto_and_reuses_normalization(self):
        engine = Engine(Path("not-read-detector"), self.model, "auto")
        torch = SimpleNamespace(load=MagicMock(return_value={}))
        transforms = MagicMock()
        classifier = MagicMock()
        with patch.dict("sys.modules", {"torch": torch,
                        "torchvision.transforms": transforms,
                        "torchvision": SimpleNamespace(transforms=transforms)}), \
                patch("app.inference.choose_device", return_value="mps") as choose, \
                patch("app.inference.verify_model", return_value="a" * 64) as verify, \
                patch("app.inference.build_classifier", return_value=classifier):
            preprocessing.load_classifier_for_comparison(engine)
        choose.assert_called_once_with("auto")
        verify.assert_called_once()
        torch.load.assert_called_once_with(engine.classifier_path, map_location="cpu", weights_only=True)
        transforms.Normalize.assert_called_once_with(IMAGENET_MEAN, IMAGENET_STD)
        classifier.to.assert_called_once_with("mps")
        self.assertEqual(engine.device, "mps")

    def test_outputs_are_exclusive_and_roll_back_new_file_on_error(self):
        self.out_json.write_text("keep")
        with self.assertRaises(FileExistsError):
            write_text_outputs({self.out_csv: "new", self.out_json: "replace"})
        self.assertFalse(self.out_csv.exists())
        self.assertEqual(self.out_json.read_text(), "keep")
        self.out_json.unlink()
        real_open = Path.open

        def fail_second(path, *args, **kwargs):
            if path == self.out_json:
                raise OSError("synthetic disk error")
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_second), self.assertRaises(OSError):
            write_text_outputs({self.out_csv: "new", self.out_json: "new"})
        self.assertFalse(self.out_csv.exists())

    def test_detection_cli_saves_empty_predictions_and_timing_without_accuracy_claim(self):
        engine = Engine(self.model, self.model)
        engine.load = MagicMock()
        engine._detect = MagicMock(return_value=[])
        with patch.object(detection, "Engine", return_value=engine):
            result = detection.main(["--image", str(self.image), "--detector", str(self.model),
                                     "--classifier", str(self.model), "--output-csv", str(self.out_csv),
                                     "--output-json", str(self.out_json)])
        self.assertEqual(result, 0)
        payload = json.loads(self.out_json.read_text())
        self.assertEqual(payload["detections"], [])
        self.assertEqual(payload["metadata"]["runs"]["tiled"]["detection_count"], 0)
        self.assertIsNone(payload["metadata"]["accuracy_metrics"])
        self.assertFalse(engine.loaded)
        with self.out_csv.open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])


if __name__ == "__main__":
    unittest.main()
