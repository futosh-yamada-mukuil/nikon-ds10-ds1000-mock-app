from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from scripts.evaluate_annotations import counts, discover, load_cached_predictions, overlap_edges, truth_boxes


class AnnotationEvaluationTests(unittest.TestCase):
    def test_cache_units_are_explicit_and_legacy_conversion_never_changes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir()
            path = root / "cache/sample.json"
            legacy = {"variants": {"legacy640": [{"score": 60.}, {"score": .5}]}}
            path.write_text(json.dumps(legacy))
            original = path.read_bytes()
            (root / "manifest.json").write_text(json.dumps({"batch_score_tolerance_0_to_100": .001}))
            rows = load_cached_predictions(root)
            self.assertEqual([c["score"] for c in rows[0]["variants"]["legacy640"]], [.60, .005])
            self.assertEqual(path.read_bytes(), original)
            (root / "manifest.json").write_text(json.dumps({"schema_version": 2, "classification_score_unit": "0_to_1"}))
            path.write_text(json.dumps(rows[0]))
            normalized = path.read_bytes()
            for _ in range(2):
                self.assertEqual(load_cached_predictions(root), rows)
            self.assertEqual(path.read_bytes(), normalized)

    def test_unknown_mixed_and_invalid_cache_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir()
            manifest = root / "manifest.json"
            path = root / "cache/sample.json"
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_cached_predictions(root)
            manifest.write_text(json.dumps({"batch_score_tolerance_0_to_100": .001}))
            path.write_text(json.dumps({"schema_version": 2, "classification_score_unit": "0_to_1", "variants": {}}))
            with self.assertRaisesRegex(ValueError, "double conversion"):
                load_cached_predictions(root)
            manifest.write_text(json.dumps({"schema_version": 2, "classification_score_unit": "0_to_1"}))
            for score in (60., float("nan"), float("inf"), -.1):
                path.write_text(json.dumps({"schema_version": 2, "classification_score_unit": "0_to_1",
                                           "variants": {"legacy640": [{"score": score}]}}))
                with self.subTest(score=score), self.assertRaisesRegex(ValueError, "Invalid"):
                    load_cached_predictions(root)

    def test_red_box_edges_are_exclusive_and_green_cells_are_not_labels(self):
        image = Image.new("RGB", (50, 40), (0, 100, 0))
        ImageDraw.Draw(image).rectangle((10, 8, 30, 28), outline="red", width=2)
        self.assertEqual(truth_boxes(image), [[10, 8, 31, 29]])
        self.assertEqual(truth_boxes(Image.new("RGB", (50, 40), (0, 255, 0))), [])

    def test_incomplete_red_annotation_is_not_silently_counted(self):
        image = Image.new("RGB", (50, 40))
        draw = ImageDraw.Draw(image)
        draw.line((10, 8, 30, 8), fill="red", width=2)
        draw.line((30, 8, 30, 28), fill="red", width=2)
        draw.line((30, 28, 10, 28), fill="red", width=2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            truth_boxes(image)

    def test_maximum_one_to_one_matching_and_threshold_boundaries(self):
        cells = [{"box": [0, 0, 20, 20], "confidence": .5, "score": 1.},
                 {"box": [0, 0, 20, 20], "confidence": .5, "score": 1.}]
        # Greedy matching would lose the second match; augmenting paths don't.
        self.assertEqual(counts(cells, {0: [0, 1], 1: [0]}, 2, .5, .5)["tp"], 2)
        result = counts(cells, overlap_edges(cells, [[0, 0, 20, 20]]), 1, .5, .5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(counts(cells, {0: [0], 1: [0]}, 1, .5, 1.)["candidates"], 0)

    def test_missing_or_ambiguous_truth_is_excluded_and_split_precedes_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for machine in ("02_DS10", "01_DS1000"):
                for folder in ("学習用", "正解"):
                    (root / machine / folder).mkdir(parents=True)
            for i in range(1, 5):
                Image.new("RGB", (40, 40)).save(root / "02_DS10/学習用" / f"1-3_{i:03d}_RGB_New.jpg")
                if i < 4:
                    Image.new("RGB", (40, 40)).save(root / "02_DS10/正解" / f"1-3_{i:03d}_RGB_40XFITC2.png")
            pairs, excluded = discover(root)
            self.assertEqual([p["split"] for p in pairs], ["calibration", "holdout", "holdout"])
            self.assertEqual(excluded[0]["reason"], "no_ground_truth")
            self.assertTrue(all(p["training_membership"] == "unknown" for p in pairs))


if __name__ == "__main__":
    unittest.main()
