"""Result export contracts, file protection, and preview/inference separation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from app.domain import CellResult, FrameResult, InferenceSettings
from app.reporting import export_bundle, render_result


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "入力.png"
        self.image = Image.new("RGB", (100, 80), (50, 60, 70))
        self.image.save(self.source)
        self.settings = InferenceSettings(detection_threshold=.29, classification_threshold=.44)
        self.result = FrameResult([
            CellResult(1, (10, 10, 30, 30), 0.81234567890123, .440000000000123, True),
            CellResult(2, (50, 50, 70, 70), 0.9, .44, False),
        ], skipped=1, elapsed_seconds=0.123456789, device="cpu")

    def export(self, **overrides):
        parameters = dict(output_dir=self.root / "結果", source=self.source, frame_index=3,
                          image=self.image, result=self.result, settings=self.settings,
                          model_metadata={"detector": {"sha256": "test-hash", "resolution": 784}})
        parameters.update(overrides)
        return export_bundle(**parameters)

    def test_export_retains_precision_and_reproduction_metadata(self):
        source_before = self.source.read_bytes()
        output = self.export()
        self.assertTrue(output["results_csv"].read_bytes().startswith(b"\xef\xbb\xbf"))
        with output["results_csv"].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["classification_score_0_to_1"], "0.440000000000123")
        self.assertNotIn("class1_score", rows[0])
        self.assertEqual(rows[1]["candidate"], "false")
        manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(source_before).hexdigest())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["classification"]["score_unit"], "0_to_1")
        self.assertEqual(manifest["classification"]["threshold_unit"], "0_to_1")
        self.assertEqual(manifest["settings"]["classification_threshold"], .44)
        self.assertEqual(manifest["displayed_frame"]["detection_count"], 3)
        self.assertEqual(manifest["displayed_frame"]["candidate_count"], 1)
        self.assertEqual(manifest["classification"]["class1_label_meaning"], "unverified")
        self.assertFalse(manifest["evaluation"]["ground_truth_compared"])
        self.assertIn("python", manifest["environment"])
        self.assertEqual(source_before, self.source.read_bytes())

    def test_null_values_are_blank_csv_and_null_json(self):
        detector_result = FrameResult([CellResult(1, (10, 10, 30, 30), 0.9, None, False)])
        output = self.export(result=detector_result, settings=InferenceSettings(mode="detection_only"))
        with output["results_csv"].open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["classification_score_0_to_1"], "")
        manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
        self.assertIsNone(manifest["displayed_frame"]["cells"][0]["score"])

    def test_video_history_records_zero_detection_frames(self):
        history = [(0, FrameResult([], elapsed_seconds=0.2)), (3, self.result)]
        output = self.export(history=history)
        manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
        self.assertEqual([record["frame_index"] for record in manifest["analyzed_frames"]], [0, 3])
        self.assertEqual(manifest["analyzed_frames"][0]["detection_count"], 0)
        with output["video_results_csv"].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["frame_index"] for row in rows], ["3", "3"])
        self.assertEqual(history[0][0], 0)

    def test_preview_and_partial_video_job_settings_are_preserved(self):
        display = {"show_all": False, "show_labels": True, "grayscale": True, "brightness": 20}
        video_job = {"is_video": True, "fps": 29.97, "frame_count": 1200,
                     "stride": 10, "start_frame": 3, "stop_reason": "cancelled"}
        output = self.export(display_settings=display, input_metadata=video_job)
        manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["display_settings"], display)
        self.assertEqual(manifest["input_metadata"], video_job)
        self.assertEqual(manifest["input_metadata"]["stop_reason"], "cancelled")
        self.assertIn("opencv-python-headless", manifest["environment"]["packages"])
        self.assertFalse(display["show_all"])

    def test_existing_result_or_unrelated_file_is_never_overwritten(self):
        target = self.root / "結果"
        target.mkdir()
        existing = target / "keep.txt"
        existing.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(existing.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(target.iterdir()), [existing])

    def test_empty_directory_is_accepted_and_failure_leaves_no_partial_bundle(self):
        target = self.root / "結果"
        target.mkdir()
        with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(list(self.root.glob(".nikon-export-*")), [])
        self.assertEqual(len(self.export()), 3)

    def test_invalid_candidate_is_rejected_before_saving(self):
        invalid = FrameResult([CellResult(1, (10, 10, 30, 30), 0.9, .44, True)])
        with self.assertRaisesRegex(ValueError, "閾値超え判定"):
            self.export(result=invalid)
        self.assertFalse((self.root / "結果").exists())

    def test_legacy_percent_and_nonfinite_scores_are_not_exported_as_probabilities(self):
        for score in (60., -0.1, 1.01, float("nan"), float("inf")):
            result = FrameResult([CellResult(1, (10, 10, 30, 30), .9, score, True)])
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "分類スコア"):
                self.export(result=result)
        self.assertFalse((self.root / "結果").exists())

    def test_metadata_serialization_failure_creates_no_output(self):
        with self.assertRaises(TypeError):
            self.export(model_metadata={"unserializable": object()})
        self.assertFalse((self.root / "結果").exists())

    def test_candidate_only_preview_and_display_options_preserve_input_and_scores(self):
        original_pixels = self.image.tobytes()
        preview = render_result(self.image, self.result, show_all=False, show_labels=False)
        self.assertEqual(preview.getpixel((10, 10)), (239, 68, 68))
        self.assertEqual(preview.getpixel((50, 50)), (50, 60, 70))
        all_cells = render_result(self.image, self.result, show_labels=False, grayscale=True, brightness=20)
        self.assertEqual(all_cells.getpixel((50, 50)), (138, 164, 188))
        self.assertEqual(original_pixels, self.image.tobytes())
        self.assertEqual(self.result.cells[0].score, .440000000000123)

    def test_detection_only_boxes_remain_visible_without_candidate_filter(self):
        result = FrameResult([CellResult(1, (10, 10, 30, 30), 0.9, None, False)])
        preview = render_result(self.image, result, show_all=False, show_labels=False)
        self.assertEqual(preview.getpixel((10, 10)), (138, 164, 188))


if __name__ == "__main__":
    unittest.main()
