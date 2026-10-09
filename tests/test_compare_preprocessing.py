import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from PIL import Image

from scripts.compare_preprocessing import compare_roi, read_rois, run, write_outputs


class ComparePreprocessingTests(unittest.TestCase):
    def test_csv_validation_rejects_missing_columns_duplicate_ids_and_bad_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rois.csv"
            path.write_text("image,cell_id,x1,y1,x2\na.jpg,c1,0,0,4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV列"):
                read_rois(path)
            path.write_text("image,cell_id,x1,y1,x2,y2\na.jpg,c1,0,0,4,4\na.jpg,c1,1,1,5,5\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重複"):
                read_rois(path)
            path.write_text("image,cell_id,x1,y1,x2,y2\na.jpg,c1,a,0,4,4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "整数"):
                read_rois(path)

    def test_compare_roi_runs_both_real_preprocessing_paths_on_same_clipped_crop(self):
        image = Image.new("RGB", (12, 10), (180, 40, 90))
        engine = MagicMock()
        engine._classify.side_effect = [.10, .20]
        direct, comparison, clipped = compare_roi(engine, image, (-2, 0, 5, 6), 83)
        self.assertEqual((direct, comparison, clipped), (.10, .20, (0, 0, 5, 6)))
        calls = [call.args[0] for call in engine._classify.call_args_list]
        self.assertEqual([im.size for im in calls], [(224, 224), (224, 224)])
        self.assertNotEqual(calls[0].tobytes(), calls[1].tobytes())

    def test_compare_roi_rejects_clipped_roi_smaller_than_four_pixels(self):
        with self.assertRaisesRegex(ValueError, "4px"):
            compare_roi(MagicMock(), Image.new("RGB", (12, 10)), (0, 0, 3, 8), 80)

    def test_jpeg_path_enlarges_original_roi_before_the_224_resize(self):
        from io import BytesIO

        image = Image.new("RGB", (19, 15))
        image.putdata([(x * 31 % 256, y * 47 % 256, (x + y) * 19 % 256)
                       for y in range(15) for x in range(19)])
        engine = MagicMock()
        engine._classify.side_effect = [.1, .2]
        compare_roi(engine, image, (1, 2, 17, 14), 83)
        buffer = BytesIO()
        image.crop((1, 2, 17, 14)).resize((640, 640), Image.Resampling.BILINEAR).save(buffer, "JPEG", quality=83)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            expected = decoded.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        self.assertEqual(engine._classify.call_args_list[1].args[0].tobytes(), expected.tobytes())

    def test_run_writes_same_image_roi_scores_and_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (12, 10), (50, 100, 150)).save(image_root / "synthetic.png")
            csv_path = root / "roi.csv"
            csv_path.write_text("image,cell_id,x1,y1,x2,y2\nsynthetic.png,cell-A,-2,0,8,10\n",
                                encoding="utf-8")
            output_csv, output_json = root / "out.csv", root / "out.json"
            args = argparse.Namespace(input_csv=csv_path, image_root=image_root, classifier=root / "fake.pth",
                                      device="cpu", jpeg_quality=82, classification_threshold=0.5,
                                      detection_threshold=0.5)
            engine = MagicMock()
            engine.device = "cpu"
            engine._classify.side_effect = [.13, .31]
            args.output_csv, args.output_json = output_csv, output_json
            rows, metadata = run(args, engine)
            write_outputs(output_csv, output_json, rows, metadata)
            with output_csv.open(newline="", encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))
            payload = json.loads(output_json.read_text())
            self.assertEqual(rows[0]["cell_id"], "cell-A")
            self.assertAlmostEqual(rows[0]["score_difference_comparison_minus_direct"], .18)
            self.assertEqual(written[0]["cell_id"], "cell-A")
            self.assertAlmostEqual(float(written[0]["score_difference_comparison_minus_direct"]), .18)
            self.assertEqual(payload["metadata"]["rows_written"], 1)
            self.assertEqual(payload["results"], rows)
            self.assertEqual(metadata["jpeg_quality"], 82)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["classification_score_unit"], "0_to_1")
            self.assertEqual(written[0]["direct_224_classification_score_0_to_1"], "0.13")
            self.assertEqual(metadata["training_time_jpeg_quality"], "unverified")
            self.assertTrue(metadata["same_roi_for_both_paths"])

    def test_run_rejects_image_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "roi.csv"
            csv_path.write_text("image,cell_id,x1,y1,x2,y2\n../outside.png,c1,0,0,4,4\n", encoding="utf-8")
            args = argparse.Namespace(input_csv=csv_path, image_root=root, classifier=root / "fake.pth",
                                      device="cpu", jpeg_quality=80, classification_threshold=0.5,
                                      detection_threshold=0.5)
            with self.assertRaisesRegex(ValueError, "画像パス"):
                run(args, MagicMock())


if __name__ == "__main__":
    unittest.main()
