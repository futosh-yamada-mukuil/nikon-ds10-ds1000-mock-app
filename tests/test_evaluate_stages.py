"""Hand-calculated synthetic measurements; never real image/model accuracy."""
from contextlib import redirect_stderr
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_stages import main, read_measurements, read_metadata, roc_auc, summarize


def group(machine="DS10", stage="L0", verified=True, index=1):
    return {"machine": machine, "stage": stage, "model_sha256": {"classifier": "a" * 64},
            "input_path": "synthetic/direct224", "positive_class_mapping_verified": verified,
            "positive_class_index": index}


class MetricTests(unittest.TestCase):
    def test_confusion_matrix_and_all_metrics_against_hand_calculation(self):
        values = [(1, .9), (1, .8), (1, .2), (0, .7), (0, .4), (0, .1)]
        result = summarize(values, group())
        self.assertEqual([result[key] for key in ("TP", "FP", "FN", "TN")], [2, 1, 1, 2])
        for metric in ("Precision", "Recall", "F1", "Accuracy", "Balanced Accuracy"):
            self.assertAlmostEqual(result[metric], 2 / 3)
        self.assertAlmostEqual(result["ROC AUC"], 7 / 9)
        self.assertEqual(result["n_evaluated"], 6)

    def test_perfect_reversed_and_tied_auc(self):
        self.assertEqual(roc_auc([(1, .9), (0, .1)]), 1.)
        self.assertEqual(roc_auc([(1, .1), (0, .9)]), 0.)
        self.assertEqual(roc_auc([(1, .5), (0, .5)]), .5)
        self.assertAlmostEqual(roc_auc([(1, .8), (1, .5), (0, .5), (0, .2)]), .875)

    def test_strict_threshold_tie_and_unrounded_value(self):
        result = summarize([(1, .5), (1, .50000000001), (0, .1)], group())
        self.assertEqual((result["TP"], result["FN"], result["TN"]), (1, 1, 1))

    def test_missing_truth_is_excluded_not_negative(self):
        result = summarize([(None, .99), (0, .1), (1, .9)], group())
        self.assertEqual((result["n_total"], result["n_evaluated"], result["excluded_missing_truth"]), (3, 2, 1))
        self.assertEqual(result["FP"], 0)
        self.assertEqual(result["Accuracy"], 1.)

    def test_unconfirmed_mapping_suppresses_metrics_and_confusion_counts(self):
        result = summarize([(0, .1), (1, .9), (None, .4)], group(verified=False))
        for name in ("TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "Accuracy", "ROC AUC"):
            self.assertIsNone(result[name])
        self.assertEqual(result["n_labeled"], 2)
        self.assertEqual(result["n_evaluated"], 0)
        self.assertIn("未確認", result["uncomputed_reasons"])

    def test_single_class_zero_denominators_and_no_labels(self):
        result = summarize([(0, .1), (0, .2)], group())
        self.assertEqual(result["TN"], 2)
        self.assertEqual(result["Accuracy"], 1.)
        for name in ("Precision", "Recall", "F1", "Balanced Accuracy", "ROC AUC"):
            self.assertIsNone(result[name])
        self.assertIn("分母0", result["uncomputed_reasons"])
        self.assertIn("片クラス", result["uncomputed_reasons"])
        missing = summarize([(None, .4)], group())
        self.assertEqual(missing["n_evaluated"], 0)
        self.assertEqual(missing["excluded_missing_truth"], 1)
        self.assertIsNone(missing["TN"])
        self.assertIsNone(missing["Accuracy"])

    def test_confirmed_class_zero_mapping_inverts_decision_and_auc(self):
        result = summarize([(1, .1), (0, .9), (1, .5)], group(index=0))
        self.assertEqual(result["Accuracy"], 1.)
        self.assertEqual(result["ROC AUC"], 1.)
        self.assertEqual(result["TP"], 2)

    def test_invalid_threshold_rejected(self):
        for value in (float("nan"), float("inf"), -1, 1.01):
            with self.assertRaises(ValueError):
                summarize([(1, .5)], group(), value)


class EvaluationCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.csv = self.root / "synthetic.csv"
        self.meta = self.root / "synthetic.json"
        self.out = self.root / "summary.csv"
        self.md = self.root / "summary.md"
        self.header = "machine,stage,image_id,cell_id,truth,score\n"
        self.metadata = {"schema_version": 1, "score_unit": "0_to_100",
                         "groups": [group("DS10", f"L{i}") for i in range(6)] +
                                   [group("DS1000", "L0", verified=False)]}
        self.meta.write_text(json.dumps(self.metadata))
        self.csv.write_text(self.header + "DS10,L0,synthetic,c1,1,90\nDS10,L0,synthetic,c2,0,10\n")
        self.args = ["--input-csv", str(self.csv), "--metadata", str(self.meta),
                     "--output-csv", str(self.out), "--output-markdown", str(self.md)]

    def test_cli_separates_machine_stage_and_exports_conditions_and_uncomputed(self):
        self.csv.write_text(self.header + "DS10,L0,synthetic,c1,1,90\nDS10,L0,synthetic,c2,0,10\n" +
                            "DS10,L1,synthetic,c1,1,20\nDS1000,L0,synthetic,c1,1,90\n")
        self.assertEqual(main(self.args), 0)
        with self.out.open() as stream:
            rows = {(row["machine"], row["stage"]): row for row in csv.DictReader(stream)}
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows["DS10", "L0"]["Accuracy"], "1.0")
        self.assertEqual(rows["DS10", "L1"]["FN"], "1")
        self.assertEqual(rows["DS1000", "L0"]["TP"], "未算出")
        self.assertEqual(rows["DS10", "L0"]["score_unit"], "0_to_100")
        report = self.md.read_text()
        for text in ("未算出", "クラス対応未確認", "片クラス", "a" * 64, "csv_sha256", "synthetic/direct224"):
            self.assertIn(text, report)

    def test_duplicate_within_stage_rejected_but_all_six_stages_allowed(self):
        metadata, groups = read_metadata(self.meta)
        self.csv.write_text(self.header + "".join(f"DS10,L{i},synthetic,c1,1,90\n" for i in range(6)))
        self.assertEqual(len(read_measurements(self.csv, metadata["score_unit"], groups)), 6)
        with self.csv.open("a") as stream:
            stream.write("DS10,L5,synthetic,c1,1,90\n")
        with self.assertRaisesRegex(ValueError, "重複"):
            read_measurements(self.csv, metadata["score_unit"], groups)

    def test_csv_validation_rejects_bad_values_ids_headers_and_groups(self):
        metadata, groups = read_metadata(self.meta)
        for line in ("DS10,L0,img,c1,1,nan", "DS10,L0,img,c1,1,101", "DS10,L0,img,c1,1,-1",
                     "DS10,L0,img,c1,positive,90", "DS10,L0,img,,1,90", "DS10,L6,img,c1,1,90",
                     "unknown,L0,img,c1,1,90", "DS10,L0,img,c1,1,90,extra", "DS10,L0,img,c1,1"):
            with self.subTest(line=line):
                self.csv.write_text(self.header + line + "\n")
                with self.assertRaises(ValueError):
                    read_measurements(self.csv, metadata["score_unit"], groups)
        self.csv.write_text("machine,stage,image_id,cell_id,truth\nDS10,L0,img,c1,1\n")
        with self.assertRaises(ValueError):
            read_measurements(self.csv, metadata["score_unit"], groups)

    def test_score_units_are_explicit_and_percent_scores_normalize(self):
        _, groups = read_metadata(self.meta)
        result = read_measurements(self.csv, "0_to_100", groups)
        self.assertEqual(result["DS10", "L0"], [(1, .9), (0, .1)])
        with self.assertRaises(ValueError):
            read_measurements(self.csv, "0_to_1", groups)
        self.csv.write_text(self.header + "DS10,L0,img,c1,1,0.5\n")
        self.assertEqual(read_measurements(self.csv, "0_to_100", groups)["DS10", "L0"], [(1, .005)])

    def test_missing_or_invalid_metadata_fails_and_writes_nothing(self):
        variants = [dict(self.metadata, score_unit="auto"), dict(self.metadata, groups=[]),
                    dict(self.metadata, groups=[dict(group(), model_sha256={"classifier": "invalid"})]),
                    dict(self.metadata, groups=[dict(group(), positive_class_mapping_verified="true")]),
                    dict(self.metadata, groups=[dict(group(), positive_class_index=None)]),
                    dict(self.metadata, groups=[group(), group()])]
        for variant in variants:
            self.meta.write_text(json.dumps(variant))
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(self.args)
            self.assertFalse(self.out.exists())
            self.assertFalse(self.md.exists())

    def test_no_overwrite_and_input_stays_intact(self):
        original = self.csv.read_bytes()
        self.out.write_text("keep existing result")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(self.args)
        self.assertEqual(self.out.read_text(), "keep existing result")
        self.assertEqual(self.csv.read_bytes(), original)
        self.assertFalse(self.md.exists())


if __name__ == "__main__":
    unittest.main()
