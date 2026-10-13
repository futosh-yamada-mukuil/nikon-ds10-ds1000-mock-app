"""Aggregate labeled cell measurements by machine and stage, without models.

CSV: machine,stage,image_id,cell_id,truth,score (truth is 0/1 or blank).
Metadata declares score units and per-group model hashes, input path and mapping.
See docs/local_measurements.md for synthetic examples and uncomputed metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
from itertools import groupby
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measurement_io import check_output_paths, write_text_outputs  # noqa: E402

MACHINES = ("DS10", "DS1000")
STAGES = tuple(f"L{index}" for index in range(6))
METRICS = ("Precision", "Recall", "F1", "Accuracy", "Balanced Accuracy", "ROC AUC")
COUNTS = ("TP", "FP", "FN", "TN")


def read_metadata(path: Path) -> tuple[dict, dict]:
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict) or metadata.get("score_unit") not in ("0_to_1", "0_to_100"):
        raise ValueError("score_unitは0_to_1または0_to_100の明示が必要です。")
    if metadata.get("schema_version") != 1 or not isinstance(metadata.get("groups"), list):
        raise ValueError("metadataにはschema_version: 1とgroups配列が必要です。")
    groups = {}
    for group in metadata["groups"]:
        if not isinstance(group, dict):
            raise ValueError("groupsの要素はオブジェクトで指定してください。")
        key = (group.get("machine"), group.get("stage"))
        if key[0] not in MACHINES or key[1] not in STAGES or key in groups:
            raise ValueError("metadataの機種・段階が不正、または重複しています。")
        hashes = group.get("model_sha256")
        if (not isinstance(hashes, dict) or "classifier" not in hashes or
                any(role not in ("classifier", "detector") or not isinstance(value, str) or
                    not re.fullmatch(r"[0-9a-fA-F]{64}", value) for role, value in hashes.items())):
            raise ValueError("model_sha256にはclassifier（任意でdetector）の64桁SHA256が必要です。")
        if not isinstance(group.get("input_path"), str) or not group["input_path"].strip():
            raise ValueError("各groupに入力経路input_pathが必要です。")
        verified = group.get("positive_class_mapping_verified")
        if type(verified) is not bool:
            raise ValueError("positive_class_mapping_verifiedはtrue/falseで指定してください。")
        if verified and (type(group.get("positive_class_index")) is not int or
                         group["positive_class_index"] not in (0, 1)):
            raise ValueError("対応確認済みの場合positive_class_indexは0または1が必要です。")
        groups[key] = group
    if not groups:
        raise ValueError("metadataのgroupsが空です。")
    return metadata, groups


def read_measurements(path: Path, score_unit: str, groups: dict) -> dict:
    limit = {"0_to_1": 1, "0_to_100": 100}[score_unit]
    required = {"machine", "stage", "image_id", "cell_id", "truth", "score"}
    grouped, seen = {}, set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or not required.issubset(reader.fieldnames) or
                len(set(reader.fieldnames)) != len(reader.fieldnames)):
            raise ValueError("CSV列にmachine,stage,image_id,cell_id,truth,scoreが必要です。")
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{line}行目: CSVの列数が一致しません。")
            row = {name: raw[name].strip() for name in required}
            key = (row["machine"], row["stage"])
            if key not in groups:
                raise ValueError(f"{line}行目: 機種・段階に対応するmetadataがありません。")
            identity = (*key, row["image_id"], row["cell_id"])
            if not row["image_id"] or not row["cell_id"] or identity in seen:
                raise ValueError(f"{line}行目: セルIDが空、または同一セルが重複しています。")
            seen.add(identity)
            if row["truth"] not in ("", "0", "1"):
                raise ValueError(f"{line}行目: truthは0/1または空欄で指定してください。")
            try:
                score = float(row["score"])
            except ValueError as exc:
                raise ValueError(f"{line}行目: scoreは数値で指定してください。") from exc
            if not math.isfinite(score) or not 0 <= score <= limit:
                raise ValueError(f"{line}行目: scoreは有限な0〜{limit}で指定してください。")
            grouped.setdefault(key, []).append((None if row["truth"] == "" else int(row["truth"]),
                                                score / limit))
    if not grouped:
        raise ValueError("CSVに測定行がありません。")
    return grouped


def roc_auc(values) -> float | None:
    """Mann-Whitney rank AUC; ties receive the average rank."""
    positives = sum(truth == 1 for truth, _ in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    rank, positive_ranks = 1, 0.0
    for _, tied in groupby(sorted(values, key=lambda item: item[1]), key=lambda item: item[1]):
        items = list(tied)
        average_rank = rank + (len(items) - 1) / 2
        positive_ranks += average_rank * sum(truth == 1 for truth, _ in items)
        rank += len(items)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def summarize(values, group: dict, threshold: float = 0.5) -> dict:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("分類しきい値は有限な0〜1で指定してください。")
    labeled = [(truth, score) for truth, score in values if truth is not None]
    verified = group["positive_class_mapping_verified"]
    result = {"machine": group["machine"], "stage": group["stage"],
              "n_total": len(values), "n_labeled": len(labeled),
              "excluded_missing_truth": len(values) - len(labeled),
              "n_evaluated": len(labeled) if verified else 0,
              "classification_threshold_0_to_1": threshold,
              **dict.fromkeys((*COUNTS, *METRICS)), "uncomputed_reasons": ""}
    if not verified or not labeled:
        result["uncomputed_reasons"] = ("全指標・混同行列: クラス対応未確認" if not verified else
                                       "全指標・混同行列: 正解付きセルなし")
        return result
    positive_index = group["positive_class_index"]
    tp = fp = fn = tn = 0
    for truth, score in labeled:
        # Preserve class-1 strict >. A confirmed class-0 positive mapping
        # inverts the decision (including equality) rather than guessing labels.
        predicted_positive = (score > threshold) == (positive_index == 1)
        tp += int(truth == 1 and predicted_positive)
        fp += int(truth == 0 and predicted_positive)
        fn += int(truth == 1 and not predicted_positive)
        tn += int(truth == 0 and not predicted_positive)
    reasons = []

    def ratio(name, numerator, denominator):
        if denominator == 0:
            reasons.append(f"{name}: 分母0")
            return None
        return numerator / denominator

    result.update(TP=tp, FP=fp, FN=fn, TN=tn)
    result["Precision"] = ratio("Precision", tp, tp + fp)
    result["Recall"] = ratio("Recall", tp, tp + fn)
    result["F1"] = ratio("F1", 2 * tp, 2 * tp + fp + fn)
    result["Accuracy"] = (tp + tn) / len(labeled)
    if tp + fn == 0 or tn + fp == 0:
        reasons.append("Balanced Accuracy・ROC AUC: 片クラス")
    else:
        result["Balanced Accuracy"] = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        oriented = [(truth, score if positive_index == 1 else -score) for truth, score in labeled]
        result["ROC AUC"] = roc_auc(oriented)
    result["uncomputed_reasons"] = "; ".join(reasons)
    return result


def export_results(rows, metadata, provenance) -> tuple[str, str]:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=(*rows[0], "score_unit", "model_sha256",
                                                "input_path", "positive_class_mapping_verified",
                                                "positive_class_index"))
    writer.writeheader()
    groups = {(group["machine"], group["stage"]): group for group in metadata["groups"]}
    for row in rows:
        group = groups[row["machine"], row["stage"]]
        writer.writerow({**{name: ("未算出" if value is None else value) for name, value in row.items()},
                         "score_unit": metadata["score_unit"],
                         "model_sha256": json.dumps(group["model_sha256"], sort_keys=True),
                         "input_path": group["input_path"],
                         "positive_class_mapping_verified": group["positive_class_mapping_verified"],
                         "positive_class_index": group.get("positive_class_index")})
    lines = ["# 機種・段階別の分類評価", "",
             "入力CSVの集計です。モデル推論・画像検査・学習来歴確認は実行していません。",
             "モデルSHA256とクラス対応は入力メタデータの申告値です。",
             f"入力スコア単位: {metadata['score_unit']}。指標は0〜1。欠損truthは未評価として除外。",
             f"class 1の判定: score > {rows[0]['classification_threshold_0_to_1']}（0〜1へ明示変換後）。",
             "クラス0が陽性と確認された場合のみ判定とAUCの向きを反転します。", ""]

    def table(columns):
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            values = []
            for column in columns:
                value = row[column]
                text = "未算出" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))
                values.append(text.replace("|", "\\|").replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    table(("machine", "stage", "n_total", "n_labeled", "excluded_missing_truth", "n_evaluated", *COUNTS))
    table(("machine", "stage", *METRICS, "uncomputed_reasons"))
    lines.extend(["## 実行条件", "", "```json", json.dumps({"metadata": metadata, "inputs": provenance},
                 ensure_ascii=False, indent=2, allow_nan=False), "```", "",
                 "検出枠の正解との照合や検出漏れはこの表の対象外です。",
                 "精度90%達成や改善原因の確定を示すものではありません。", ""])
    return stream.getvalue(), "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--classification-threshold", type=float, default=0.5,
                        help="Normalized 0-1 threshold, default 0.5 (50%%), regardless of input units")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser())
    try:
        check_output_paths(args.output_csv, args.output_markdown)
        input_bytes = (args.input_csv.read_bytes(), args.metadata.read_bytes())
        metadata, groups = read_metadata(args.metadata)
        grouped = read_measurements(args.input_csv, metadata["score_unit"], groups)
        rows = [summarize(values, groups[key], args.classification_threshold)
                for key, values in sorted(grouped.items())]
        provenance = {"csv_sha256": hashlib.sha256(input_bytes[0]).hexdigest(),
                      "metadata_sha256": hashlib.sha256(input_bytes[1]).hexdigest()}
        csv_text, markdown = export_results(rows, metadata, provenance)
        if input_bytes != (args.input_csv.read_bytes(), args.metadata.read_bytes()):
            raise ValueError("集計中に入力が変化したため保存を中止しました。")
        write_text_outputs({args.output_csv: csv_text, args.output_markdown: markdown})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"{len(rows)}組の機種・段階別集計を保存しました。実モデル推論は行っていません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
