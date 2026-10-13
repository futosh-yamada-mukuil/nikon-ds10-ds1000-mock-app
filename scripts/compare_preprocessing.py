"""Compare two classifier preprocessing paths on identical, user-specified ROIs.

Example CSV columns: image,cell_id,x1,y1,x2,y2. Coordinates are integer xyxy
with exclusive right/bottom edges. Images are read locally and never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import InferenceSettings  # noqa: E402
from app.inference import (Engine, crop_and_resize, file_sha256, IMAGENET_MEAN,
                           IMAGENET_STD)  # noqa: E402
from scripts.measurement_io import check_output_paths, runtime_metadata, write_text_outputs  # noqa: E402


def read_rois(csv_path: Path) -> list[dict]:
    required = {"image", "cell_id", "x1", "y1", "x2", "y2"}
    rows = []
    seen: set[tuple[str, str]] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if (reader.fieldnames is None or not required.issubset(reader.fieldnames)
                    or len(set(reader.fieldnames)) != len(reader.fieldnames)):
                raise ValueError("CSV列に image,cell_id,x1,y1,x2,y2 が必要です。")
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{line}行目: CSVの列数が一致しません。")
                image_value = (row.get("image") or "").strip()
                cell_id = (row.get("cell_id") or "").strip()
                if not image_value or not cell_id:
                    raise ValueError(f"{line}行目: imageとcell_idは必須です。")
                key = (image_value, cell_id)
                if key in seen:
                    raise ValueError(f"{line}行目: 同じ画像のcell_idが重複しています: {cell_id}")
                seen.add(key)
                try:
                    box = tuple(int(row[name]) for name in ("x1", "y1", "x2", "y2"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{line}行目: xyxy座標は整数で指定してください。") from exc
                rows.append({"image": image_value, "cell_id": cell_id, "box": box})
    except OSError as exc:
        raise ValueError(f"CSVを読めません: {csv_path}") from exc
    if not rows:
        raise ValueError("CSVにROI行がありません。")
    return rows


def compare_roi(engine: Engine, image: Image.Image, box: Iterable[int], jpeg_quality: int) -> tuple[float, float, tuple[int, int, int, int]]:
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 95:
        raise ValueError("JPEG品質は1〜95の整数で指定してください。")
    crop, clipped = crop_and_resize(image.convert("RGB"), box)
    if crop is None or clipped is None:
        raise ValueError("クリップ後のROIは幅・高さ4px以上必要です。")
    direct_score = engine._classify(crop)

    # The comparison path models enlargement of the same ROI, JPEG round-trip
    # in memory, then the classifier's normal 224×224 input size.
    enlarged = image.convert("RGB").crop(clipped).resize((640, 640), Image.Resampling.BILINEAR)
    buffer = BytesIO()
    enlarged.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        comparison = decoded.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    comparison_score = engine._classify(comparison)
    if any(not math.isfinite(score) or not 0 <= score <= 1
           for score in (direct_score, comparison_score)):
        raise ValueError("Engineの分類スコアは有限な0〜1である必要があります。")
    return direct_score, comparison_score, clipped


def prepare_inputs(args: argparse.Namespace) -> tuple[list[dict], dict[Path, Image.Image], dict[str, str]]:
    """Validate every row/image/ROI before loading any checkpoint."""
    InferenceSettings(detection_threshold=args.detection_threshold,
                      classification_threshold=args.classification_threshold)
    if type(args.jpeg_quality) is not int or not 1 <= args.jpeg_quality <= 95:
        raise ValueError("JPEG品質は1〜95の整数で指定してください。")
    rows = read_rois(args.input_csv)
    image_cache: dict[Path, Image.Image] = {}
    hashes = {}
    seen = set()
    input_root = args.image_root.resolve()
    for row in rows:
        image_path = (input_root / row["image"]).resolve()
        if not image_path.is_relative_to(input_root) or not image_path.is_file():
            raise ValueError(f"画像パスが無効か見つかりません: {row['image']}")
        key = (image_path, row["cell_id"])
        if key in seen:
            raise ValueError("同じ画像のcell_idが重複しています（パス表記違いを含む）。")
        seen.add(key)
        if image_path not in image_cache:
            # Decode and fingerprint the same bytes, not two reads of a changing file.
            content = image_path.read_bytes()
            hashes[str(image_path)] = hashlib.sha256(content).hexdigest()
            with Image.open(BytesIO(content)) as source:
                image_cache[image_path] = source.convert("RGB")
        if crop_and_resize(image_cache[image_path], row["box"])[0] is None:
            raise ValueError(f"{row['cell_id']}: クリップ後のROIは幅・高さ4px以上必要です。")
    return rows, image_cache, hashes


def run(args: argparse.Namespace, engine: Engine, prepared=None) -> tuple[list[dict], dict]:
    rows, image_cache, hashes = prepared if prepared is not None else prepare_inputs(args)
    outputs = []
    input_root = args.image_root.resolve()
    for row in rows:
        image_path = (input_root / row["image"]).resolve()
        direct, comparison, clipped = compare_roi(engine, image_cache[image_path], row["box"], args.jpeg_quality)
        outputs.append({
            "image": row["image"], "cell_id": row["cell_id"],
            "original_x1": row["box"][0], "original_y1": row["box"][1],
            "original_x2": row["box"][2], "original_y2": row["box"][3],
            "x1": clipped[0], "y1": clipped[1], "x2": clipped[2], "y2": clipped[3],
            "direct_224_classification_score_0_to_1": direct,
            "jpeg_640_then_224_classification_score_0_to_1": comparison,
            "score_difference_comparison_minus_direct": comparison - direct,
            "classification_threshold": args.classification_threshold,
            "jpeg_quality": args.jpeg_quality,
            "direct_candidate": direct > args.classification_threshold,
            "comparison_candidate": comparison > args.classification_threshold,
        })
    metadata = {
        "schema_version": 2,
        "classification_score_unit": "0_to_1",
        "input_csv": str(args.input_csv.resolve()),
        "image_root": str(input_root),
        "classifier_model": str(args.classifier.resolve()),
        "classifier_sha256": engine._hashes.get("classifier"),
        "image_sha256": hashes,
        "environment": runtime_metadata(),
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "device_requested": args.device,
        "device_used": engine.device,
        "classification_threshold_score_0_to_1": args.classification_threshold,
        "detection_threshold": args.detection_threshold,
        "same_roi_for_both_paths": True,
        "paths": {
            "direct": "clipped ROI -> 224x224 bilinear -> existing Engine classifier transform",
            "comparison": "same clipped ROI -> 640x640 bilinear -> in-memory JPEG -> 224x224 bilinear -> existing Engine classifier transform",
        },
        "jpeg_quality": args.jpeg_quality,
        "training_time_jpeg_quality": "unverified",
        "scores": "softmax class 1 probability (0–1); class mapping independently unverified",
        "detection_threshold_applied": False,
        "labels_created": False,
        "ground_truth_compared": False,
        "score_source": "Engine._classify (uncalibrated class 1)",
    }
    return outputs, metadata


def write_outputs(output_csv: Path, output_json: Path, rows: list[dict], metadata: dict) -> None:
    check_output_paths(output_csv, output_json)
    if not rows:
        raise ValueError("出力するROIがありません。")
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payload = json.dumps({"metadata": {**metadata, "rows_written": len(rows)}, "results": rows},
                         ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_text_outputs({output_csv: stream.getvalue(), output_json: payload})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--jpeg-quality", type=int, choices=range(1, 96), required=True)
    parser.add_argument("--classification-threshold", type=float, default=0.5,
                        help="class 1 score threshold on 0-1 scale (default: 0.5 = 50%%)")
    parser.add_argument("--detection-threshold", type=float, default=0.5,
                        help="recorded reference setting; no detector is run")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser())
    if not 0 <= args.classification_threshold <= 1:
        parser.error("--classification-threshold は0〜1です。")
    if not 0 <= args.detection_threshold <= 1:
        parser.error("--detection-threshold は0〜1です。")
    if not args.classifier.is_file():
        parser.error("--classifier のファイルが見つかりません。")
    try:
        check_output_paths(args.output_csv, args.output_json)
        input_digest = file_sha256(args.input_csv)
        prepared = prepare_inputs(args)
        engine = Engine(Path("unused-detector-not-loaded"), args.classifier, args.device)
        # Reuse Engine's classifier implementation without loading detector weights.
        load_classifier_for_comparison(engine)
        outputs, metadata = run(args, engine, prepared)
        if file_sha256(args.input_csv) != input_digest:
            raise ValueError("測定中に入力CSVが変化したため保存を中止しました。")
        metadata["input_csv_sha256"] = input_digest
        write_outputs(args.output_csv, args.output_json, outputs, metadata)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        if "engine" in locals():
            engine.close()
    print(f"比較完了: {len(outputs)} ROI; CSV={args.output_csv}; JSON={args.output_json}")
    return 0


def load_classifier_for_comparison(engine: Engine) -> None:
    """Load only the classifier checkpoint; detector weights are never read."""
    import torch
    import torchvision.transforms as transforms

    from app.inference import CLASSIFIER_SHA256, build_classifier, choose_device, verify_model
    engine.device = choose_device(engine.requested_device)
    engine._hashes["classifier"] = verify_model(engine.classifier_path, CLASSIFIER_SHA256)
    state = torch.load(engine.classifier_path, map_location="cpu", weights_only=True)
    engine._classifier = build_classifier(state).to(engine.device)
    engine._transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    engine.loaded = True


if __name__ == "__main__":
    raise SystemExit(main())
