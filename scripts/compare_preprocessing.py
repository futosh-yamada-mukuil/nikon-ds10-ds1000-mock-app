"""Compare two classifier preprocessing paths on identical, user-specified ROIs.

Example CSV columns: image,cell_id,x1,y1,x2,y2. Coordinates are integer xyxy
with exclusive right/bottom edges. Images are read locally and never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import InferenceSettings  # noqa: E402
from app.inference import Engine, crop_and_resize  # noqa: E402


def read_rois(csv_path: Path) -> list[dict]:
    required = {"image", "cell_id", "x1", "y1", "x2", "y2"}
    rows = []
    seen: set[tuple[str, str]] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("CSV列に image,cell_id,x1,y1,x2,y2 が必要です。")
            for line, row in enumerate(reader, 2):
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
    crop, clipped = crop_and_resize(image.convert("RGB"), box)
    if crop is None or clipped is None:
        raise ValueError("クリップ後のROIは幅・高さ4px以上必要です。")
    direct_score = engine._classify(crop)

    # The comparison path models enlargement of the same ROI, JPEG round-trip
    # in memory, then the classifier's normal 224×224 input size.
    enlarged = crop.resize((640, 640), Image.Resampling.BILINEAR)
    from io import BytesIO
    buffer = BytesIO()
    enlarged.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        comparison = decoded.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    comparison_score = engine._classify(comparison)
    return direct_score, comparison_score, clipped


def run(args: argparse.Namespace, engine: Engine) -> tuple[list[dict], dict]:
    rows = read_rois(args.input_csv)
    outputs = []
    image_cache: dict[Path, Image.Image] = {}
    input_root = args.image_root.resolve()
    for row in rows:
        image_path = (input_root / row["image"]).resolve()
        if not image_path.is_relative_to(input_root) or not image_path.is_file():
            raise ValueError(f"画像パスが無効か見つかりません: {row['image']}")
        if image_path not in image_cache:
            with Image.open(image_path) as source:
                image_cache[image_path] = source.convert("RGB")
        direct, comparison, clipped = compare_roi(engine, image_cache[image_path], row["box"], args.jpeg_quality)
        outputs.append({
            "image": row["image"], "cell_id": row["cell_id"],
            "x1": clipped[0], "y1": clipped[1], "x2": clipped[2], "y2": clipped[3],
            "direct_224_class1_score": direct,
            "jpeg_640_then_224_class1_score": comparison,
            "score_difference_comparison_minus_direct": comparison - direct,
            "classification_threshold": args.classification_threshold,
            "direct_candidate": direct > args.classification_threshold,
            "comparison_candidate": comparison > args.classification_threshold,
        })
    metadata = {
        "input_csv": str(args.input_csv.resolve()),
        "image_root": str(input_root),
        "classifier_model": str(args.classifier.resolve()),
        "device_requested": args.device,
        "device_used": engine.device,
        "classification_threshold_score_0_to_100": args.classification_threshold,
        "detection_threshold": args.detection_threshold,
        "same_roi_for_both_paths": True,
        "paths": {
            "direct": "clipped ROI -> 224x224 bilinear -> existing Engine classifier transform",
            "comparison": "same clipped ROI -> 640x640 bilinear -> in-memory JPEG -> 224x224 bilinear -> existing Engine classifier transform",
        },
        "jpeg_quality": args.jpeg_quality,
        "training_time_jpeg_quality": "unverified",
        "scores": "softmax class 1 multiplied by 100; class mapping independently unverified",
        "detection_threshold_applied": False,
        "labels_created": False,
        "model_predictions_measured": True,
    }
    return outputs, metadata


def write_outputs(output_csv: Path, output_json: Path, rows: list[dict], metadata: dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata["rows_written"] = len(rows)
    output_json.write_text(json.dumps({"metadata": metadata, "results": rows},
                                      ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


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
                        help="class 1 score threshold on 0-100 scale (default: 0.5)")
    parser.add_argument("--detection-threshold", type=float, default=0.5,
                        help="recorded reference setting; no detector is run")
    args = parser.parse_args(argv)
    if not 0 <= args.classification_threshold <= 100:
        parser.error("--classification-threshold は0〜100です。")
    if not 0 <= args.detection_threshold <= 1:
        parser.error("--detection-threshold は0〜1です。")
    if not args.classifier.is_file():
        parser.error("--classifier のファイルが見つかりません。")
    try:
        engine = Engine.__new__(Engine)
        engine.classifier_path = args.classifier.expanduser().resolve()
        engine.device = "cpu" if args.device == "auto" else args.device
        engine.loaded = False
        engine._hashes = {}
        engine._classifier = None
        engine._transform = None
        # Reuse Engine's classifier implementation without loading detector weights.
        load_classifier_for_comparison(engine)
        outputs, metadata = run(args, engine)
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
    engine.device = choose_device(engine.device)
    engine._hashes["classifier"] = verify_model(engine.classifier_path, CLASSIFIER_SHA256)
    state = torch.load(engine.classifier_path, map_location="cpu", weights_only=True)
    engine._classifier = build_classifier(state).to(engine.device)
    engine._transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    engine.loaded = True


if __name__ == "__main__":
    raise SystemExit(main())
