"""Compare the default detector and experimental tiles on one local image.

Writes predictions and timing only; counts are not an accuracy measurement.
"""
from __future__ import annotations

import argparse
import csv
from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import InferenceSettings  # noqa: E402
from app.inference import Engine  # noqa: E402
from app.tiled_detection import detect_tiled, tile_boxes  # noqa: E402
from scripts.measurement_io import check_output_paths, runtime_metadata, write_text_outputs  # noqa: E402


def compare(engine: Engine, image: Image.Image, threshold=0.5, *,
            tile_size=640, overlap=0.25, nms_iou=0.5) -> tuple[list[dict], dict]:
    InferenceSettings(detection_threshold=threshold)
    tiles = list(tile_boxes(image.width, image.height, tile_size, overlap))
    if not 0 <= nms_iou <= 1:
        raise ValueError("NMSのIoUは0〜1です。")
    rows, runs = [], {}
    for name in ("baseline_640", "tiled"):
        started = time.perf_counter()
        detections = (engine._detect(image, threshold) if name == "baseline_640" else
                      detect_tiled(engine, image, threshold, tile_size=tile_size,
                                   overlap=overlap, nms_iou=nms_iou))
        runs[name] = {"elapsed_seconds": time.perf_counter() - started,
                      "detection_count": len(detections)}
        rows.extend({"path": name, "detection_id": index, "x1": box[0], "y1": box[1],
                     "x2": box[2], "y2": box[3], "confidence": score}
                    for index, (box, score) in enumerate(detections, 1))
    return rows, {"detection_threshold": threshold, "candidate_rule": "confidence >= threshold",
                  "tile_size": tile_size, "overlap": overlap, "tile_count": len(tiles),
                  "nms_iou": nms_iou, "nms_rule": "IoU > threshold, different tiles only",
                  "runs": runs, "ground_truth_compared": False, "accuracy_metrics": None,
                  "classification_performed": False,
                  "limitations": "Dense adjacent cells may be suppressed; requires labeled evaluation."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True,
                        help="Required by the existing Engine loader; no classification is run.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser())
    engine = None
    try:
        check_output_paths(args.output_csv, args.output_json)
        InferenceSettings(detection_threshold=args.detection_threshold)
        if not 0 <= args.nms_iou <= 1:
            raise ValueError("NMSのIoUは0〜1です。")
        for path in (args.image, args.detector, args.classifier):
            if not path.is_file():
                raise ValueError(f"入力ファイルが見つかりません: {path}")
        content = args.image.read_bytes()
        with Image.open(BytesIO(content)) as source:
            image = source.convert("RGB")
        list(tile_boxes(image.width, image.height, args.tile_size, args.overlap))
        engine = Engine(args.detector, args.classifier, args.device)
        engine.load()
        rows, metadata = compare(engine, image, args.detection_threshold,
                                 tile_size=args.tile_size, overlap=args.overlap, nms_iou=args.nms_iou)
        metadata.update({"schema_version": 1, "image": str(args.image.resolve()),
                         "image_sha256": hashlib.sha256(content).hexdigest(),
                         "image_size": image.size, "models": engine.get_model_metadata(),
                         "device_requested": args.device, "environment": runtime_metadata()})
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=("path", "detection_id", "x1", "y1", "x2", "y2", "confidence"))
        writer.writeheader()
        writer.writerows(rows)
        payload = json.dumps({"metadata": metadata, "detections": rows}, ensure_ascii=False,
                             indent=2, allow_nan=False) + "\n"
        write_text_outputs({args.output_csv: stream.getvalue(), args.output_json: payload})
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        if engine is not None:
            engine.close()
    print("検出比較を保存しました。正解との精度比較は未実施です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
