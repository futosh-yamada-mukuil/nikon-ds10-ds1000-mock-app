"""Local positive-annotation comparison; no retraining or source-image changes.

Measures both the app's 640px detector input and original RGB input. Classifier
scores are cached once; threshold search never repeats model inference. Red
rectangles are positive annotations, not an all-cell detection ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import Engine, crop_and_resize, file_sha256
from app.domain import classification_score_from_percent


def load_cached_predictions(output: Path) -> list[dict]:
    """Read versioned scores or explicitly identified legacy percentages, in memory."""
    metadata = json.loads((output / "manifest.json").read_text())
    unit = metadata.get("classification_score_unit")
    if metadata.get("schema_version") == 2 and unit == "0_to_1":
        legacy = False
    elif (metadata.get("schema_version") in (None, 1) and unit in (None, "0_to_100")
          and "batch_score_tolerance_0_to_100" in metadata):
        legacy = True
    else:
        raise ValueError("Prediction cache score units are unknown or inconsistent")
    records = []
    for path in sorted((output / "cache").glob("*.json")):
        row = json.loads(path.read_text())
        if legacy:
            if (row.get("schema_version") not in (None, 1) or
                    row.get("classification_score_unit") not in (None, "0_to_100")):
                raise ValueError("Legacy cache contains normalized scores; refusing double conversion")
        elif row.get("schema_version") != 2 or row.get("classification_score_unit") != "0_to_1":
            raise ValueError("Normalized cache record is missing its score unit")
        for cells in row["variants"].values():
            for cell in cells:
                score = cell["score"]
                if legacy:
                    score = classification_score_from_percent(score)
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid normalized classifier score")
                cell["score"] = score
        row.update(schema_version=2, classification_score_unit="0_to_1")
        records.append(row)
    return records


def truth_boxes(image: Image.Image) -> list[list[int]]:
    """Extract closed red rectangles; use exclusive right/bottom pixel edges."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    red = ((rgb[..., 0] >= 220) & (rgb[..., 1] <= 80) & (rgb[..., 2] <= 80)).astype("uint8")
    _, _, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if w < 10 or h < 10 or area < w + h:
            continue
        borders = ((red[y:y + 4, x:x + w].max(axis=0).mean(), y == 0),
                   (red[y + h - 4:y + h, x:x + w].max(axis=0).mean(), y + h == red.shape[0]),
                   (red[y:y + h, x:x + 4].max(axis=1).mean(), x == 0),
                   (red[y:y + h, x + w - 4:x + w].max(axis=1).mean(), x + w == red.shape[1]))
        if any(coverage < .8 and not edge for coverage, edge in borders):
            raise ValueError(f"Incomplete/merged annotation rectangle: {(x, y, w, h)}")
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def overlap_edges(cells: list[dict], truth: list[list[float]]) -> dict[int, list[int]]:
    edges = {}
    for index, cell in enumerate(cells):
        a = cell["box"]
        overlaps = []
        for j, b in enumerate(truth):
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
            iou = intersection / max(1., union)
            if iou >= .20:
                overlaps.append((j, iou))
        edges[index] = [j for j, _ in sorted(overlaps, key=lambda pair: (-pair[1], pair[0]))]
    return edges


def counts(cells: list[dict], edges: dict[int, list[int]], n_truth: int,
           detection: float, classification: float | None) -> dict:
    selected = [i for i, c in enumerate(cells) if c["confidence"] >= detection and
                (classification is None or c["score"] > classification)]
    matched = {}

    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or visit(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in selected:
        visit(i, set())
    tp = len(matched)
    return {"candidates": len(selected), "truth": n_truth, "tp": tp,
            "fp": len(selected) - tp, "fn": n_truth - tp}


def aggregate(records: list[dict], variant: str, detection: float, classification: float | None) -> dict:
    totals = {key: 0 for key in ("candidates", "truth", "tp", "fp", "fn")}
    for row in records:
        cells = row["variants"][variant]
        result = counts(cells, overlap_edges(cells, row["truth_boxes"]), len(row["truth_boxes"]), detection, classification)
        for key in totals:
            totals[key] += result[key]
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    return dict(totals, images=len(records), precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn), error=fp + fn)


def discover(data_root: Path) -> tuple[list[dict], list[dict]]:
    pairs, missing = [], []
    for machine, folder, source_pattern, truth_pattern in (
        ("DS10", "02_DS10", r"1-3_(\d+)_RGB_New", r"1-3_(\d+)_RGB_40XFITC2"),
        ("DS1000", "01_DS1000", r"NIS_L_Image_(\d+)", r"NIS_L_Image_(\d+)_after"),
    ):
        truths = {int(re.fullmatch(truth_pattern, p.stem)[1]): p
                  for p in (data_root / folder / "正解").glob("*.png") if re.fullmatch(truth_pattern, p.stem)}
        sources = sorted((p for p in (data_root / folder / "学習用").glob("*.jpg")
                          if re.fullmatch(source_pattern, p.stem)), key=lambda p: int(re.fullmatch(source_pattern, p.stem)[1]))
        index = 0
        for source in sources:
            number = int(re.fullmatch(source_pattern, source.stem)[1])
            if number not in truths:
                missing.append({"machine": machine, "image": source.name, "reason": "no_ground_truth"})
                continue
            try:
                with Image.open(truths[number]) as annotation:
                    truth_boxes(annotation)
            except ValueError as exc:
                missing.append({"machine": machine, "image": source.name,
                                "reason": "ambiguous_ground_truth_rectangles", "detail": str(exc)})
                continue
            # Fixed by paired image order BEFORE predictions. 1/3 calibration,
            # 2/3 held-out; original model training membership remains unknown.
            split = "calibration" if index % 3 == 0 else "holdout"
            pairs.append({"machine": machine, "image": source.name, "source": str(source),
                          "annotation": str(truths[number]), "split": split, "training_membership": "unknown"})
            index += 1
    return pairs, missing


def classify_batches(engine: Engine, image: Image.Image, detections, batch_size=16):
    import torch

    cells, crops = [], []
    for box, confidence in detections:
        crop, clipped = crop_and_resize(image, box)
        if crop is not None:
            crops.append(crop)
            cells.append({"box": list(clipped), "confidence": float(confidence)})
    for offset in range(0, len(crops), batch_size):
        tensor = torch.stack([engine._transform(crop) for crop in crops[offset:offset + batch_size]]).to(engine.device)
        with torch.no_grad():
            scores = torch.softmax(engine._classifier(tensor), dim=-1)[:, 1].cpu().tolist()
        for cell, score in zip(cells[offset:offset + batch_size], scores):
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid classifier score")
            cell["score"] = score
    if crops:
        # Real comparison with the app's single-cell path, per image/variant.
        difference = abs(engine._classify(crops[0]) - cells[0]["score"])
        if difference > .00001:
            raise ValueError(f"Batch/single-cell score difference {difference} exceeds .00001 (0–1)")
    return cells


def measure(args):
    import torch

    pairs, missing = discover(args.data_root)
    config = json.loads(args.model_config.read_text())
    engine = Engine(Path(config["detection"]["path"]), Path(config["classification"]["path"]), log=print)
    engine.load()
    for index, pair in enumerate(pairs, 1):
        path = args.output / "cache" / f"{pair['machine']}_{Path(pair['image']).stem}.json"
        if path.exists():
            raise ValueError(f"Refusing to overwrite inference cache: {path}")
        with Image.open(pair["source"]) as f:
            image = f.convert("RGB")
        with Image.open(pair["annotation"]) as f:
            truth_size = f.size
            boxes = truth_boxes(f)
        sx, sy = image.width / truth_size[0], image.height / truth_size[1]
        pair.update(schema_version=2, classification_score_unit="0_to_1",
                    source_size=list(image.size), truth_size=list(truth_size),
                    truth_boxes=[[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes],
                    source_sha256=file_sha256(Path(pair["source"])),
                    annotation_sha256=file_sha256(Path(pair["annotation"])), variants={})
        pair["variants"]["legacy640"] = classify_batches(engine, image, engine._detect(image, .10))
        with torch.no_grad():
            native = engine._detector.predict(image, threshold=0.0)
        xyxy, confidence = np.asarray(native.xyxy), np.asarray(native.confidence)
        if xyxy.shape != (len(confidence), 4) or not np.isfinite(xyxy).all() or not np.isfinite(confidence).all():
            raise ValueError("Invalid native detector output")
        pair["variants"]["native"] = classify_batches(engine, image, [(b, s) for b, s in zip(xyxy, confidence) if s >= .10])
        path.write_text(json.dumps(pair, ensure_ascii=False, allow_nan=False))
        print(f"{index}/{len(pairs)} {pair['machine']} {pair['image']} truth={len(boxes)}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 2, "classification_score_unit": "0_to_1",
        "models": engine.get_model_metadata(), "missing": missing,
        "matching_iou": .20, "annotation_convention": "exclusive right/bottom; scaled independently on x/y",
        "positive_truth_confirmed_by_user": True, "classifier_biology_independently_verified": False,
        "batch_score_tolerance_0_to_1": .00001, "minimum_cached_detection_confidence": .10}, ensure_ascii=False, indent=2))
    engine.close()


def summarize(args):
    records = load_cached_predictions(args.output)
    pairs, _ = discover(args.data_root)
    if {(r["machine"], r["image"]) for r in records} != {(r["machine"], r["image"]) for r in pairs}:
        raise ValueError("Incomplete inference cache; no accuracy result published")
    summary, image_rows = {}, []
    for machine in ("DS10", "DS1000"):
        train = [r for r in records if r["machine"] == machine and r["split"] == "calibration"]
        test = [r for r in records if r["machine"] == machine and r["split"] == "holdout"]
        baseline = aggregate(train, "legacy640", .5, .5)
        baseline_detection = aggregate(train, "legacy640", .5, None)
        trials = []
        for variant in ("legacy640", "native"):
            for detection in (.10, .15, .20, .29, .40, .50, .60, .70, .80, .90):
                detection_metric = aggregate(train, variant, detection, None)
                if detection_metric["fn"] > baseline_detection["fn"]:
                    continue
                for classification in (0., .0001, .0005, .001, .002, .005, .10, .20, .30, .40, .44, .50, .60, .70, .80, .90, .95, .97, .99, .995, .999):
                    metric = aggregate(train, variant, detection, classification)
                    if metric["fn"] > baseline["fn"] or metric["fp"] > baseline["fp"]:
                        continue
                    trials.append((metric["f1"], -metric["error"], metric["recall"], variant == "legacy640", detection, classification, variant, metric))
        best = max(trials)
        detection, classification, variant = best[4:7]
        summary[machine] = {"schema_version": 2, "classification_score_unit": "0_to_1",
            "selection": {"variant": variant, "detection_threshold": detection, "classification_threshold": classification},
            "calibration_selected": best[7], "calibration_baseline": aggregate(train, "legacy640", .5, .5),
            "holdout_baseline": aggregate(test, "legacy640", .5, .5),
            "holdout_same_threshold_native": aggregate(test, "native", .5, .5),
            "holdout_selected": aggregate(test, variant, detection, classification),
            "positive_detection_coverage_baseline": aggregate(test, "legacy640", .5, None),
            "positive_detection_coverage_selected": aggregate(test, variant, detection, None)}
        before, after = summary[machine]["holdout_baseline"], summary[machine]["holdout_selected"]
        detection_before = summary[machine]["positive_detection_coverage_baseline"]
        detection_after = summary[machine]["positive_detection_coverage_selected"]
        summary[machine]["eligible_for_app"] = (after["fn"] <= before["fn"] and after["fp"] <= before["fp"]
            and detection_after["fn"] <= detection_before["fn"] and after["f1"] > before["f1"])
        summary[machine]["selection_rule"] = "Calibration F1 maximum with FP/FN and positive-detection FN no worse than baseline; held-out degradation prevents adoption"
        for row in train + test:
            for label, v, d, c in (("baseline", "legacy640", .5, .5), ("selected", variant, detection, classification)):
                cells = row["variants"][v]
                metric = counts(cells, overlap_edges(cells, row["truth_boxes"]), len(row["truth_boxes"]), d, c)
                image_rows.append(dict(machine=machine, image=row["image"], split=row["split"], condition=label,
                                       classification_score_unit="0_to_1",
                                       variant=v, detection=d, classification=c, **metric))
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (args.output / "per-image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "config/models.local.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure", action="store_true", help="Run actual models once; otherwise analyze the existing cache")
    args = parser.parse_args()
    (args.output / "cache").mkdir(parents=True, exist_ok=True)
    if args.measure:
        measure(args)
    summarize(args)


if __name__ == "__main__":
    main()
