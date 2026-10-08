"""Fit one small, machine-specific linear head on frozen ResNet features.

Original model weights stay unchanged. Only calibration-split images fit the
head/threshold; held-out images gate adoption. Unmarked ROIs are assumed negative
under the supplied positive annotations; near-annotation ambiguous ROIs are
excluded from training. This is internal validation, not a clinical estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import Engine, crop_and_resize, file_sha256
from app.calibration import score_features
from scripts.evaluate_annotations import aggregate, overlap_edges


def features(args, rows):
    import torch

    config = json.loads(args.model_config.read_text())
    engine = Engine(Path(config["detection"]["path"]), Path(config["classification"]["path"]), log=print)
    engine.load()
    captured = []
    hook = engine._classifier.fc[5].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy()))
    try:
        for index, row in enumerate(rows, 1):
            path = args.output / "features" / f"{row['machine']}_{Path(row['image']).stem}.npy"
            if path.exists():
                continue
            if file_sha256(Path(row["source"])) != row["source_sha256"]:
                raise ValueError("Source image changed since baseline measurement")
            with Image.open(row["source"]) as f:
                image = f.convert("RGB")
            cells = row["variants"]["legacy640"]
            crops = [crop_and_resize(image, c["box"])[0] for c in cells]
            captured.clear()
            for offset in range(0, len(crops), 16):
                tensor = torch.stack([engine._transform(c) for c in crops[offset:offset + 16]]).to(engine.device)
                with torch.no_grad():
                    engine._classifier(tensor)
            matrix = np.concatenate(captured) if captured else np.zeros((0, 128), dtype=np.float32)
            if matrix.shape != (len(cells), 128) or not np.isfinite(matrix).all():
                raise ValueError("Invalid frozen-feature matrix")
            np.save(path, matrix, allow_pickle=False)
            print(f"features {index}/{len(rows)} {row['image']}", flush=True)
    finally:
        hook.remove()
        engine.close()


def fit_head(x, y):
    import torch

    if not y.any() or y.all():
        raise ValueError("Both positive and negative calibration ROIs are required")
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
    tensor = torch.tensor(np.clip((x - mean) / std, -6, 6), dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=.03)
    positive_weight = torch.tensor(float((y == 0).sum() / y.sum()))
    # Fixed settings and one fit per machine; no hyperparameter search.
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(tensor @ weight + bias, labels,
                                                                    pos_weight=positive_weight)
        loss = loss + .01 * weight.square().sum()
        loss.backward()
        optimizer.step()
    return {"mean": mean.tolist(), "std": std.tolist(), "weight": weight.detach().tolist(),
            "bias": bias.item(), "training_rois": len(y), "positive_rois": int(y.sum()),
            "negative_rois": int((y == 0).sum())}


def score_head(matrix, head):
    return score_features(matrix, head)


def fit_and_evaluate(args, rows):
    metadata = json.loads((args.output / "manifest.json").read_text())
    result = {"schema_version": 1, "detector_sha256": metadata["models"]["detector"]["sha256"],
              "classifier_sha256": metadata["models"]["classifier"]["sha256"], "machines": {},
              "feature_layer": "fc.5 input", "feature_size": 128, "score_scale": "sigmoid linear head x 100",
              "training": {"steps": 300, "optimizer": "Adam", "learning_rate": .03, "l2": .01,
                  "near_truth_negatives_excluded": True, "unmarked_rois_assumed_negative": True,
                  "base_model_training_membership": "unknown", "validation": "internal held-out images"}}
    for machine in ("DS10", "DS1000"):
        calibration = [r for r in rows if r["machine"] == machine and r["split"] == "calibration"]
        holdout = [r for r in rows if r["machine"] == machine and r["split"] == "holdout"]
        x, y = [], []
        for row in calibration:
            cells = row["variants"]["legacy640"]
            matrix = np.load(args.output / "features" / f"{machine}_{Path(row['image']).stem}.npy", allow_pickle=False)
            positives = overlap_edges(cells, row["truth_boxes"])
            for i, cell in enumerate(cells):
                # Avoid interpreting a shifted/partial positive crop as negative.
                a = cell["box"]
                touches = any(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]) for b in row["truth_boxes"])
                if positives[i] or not touches:
                    x.append(matrix[i]); y.append(bool(positives[i]))
        head = fit_head(np.asarray(x), np.asarray(y))
        for row in calibration + holdout:
            matrix = np.load(args.output / "features" / f"{machine}_{Path(row['image']).stem}.npy", allow_pickle=False)
            row["variants"]["calibrated"] = [dict(c, score=float(score)) for c, score in
                                                zip(row["variants"]["legacy640"], score_head(matrix, head))]
        baseline = aggregate(calibration, "legacy640", .5, .5)
        # Prefer recovering positives: calibration recall target is 10 points
        # above baseline, capped by detector coverage at the unchanged .5.
        coverage = aggregate(calibration, "legacy640", .5, None)
        target_tp = min(coverage["tp"], int(np.ceil(baseline["tp"] + .10 * baseline["truth"])))
        trials = []
        for threshold in (0., .5, *range(1, 100)):
            metric = aggregate(calibration, "calibrated", .5, threshold)
            if metric["tp"] >= target_tp:
                trials.append((metric["f1"], -metric["fp"], threshold, metric))
        best = max(trials)
        threshold, cal_result = best[2:]
        before = aggregate(holdout, "legacy640", .5, .5)
        after = aggregate(holdout, "calibrated", .5, threshold)
        eligible = after["fn"] <= before["fn"] and after["fp"] < before["fp"] and after["f1"] > before["f1"]
        result["machines"][machine] = dict(head, detection_threshold=.5, classification_threshold=threshold,
            calibration_result=cal_result, holdout_baseline=before, holdout_calibrated=after, eligible_for_app=eligible)
        print(machine, "eligible", eligible, "threshold", threshold, "before", before, "after", after, flush=True)
    (args.output / "cell-calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory with evaluate_annotations cache")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config/models.local.json")
    args = parser.parse_args()
    (args.output / "features").mkdir(exist_ok=True)
    rows = [json.loads(path.read_text()) for path in sorted((args.output / "cache").glob("*.json"))]
    if not rows or not (args.output / "manifest.json").exists():
        raise ValueError("Complete baseline measurement is required")
    features(args, rows)
    fit_and_evaluate(args, rows)


if __name__ == "__main__":
    main()
