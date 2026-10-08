"""Optional local DS1000 head, pinned to the measured artifact and base models."""
import hashlib
import json
from pathlib import Path

import numpy as np

CALIBRATION_SHA256 = "43c3489c9c1b5d7a654d36c4d022e67f7568f64d68fc15a23ec9e15aa5acc01f"


def load_calibration(path: Path, detector_sha256: str, classifier_sha256: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CALIBRATION_SHA256:
        raise ValueError("分類補正ファイルのSHA256が検証済みの内容と一致しません。")
    data = json.loads(raw)
    if (data.get("schema_version") != 1 or data.get("feature_size") != 128 or
            data.get("detector_sha256") != detector_sha256 or data.get("classifier_sha256") != classifier_sha256):
        raise ValueError("分類補正と元モデルの組み合わせが一致しません。")
    if set(data["machines"]) != {"DS1000"}:
        raise ValueError("検証済みの分類補正はDS1000専用です。")
    head = data["machines"]["DS1000"]
    for key in ("mean", "std", "weight"):
        values = np.asarray(head[key], dtype=float)
        if values.shape != (128,) or not np.isfinite(values).all():
            raise ValueError("分類補正の係数が不正です。")
    if (not head["eligible_for_app"] or (np.asarray(head["std"]) <= 0).any() or
            not np.isfinite(head["bias"]) or not 0 <= head["classification_threshold"] <= 100 or
            not 0 <= head["detection_threshold"] <= 1):
        raise ValueError("分類補正の条件が不正です。")
    return data


def score_features(matrix: np.ndarray, head: dict) -> np.ndarray:
    normalized = np.clip((matrix - np.asarray(head["mean"])) / np.asarray(head["std"]), -6, 6)
    logits = normalized @ np.asarray(head["weight"]) + head["bias"]
    return 100 / (1 + np.exp(-np.clip(logits, -60, 60)))
