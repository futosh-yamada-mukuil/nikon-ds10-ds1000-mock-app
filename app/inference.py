"""Local RF-DETR and ResNet101 inference, preserving the reference app's inputs.

The two supplied checkpoints are identified by content, never by a recent filename.
RF-DETR 1.5.0 performs its own unrestricted pickle load, so only the exact supplied
detector is passed to that library after SHA256 verification and a safe read.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# All runtime model files are local; a missing cache must not initiate a download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image

from .domain import CellResult, FrameResult, InferenceSettings
from .calibration import CALIBRATION_SHA256


DETECTOR_SHA256 = "b7a6beaacdf36f44efbe2c42afa4d9ad686fc08d8e03a299b5f294675513da04"
CLASSIFIER_SHA256 = "abad46a8728ad29b3e05b377916b842e059fad42151e2bddb8871b1db7edfe4c"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelError(RuntimeError):
    """Model absence, identity mismatch or a failed real model operation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, expected: str) -> str:
    if not path.is_file():
        raise ModelError(f"モデルファイルがありません: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ModelError(f"今回指定されたモデルと内容が異なります: {path.name}\nSHA256: {actual}")
    return actual


def strip_state_dict_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = list(state)
    for prefix in ("module.", "model.", "backbone."):
        if keys and all(key.startswith(prefix) for key in keys):
            return {key[len(prefix):]: value for key, value in state.items()}
    return dict(state)


def build_classifier(state: Mapping[str, Any]):
    import torch.nn as nn
    from torchvision.models import resnet101

    state = strip_state_dict_prefix(state)
    if "fc.1.weight" not in state or "fc.5.weight" not in state:
        raise ModelError("提供されたResNet101の分類層が見つかりません。")
    hidden, input_features = state["fc.1.weight"].shape
    classes, output_hidden = state["fc.5.weight"].shape
    if input_features != 2048 or hidden != output_hidden or classes != 2:
        raise ModelError("提供されたResNet101の分類層の形状が一致しません。")
    model = resnet101(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(input_features, hidden), nn.ReLU(inplace=True),
        nn.BatchNorm1d(hidden), nn.Dropout(p=0.5), nn.Linear(hidden, classes),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def choose_device(requested: str = "auto") -> str:
    import torch

    if os.getenv("FORCE_CPU", "false").lower() == "true":
        return "cpu"
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if mps else "cpu")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"対応していないデバイスです: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelError("CUDAを指定しましたが、この環境では利用できません。")
    if requested == "mps" and not mps:
        raise ModelError("MPSを指定しましたが、この環境では利用できません。")
    return requested


def crop_and_resize(image: Image.Image, box) -> tuple[Image.Image | None, tuple[int, int, int, int] | None]:
    """Reference contract: zero padding, minimum 4 px, bilinear 224×224."""
    width, height = image.size
    x1, y1, x2, y2 = [int(value) for value in box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # PIL crop uses an exclusive right/bottom edge, so valid endpoints include
    # the image width and height themselves.
    x1, x2 = [max(0, min(x, width)) for x in (x1, x2)]
    y1, y2 = [max(0, min(y, height)) for y in (y1, y2)]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None
    clipped = (x1, y1, x2, y2)
    return image.crop(clipped).resize((224, 224), Image.Resampling.BILINEAR), clipped


class Engine:
    def __init__(self, detector_path: Path, classifier_path: Path, device: str = "auto",
                 log: Callable[[str], None] | None = None, calibration_path: Path | None = None) -> None:
        self.detector_path = Path(detector_path).expanduser().resolve()
        self.classifier_path = Path(classifier_path).expanduser().resolve()
        self.requested_device = device
        self.device = "cpu"
        self.log = log or (lambda message: None)
        self.loaded = False
        self._detector = None
        self._classifier = None
        self._transform = None
        self._hashes: dict[str, str] = {}
        self.calibration_path = calibration_path
        self.calibration = None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            self.log("提供モデルのSHA256を確認しています。")
            self._hashes = {
                "detector": verify_model(self.detector_path, DETECTOR_SHA256),
                "classifier": verify_model(self.classifier_path, CLASSIFIER_SHA256),
            }
            import torch
            import torchvision.transforms as transforms
            from rfdetr import RFDETRBase

            self.device = choose_device(self.requested_device)
            self.log(f"物体検知モデルを読み込んでいます ({self.device})。")
            with torch.serialization.safe_globals([argparse.Namespace]):
                detector_checkpoint = torch.load(self.detector_path, map_location="cpu", weights_only=True)
            state = detector_checkpoint["model"]
            classes = int(state["class_embed.bias"].shape[0]) - 1
            resolution = int(getattr(detector_checkpoint["args"], "resolution"))
            if classes != 1 or resolution != 784:
                raise ModelError("提供されたRF-DETRの構成が一致しません。")
            detector = RFDETRBase(device=self.device, num_classes=classes, resolution=resolution,
                                 pretrain_weights=str(self.detector_path))
            detector.model.model.eval()
            del detector_checkpoint
            self.log("細胞分類モデルを読み込んでいます。")
            classifier_state = torch.load(self.classifier_path, map_location="cpu", weights_only=True)
            classifier = build_classifier(classifier_state).to(self.device)
            self._detector = detector
            self._classifier = classifier
            self._transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            self.loaded = True
            if self.calibration_path and self.calibration_path.is_file():
                from .calibration import load_calibration
                try:
                    self.calibration = load_calibration(self.calibration_path, self._hashes["detector"], self._hashes["classifier"])
                    self.log("DS1000の分類補正を検証しました（適用は改善設定ボタンから）。")
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    self.log(f"分類補正は使用できません。標準モデルを使用します: {exc}")
            self.log(f"モデル読み込み完了 ({self.device})。")
        except Exception as exc:
            self.close()
            if isinstance(exc, ModelError):
                raise
            raise ModelError(f"モデル読み込みに失敗しました: {exc}") from exc

    def _classify(self, image: Image.Image) -> float:
        import torch

        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._classifier(tensor)
            if output.ndim != 2 or output.shape != (1, 2):
                raise ModelError("分類モデルの出力形状が一致しません。")
            score = float(torch.softmax(output, dim=-1)[0, 1].item() * 100.0)
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ModelError("分類モデルから不正なスコアが返されました。")
        return score

    def _classify_calibrated(self, image: Image.Image, machine: str) -> float:
        import numpy as np
        import torch
        from .calibration import score_features

        if not self.calibration or machine not in self.calibration["machines"]:
            raise ModelError("この機種の検証済み分類補正がありません。")
        captured = []
        hook = self._classifier.fc[5].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy()))
        try:
            with torch.no_grad():
                self._classifier(self._transform(image).unsqueeze(0).to(self.device))
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (1, 128) or not np.isfinite(captured[0]).all():
            raise ModelError("分類補正の特徴量が不正です。")
        score = float(score_features(captured[0], self.calibration["machines"][machine])[0])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ModelError("分類補正のスコアが不正です。")
        return score

    def _detect(self, image: Image.Image, threshold: float):
        import cv2
        import numpy as np
        import torch

        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        scale_x = scale_y = 1.0
        if max(height, width) > 640:
            scale = 640 / float(max(height, width))
            resized_width = max(1, int(width * scale))
            resized_height = max(1, int(height * scale))
            scale_x = resized_width / float(width)
            scale_y = resized_height / float(height)
            bgr = cv2.resize(bgr, (resized_width, resized_height),
                             interpolation=cv2.INTER_AREA)
        resized = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            detected = self._detector.predict(resized, threshold=0.0)
        if not hasattr(detected, "xyxy") or detected.xyxy is None:
            raise ModelError("検出モデルの出力形式が一致しません。")
        boxes = np.asarray(detected.xyxy, dtype=float).copy()
        if len(boxes) == 0:
            return []
        confidence = np.asarray(detected.confidence, dtype=float).reshape(-1)
        if boxes.shape != (len(confidence), 4) or not np.isfinite(boxes).all():
            raise ModelError("検出モデルから不正な座標が返されました。")
        # Preserve the existing app's NaN confidence sanitization.
        confidence = np.nan_to_num(confidence, nan=0.0)
        if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise ModelError("検出モデルから不正な信頼度が返されました。")
        if scale_x != 1 or scale_y != 1:
            boxes[:, (0, 2)] /= scale_x
            boxes[:, (1, 3)] /= scale_y
        return [(tuple(int(value) for value in box), float(score))
                for box, score in zip(boxes, confidence) if score >= threshold]

    def analyze(self, image: Image.Image, settings: InferenceSettings) -> FrameResult:
        if not self.loaded:
            raise ModelError("モデルを読み込んでから解析してください。")
        started = time.perf_counter()
        image = image.convert("RGB")
        cells: list[CellResult] = []
        skipped = 0
        try:
            if settings.mode == "classification_only":
                score = self._classify(image.resize((224, 224), Image.Resampling.BILINEAR))
                cells.append(CellResult(1, (0, 0, image.width, image.height), None, score,
                                        score > settings.classification_threshold))
            else:
                for index, (box, confidence) in enumerate(self._detect(image, settings.detection_threshold), 1):
                    crop, clipped = crop_and_resize(image, box)
                    if crop is None:
                        skipped += 1
                        continue
                    score = (self._classify_calibrated(crop, settings.machine) if settings.use_calibration
                             else self._classify(crop)) if settings.mode == "detection_classification" else None
                    candidate = score is not None and score > settings.classification_threshold
                    cells.append(CellResult(index, clipped, confidence, score, candidate))
        except Exception as exc:
            if isinstance(exc, ModelError):
                raise
            raise ModelError(f"解析に失敗しました ({self.device}): {exc}") from exc
        return FrameResult(cells, skipped, time.perf_counter() - started, self.device)

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "detector": {"file": self.detector_path.name, "sha256": self._hashes.get("detector"),
                         "expected_sha256": DETECTOR_SHA256, "architecture": "RFDETRBase",
                         "resolution": 784, "classes": 1},
            "classifier": {"file": self.classifier_path.name, "sha256": self._hashes.get("classifier"),
                           "expected_sha256": CLASSIFIER_SHA256, "architecture": "ResNet101",
                           "classes": 2, "score_class_index": 1,
                           "class_mapping_verified": False},
            "device": self.device, "loaded": self.loaded,
            "preprocessing": {"detector_long_side": 640, "crop_size": [224, 224],
                              "crop_padding_ratio": 0.0, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "classification_score": "softmax class 1 × 100; class meaning not independently verified",
            "calibration": {"available": self.calibration is not None,
                            "sha256": CALIBRATION_SHA256 if self.calibration else None,
                            "score_definition": "local frozen-feature linear head sigmoid × 100 when use_calibration=true",
                            "training_membership": "unknown for original model; local head fitted on calibration split only"},
        }

    def close(self) -> None:
        self.loaded = False
        self._detector = self._classifier = self._transform = None
        self.calibration = None
