"""Values shared by the UI, workers and the real inference service."""
from __future__ import annotations

from dataclasses import dataclass
import math


MODES = ("detection_classification", "detection_only", "classification_only")


@dataclass(frozen=True)
class InferenceSettings:
    detection_threshold: float = 0.5
    classification_threshold: float = 0.5
    machine: str = "DS10"
    mode: str = "detection_classification"

    def __post_init__(self) -> None:
        if not math.isfinite(self.detection_threshold) or not 0 <= self.detection_threshold <= 1:
            raise ValueError("検出しきい値は0〜1で指定してください。")
        if not math.isfinite(self.classification_threshold) or not 0 <= self.classification_threshold <= 100:
            raise ValueError("分類しきい値は0〜100で指定してください。")
        if self.machine not in ("DS10", "DS1000"):
            raise ValueError("機種はDS10またはDS1000を指定してください。")
        if self.mode not in MODES:
            raise ValueError(f"対応していない解析モードです: {self.mode}")


@dataclass(frozen=True)
class CellResult:
    id: int
    box: tuple[int, int, int, int]
    confidence: float | None
    score: float | None
    candidate: bool


@dataclass
class FrameResult:
    cells: list[CellResult]
    skipped: int = 0
    elapsed_seconds: float = 0.0
    device: str = "cpu"

    @property
    def detection_count(self) -> int:
        """Includes detections whose crop was too small, as in the reference app."""
        return len(self.cells) + self.skipped

    @property
    def candidate_count(self) -> int:
        return sum(cell.candidate for cell in self.cells)
