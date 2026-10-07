"""File-only image/video input; inference and camera access are separate."""
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path

from PIL import Image, UnidentifiedImageError


VIDEO_SUFFIXES = frozenset((".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"))
IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))


class MediaError(RuntimeError):
    pass


class MediaSource:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._capture = None
        self._image = None
        self._closed = False
        self._next_index = 0
        if not self.path.is_file():
            raise MediaError(f"入力ファイルがありません: {self.path}")
        suffix = self.path.suffix.lower()
        self.is_video = suffix in VIDEO_SUFFIXES
        if not self.is_video and suffix not in IMAGE_SUFFIXES:
            raise MediaError("対応する画像または動画ファイルを選択してください。")
        if self.is_video:
            import cv2

            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                capture.release()
                raise MediaError(f"動画を開けません: {self.path.name}")
            self._capture = capture
            self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.frame_count < 1 or not math.isfinite(self.fps) or self.fps <= 0 or self.width < 1 or self.height < 1:
                self.close()
                raise MediaError("動画のフレーム数・FPS・サイズを読み取れません。")
        else:
            try:
                with Image.open(self.path) as image:
                    self._image = image.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise MediaError(f"画像を開けません: {self.path.name}") from exc
            self.frame_count, self.fps = 1, 0.0
            self.width, self.height = self._image.size
        self.duration = self.frame_count / self.fps if self.is_video else 0.0

    def read(self, index: int = 0) -> Image.Image:
        if self._closed:
            raise MediaError("入力ファイルは閉じられています。")
        if not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < self.frame_count:
            raise IndexError(f"フレーム番号は0〜{self.frame_count - 1}で指定してください。")
        if not self.is_video:
            return self._image.copy()
        import cv2

        if index != self._next_index:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                raise MediaError(f"フレーム{index}へ移動できません。")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise MediaError(f"フレーム{index}を読み取れません。")
        self._next_index = int(index) + 1
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._image = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
