"""Experimental local tiled inference using the existing Engine detector.

Not connected to GUI defaults. Model configuration and classifier inputs are
unchanged. NMS is applied only between different tiles of the one-class model.
"""
from __future__ import annotations

import math

from PIL import Image

from .inference import Engine


def tile_boxes(width: int, height: int, tile_size: int = 640, overlap: float = 0.25):
    if type(tile_size) is not int or tile_size < 4:
        raise ValueError("タイルサイズは4以上の整数で指定してください。")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("重なり率は0以上1未満です。")
    if width <= 0 or height <= 0:
        raise ValueError("画像の幅・高さは正である必要があります。")
    step = max(1, int(tile_size * (1 - overlap)))

    def starts(length):
        end = max(0, length - tile_size)
        positions = list(range(0, end + 1, step))
        if positions[-1] != end:
            positions.append(end)
        return positions

    for y in starts(height):
        for x in starts(width):
            yield x, y, min(x + tile_size, width), min(y + tile_size, height)


def box_iou(first, second) -> float:
    intersection = (max(0, min(first[2], second[2]) - max(first[0], second[0])) *
                    max(0, min(first[3], second[3]) - max(first[1], second[1])))
    area1 = (first[2] - first[0]) * (first[3] - first[1])
    area2 = (second[2] - second[0]) * (second[3] - second[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def suppress_tile_duplicates(detections, iou_threshold: float = 0.5):
    """Input/output: (exclusive xyxy, confidence, tile index), stable tie order."""
    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("NMSのIoUは0〜1です。")
    kept = []
    for item in sorted(detections, key=lambda detection: -detection[1]):
        if not any(item[2] != other[2] and box_iou(item[0], other[0]) > iou_threshold
                   for other in kept):
            kept.append(item)
    return kept


def detect_tiled(engine: Engine, image: Image.Image, threshold: float = 0.5, *,
                 tile_size: int = 640, overlap: float = 0.25, nms_iou: float = 0.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("検出しきい値は0〜1です。")
    if not math.isfinite(nms_iou) or not 0 <= nms_iou <= 1:
        raise ValueError("NMSのIoUは0〜1です。")
    detections = []
    for tile_index, (left, top, right, bottom) in enumerate(
            tile_boxes(image.width, image.height, tile_size, overlap)):
        tile = image.crop((left, top, right, bottom))
        for box, confidence in engine._detect(tile, threshold):
            # Clip in tile coordinates first: detections outside a tile must not
            # become invented boxes in its neighbour after adding the offset.
            x1, x2 = sorted((box[0], box[2]))
            y1, y2 = sorted((box[1], box[3]))
            x1, x2 = (max(0, min(tile.width, x)) for x in (x1, x2))
            y1, y2 = (max(0, min(tile.height, y)) for y in (y1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(((left + x1, top + y1, left + x2, top + y2),
                               confidence, tile_index))
    return [(box, confidence) for box, confidence, _ in
            suppress_tile_duplicates(detections, nms_iou)]
