"""Local result rendering and reproducible exports; source files are never changed."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .domain import FrameResult, InferenceSettings


CANDIDATE_COLOR = "#ef4444"
DETECTION_COLOR = "#8aa4bc"
CSV_COLUMNS = ("cell_id", "x1", "y1", "x2", "y2", "detection_confidence", "classification_score_0_to_1", "candidate")


def render_result(
    image: Image.Image,
    result: FrameResult,
    show_all: bool = True,
    show_labels: bool = True,
    grayscale: bool = False,
    brightness: int = 0,
    *,
    box_color: str | None = None,
) -> Image.Image:
    """Draw a separate RGB preview; appearance options do not change inference.

    ``show_all=False`` hides classified cells below the threshold. Detection-only
    boxes remain visible because those cells do not have classification scores.
    ``brightness`` is a display-only RGB offset in the range -255 to 255.
    ``box_color`` can give all visible boxes the same display color.
    """
    rendered = image.convert("RGB").copy()
    if grayscale:
        rendered = ImageOps.grayscale(rendered).convert("RGB")
    if brightness:
        offset = max(-255, min(255, int(brightness)))
        lookup = [max(0, min(255, value + offset)) for value in range(256)]
        rendered = rendered.point(lookup * 3)

    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) / 700))
    try:
        font = ImageFont.load_default(size=max(12, line_width * 5))
    except TypeError:  # Pillow before the size parameter was added.
        font = ImageFont.load_default()

    for cell in result.cells:
        if cell.score is not None and not cell.candidate and not show_all:
            continue
        x1, y1, x2, y2 = cell.box
        box = (
            max(0, min(rendered.width - 1, x1)),
            max(0, min(rendered.height - 1, y1)),
            max(0, min(rendered.width - 1, x2)),
            max(0, min(rendered.height - 1, y2)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        color = box_color or (CANDIDATE_COLOR if cell.candidate else DETECTION_COLOR)
        draw.rectangle(box, outline=color, width=line_width)
        if show_labels:
            parts = [f"#{cell.id}"]
            if cell.confidence is not None:
                parts.append(f"det:{cell.confidence:.2f}")
            if cell.score is not None:
                parts.append(f"score:{cell.score:.2f}")
            label = " ".join(parts)
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_x = max(0, min(box[0], rendered.width - text_width - 4))
            text_y = max(0, box[1] - text_height - 6)
            draw.rectangle(
                (text_x, text_y, min(rendered.width - 1, text_x + text_width + 4), text_y + text_height + 4),
                fill=color,
            )
            draw.text((text_x + 2, text_y + 2 - text_box[1]), label, fill="#ffffff", font=font)
    return rendered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("PySide6", "Pillow", "numpy", "torch", "torchvision", "rfdetr", "transformers", "opencv-python", "opencv-python-headless"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "packages": packages,
    }


def _cell_row(cell: Any) -> list[Any]:
    return [cell.id, *cell.box, cell.confidence, cell.score, str(cell.candidate).lower()]


def _frame_record(index: int, result: FrameResult, settings: InferenceSettings) -> dict[str, Any]:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("フレーム番号は0以上の整数で指定してください。")
    if not math.isfinite(result.elapsed_seconds) or result.elapsed_seconds < 0:
        raise ValueError("処理時間が不正です。")
    for cell in result.cells:
        if cell.score is not None and (not math.isfinite(cell.score) or not 0 <= cell.score <= 1):
            raise ValueError("分類スコアが不正です。")
        if cell.confidence is not None and (not math.isfinite(cell.confidence) or not 0 <= cell.confidence <= 1):
            raise ValueError("検出信頼度が不正です。")
        expected_candidate = cell.score is not None and cell.score > settings.classification_threshold
        if cell.candidate != expected_candidate:
            raise ValueError("分類スコアと閾値超え判定が一致しません。")
    return {
        "frame_index": index,
        "detection_count": result.detection_count,
        "candidate_count": result.candidate_count,
        "skipped": result.skipped,
        "elapsed_seconds": result.elapsed_seconds,
        "device": result.device,
        "cells": [asdict(cell) for cell in result.cells],
    }


def export_bundle(
    output_dir: Path,
    source: Path,
    frame_index: int,
    image: Image.Image,
    result: FrameResult,
    settings: InferenceSettings,
    model_metadata: dict[str, Any],
    rendered_image: Image.Image | None = None,
    history: list[tuple[int, FrameResult]] | None = None,
    display_settings: dict[str, Any] | None = None,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save one complete bundle by a directory rename, without overwriting files.

    An existing empty output directory is accepted. Existing results or any
    other files in the target directory cause an error. Video history contains
    only frames that were actually processed, including frames with zero cells.
    CSV and JSON retain unrounded scores; rounding is for preview labels only.
    Display settings and input/job metadata are explicit snapshots supplied by
    the UI, including video stride and completed/cancelled/error status.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("入力元はファイルを指定してください。")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"保存先が空ではありません: {output_dir}")
    preview = rendered_image if rendered_image is not None else render_result(image, result)
    if preview.size != image.size:
        raise ValueError("保存する結果画像と入力画像のサイズが一致しません。")

    displayed = _frame_record(frame_index, result, settings)
    frames = list(history) if history is not None else [(frame_index, result)]
    indices = [index for index, _ in frames]
    if len(indices) != len(set(indices)):
        raise ValueError("解析履歴のフレーム番号が重複しています。")
    if frame_index not in indices:
        frames.append((frame_index, result))
    frames.sort(key=lambda item: item[0])
    records = [_frame_record(index, frame_result, settings) for index, frame_result in frames]
    source_stat = source.stat()
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "size_bytes": source_stat.st_size,
            "modified_at_ns": source_stat.st_mtime_ns,
        },
        "image": {"width": image.width, "height": image.height, "mode": "RGB"},
        "settings": asdict(settings),
        "display_settings": display_settings,
        "input_metadata": input_metadata,
        "models": model_metadata,
        "environment": _environment(),
        "displayed_frame": displayed,
        "analyzed_frames": records,
        "classification": {
            "score_unit": "0_to_1",
            "threshold_unit": "0_to_1",
            "score_definition": ("local frozen-feature linear head sigmoid"
                                 if settings.use_calibration else "softmax class index 1 probability"),
            "candidate_rule": "classification_score_0_to_1 > classification_threshold (strict; unrounded)",
            "class1_label_meaning": ("positive under supplied red annotations; unmarked ROIs assumed negative"
                                    if settings.use_calibration else "unverified"),
            "medical_interpretation": "not established; score is not a measured DFI percentage",
        },
        "evaluation": {"ground_truth_compared": False, "accuracy_metrics": None},
        "exports": {"results_csv_frame": frame_index, "annotated_image_frame": frame_index},
    }
    # Validate all metadata before creating any result files.
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nikon-export-", dir=output_dir.parent))
    names = {"results_csv": "results.csv", "annotated_image": "annotated.png", "manifest": "manifest.json"}
    if history is not None:
        names["video_results_csv"] = "video_results.csv"
    try:
        with (staging / names["results_csv"]).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(_cell_row(cell) for cell in result.cells)
        if history is not None:
            with (staging / names["video_results_csv"]).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("frame_index", *CSV_COLUMNS))
                for index, frame_result in frames:
                    writer.writerows([index, *_cell_row(cell)] for cell in frame_result.cells)
        preview.convert("RGB").save(staging / names["annotated_image"], format="PNG")
        (staging / names["manifest"]).write_text(manifest_text, encoding="utf-8")
        # rmdir fails if another process added a file; rename never replaces a
        # nonempty directory. No per-file replacement can leave a mixed bundle.
        if output_dir.exists():
            output_dir.rmdir()
        os.rename(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {key: output_dir / name for key, name in names.items()}
