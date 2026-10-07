"""Opt-in real GUI verification, also available in the packaged application."""
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

from PySide6.QtCore import QTimer


def schedule_verification(window, application, input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {"phase": "loading", "heartbeats": 0}
    expected = None

    def poll():
        nonlocal expected
        try:
            if time.monotonic() - started > 180:
                raise TimeoutError("Integration validation exceeded 180 seconds")
            if state["phase"] == "loading" and window.loader is None:
                if window.engine is None or window.media is None:
                    raise RuntimeError(window.log.toPlainText())
                expected = window.engine.analyze(window.original_image, window.current_settings())
                # Videos are sampled at start/middle/end for this smoke check.
                if window.media.is_video:
                    window.stride.setValue(max(1, math.ceil(window.media.frame_count / 3)))
                window.start_button.click()
                state["phase"] = "analyzing"
            elif state["phase"] == "analyzing":
                state["heartbeats"] += 1
                if window.worker is not None:
                    return
                if window.result is None or window.run_metadata["stop_reason"] != "completed":
                    raise RuntimeError(window.log.toPlainText())
                first = window.history[0][1]
                assert first.detection_count == expected.detection_count
                assert first.candidate_count == expected.candidate_count
                assert [cell.box for cell in first.cells] == [cell.box for cell in expected.cells]
                score_difference = max((abs(a.score - b.score) for a, b in zip(first.cells, expected.cells)
                                        if a.score is not None and b.score is not None), default=0)
                assert score_difference < 0.001, score_difference
                with patch.object(window, "export_directory", output_dir), patch.object(window, "download_opener", return_value=True):
                    window.export_button.click()
                manifests = list(output_dir.glob("*/manifest.json"))
                assert manifests, "GUI export did not create a manifest"
                window.grab().save(str(output_dir / "ui_result.png"))
                report = {"source": str(input_path.resolve()), "device": window.engine.device,
                          "model_metadata": window.engine.get_model_metadata(),
                          "gui_detections_first_frame": first.detection_count,
                          "gui_candidates_first_frame": first.candidate_count,
                          "same_engine_direct_gui_counts_and_boxes_equal": True,
                          "max_class_score_difference": score_difference,
                          "gui_timer_ticks_during_worker": state["heartbeats"],
                          "processed_frames": [index for index, _ in window.history],
                          "elapsed_seconds": time.monotonic() - started,
                          "scope": "programmatic Qt GUI workflow; no ground-truth evaluation"}
                (output_dir / "smoke_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (output_dir / "gui_log.txt").write_text(window.log.toPlainText(), encoding="utf-8")
                print(json.dumps(report, ensure_ascii=False), flush=True)
                timer.stop()
                window.close()
                application.exit(0)
        except Exception as error:
            timer.stop()
            print(f"Integration validation failed: {error}", file=sys.stderr, flush=True)
            window.close()
            application.exit(1)

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    window.verification_timer = timer
