"""Opt-in real model and programmatic GUI verification; no accuracy benchmark."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from app.ui import MainWindow
from app.verification import schedule_verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    application = QApplication(sys.argv[:1])
    window = MainWindow(args.config, args.device)
    window.show()
    window.open_source(args.input)
    schedule_verification(window, application, args.input, args.output)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
