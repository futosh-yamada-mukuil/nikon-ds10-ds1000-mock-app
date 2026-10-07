import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def main():
    parser = argparse.ArgumentParser(description="DS10 / DS1000 desktop mock app")
    parser.add_argument("--input", type=Path, help="Local image or video file")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--screenshot", type=Path, help="Render the initial UI, save it and exit (no inference)")
    parser.add_argument("--verify-output", type=Path, help="Run real model/GUI/export verification and exit")
    args = parser.parse_args()
    if args.verify_output and (not args.input or args.screenshot):
        parser.error("--verify-output requires --input and cannot be combined with --screenshot")
    from .ui import MainWindow

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Nikon Cell Analysis Mock")
    application.setOrganizationName("MUKUiL")
    window = MainWindow(args.config, args.device, autoload=args.screenshot is None)
    window.show()
    if args.input:
        window.open_source(args.input)
    if args.verify_output:
        from .verification import schedule_verification
        schedule_verification(window, application, args.input, args.verify_output)
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            success = window.grab().save(str(args.screenshot))
            application.exit(0 if success else 1)
        QTimer.singleShot(250, capture)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
