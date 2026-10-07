"""Check the setup target before creating or changing its virtual environment."""

import argparse
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("windows", "macos"))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.exit(1, "Python 3.12 is required.\n")
    target = (platform.system(), platform.machine().lower())
    allowed = {"windows": {("Windows", "amd64"), ("Windows", "x86_64")},
               "macos": {("Darwin", "arm64")}}
    if target not in allowed[args.target]:
        parser.exit(1, "Use Windows x64 or Apple Silicon macOS. Other targets require a separately validated dependency set.\n")
    print(f"Setup target verified: {platform.system()} {platform.machine()} Python {platform.python_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
