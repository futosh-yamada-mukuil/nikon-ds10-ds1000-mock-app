"""Build on the target OS. No models or local configuration are bundled."""

from __future__ import annotations

import platform
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    if platform.system() not in {"Windows", "Darwin"}:
        raise SystemExit("This distribution profile supports Windows and macOS.")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(project / "dist"), "--workpath", str(project / "build"),
         str(project / "packaging" / "nikon_mock_app.spec")],
        cwd=project,
        check=True,
    )
    app_dir = project / "dist" / "NikonMockApp"
    if platform.system() == "Darwin":
        # Local settings stay outside the signed .app bundle.
        app_dir = project / "dist"
    config_dir = app_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project / "config" / "models.example.json", config_dir / "models.example.json")
    shutil.copy2(project / "docs" / "DISTRIBUTION.md", project / "dist" / "DISTRIBUTION.md")
    print("Build completed. Validate startup, file input, inference and export on this OS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
