"""Small, local-only helpers for measurement exports."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform


def check_output_paths(*paths: Path) -> None:
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("出力先は別々のファイルを指定してください。")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"既存ファイルは上書きしません: {path}")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"出力先の親がフォルダーではありません: {parent}")


def write_text_outputs(outputs: dict[Path, str]) -> None:
    """Exclusive creation; clean up only files created by this invocation."""
    check_output_paths(*outputs)
    created = []
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as stream:
                created.append(path)
                stream.write(content)
    except Exception:
        for path in created:
            path.unlink()
        raise


def runtime_metadata() -> dict:
    packages = {}
    for name in ("Pillow", "numpy", "torch", "torchvision", "rfdetr", "opencv-python-headless"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "os": platform.system(),
            "architecture": platform.machine(), "packages": packages}
