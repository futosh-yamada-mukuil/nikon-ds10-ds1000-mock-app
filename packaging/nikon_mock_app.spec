# Build this specification on the target OS with its release environment.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project = Path(SPECPATH).parent
datas = [(str(project / "config" / "models.example.json"), "config")]
# RF-DETR's @torch.jit.script functions need inspectable Python source files.
datas += collect_data_files("rfdetr", include_py_files=True)
datas += collect_data_files("transformers")
hiddenimports = collect_submodules("rfdetr")
hiddenimports += ["app", "transformers", "torchvision.models.resnet"]

a = Analysis(
    [str(project / "packaging" / "entrypoint.py")],
    pathex=[str(project)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NikonMockApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="NikonMockApp")
if __import__("sys").platform == "darwin":
    app = BUNDLE(
        coll,
        name="NikonMockApp.app",
        bundle_identifier="jp.mukuil.nikon-mock-app",
    )
