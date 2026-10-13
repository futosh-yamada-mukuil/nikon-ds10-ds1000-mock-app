"""Update explicitly named Python source records; never inspect model weights."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_sbom import file_component  # noqa: E402


def update_sources(sbom_path: Path, sources: list[Path]) -> int:
    original = sbom_path.read_bytes()
    before = json.loads(original)
    after = deepcopy(before)
    names = set()
    replacements = {}
    for source in sources:
        path = source.resolve(strict=True)
        if (path.suffix != ".py" or not any(path.is_relative_to(ROOT / folder) for folder in ("app", "scripts"))
                or path.name in names):
            raise ValueError("重複しないapp/またはscripts/内のPythonソースのみ指定してください。")
        names.add(path.name)
        replacements[path.name] = file_component("source", path)
    if not replacements:
        raise ValueError("更新対象ソースを指定してください。")

    def target(item):
        return (item.get("type") == "file" and item.get("name") in names and
                {"name": "nikon:file-role", "value": "source"} in item.get("properties", []))

    matched = set()
    for index, item in enumerate(after["components"]):
        if target(item):
            if item["name"] in matched:
                raise ValueError("既存SBOMのソース名が重複しています。")
            matched.add(item["name"])
            after["components"][index] = replacements[item["name"]]
    after["components"].extend(replacements[name] for name in sorted(names - matched))
    # Compare against the actual pre-update snapshot, never HEAD or a rebuilt
    # environment inventory. All other records and top-level metadata stay exact.
    if ([item for item in before["components"] if not target(item)] !=
            [item for item in after["components"] if not target(item)]):
        raise ValueError("対象外のSBOMレコードが変化しました。")
    if ({key: value for key, value in before.items() if key != "components"} !=
            {key: value for key, value in after.items() if key != "components"}):
        raise ValueError("SBOMのメタデータが変化しました。")
    text = json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=sbom_path.parent,
                                         prefix=".sbom-source-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        if sbom_path.read_bytes() != original:
            raise ValueError("SBOMが並行して変更されたため更新を中止しました。")
        temporary.chmod(sbom_path.stat().st_mode & 0o777)
        os.replace(temporary, sbom_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    if json.loads(sbom_path.read_bytes()) != after:
        raise ValueError("SBOMの保存後確認に失敗しました。")
    return len(replacements)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbom", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    args = parser.parse_args(argv)
    try:
        count = update_sources(args.sbom, args.source)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"ソース{count}件を更新。対象外レコードを保持しました。モデルファイルの読込なし。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
