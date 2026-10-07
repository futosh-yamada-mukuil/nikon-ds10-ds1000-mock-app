"""Create a deterministic CycloneDX 1.5 inventory using Python's standard library.

This inventories the executing Python environment, not a final frozen executable.
No package location, source URL, environment variables or model contents are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from typing import Iterable
from urllib.parse import quote
import uuid


LICENSE_ALIASES = {
    "mit": "MIT", "mit license": "MIT",
    "apache-2.0": "Apache-2.0", "apache 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause", "bsd-3-clause": "BSD-3-Clause",
    "lgpl-3.0-only": "LGPL-3.0-only", "lgpl-3.0-or-later": "LGPL-3.0-or-later",
    "gpl-3.0-only": "GPL-3.0-only", "gpl-3.0-or-later": "GPL-3.0-or-later",
    "psf-2.0": "PSF-2.0", "python software foundation license": "PSF-2.0",
    "isc": "ISC", "isc license (iscl)": "ISC",
    "mpl-2.0": "MPL-2.0",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_license(package_metadata) -> str | None:
    # Complex expressions and ambiguous labels stay unconfirmed.
    for field in ("License-Expression", "License"):
        value = (package_metadata.get(field) or "").strip().lower()
        if value in LICENSE_ALIASES:
            return LICENSE_ALIASES[value]
    for classifier in package_metadata.get_all("Classifier", []):
        if classifier.startswith("License :: OSI Approved :: "):
            value = classifier.rsplit(" :: ", 1)[-1].strip().lower()
            if value in LICENSE_ALIASES:
                return LICENSE_ALIASES[value]
    return None


def package_component(distribution) -> dict | None:
    name = (distribution.metadata.get("Name") or "").strip()
    version = str(distribution.version or "").strip()
    if not name or not version:
        return None
    normalized = package_name(name)
    # Package metadata names are identifiers, never arbitrary paths/URLs.
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized):
        return None
    purl = f"pkg:pypi/{normalized}@{quote(version, safe='')}"
    license_id = declared_license(distribution.metadata)
    component = {
        "type": "library", "bom-ref": purl, "name": normalized,
        "version": version, "purl": purl,
        "properties": [{"name": "nikon:license-status",
                        "value": "declared-in-package-metadata" if license_id else "unconfirmed"}],
    }
    if license_id:
        component["licenses"] = [{"license": {"id": license_id}}]
    return component


def file_component(kind: str, path: Path) -> dict:
    digest = file_sha256(path)
    return {
        "type": "file",
        "bom-ref": f"file:{kind}:{quote(path.name, safe='')}:{digest}",
        "name": path.name,
        "hashes": [{"alg": "SHA-256", "content": digest}],
        "properties": [
            {"name": "nikon:file-role", "value": kind},
            {"name": "nikon:license-status", "value": "unconfirmed"},
        ],
    }


def build_sbom(
    distributions: Iterable | None = None,
    model_files: Iterable[tuple[str, Path]] = (),
    source_files: Iterable[Path] = (),
    scope: str = "development-python-environment",
) -> dict:
    records = {}
    for distribution in metadata.distributions() if distributions is None else distributions:
        component = package_component(distribution)
        if component:
            records[component["bom-ref"]] = component
    for role, path in model_files:
        component = file_component(f"model:{role}", Path(path))
        records[component["bom-ref"]] = component
    for path in source_files:
        component = file_component("source", Path(path))
        records[component["bom-ref"]] = component
    components = [records[key] for key in sorted(records)]
    generator_hash = file_sha256(Path(__file__))
    identity = json.dumps({"scope": scope, "components": components,
                           "python": platform.python_version(), "os": platform.system(),
                           "architecture": platform.machine(), "generator": generator_hash},
                          ensure_ascii=False, sort_keys=True)
    return {
        "$schema": "http://cyclonedx.org/schema/bom-1.5.schema.json",
        "bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}",
        "metadata": {
            "tools": [{"vendor": "MUKUiL", "name": "nikon-mock-app-sbom", "version": "1"}],
            "component": {"type": "application", "bom-ref": "nikon-mock-app",
                          "name": "nikon-ds10-ds1000-mock-app", "version": "0.1.0"},
            "properties": [
                {"name": "nikon:scope", "value": scope},
                {"name": "nikon:python-version", "value": platform.python_version()},
                {"name": "nikon:os", "value": platform.system()},
                {"name": "nikon:architecture", "value": platform.machine()},
                {"name": "nikon:inventory-method", "value": "installed-distribution-metadata"},
                {"name": "nikon:dependency-graph", "value": "not-included"},
                {"name": "nikon:generator-sha256", "value": generator_hash},
            ],
        },
        "components": components,
    }


def model_argument(value: str) -> tuple[str, Path]:
    role, separator, filename = value.partition("=")
    if not separator or role not in {"detection", "classification"} or not filename:
        raise argparse.ArgumentTypeError("Use --model detection=PATH or classification=PATH.")
    path = Path(filename).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError("The model file does not exist.")
    return role, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", type=model_argument, default=[])
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--scope", choices=("development-python-environment", "release-python-environment"),
                        default="development-python-environment")
    args = parser.parse_args(argv)
    try:
        sbom = build_sbom(model_files=args.model, source_files=args.source, scope=args.scope)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sbom, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"SBOM could not be written ({exc.__class__.__name__}).\n")
    print(f"SBOM written: {len(sbom['components'])} components; scope={args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
