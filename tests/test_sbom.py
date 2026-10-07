import argparse
from email.message import Message
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_sbom.py"
SPEC = importlib.util.spec_from_file_location("nikon_generate_sbom", SCRIPT)
SBOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SBOM)


def distribution(name, version, license_text=None):
    info = Message()
    info["Name"] = name
    if license_text:
        info["License"] = license_text
    return SimpleNamespace(metadata=info, version=version)


class SbomTests(unittest.TestCase):
    def test_deterministic_inventory_and_valid_purl_encoding(self):
        packages = [distribution("Demo_pkg", "1.0+cpu", "MIT"), distribution("Other", "2.0")]
        first = SBOM.build_sbom(packages)
        second = SBOM.build_sbom(reversed(packages))
        self.assertEqual(first, second)
        self.assertEqual(first["bomFormat"], "CycloneDX")
        self.assertEqual(first["specVersion"], "1.5")
        self.assertEqual(first["components"][0]["purl"], "pkg:pypi/demo-pkg@1.0%2Bcpu")
        self.assertEqual(first["components"][0]["licenses"], [{"license": {"id": "MIT"}}])
        self.assertNotIn("licenses", first["components"][1])

    def test_model_and_source_hashes_without_path_or_contents(self):
        with TemporaryDirectory(prefix="private-user-token-") as temporary:
            model = Path(temporary) / "checkpoint.pt"
            source = Path(temporary) / "source.py"
            model.write_bytes(b"private-model-content")
            source.write_bytes(b"secret_token = 'not-for-output'")
            bom = SBOM.build_sbom([], [("detection", model)], [source])
            payload = json.dumps(bom)
            self.assertNotIn(temporary, payload)
            self.assertNotIn("private-model-content", payload)
            self.assertNotIn("not-for-output", payload)
            files = {component["name"]: component for component in bom["components"]}
            self.assertEqual(files["checkpoint.pt"]["hashes"][0]["content"],
                             hashlib.sha256(b"private-model-content").hexdigest())
            generator = {p["name"]: p["value"] for p in bom["metadata"]["properties"]}
            self.assertEqual(generator["nikon:generator-sha256"], SBOM.file_sha256(SCRIPT))

    def test_unknown_and_complex_licenses_remain_unconfirmed(self):
        bom = SBOM.build_sbom([distribution("Ambiguous", "1", "BSD License"),
                               distribution("Expression", "1", "MIT OR Apache-2.0")])
        for component in bom["components"]:
            self.assertNotIn("licenses", component)
            self.assertEqual(component["properties"][0]["value"], "unconfirmed")

    def test_duplicate_metadata_and_invalid_identifiers(self):
        valid = distribution("Demo", "1")
        bom = SBOM.build_sbom([valid, valid, distribution("https://user:token@example", "1")])
        self.assertEqual(len(bom["components"]), 1)
        self.assertNotIn("token", json.dumps(bom))

    def test_cli_writes_json_and_records_scope(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "sbom.json"
            self.assertEqual(SBOM.main(["--output", str(output), "--scope", "release-python-environment"]), 0)
            bom = json.loads(output.read_text(encoding="utf-8"))
            properties = {p["name"]: p["value"] for p in bom["metadata"]["properties"]}
            self.assertEqual(properties["nikon:scope"], "release-python-environment")

    def test_missing_model_is_rejected(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(argparse.ArgumentTypeError):
                SBOM.model_argument(f"detection={temporary}/missing.pt")


if __name__ == "__main__":
    unittest.main()
