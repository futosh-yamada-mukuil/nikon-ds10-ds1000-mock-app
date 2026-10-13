import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.generate_sbom import file_component
from scripts.update_source_sbom import ROOT, update_sources


class SourceSBOMTests(unittest.TestCase):
    def test_updates_only_named_source_and_preserves_preexisting_model_library_changes(self):
        source = ROOT / "scripts/evaluate_stages.py"
        old = file_component("source", source)
        old["hashes"][0]["content"] = "old synthetic source digest"
        model = {"type": "file", "name": "missing-model.pth", "bom-ref": "synthetic-model",
                 "properties": [{"name": "nikon:file-role", "value": "model:classification"}],
                 "hashes": [{"alg": "SHA-256", "content": "b" * 64}]}
        library = {"type": "library", "name": "synthetic-dependency", "version": "existing-local-change"}
        before = {"serialNumber": "existing-identity", "metadata": {"local_change": True},
                  "components": [model, library, old]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-sbom.json"
            path.write_text(json.dumps(before))
            with patch("scripts.update_source_sbom.file_component", wraps=file_component) as component:
                self.assertEqual(update_sources(path, [source]), 1)
            component.assert_called_once_with("source", source.resolve())
            after = json.loads(path.read_text())
            self.assertEqual(after["components"][:2], before["components"][:2])
            self.assertEqual(after["metadata"], before["metadata"])
            self.assertEqual(after["serialNumber"], before["serialNumber"])
            self.assertEqual(after["components"][2], file_component("source", source))

    def test_duplicate_sources_do_not_modify_inventory(self):
        source = ROOT / "scripts/evaluate_stages.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-sbom.json"
            original = '{"components": []}\n'
            path.write_text(original)
            with self.assertRaises(ValueError):
                update_sources(path, [source, source])
            self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
