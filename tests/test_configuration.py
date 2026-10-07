import unittest
from pathlib import Path
from unittest.mock import patch

from app.ui import project_directory


class DistributionPaths(unittest.TestCase):
    def test_macos_settings_are_outside_signed_app_bundle(self):
        executable = "/tmp/Nikon Mock/NikonMockApp.app/Contents/MacOS/NikonMockApp"
        with patch("sys.frozen", True, create=True), patch("sys.platform", "darwin"), patch("sys.executable", executable):
            self.assertEqual(project_directory(), Path("/tmp/Nikon Mock").resolve())

    def test_windows_onedir_settings_are_next_to_executable(self):
        executable = "/tmp/Nikon Mock/NikonMockApp.exe"
        with patch("sys.frozen", True, create=True), patch("sys.platform", "win32"), patch("sys.executable", executable):
            self.assertEqual(project_directory(), Path("/tmp/Nikon Mock").resolve())

    def test_source_settings_are_in_project_root(self):
        with patch("sys.frozen", False, create=True):
            self.assertEqual(project_directory(), Path(__file__).resolve().parents[1])
