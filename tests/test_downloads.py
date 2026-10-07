"""Browser HTTP attachment and actual GUI export tests, using temporary files."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import csv
import json
import socket
from pathlib import Path
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from PIL import Image
from PySide6.QtWidgets import QApplication
from app.domain import CellResult, FrameResult, InferenceSettings
from app.downloads import CsvDownload
from app.ui import MainWindow

APP = QApplication.instance() or QApplication([])


class DownloadTests(unittest.TestCase):
    def test_attachment_serves_exact_csv_and_refuses_other_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "検索結果.csv"; data = b'\xef\xbb\xbfcell_id,class1_score\r\n1,0.6\r\n'
            path.write_bytes(data); download = CsvDownload(path)
            try:
                with urlopen(download.url, timeout=3) as response:
                    self.assertIn("attachment", response.headers['Content-Disposition'])
                    self.assertIn("filename*=UTF-8", response.headers['Content-Disposition'])
                    self.assertEqual(response.read(), data)
                with self.assertRaises(HTTPError) as error:
                    urlopen(download.url.rsplit('/', 1)[0] + '/../../', timeout=3)
                self.assertEqual(error.exception.code, 404)
                self.assertEqual(download.server.server_address[0], '127.0.0.1')
            finally: download.close()
            self.assertTrue(download.closed)

    def test_gui_saves_and_dispatches_browser_with_visible_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'input.png'; Image.new('RGB', (80,60)).save(source)
            window = MainWindow(config=root/'absent.json', autoload=False)
            try:
                self.assertEqual(window.det_threshold.value(), .5)
                self.assertEqual(window.class_threshold.value(), .5)
                self.assertEqual(InferenceSettings().classification_threshold, .5)
                window.open_source(source); window.run_settings = InferenceSettings()
                window.result = FrameResult([CellResult(1,(4,4,20,20),.9,.6,True)])
                class Engine:
                    def get_model_metadata(self): return {'test': True}
                window.engine = Engine(); window.export_directory = root/'downloads'; received = []
                def opener(url):
                    with urlopen(url.toString(), timeout=3) as response: received.append(response.read())
                    return True
                window.download_opener = opener; window.set_busy(False); window.export_button.click()
                files = list(window.export_directory.glob('*/results.csv')); self.assertEqual(len(files),1)
                self.assertEqual(received,[files[0].read_bytes()])
                self.assertIn('ダウンロードを開始',window.export_feedback.text())
                self.assertIn('保存済みCSVを開く',window.export_feedback.text())
                manifest=json.loads((files[0].parent/'manifest.json').read_text())
                self.assertEqual(manifest['settings']['classification_threshold'],.5)
                with files[0].open(encoding='utf-8-sig') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))),1)
                window.download_opener=lambda url: False; window.export_button.click()
                self.assertIn('ブラウザーを開けません',window.export_feedback.text())
            finally: window.close(); APP.processEvents()

    def test_incomplete_browser_connection_does_not_block_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data.csv'; path.write_text('id\n1\n')
            download=CsvDownload(path)
            with socket.create_connection(download.server.server_address,timeout=3) as connection:
                connection.sendall(b'GET /')
                time.sleep(.05)
                started=time.monotonic()
                download.close()
                self.assertLess(time.monotonic()-started,5)
            self.assertTrue(download.closed)
