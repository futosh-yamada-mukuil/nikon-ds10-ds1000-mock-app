"""Serve a single local CSV to a browser, never a directory or external host."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import threading
from urllib.parse import quote


class CsvDownload:
    def __init__(self, path: Path, timeout=300):
        path = Path(path).resolve(strict=True)
        if not path.is_file() or path.suffix != ".csv":
            raise ValueError("CSVファイルを指定してください。")
        route = "/" + secrets.token_urlsafe(24)
        filename = path.parent.name + ".csv"

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def do_GET(self):
                if self.path != route:
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=results.csv; filename*=UTF-8''" + quote(filename))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}{route}"
        self.closed = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.timer = threading.Timer(timeout, self.close)
        self.timer.daemon = True
        self.timer.start()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.timer.cancel()
        self.server.shutdown()
        self.server.server_close()
