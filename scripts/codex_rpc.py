"""Small stdio client for the installed Codex App Server; no inference on connect."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import time


class AppServer:
    def __init__(self, executable, log_path, on_event=None):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(executable), "app-server", "--listen", "stdio://", "--disable", "multi_agent", "--disable", "fast_mode", "--disable", "multi_agent_v2"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1,
        )
        self.messages = queue.Queue()
        self.next_id = 0
        self.on_event = on_event or (lambda event: None)
        threading.Thread(target=self._read, daemon=True).start()
        self.request("initialize", {"clientInfo": {"name": "nikon_nightly_guard", "title": "Nikon nightly guard", "version": "1"}})
        self.send({"method": "initialized"})

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        finally:
            self.messages.put(None)

    def send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def request(self, method, params=None, timeout=15):
        self.next_id += 1
        request_id = self.next_id
        self.send({"id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self.messages.get(timeout=max(.01, deadline - time.monotonic()))
            if event is None:
                raise RuntimeError("Codex App Server disconnected")
            if event.get("id") == request_id and "method" not in event:
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                return event["result"]
            self.on_event(event)
        raise TimeoutError(method)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.log.close()
