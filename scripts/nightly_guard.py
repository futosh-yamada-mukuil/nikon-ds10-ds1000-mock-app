"""Fail-closed quota gate; notifications and interruption never invoke an AI."""
from datetime import datetime, timezone
import json
import queue
from pathlib import Path
import subprocess
import time


class QuotaGuard:
    def __init__(self, state_path, start_remaining=85, stop_remaining=80, max_age=120, clock=time.time):
        if not 0 <= stop_remaining < start_remaining <= 100 or max_age <= 0:
            raise ValueError("Invalid quota policy")
        self.path = Path(state_path)
        self.start_remaining, self.stop_remaining, self.max_age = start_remaining, stop_remaining, max_age
        self.clock = clock
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"stopped": False}
        self.snapshot = None
        self.received_at = None
        self.windows = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def stop(self, reason):
        if not self.state.get("stopped"):
            self.state.update(stopped=True, stopped_at=datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
                              reason=reason, quota_windows=self.windows, last_usage_received_at=self.received_at)
            self.save()
        return False

    def manual_resume(self):
        # Only the explicit operator command calls this. Daily limits remain.
        self.state["stopped"] = False
        self.state["resumed_at"] = datetime.fromtimestamp(self.clock(), timezone.utc).isoformat()
        self.save()

    def update(self, payload, received_at=None, partial=False):
        if not isinstance(payload, dict):
            return self.stop("invalid_usage_payload")
        if partial:
            if self.snapshot is None or not isinstance(payload, dict):
                return self.stop("usage_update_without_full_snapshot")
            bucket = payload.get("rateLimits")
            if not isinstance(bucket, dict) or not bucket.get("limitId"):
                return self.stop("unknown_usage_update_scope")
            buckets = self.snapshot["rateLimitsByLimitId"]
            if bucket["limitId"] not in buckets:
                return self.stop("unknown_usage_update_bucket")
            buckets[bucket["limitId"]].update(bucket)
        else:
            if (self.snapshot and self.snapshot.get("accountId") and
                    payload.get("accountId") != self.snapshot["accountId"]):
                return self.stop("account_identity_changed_or_unknown")
            self.received_at = self.clock() if received_at is None else received_at
            self.snapshot = payload
        return self.check(start=False)

    def check(self, start=True):
        if self.state.get("stopped"):
            return False
        try:
            now = self.clock()
            age = now - self.received_at
            if not 0 <= age <= self.max_age:
                raise ValueError("stale_usage")
            if not isinstance(self.snapshot, dict) or self.snapshot.get("ordinaryUsageAllowed") is not True:
                raise ValueError("included_usage_permission_unknown_or_denied")
            buckets = self.snapshot.get("rateLimitsByLimitId")
            if buckets is None:
                legacy = self.snapshot.get("rateLimits")
                if not isinstance(legacy, dict) or not legacy.get("limitId"):
                    raise ValueError("unknown_usage_scope")
                buckets = {legacy["limitId"]: legacy}
                self.snapshot["rateLimitsByLimitId"] = buckets
            if not isinstance(buckets, dict) or not buckets:
                raise ValueError("missing_usage_buckets")
            windows = []
            for key, bucket in buckets.items():
                if not isinstance(bucket, dict) or bucket.get("limitId") != key:
                    raise ValueError("unknown_usage_scope")
                if bucket.get("spendControlReached") is True or bucket.get("rateLimitReachedType"):
                    raise ValueError("backend_limit_reached")
                count = 0
                for name in ("primary", "secondary"):
                    if name not in bucket:
                        raise ValueError("incomplete_usage_snapshot")
                    window = bucket[name]
                    if window is None:
                        continue  # Explicitly unadvertised, never assumed unlimited.
                    if not isinstance(window, dict):
                        raise ValueError("invalid_usage_window")
                    used, duration, reset = (window.get(k) for k in ("usedPercent", "windowDurationMins", "resetsAt"))
                    if type(used) is not int or not 0 <= used <= 100:
                        raise ValueError("invalid_used_percent")
                    if type(duration) is not int or duration <= 0 or type(reset) is not int or reset <= now:
                        raise ValueError("missing_or_expired_window_metadata")
                    windows.append({"limit_id": key, "window": name, "remaining_percent": 100 - used,
                                    "window_duration_mins": duration, "resets_at": reset})
                    count += 1
                if not count:
                    raise ValueError("no_confirmed_quota_windows")
            self.windows = windows
            self.state.update(last_quota_windows=windows, last_usage_received_at=self.received_at)
            if any(w["remaining_percent"] <= self.stop_remaining for w in windows):
                return self.stop("remaining_at_or_below_stop_threshold")
            if start and any(w["remaining_percent"] <= self.start_remaining for w in windows):
                return self.stop("preventive_start_limit")
            return True
        except (TypeError, ValueError, KeyError) as error:
            return self.stop(str(error))


class GuardedSession:
    """Owns exactly one nightly thread; never interrupts other conversations."""
    def __init__(self, rpc, guard, project, model, effort, end_time=None):
        self.rpc, self.guard = rpc, guard
        self.project, self.model, self.effort = Path(project), model, effort
        self.end_time = end_time
        self.thread_id = self.turn_id = None
        self.interrupt_sent = self.completed = False
        self.status = None
        rpc.on_event = self.event

    def read_usage(self):
        try:
            return self.guard.update(self.rpc.request("account/rateLimits/read"))
        except Exception:
            self.guard.stop("usage_read_failed")
            self.interrupt()
            return False

    def interrupt(self):
        if self.guard.state.get("stopped") and self.thread_id and self.turn_id and not self.interrupt_sent:
            self.interrupt_sent = True
            self.rpc.next_id += 1
            self.rpc.send({"id": self.rpc.next_id, "method": "turn/interrupt",
                           "params": {"threadId": self.thread_id, "turnId": self.turn_id}})
            self.guard.state["interrupt_requested"] = True
            self.guard.save()

    def event(self, event):
        method, params = event.get("method"), event.get("params", {})
        if method == "account/rateLimits/updated":
            self.guard.update(params, partial=True)
            self.interrupt()
        elif method == "account/updated":
            self.guard.stop("account_changed")
            self.interrupt()
        elif method == "turn/started" and params.get("threadId") == self.thread_id:
            self.turn_id = params["turn"]["id"]
            self.interrupt()
        elif method == "turn/completed" and params.get("threadId") == self.thread_id and params["turn"].get("id") == self.turn_id:
            self.status = params["turn"]["status"]
            self.completed = True
            self.guard.state["last_turn_status"] = self.status
            self.guard.save()
        elif method == "item/completed" and params.get("threadId") == self.thread_id and params.get("turnId") == self.turn_id:
            item = params.get("item", {})
            if item.get("type") == "agentMessage" and item.get("phase") == "final_answer":
                self.guard.state.setdefault("task_summaries", []).append(item.get("text", "")[:3500])
                self.guard.save()
        elif "id" in event and method:
            # Never grant permissions or send external messages on behalf of Dot.
            self.guard.stop("unexpected_server_request_requires_operator")
            self.rpc.send({"id": event["id"], "error": {"code": -32603, "message": "Unattended approval denied"}})
            self.interrupt()

    def run_turn(self, prompt):
        if self.end_time is not None and self.guard.clock() >= self.end_time:
            return self.guard.stop("outside_night_window")
        if not self.read_usage() or not self.guard.check(start=True):
            return False
        if self.thread_id is None:
            thread = self.rpc.request("thread/start", {"model": self.model, "modelProvider": "openai", "cwd": str(self.project),
                "approvalPolicy": "never", "sandbox": "workspace-write", "serviceTier": "default",
                "developerInstructions": "Follow AGENTS.md and docs/dots-nightly.md. Work locally only. No agents, git index changes, commits, push, PR, external messages, paid API, credits, or guard/config changes. Read only relevant files. Run necessary local tests. One attempt and one retry maximum."})
            self.thread_id = thread["thread"]["id"]
        if not self.read_usage() or not self.guard.check(start=True):
            return False
        self.completed = self.interrupt_sent = False
        self.status = None
        self.guard.state.update(last_turn_status=None, interrupt_requested=False)
        self.guard.save()
        if self.end_time is not None and self.guard.clock() >= self.end_time:
            return self.guard.stop("outside_night_window")
        turn = self.rpc.request("turn/start", {"threadId": self.thread_id, "model": self.model,
            "effort": self.effort, "serviceTier": "default", "input": [{"type": "text", "text": prompt}]})
        self.turn_id = turn["turn"]["id"]
        self.interrupt()
        # Notifications first; refresh at most once per freshness interval, without AI.
        deadline = time.monotonic() + 1800
        stop_deadline = None
        while not self.completed and time.monotonic() < deadline:
            try:
                event = self.rpc.messages.get(timeout=1)
                if event is None:
                    self.guard.stop("app_server_disconnected")
                    break
                self.event(event)
            except queue.Empty:
                pass
            if self.end_time is not None and self.guard.clock() >= self.end_time:
                self.guard.stop("night_window_ended")
                self.interrupt()
            if self.guard.state.get("stopped"):
                stop_deadline = stop_deadline or time.monotonic() + 30
                if time.monotonic() >= stop_deadline:
                    break
            if not self.guard.state.get("stopped") and self.guard.clock() - self.guard.received_at >= self.guard.max_age:
                self.read_usage()
                self.interrupt()
        if not self.completed:
            self.guard.stop("turn_completion_not_confirmed")
            self.interrupt()
        if self.completed:
            self.turn_id = None
        if not self.guard.state.get("stopped"):
            self.read_usage()
        return self.status == "completed" and not self.guard.state.get("stopped")

    def local_check(self, command, log_path):
        """Keep processing usage notifications while a normal local test runs."""
        with Path(log_path).open("w") as log:
            process = subprocess.Popen(command, cwd=self.project, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 300
            while process.poll() is None:
                try:
                    event = self.rpc.messages.get(timeout=.2)
                    if event is None:
                        self.guard.stop("app_server_disconnected")
                    else:
                        self.event(event)
                except queue.Empty:
                    pass
                if self.end_time is not None and self.guard.clock() >= self.end_time:
                    self.guard.stop("night_window_ended")
                if not self.guard.state.get("stopped") and self.guard.clock() - self.guard.received_at >= self.guard.max_age:
                    self.read_usage()
                if time.monotonic() >= deadline:
                    self.guard.stop("local_check_timeout")
                if self.guard.state.get("stopped"):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            return process.wait()


def change_record(project):
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=project, capture_output=True, text=True)
    return result.stdout.splitlines() if result.returncode == 0 else ["Git status unavailable"]
