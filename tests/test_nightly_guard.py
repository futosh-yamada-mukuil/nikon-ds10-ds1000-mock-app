"""Quota boundary and owned interruption tests; no models or inference requests."""
import copy
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nightly_guard import QuotaGuard, GuardedSession


def quota(remaining=90):
    bucket = {"limitId": "codex", "primary": {"usedPercent": 100-remaining, "windowDurationMins": 10080, "resetsAt": 999999}, "secondary": None}
    return {"ordinaryUsageAllowed": True, "rateLimits": bucket, "rateLimitsByLimitId": {"codex": bucket}}


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"
        self.now = 1000
        self.guard = QuotaGuard(self.path, clock=lambda: self.now)

    def test_90_starts_and_85_prevents_start(self):
        self.assertTrue(self.guard.update(quota(90)))
        self.assertTrue(self.guard.check())
        self.assertTrue(self.guard.update(quota(85)))
        self.assertFalse(self.guard.check())
        self.assertEqual(self.guard.state["reason"], "preventive_start_limit")

    def test_80_and_79_stop_during_execution(self):
        for remaining in (80, 79):
            with self.subTest(remaining=remaining):
                guard = QuotaGuard(self.path.with_name(f"{remaining}.json"), clock=lambda: self.now)
                self.assertFalse(guard.update(quota(remaining)))
                self.assertTrue(guard.state["stopped"])

    def test_any_window_or_bucket_can_stop(self):
        data = quota(90)
        bucket = copy.deepcopy(data["rateLimitsByLimitId"]["codex"])
        bucket["limitId"] = "other_codex_bucket"
        bucket["secondary"] = {"usedPercent": 20, "windowDurationMins": 15, "resetsAt": 2000}
        data["rateLimitsByLimitId"][bucket["limitId"]] = bucket
        self.assertFalse(self.guard.update(data))
        self.assertEqual(min(w["remaining_percent"] for w in self.guard.windows), 80)

    def test_missing_stale_invalid_and_expired_are_fail_closed(self):
        cases = [None, {}, quota()]
        cases[-1]["rateLimitsByLimitId"]["codex"]["primary"]["usedPercent"] = float("nan")
        for index, data in enumerate(cases):
            guard = QuotaGuard(self.path.with_name(f"invalid{index}.json"), clock=lambda: self.now)
            self.assertFalse(guard.update(data))
        guard = QuotaGuard(self.path.with_name("stale.json"), clock=lambda: self.now)
        self.assertFalse(guard.update(quota(), received_at=self.now-121))
        data = quota(); data["rateLimitsByLimitId"]["codex"]["primary"]["resetsAt"] = 999
        self.assertFalse(self.guard.update(data))

    def test_bool_out_of_range_and_unknown_scope_are_rejected(self):
        for index, used in enumerate((True, -1, 101, None, "10")):
            data = quota(); data["rateLimitsByLimitId"]["codex"]["primary"]["usedPercent"] = used
            guard = QuotaGuard(self.path.with_name(f"value{index}.json"), clock=lambda: self.now)
            self.assertFalse(guard.update(data))
        data = quota(); data["rateLimitsByLimitId"]["codex"]["limitId"] = "unknown"
        self.assertFalse(self.guard.update(data))

    def test_no_numeric_windows_is_not_assumed_unlimited(self):
        data = quota(); bucket = data["rateLimitsByLimitId"]["codex"]
        bucket["primary"] = None; bucket["credits"] = {"unlimited": True}
        self.assertFalse(self.guard.update(data))

    def test_stop_persists_across_recovery_restart_and_reset(self):
        self.guard.update(quota(80))
        self.assertFalse(self.guard.update(quota(100)))
        restarted = QuotaGuard(self.path, clock=lambda: self.now)
        self.assertFalse(restarted.update(quota(100)))
        restarted.manual_resume()
        self.assertTrue(restarted.update(quota(90)))
        self.assertTrue(restarted.check())

    def test_partial_update_keeps_other_windows_and_staleness(self):
        self.guard.update(quota(90))
        self.assertFalse(self.guard.update({"rateLimits": {"limitId": "codex", "primary": {"usedPercent": 20, "windowDurationMins": 30, "resetsAt": 2000}}}, partial=True))

    def test_account_switch_does_not_restore_permission(self):
        data = quota(); data["accountId"] = "account-a"
        self.guard.update(data)
        changed = quota(); changed["accountId"] = "account-b"
        self.assertFalse(self.guard.update(changed))

    def test_interrupt_is_owned_once_and_no_ai_restart(self):
        class RPC:
            def __init__(self):
                self.next_id = 0; self.messages = queue.Queue(); self.calls = []; self.sent = []
            def send(self, event): self.sent.append(event)
            def request(self, method, params=None):
                self.calls.append(method)
                if method == "account/rateLimits/read": return quota(90)
                if method == "thread/start": return {"thread": {"id": "owned"}}
                if method == "turn/start":
                    self.on_event({"method": "turn/started", "params": {"threadId": "owned", "turn": {"id": "turn-1"}}})
                    self.on_event({"method": "account/rateLimits/updated", "params": {"rateLimits": quota(80)["rateLimits"]}})
                    self.messages.put({"method": "turn/completed", "params": {"threadId": "owned", "turn": {"id": "turn-1", "status": "interrupted"}}})
                    return {"turn": {"id": "turn-1"}}
                raise AssertionError(method)
        rpc = RPC(); session = GuardedSession(rpc, self.guard, self.temp.name, "test-model", "low")
        self.assertFalse(session.run_turn("test task"))
        self.assertEqual(rpc.sent, [{"id": 1, "method": "turn/interrupt", "params": {"threadId": "owned", "turnId": "turn-1"}}])
        self.assertFalse(session.run_turn("another task"))
        self.assertEqual(rpc.calls.count("turn/start"), 1)
        self.assertTrue(json.loads(self.path.read_text())["interrupt_requested"])

    def test_other_thread_events_do_not_interrupt_or_finish(self):
        class RPC:
            next_id = 0
            def send(self, event): raise AssertionError("Must not interrupt other threads")
        session = GuardedSession(RPC(), self.guard, self.temp.name, "test-model", "low")
        session.thread_id = "owned"
        session.event({"method": "turn/started", "params": {"threadId": "interactive", "turn": {"id": "other"}}})
        session.event({"method": "turn/completed", "params": {"threadId": "interactive", "turn": {"status": "completed"}}})
        self.assertIsNone(session.turn_id)
        self.assertFalse(session.completed)

    def test_read_failure_stops_without_ai_retry(self):
        class RPC:
            def request(self, method): raise OSError("offline")
        self.assertFalse(GuardedSession(RPC(), self.guard, self.temp.name, "test-model", "low").run_turn("task"))
        self.assertEqual(self.guard.state["reason"], "usage_read_failed")

    def test_usage_notification_stops_owned_local_test_without_ai(self):
        class RPC:
            def __init__(self): self.messages=queue.Queue()
            def request(self, method): raise AssertionError("No extra request needed")
        self.guard.update(quota(90)); rpc=RPC()
        session=GuardedSession(rpc,self.guard,self.temp.name,'test-model','low')
        rpc.messages.put({'method':'account/rateLimits/updated','params':{'rateLimits':quota(80)['rateLimits']}})
        status=session.local_check([sys.executable,'-c','import time; time.sleep(30)'],self.path.with_name('test.log'))
        self.assertNotEqual(status,0)
        self.assertTrue(self.guard.state['stopped'])

    def test_window_expiring_during_preflight_cannot_start_ai(self):
        owner=self
        class RPC:
            def __init__(self): self.calls=[];self.reads=0
            def request(self,method,params=None):
                self.calls.append(method)
                if method=='account/rateLimits/read':
                    self.reads+=1
                    if self.reads==2:owner.now=1002
                    return quota(90)
                if method=='thread/start':return {'thread':{'id':'owned'}}
                raise AssertionError('Must not start a turn after the window ends')
        rpc=RPC();session=GuardedSession(rpc,self.guard,self.temp.name,'test-model','low',end_time=1001)
        self.assertFalse(session.run_turn('task'))
        self.assertNotIn('turn/start',rpc.calls)
