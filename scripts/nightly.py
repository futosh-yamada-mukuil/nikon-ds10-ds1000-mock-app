"""One local nightly invocation. Disabled until unattended cancellation is validated."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from codex_rpc import AppServer
from nightly_guard import QuotaGuard, GuardedSession, change_record


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config/nightly.example.json")
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="Read account, usage and catalog only; no AI")
    parser.add_argument("--manual-resume", action="store_true", help="Explicit operator action; never automatic")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    local = root / ".nightly-local"
    local.mkdir(mode=0o700, exist_ok=True)
    guard = QuotaGuard(local / "state.json", config["start_remaining_percent"], config["stop_remaining_percent"], config["freshness_seconds"])
    if args.manual_resume:
        if (local / "running.lock").exists():
            print("夜間処理の終了確認後に手動再開してください。")
            return 1
        guard.manual_resume()
        print("停止状態を手動解除しました。自動実行の有効化・当日の実行回数は変更していません。")
        return 0
    if not args.probe and (config.get("enabled") is not True or config.get("validated_unattended") is not True):
        print("夜間自動実行は無効です。監視・中断の実機確認が必要です。")
        return 0
    if not args.probe and guard.state.get("stopped"):
        print("停止状態を維持しています。本人の明示的な再開が必要です。")
        return 1
    lock = local / "running.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print("既存の実行ロックがあります。並列起動しません。")
        return 1
    rpc = session = None
    try:
        now = datetime.now(ZoneInfo(config["timezone"]))
        begin, end = [datetime.strptime(t, "%H:%M").time() for t in config["time_window"]]
        if begin >= end:
            raise ValueError("Use the confirmed same-day midnight-to-morning window")
        if not args.probe and not begin <= now.time() < end:
            print("設定された夜間時間帯ではありません。AI処理を開始しません。")
            return 0
        if not args.probe and guard.state.get("last_run_date") == now.date().isoformat():
            print("本日は実行済みです。1日1回の制限を維持します。")
            return 0
        rpc = AppServer(args.codex, local / "app_server.log")
        account = rpc.request("account/read", {"refreshToken": False})
        if (account.get("account") or {}).get("type") != "chatgpt":
            guard.stop("chatgpt_subscription_auth_required")
            return 1
        if not guard.update(rpc.request("account/rateLimits/read")):
            print("利用枠の停止条件を検知しました。停止状態を維持します。")
            return 1
        models = rpc.request("model/list", {"limit": 100, "includeHidden": False})
        selected = next((m for m in models["data"] if m["model"] == config["model"]), None)
        supported = selected and config["effort"] in [e["reasoningEffort"] for e in selected["supportedReasoningEfforts"]]
        if not supported:
            guard.stop("configured_model_or_effort_unavailable")
            return 1
        if args.probe:
            print(json.dumps({"auth": "chatgpt", "windows": guard.windows, "configured_model": config["model"],
                              "effort": config["effort"], "catalog_present": True, "inference_access_tested": False,
                              "stopped": guard.state.get("stopped"), "unattended_enabled": False}, ensure_ascii=False))
            return 0
        tasks = config.get("tasks", [])
        if not tasks:
            print("依頼がありません。AI処理を開始しません。")
            return 0
        if not isinstance(tasks, list) or not 1 <= config["max_tasks"] <= 3 or len(tasks) > config["max_tasks"]:
            raise ValueError("At most three explicit tasks are accepted")
        if any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in tasks):
            raise ValueError("Tasks must be concise, explicit text")
        if len(set(tasks)) != len(tasks):
            raise ValueError("Duplicate tasks must not bypass the retry limit")
        if not guard.check(start=True):
            return 1
        guard.state.update(last_run_date=now.date().isoformat(), unfinished_tasks=list(tasks), executed_tests=[])
        guard.save()
        end_time = datetime.combine(now.date(), end, now.tzinfo).timestamp()
        session = GuardedSession(rpc, guard, root, config["model"], config["effort"], end_time)
        for task in tasks:
            success = False
            prompt = task
            for attempt in range(2):
                if not session.run_turn(prompt):
                    if guard.state.get("stopped"):
                        break
                    prompt = task + "\n初回が失敗しました。この1回だけ再試行し、解決できなければ保留してください。"
                    continue
                if not session.read_usage() or not guard.check(start=False):
                    break
                test_log = local / f"tests_{now.date()}_{tasks.index(task)}_{attempt}.log"
                command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
                test_record = {"command": command, "exit_code": None, "log": str(test_log), "status": "started"}
                guard.state["executed_tests"].append(test_record)
                guard.save()
                exit_code = session.local_check(command, test_log)
                test_record.update(exit_code=exit_code, status="interrupted" if guard.state.get("stopped") else "finished")
                guard.save()
                if not session.read_usage() or not guard.check(start=False):
                    break
                if exit_code == 0:
                    success = True
                    break
                prompt = task + "\n必要なテストが失敗しました。再試行は今回だけです。\n" + "\n".join(test_log.read_text()[-3000:].splitlines()[-30:])
            if success:
                guard.state["unfinished_tasks"].remove(task)
                guard.save()
            if guard.state.get("stopped"):
                break
        return 1 if guard.state.get("stopped") else 0
    except Exception as error:
        guard.stop("controller_error: " + type(error).__name__)
        if session:
            session.interrupt()
        print("夜間制御を停止しました。ローカル記録を確認してください。")
        return 1
    finally:
        if rpc:
            rpc.close()
        guard.state["changed_files_status"] = change_record(root)
        guard.save()
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
