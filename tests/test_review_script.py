"""Review script contracts without staging, committing, pushing, or dependencies."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


SOURCE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review.sh"
BASH = "/bin/bash"
GIT = shutil.which("git")


@unittest.skipUnless(GIT and Path(BASH).is_file(), "Bash and Git are required")
class ReviewScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nikon-review-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        scripts = self.project / "scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(SOURCE_SCRIPT, scripts / "review.sh")
        self.env = os.environ.copy()
        # Isolate temporary repositories from user Git configuration.
        self.env.update({
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.env.pop(key, None)

    def init_repo(self, directory=None):
        directory = directory or self.project
        subprocess.run(
            [GIT, "-c", "init.defaultBranch=main", "init", "--quiet", str(directory)],
            env=self.env, check=True, capture_output=True, text=True,
        )

    def run_review(self, cwd=None, env=None):
        return subprocess.run(
            [BASH, str(self.project / "scripts" / "review.sh")],
            cwd=cwd or self.root, env=env or self.env,
            capture_output=True, text=True,
        )

    def snapshot(self, directory=None):
        directory = directory or self.project
        result = {}
        for path in sorted(directory.rglob("*")):
            info = path.lstat()
            entry = [stat.S_IMODE(info.st_mode), info.st_mtime_ns]
            if path.is_file():
                entry.append(hashlib.sha256(path.read_bytes()).hexdigest())
            result[str(path.relative_to(directory))] = entry
        return result

    def real_status(self):
        return subprocess.run(
            [GIT, "--no-pager", "-c", "core.fsmonitor=false", "-C",
             str(self.project), "status", "--porcelain=v1", "--untracked-files=all"],
            env=self.env, check=True, capture_output=True, text=True,
        ).stdout

    def install_git_stub(self, has_head=True, status_failure=False):
        bin_dir = self.root / "stub-bin"
        bin_dir.mkdir()
        log = self.root / "stub-calls.jsonl"
        stub = bin_dir / "git"
        stub.write_text(
            "#!" + sys.executable + "\n" + r'''
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["REVIEW_TEST_CALLS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"args": args, "optional_locks": os.environ.get("GIT_OPTIONAL_LOCKS")}) + "\n")
commands = [a for a in args if a in ("rev-parse", "status", "diff", "ls-files")]
if len(commands) != 1:
    sys.exit(70)
command = commands[0]
if command == "rev-parse":
    if "--is-inside-work-tree" in args:
        print("true")
    elif "--show-toplevel" in args:
        print(os.environ["REVIEW_TEST_ROOT"])
    elif "HEAD" in args:
        sys.exit(0 if os.environ["REVIEW_TEST_HEAD"] == "yes" else 1)
    else:
        sys.exit(71)
elif command == "status":
    if os.environ["REVIEW_TEST_STATUS_FAILURE"] == "yes":
        print("stub status failure", file=sys.stderr)
        sys.exit(9)
    print(" M app/example.py\nM  docs/staged.md\n?? notes.txt")
elif command == "diff":
    if "--cached" in args:
        print(" docs/staged.md | 1 +" if "--stat" in args else "diff --git a/docs/staged.md b/docs/staged.md\n+STAGED_CHANGE")
    else:
        print(" app/example.py | 1 +" if "--stat" in args else "diff --git a/app/example.py b/app/example.py\n+WORKING_CHANGE")
elif command == "ls-files":
    print("notes.txt")
''', encoding="utf-8",
        )
        stub.chmod(0o755)
        env = self.env.copy()
        env.update({
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "REVIEW_TEST_CALLS": str(log),
            "REVIEW_TEST_ROOT": str(self.project.resolve()),
            "REVIEW_TEST_HEAD": "yes" if has_head else "no",
            "REVIEW_TEST_STATUS_FAILURE": "yes" if status_failure else "no",
        })
        return env, log

    def test_uninitialized_directory_reports_reason_without_changes(self):
        before = self.snapshot()
        result = self.run_review()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Git作業ツリーではありません", result.stderr)
        self.assertIn("このスクリプトはgit initを実行しません", result.stderr)
        self.assertFalse((self.project / ".git").exists())
        self.assertEqual(before, self.snapshot())

    def test_before_first_commit_lists_untracked_files(self):
        self.init_repo()
        (self.project / "notes.txt").write_text("review me\n", encoding="utf-8")
        result = self.run_review()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("初回コミット前", result.stdout)
        self.assertIn("notes.txt", result.stdout)
        self.assertIn("未追跡ファイル一覧", result.stdout)
        self.assertIn("差分が空でも「変更なし」とは限りません", result.stdout)

    def test_runs_from_another_directory(self):
        self.init_repo()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        result = self.run_review(cwd=elsewhere)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.project.resolve()), result.stdout)
        self.assertIn("scripts/review.sh", result.stdout)

    def test_project_and_file_paths_with_spaces(self):
        renamed = self.root / "project with spaces"
        self.project.rename(renamed)
        self.project = renamed
        self.init_repo()
        (self.project / "review notes.txt").write_text("notes\n", encoding="utf-8")
        result = self.run_review()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(renamed.resolve()), result.stdout)
        self.assertIn("review notes.txt", result.stdout)

    def test_parent_repository_is_rejected_and_preserved(self):
        self.init_repo(self.root)
        before = self.snapshot(self.root)
        result = self.run_review()
        self.assertEqual(result.returncode, 2)
        self.assertIn("親の別リポジトリはレビューしません", result.stderr)
        self.assertNotIn("=== Git状態の概要 ===", result.stdout)
        self.assertEqual(before, self.snapshot(self.root))

    def test_review_preserves_files_and_git_state(self):
        self.init_repo()
        (self.project / "notes.txt").write_text("unchanged\n", encoding="utf-8")
        before_status = self.real_status()
        before_files = self.snapshot()
        result = self.run_review()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before_files, self.snapshot())
        self.assertEqual(before_status, self.real_status())
        self.assertFalse((self.project / ".git" / "index").exists())

    def test_git_environment_override_is_rejected(self):
        other = self.root / "other-project"
        other.mkdir()
        self.init_repo(other)
        before = self.snapshot(self.root)
        env = self.env.copy()
        env["GIT_DIR"] = str(other / ".git")
        result = self.run_review(env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("対象変更が設定されています", result.stderr)
        self.assertEqual(before, self.snapshot(self.root))

    def test_missing_git_reports_actionable_error(self):
        empty_bin = self.root / "empty-bin"
        empty_bin.mkdir()
        env = self.env.copy()
        env["PATH"] = str(empty_bin)
        result = self.run_review(env=env)
        self.assertEqual(result.returncode, 127)
        self.assertIn("Gitが見つかりません", result.stderr)
        self.assertIn("PATH", result.stderr)

    def test_staged_and_unstaged_diffs_are_separate_with_stub(self):
        env, _ = self.install_git_stub()
        before = self.snapshot()
        result = self.run_review(env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        working = result.stdout.index("=== 未ステージ変更の差分 ===")
        staged = result.stdout.index("=== ステージ済み変更の差分 ===")
        self.assertIn("+WORKING_CHANGE", result.stdout[working:staged])
        self.assertIn("+STAGED_CHANGE", result.stdout[staged:])
        self.assertNotIn("初回コミット前", result.stdout)
        self.assertEqual(before, self.snapshot())

    def test_staged_diff_before_first_commit_with_stub(self):
        env, _ = self.install_git_stub(has_head=False)
        result = self.run_review(env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("初回コミット前", result.stdout)
        self.assertIn("+STAGED_CHANGE", result.stdout)

    def test_git_failure_does_not_report_success_with_stub(self):
        env, _ = self.install_git_stub(status_failure=True)
        result = self.run_review(env=env)
        self.assertEqual(result.returncode, 9)
        self.assertIn("stub status failure", result.stderr)
        self.assertNotIn("=== 未ステージ変更の差分 ===", result.stdout)

    def test_only_read_commands_and_disabled_optional_writes_with_stub(self):
        env, log = self.install_git_stub()
        result = self.run_review(env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(calls)
        allowed = {"rev-parse", "status", "diff", "ls-files"}
        forbidden = {"add", "commit", "push", "reset", "clean", "checkout", "restore", "update-index", "init"}
        for call in calls:
            args = call["args"]
            self.assertEqual(call["optional_locks"], "0")
            self.assertTrue(allowed.intersection(args))
            self.assertFalse(forbidden.intersection(args))
            self.assertIn("--no-pager", args)
            self.assertIn("core.fsmonitor=false", args)
            if "diff" in args:
                self.assertIn("--no-ext-diff", args)
                self.assertIn("--no-textconv", args)


if __name__ == "__main__":
    unittest.main()
