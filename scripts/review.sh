#!/bin/bash
# Read-only project review. Compatible with macOS Bash 3.2.
set -euo pipefail

case "${BASH_SOURCE[0]}" in
  */*) script_parent="${BASH_SOURCE[0]%/*}" ;;
  *) script_parent="." ;;
esac
script_dir=$(CDPATH= cd -P -- "$script_parent" && pwd -P)
project_root=$(CDPATH= cd -P -- "$script_dir/.." && pwd -P)

printf '対象プロジェクト: %s\n' "$project_root"

if ! command -v git >/dev/null 2>&1; then
  printf 'エラー: Gitが見つかりません。Gitを利用できるPATHと実行環境を確認してください。\n' >&2
  exit 127
fi

if [ -n "${GIT_DIR-}" ] || [ -n "${GIT_WORK_TREE-}" ] || [ -n "${GIT_INDEX_FILE-}" ]; then
  printf 'エラー: GIT_DIR・GIT_WORK_TREE・GIT_INDEX_FILEによる対象変更が設定されています。\n' >&2
  printf '対処: Gitの対象を変更する環境変数を外したシェルで、対象プロジェクトを確認して実行してください。\n' >&2
  exit 2
fi

# Avoid optional index refresh writes, pagers, and filesystem-monitor hooks.
export GIT_OPTIONAL_LOCKS=0
run_git() {
  git --no-pager -c core.fsmonitor=false -c core.quotepath=false -C "$project_root" "$@"
}

if ! inside_work_tree=$(run_git rev-parse --is-inside-work-tree 2>/dev/null) || [ "$inside_work_tree" != "true" ]; then
  printf 'エラー: 対象フォルダーはGit作業ツリーではありません。\n' >&2
  printf '対処: 対象パスと親のGitルートを確認し、必要な初期化は別の依頼で行ってください。このスクリプトはgit initを実行しません。\n' >&2
  exit 2
fi

if ! git_root=$(run_git rev-parse --show-toplevel 2>/dev/null); then
  printf 'エラー: Gitルートを確認できません。Gitリポジトリの状態を確認してください。\n' >&2
  exit 2
fi
git_root=$(CDPATH= cd -P -- "$git_root" && pwd -P)
if [ "$git_root" != "$project_root" ]; then
  printf 'エラー: 対象フォルダーのGitルートは別の場所です: %s\n' "$git_root" >&2
  printf '対処: 親の別リポジトリはレビューしません。プロジェクトの配置と初期化方針を確認してください。\n' >&2
  exit 2
fi

printf '\n=== Git状態の概要 ===\n'
run_git status --short --untracked-files=normal

if ! run_git rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
  printf '\n注意: 初回コミット前です。HEADがなくても状態と差分を確認します。\n'
fi

printf '\n=== 未ステージ変更の差分統計 ===\n'
run_git diff --no-ext-diff --no-textconv --no-color --stat
printf '\n=== 未ステージ変更の差分 ===\n'
run_git diff --no-ext-diff --no-textconv --no-color

printf '\n=== ステージ済み変更の差分統計 ===\n'
run_git diff --cached --no-ext-diff --no-textconv --no-color --stat
printf '\n=== ステージ済み変更の差分 ===\n'
run_git diff --cached --no-ext-diff --no-textconv --no-color

printf '\n=== 未追跡ファイル一覧 ===\n'
untracked_files=$(run_git ls-files --others --exclude-standard)
if [ -n "$untracked_files" ]; then
  printf '%s\n' "$untracked_files"
else
  printf '未追跡ファイルはありません。\n'
fi
printf '\n未追跡ファイルの内容は通常のgit diffに含まれません。一覧にあるファイルを別途読んで確認してください。\n'
printf '差分が空でも「変更なし」とは限りません。ステージ状態と未追跡ファイルを合わせて確認してください。\n'
