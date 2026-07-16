#!/bin/bash
# 在新机器上 clone hermes-agent 后运行此脚本，一键启用 pre-push 守卫。
# 用法：cd hermes-agent && ./githooks/setup-hooks.sh
# 效果：设置 core.hooksPath=githooks，禁止任何 agent/人工 push 到远端，
#       防止再次误推 LLM API key 等敏感配置。
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || {
  echo "❌ 不在 git 仓库内，请在 hermes-agent 根目录运行。" >&2
  exit 1
}

if [[ ! -f githooks/pre-push ]]; then
  echo "❌ 未找到 githooks/pre-push，请确认在 hermes-agent 根目录。" >&2
  exit 1
fi

chmod +x githooks/pre-push
git config core.hooksPath githooks

echo "✅ 已启用 pre-push 守卫：core.hooksPath = $(git config --get core.hooksPath)"
echo "   规则：禁止任何 push 到远端（含 codex/claude 等 agent）。"
echo "   紧急放行（仅人工）：BYPASS_HERMES_PUSH=1 git push ..."
