#!/bin/bash
# deploy-hermes.sh - 一键部署 Hermes 伴侣记忆引擎服务至远程服务器
#
# 改造要点（2026-07-16）：
#   1. tar 工作区打包 → git archive HEAD（只部署已提交代码，避免未提交改动误上线）
#   2. 硬编码密码/API key → 读 .env.deploy（gitignore 忽略，不入 git）
#   3. 加提交检查、远端 py_compile 语法兼容检查、FTS trigram 降级逻辑预检
#   4. 解压到独立目录而非直接覆盖生产；rsync 同步排除运行时数据（profiles/venv/.env）
#   5. ERR trap 回滚（部署失败自动拉起服务）+ 端口 8009 占用检查
#   6. 对齐 VelvetChat deploy-backend.sh 的安全门禁模式
set -e

echo "========================================="
echo "Hermes Engine Deployment Script"
echo "========================================="

# === 配置（敏感值从 .env.deploy 读取，不硬编码） ===
DEPLOY_ENV_FILE="$(cd "$(dirname "$0")" && pwd)/.env.deploy"
if [ ! -f "$DEPLOY_ENV_FILE" ]; then
  echo "❌ 缺少 $DEPLOY_ENV_FILE，请参考 .env.deploy.example 创建并填入真实值"
  exit 1
fi
# shellcheck disable=SC1090
source "$DEPLOY_ENV_FILE"

: "${HERMES_REMOTE_PASSWORD:?HERMES_REMOTE_PASSWORD 未在 .env.deploy 中设置}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY 未在 .env.deploy 中设置}"

REMOTE_USER="root"
REMOTE_HOST="146.56.229.151"
SERVICE_NAME="airi-love-hermes.service"
LOCAL_TAR="/tmp/hermes-agent-deploy.tar.gz"
# 实际远端部署路径在下方 ENDSSH 内定义为 /var/www/hermes-agent

SSH_ARGS=(
  -F /dev/null
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o PubkeyAuthentication=no
  -o PreferredAuthentications=password
  -o PasswordAuthentication=yes
  -o KbdInteractiveAuthentication=no
  -o IdentitiesOnly=yes
  -o IdentityFile=/dev/null
)

# === Step 1: 提交检查 + 打包 ===
echo "📦 Step 1: 检查工作区并打包代码（仅已提交的 git 内容）..."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ 当前目录不是 git 仓库，无法安全打包"
  exit 1
fi

# 强制要求 working tree clean（对齐 VelvetChat deploy-backend.sh）
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ 检测到未提交的修改（tracked files）。请先提交后再部署："
  git status --porcelain=v1
  exit 1
fi

# 收集本次提交变更的 Python 文件（用于远端 py_compile 兼容检查）
if git rev-parse --verify HEAD^ >/dev/null 2>&1; then
  CHANGED_PY_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD | grep '\.py$' || true)
else
  CHANGED_PY_FILES=$(git ls-files | grep '\.py$' || true)
fi
CHANGED_PY_FILES=$(printf '%s\n' "$CHANGED_PY_FILES" | sed '/^$/d' || true)

if [ -n "$CHANGED_PY_FILES" ]; then
  echo "🧾 本次提交涉及的 Python 文件："
  printf '  - %s\n' $CHANGED_PY_FILES
  CHANGED_PY_FILES_B64=$(printf '%s\n' "$CHANGED_PY_FILES" | base64 | tr -d '\n')
else
  echo "🧾 本次提交未修改 Python 文件，跳过远端语法检查。"
  CHANGED_PY_FILES_B64=""
fi

# git archive HEAD（Hermes 仓库根即部署源码，archive 整个根）
git archive --format=tar.gz -o "$LOCAL_TAR" HEAD .

echo "✅ 代码打包完成"

# === Step 2: 上传 ===
echo "📤 Step 2: 上传代码到远程服务器..."
sshpass -p "$HERMES_REMOTE_PASSWORD" scp "${SSH_ARGS[@]}" \
  "$LOCAL_TAR" \
  $REMOTE_USER@$REMOTE_HOST:/root/hermes-agent-deploy.tar.gz
echo "✅ 上传完成"

# === Step 3: 远端部署 ===
echo "🚀 Step 3: 远端部署..."
SSH_RUN=(sshpass -p "$HERMES_REMOTE_PASSWORD" ssh "${SSH_ARGS[@]}" $REMOTE_USER@$REMOTE_HOST)

# 通过环境变量传递敏感值和变更文件列表到远端
"${SSH_RUN[@]}" \
  "DEEPSEEK_API_KEY='$DEEPSEEK_API_KEY' CHANGED_PY_FILES_B64='$CHANGED_PY_FILES_B64' bash -s" << 'ENDSSH'
set -e

REMOTE_PATH="/var/www/hermes-agent"
SERVICE_NAME="airi-love-hermes.service"
python3_12_path="/usr/local/bin/python3.12"
SERVICE_STOPPED=0

restore_service() {
  if [ "$SERVICE_STOPPED" = "1" ]; then
    echo "🛟 部署失败，尝试恢复 $SERVICE_NAME ..."
    systemctl start "$SERVICE_NAME" || true
  fi
}

trap 'restore_service' ERR

# === 3a: 解压到独立目录 ===
echo "📂 解压代码到独立目录（避免直接覆盖生产）..."
EXTRACT_ROOT="/root/hermes_extract_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXTRACT_ROOT"
cd /root
tar xzf hermes-agent-deploy.tar.gz -C "$EXTRACT_ROOT"
# 清理 macOS 隐藏 metadata 文件，防止 python 加载 UnicodeDecodeError 崩溃
find "$EXTRACT_ROOT" -name "._*" -delete
echo "  ✅ 已解压到: $EXTRACT_ROOT"

# === 3b: 远端 Python 语法兼容检查（py_compile） ===
if [ -n "${CHANGED_PY_FILES_B64:-}" ]; then
  echo "🐍 远端 Python 语法兼容检查..."
  "$REMOTE_PATH/venv/bin/python3" --version 2>/dev/null || python3 --version
  EXTRACT_ROOT="$EXTRACT_ROOT" CHANGED_PY_FILES_B64="$CHANGED_PY_FILES_B64" \
    "${REMOTE_PATH}/venv/bin/python3" - <<'PYEOF'
import base64
import os
import py_compile

extract_root = os.environ["EXTRACT_ROOT"]
encoded = os.environ.get("CHANGED_PY_FILES_B64", "")
raw = base64.b64decode(encoded.encode("utf-8")).decode("utf-8") if encoded else ""
files = [line.strip() for line in raw.splitlines() if line.strip()]
targets, missing = [], []
for rel_path in files:
    target = os.path.join(extract_root, rel_path)
    if os.path.isfile(target):
        targets.append(target)
    else:
        missing.append(rel_path)
if missing:
    print("⚠️ 以下文件未在解压目录中找到，已跳过：")
    for item in missing:
        print(f"  - {item}")
failed = []
for path in targets:
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✅ {os.path.relpath(path, extract_root)}")
    except Exception as exc:
        failed.append((path, str(exc)))
if failed:
    print("❌ 远端 Python 语法检查失败：")
    for path, error in failed:
        print(f"  - {os.path.relpath(path, extract_root)} -> {error}")
    raise SystemExit(1)
print("  ✅ 远端 Python 语法检查通过")
PYEOF
fi

# === 3c: FTS trigram 降级逻辑预检（替代 schema migration 预检） ===
# Hermes 用 SQLite，无 Alembic；FTS schema 在代码 _ensure_fts_schema() 里 ensure。
# 检查 trigram 降级补丁存在，避免部署旧版本导致 500 复发。
echo "🔍 检查 FTS trigram 降级逻辑..."
if grep -q "no such tokenizer" "$EXTRACT_ROOT/hermes_state.py"; then
  echo "  ✅ trigram 降级补丁存在（_is_fts5_unavailable_error 含 no such tokenizer 分支）"
else
  echo "  ⚠️ trigram 降级补丁未找到，老 SQLite 环境可能复现 500 故障，请确认代码版本"
fi

# === 3d: 停服务前确保部署目录与 venv 存在 ===
mkdir -p "$REMOTE_PATH"

# 确保有 venv（首次部署兜底）
if [ ! -d "$REMOTE_PATH/venv" ]; then
  echo "🐍 正在创建 Python 3.12 虚拟环境..."
  "$python3_12_path" -m venv "$REMOTE_PATH/venv"
  echo "  ✅ 虚拟环境创建成功"
fi

# === 3e: 安装依赖（停服前装，避免服务带旧依赖启动） ===
echo "🐍 安装项目 [web,messaging] 依赖..."
"$REMOTE_PATH/venv/bin/pip" install --upgrade pip
"$REMOTE_PATH/venv/bin/pip" install -e "$EXTRACT_ROOT[web,messaging]"
echo "  ✅ 依赖安装完成"

# === 3f: 停服务 + 端口检查 ===
echo "🔄 停止 $SERVICE_NAME..."
systemctl stop "$SERVICE_NAME" || true
SERVICE_STOPPED=1
sleep 1

if command -v ss >/dev/null 2>&1; then
  if ss -ltnp | grep -q ':8009 '; then
    echo "  ⚠️ 端口 8009 仍被占用（可能是残留进程）："
    ss -ltnp | grep ':8009 ' || true
  fi
fi

# === 3g: rsync 同步代码到生产目录（排除运行时数据，不覆盖 profiles/venv/.env） ===
echo "🔄 同步代码到生产目录..."
rsync -a --delete \
  --exclude='venv' \
  --exclude='.env' \
  --exclude='profiles' \
  --exclude='.plans' \
  --exclude='.omx' \
  --exclude='.loop-state' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='hermes_agent.egg-info' \
  --exclude='*.bak' \
  --exclude='*.bak.*' \
  --exclude='*.bak-*' \
  --exclude='logs' \
  --exclude='.git' \
  "$EXTRACT_ROOT/" "$REMOTE_PATH/"
echo "  ✅ 代码同步完成"

# === 3g-fix: 在持久目录重装 editable，确保 editable finder 的 MAPPING 指向 REMOTE_PATH
# 而非临时 EXTRACT_ROOT（部署后会被删除，导致 cron tick 的 hermes_cli 找不到模块）===
echo "🐍 在生产目录重装 editable（修正 editable finder 映射）..."
"$REMOTE_PATH/venv/bin/pip" install -e "$REMOTE_PATH[web,messaging]" --no-deps -q
echo "  ✅ editable 映射已指向 $REMOTE_PATH"

# === 3h: 同步 Savana 自进化脚本与技能到 Hermes 运行态目录 ===
echo "🧬 同步 Savana 自进化脚本与技能..."
mkdir -p /root/.hermes/scripts /root/.hermes/skills
if [ -f "$REMOTE_PATH/scripts/extract_recent_dialogues.py" ]; then
  cp "$REMOTE_PATH/scripts/extract_recent_dialogues.py" /root/.hermes/scripts/extract_recent_dialogues.py
  chmod 700 /root/.hermes/scripts
  chmod 700 /root/.hermes/scripts/extract_recent_dialogues.py
fi
for skill_name in savana-companion-evolution savana-companion-evolution-guarded; do
  if [ -d "$REMOTE_PATH/skills/$skill_name" ]; then
    rm -rf "/root/.hermes/skills/$skill_name"
    cp -R "$REMOTE_PATH/skills/$skill_name" "/root/.hermes/skills/$skill_name"
    chmod -R go-rwx "/root/.hermes/skills/$skill_name"
  fi
done

# === 3i: 微信桥接配置（从 airi-love-backend/.env 读取，保留原逻辑） ===
WECHAT_BRIDGE_ENV_FILE="/var/www/airi-love-backend/.env"
WECHAT_BRIDGE_ENABLED="true"
WECHAT_BRIDGE_URL="http://127.0.0.1:8005/api/v1/wechat-role-binding/bridge/inbound"
WECHAT_BRIDGE_SECRET=""

if [ -f "$WECHAT_BRIDGE_ENV_FILE" ]; then
  echo "🔗 从 $WECHAT_BRIDGE_ENV_FILE 同步微信桥接配置..."
  bridge_enabled_raw=$(grep '^WECHAT_INBOUND_BRIDGE_ENABLED=' "$WECHAT_BRIDGE_ENV_FILE" | head -n1 | cut -d= -f2- || true)
  bridge_secret_raw=$(grep '^WECHAT_INBOUND_BRIDGE_SECRET=' "$WECHAT_BRIDGE_ENV_FILE" | head -n1 | cut -d= -f2- || true)
  if [ -n "$bridge_enabled_raw" ]; then
    WECHAT_BRIDGE_ENABLED=$(printf '%s' "$bridge_enabled_raw" | tr '[:upper:]' '[:lower:]')
  fi
  WECHAT_BRIDGE_SECRET=$(printf '%s' "$bridge_secret_raw" | sed 's/^["'\'']//; s/["'\'']$//')
fi

# === 3j: 写 systemd unit（DEEPSEEK_API_KEY 从环境变量注入，不硬编码） ===
echo "⚙️  写 systemd 配置..."
cat <<EOF > /etc/systemd/system/airi-love-hermes.service
[Unit]
Description=Airi Love Hermes Companion Service
After=network.target

[Service]
User=root
WorkingDirectory=/var/www/hermes-agent
Environment=DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
Environment=DEEPSEEK_API_BASE=https://api.deepseek.com/v1
Environment="SAVANA_WECHAT_BRIDGE_ENABLED=${WECHAT_BRIDGE_ENABLED}"
Environment="SAVANA_WECHAT_BRIDGE_URL=${WECHAT_BRIDGE_URL}"
Environment="SAVANA_WECHAT_BRIDGE_SECRET=${WECHAT_BRIDGE_SECRET}"
ExecStart=/var/www/hermes-agent/venv/bin/python -m uvicorn hermes_cli.web_server:app --port 8009 --host 127.0.0.1 --workers 4
Restart=always
RestartSec=2
WatchdogSec=30

[Install]
WantedBy=multi-user.target
EOF

# === 3k: 重启 + cron ===
systemctl daemon-reload
systemctl enable airi-love-hermes.service
systemctl restart airi-love-hermes.service
SERVICE_STOPPED=0

echo "⏰ 确保 Hermes cron tick 以 5 分钟频率运行（DEEPSEEK_API_KEY 从环境变量注入）..."
CRON_LINE="*/5 * * * * DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY} /var/www/hermes-agent/venv/bin/hermes cron tick >> /root/.hermes/logs/cron_tick.log 2>&1"
mkdir -p /root/.hermes/logs
TMP_CRON=$(mktemp)
crontab -l 2>/dev/null | grep -v '/var/www/hermes-agent/venv/bin/hermes cron tick' > "$TMP_CRON" || true
printf '%s\n' "$CRON_LINE" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"

# === 3l: 健康检查 ===
echo "🔍 检查服务运行状态..."
sleep 2
systemctl is-active --quiet "$SERVICE_NAME" || (systemctl status "$SERVICE_NAME" --no-pager && exit 1)
echo "  ✅ $SERVICE_NAME 状态正常，已在 8009 端口运行"
echo "  ✅ Hermes cron tick 已配置为每 5 分钟执行一次"

echo "🩺 验证本地回显..."
curl_out=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8009/companion/v1/chat || echo "FAILED")
echo "  ✅ HTTP response code from localhost:8009 (expected 405/422): $curl_out"

# 清理
rm -f /root/hermes-agent-deploy.tar.gz
rm -rf "$EXTRACT_ROOT"

ENDSSH

# 清理本地临时压缩包
rm -f "$LOCAL_TAR"

echo ""
echo "========================================="
echo "✅ Hermes 远程部署执行完成！"
echo "服务地址: http://127.0.0.1:8009 (仅本机)"
echo "========================================="
