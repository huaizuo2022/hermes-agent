#!/bin/bash
# deploy-hermes.sh - 一键部署 Hermes 伴侣记忆引擎服务至远程服务器
set -e

echo "========================================="
echo "Hermes Engine Deployment Script"
echo "========================================="

REMOTE_USER="root"
REMOTE_HOST="146.56.229.151"
REMOTE_PASSWORD="8spCTD8uFm.BZQe"
REMOTE_PATH="/var/www/hermes-agent"
LOCAL_TAR="/tmp/hermes-agent-deploy.tar.gz"

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

# Step 1: 本地打包代码
echo "📦 Step 1: 打包本地 hermes-agent 代码..."
tar --exclude='venv' --exclude='.venv' --exclude='.git' \
    --exclude='.plans' --exclude='__pycache__' \
    -czf "$LOCAL_TAR" -C "/Users/shang/Dev/hermes-agent" .

echo "✅ 打包完成"

# Step 2: 上传代码至远程
echo "📤 Step 2: 上传部署包至远程服务器..."
sshpass -p "$REMOTE_PASSWORD" scp "${SSH_ARGS[@]}" \
  "$LOCAL_TAR" \
  $REMOTE_USER@$REMOTE_HOST:/root/hermes-agent-deploy.tar.gz

echo "✅ 上传完成"

# Step 3: 远端解压、安装依赖并重启服务
echo "🚀 Step 3: 远端部署依赖及服务配置..."
sshpass -p "$REMOTE_PASSWORD" ssh "${SSH_ARGS[@]}" $REMOTE_USER@$REMOTE_HOST "bash -s" << 'ENDSSH'
set -e

REMOTE_PATH="/var/www/hermes-agent"
python3_12_path="/usr/local/bin/python3.12"

echo "📂 创建和清理远端部署目录..."
mkdir -p "$REMOTE_PATH"

echo "📂 解压缩代码到 $REMOTE_PATH ..."
tar xzf /root/hermes-agent-deploy.tar.gz -C "$REMOTE_PATH"
# 清理所有 macOS 隐藏 metadata 文件，防止 python3 加载发生 UnicodeDecodeError 崩溃
find "$REMOTE_PATH" -name "._*" -delete

echo "🧬 同步 Savana 自进化脚本与技能到 Hermes 运行态目录..."
mkdir -p /root/.hermes/scripts /root/.hermes/skills
cp "$REMOTE_PATH/scripts/extract_recent_dialogues.py" /root/.hermes/scripts/extract_recent_dialogues.py
rm -rf /root/.hermes/skills/savana-companion-evolution
cp -R "$REMOTE_PATH/skills/savana-companion-evolution" /root/.hermes/skills/savana-companion-evolution
chmod 700 /root/.hermes/scripts
chmod 600 /root/.hermes/scripts/extract_recent_dialogues.py
chmod -R go-rwx /root/.hermes/skills/savana-companion-evolution

# 确保有 venv
if [ ! -d "$REMOTE_PATH/venv" ]; then
  echo "🐍 正在创建 Python 3.12 虚拟环境..."
  "$python3_12_path" -m venv "$REMOTE_PATH/venv"
  echo "  ✅ 虚拟环境创建成功"
fi

echo "🐍 正在激活虚拟环境并安装项目 [web,messaging] 依赖..."
"$REMOTE_PATH/venv/bin/pip" install --upgrade pip
"$REMOTE_PATH/venv/bin/pip" install -e "$REMOTE_PATH[web,messaging]"
echo "  ✅ 依赖安装成功"

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

echo "⚙️  正在创建 systemd 配置文件 /etc/systemd/system/airi-love-hermes.service ..."
cat <<EOF > /etc/systemd/system/airi-love-hermes.service
[Unit]
Description=Airi Love Hermes Companion Service
After=network.target

[Service]
User=root
WorkingDirectory=/var/www/hermes-agent
Environment=DEEPSEEK_API_KEY=sk-fe3ca6dcd52f4569bc013a4069879c18
Environment=DEEPSEEK_API_BASE=https://api.deepseek.com/v1
Environment="SAVANA_WECHAT_BRIDGE_ENABLED=${WECHAT_BRIDGE_ENABLED}"
Environment="SAVANA_WECHAT_BRIDGE_URL=${WECHAT_BRIDGE_URL}"
Environment="SAVANA_WECHAT_BRIDGE_SECRET=${WECHAT_BRIDGE_SECRET}"
ExecStart=/var/www/hermes-agent/venv/bin/python -m uvicorn hermes_cli.web_server:app --port 8009 --host 127.0.0.1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "🔄 正在重启 systemd 服务并启用开机启动..."
systemctl daemon-reload
systemctl enable airi-love-hermes.service
systemctl restart airi-love-hermes.service

echo "🔍 检查服务运行状态..."
sleep 2
systemctl is-active --quiet airi-love-hermes.service || (systemctl status airi-love-hermes.service --no-pager && exit 1)
echo "  ✅ airi-love-hermes.service 状态正常，已启动并在 8009 端口运行"

echo "🩺 验证本地回显..."
curl_out=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8009/companion/v1/chat || echo "FAILED")
echo "  ✅ HTTP response code from localhost:8009 (expected 405 Method Not Allowed/422): $curl_out"

# 清理远端压缩包
rm -f /root/hermes-agent-deploy.tar.gz

ENDSSH

# 清理本地临时压缩包
rm -f "$LOCAL_TAR"

echo ""
echo "========================================="
echo "✅ Hermes 远程部署执行完成！"
echo "========================================="
