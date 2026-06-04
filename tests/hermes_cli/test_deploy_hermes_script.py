from pathlib import Path


def test_deploy_hermes_script_injects_wechat_bridge_envs():
    script = Path(__file__).resolve().parents[2] / "deploy-hermes.sh"
    content = script.read_text(encoding="utf-8")

    assert "WECHAT_BRIDGE_ENV_FILE=\"/var/www/airi-love-backend/.env\"" in content
    assert "SAVANA_WECHAT_BRIDGE_ENABLED" in content
    assert "SAVANA_WECHAT_BRIDGE_URL" in content
    assert "SAVANA_WECHAT_BRIDGE_SECRET" in content
    assert "WECHAT_INBOUND_BRIDGE_SECRET" in content
