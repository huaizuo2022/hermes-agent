import os
import shutil

from starlette.testclient import TestClient

from hermes_cli.companion_api import sync_soul_file
from hermes_cli.web_server import app

def test_sync_soul_file_appends_relationship(tmp_path):
    profile_dir = tmp_path / "profile_test"
    profile_dir.mkdir()
    
    profile_data = {
        "name": "小雪",
        "personality": "温柔的学妹",
        "speaking_style": "学妹语气",
        "relationship": {
            "relationship_stage": "ambiguous",
            "intimacy_score": 8,
            "trust_score": 7,
            "preferred_nickname": "学长",
            "persona_profile": "程序员学长",
            "persona_prompt_constraints": "不进行越界互动"
        }
    }
    
    sync_soul_file(str(profile_dir), profile_data)
    
    soul_path = profile_dir / "SOUL.md"
    assert soul_path.exists()
    
    content = soul_path.read_text(encoding="utf-8")
    assert "## Relationship with User" in content
    assert "- Current Stage: ambiguous (暧昧期，关系推拉有张力)" in content
    assert "- Intimacy level: 8/10" in content
    assert "- Trust level: 7/10" in content
    assert "- Preferred Nickname: 学长" in content
    assert "- User Profile: 程序员学长" in content
    assert "- Prompt Constraints: 不进行越界互动" in content


def test_weixin_qr_start_returns_scannable_qr_payload(monkeypatch):
    async def _fake_start(hermes_home, session_id, bot_type="3", timeout_seconds=480):
        assert session_id == "savana_user_char"
        return {
            "qrcode": "qr-token-1",
            "qrcode_img_content": "https://example.com/ilink-qr",
            "status": "wait",
        }

    monkeypatch.setattr(
        "hermes_cli.companion_api._start_weixin_qr_session",
        _fake_start,
    )

    client = TestClient(app)
    response = client.post(
        "/companion/v1/weixin/qr/start",
        json={"session_id": "savana_user_char"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["qrcode"] == "qr-token-1"
    assert payload["qrcode_img_content"] == "https://example.com/ilink-qr"


def test_weixin_qr_status_returns_confirmed_payload(monkeypatch):
    async def _fake_status(hermes_home, session_id, qrcode, base_url=None):
        assert session_id == "savana_user_char"
        assert qrcode == "qr-token-1"
        return {
            "status": "confirmed",
            "ilink_bot_id": "bot-account",
            "bot_token": "bot-token",
            "baseurl": "https://ilinkai.weixin.qq.com",
            "ilink_user_id": "wxid_bound_user",
        }

    monkeypatch.setattr(
        "hermes_cli.companion_api._query_weixin_qr_status",
        _fake_status,
    )

    client = TestClient(app)
    response = client.post(
        "/companion/v1/weixin/qr/status",
        json={
            "session_id": "savana_user_char",
            "qrcode": "qr-token-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmed"
    assert payload["ilink_user_id"] == "wxid_bound_user"
