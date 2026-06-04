import os
import shutil
import asyncio
import pytest
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


def test_sync_soul_file_preserves_evolved_persona(tmp_path):
    profile_dir = tmp_path / "profile_test"
    profile_dir.mkdir()
    
    # 1. Create a SOUL.md with existing Evolved Persona
    soul_path = profile_dir / "SOUL.md"
    soul_path.write_text(
        "# 小雪\n\n## Personality\n温柔的学妹\n\n## Evolved Persona\n开始经常对用户撒娇\n",
        encoding="utf-8"
    )
    
    # 2. Call sync_soul_file with new base profile (which would normally overwrite it)
    profile_data = {
        "name": "小雪",
        "personality": "温柔的学妹（基础人设更新）",
        "speaking_style": "学妹语气",
    }
    
    sync_soul_file(str(profile_dir), profile_data)
    
    # 3. Read it back and assert that Evolved Persona is preserved while basic traits are updated
    content = soul_path.read_text(encoding="utf-8")
    assert "## Personality\n温柔的学妹（基础人设更新）" in content
    assert "## Evolved Persona\n开始经常对用户撒娇" in content


def test_sync_soul_file_initializes_default_evolved_persona(tmp_path):
    profile_dir = tmp_path / "profile_test"
    profile_dir.mkdir()
    
    # Test case 1: SOUL.md does not exist
    profile_data = {
        "name": "小雪",
        "personality": "温柔的学妹",
        "speaking_style": "学妹语气",
    }
    sync_soul_file(str(profile_dir), profile_data)
    
    soul_path = profile_dir / "SOUL.md"
    assert soul_path.exists()
    content = soul_path.read_text(encoding="utf-8")
    assert "## Evolved Persona\n(暂无自主进化，人设遵循基础设定)" in content


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


def test_relay_weixin_message_bridges_to_backend_and_sends_reply(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "success": True,
                "data": {
                    "reply": "微信回复测试",
                },
            }

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            captured["json"] = dict(json or {})
            return _FakeResponse()

    sent_payload = {}

    async def _fake_send_weixin_direct(*, extra, token, chat_id, message, media_files=None):
        sent_payload["extra"] = dict(extra)
        sent_payload["token"] = token
        sent_payload["chat_id"] = chat_id
        sent_payload["message"] = message
        sent_payload["media_files"] = media_files
        return {"success": True}

    monkeypatch.setattr(
        "hermes_cli.companion_api.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(),
    )
    monkeypatch.setattr(
        "gateway.platforms.weixin.send_weixin_direct",
        _fake_send_weixin_direct,
    )

    from hermes_cli.companion_api import _relay_inbound_weixin_message

    reply = asyncio.run(
        _relay_inbound_weixin_message(
            hermes_home=str(tmp_path),
            bridge_url="http://127.0.0.1:8005/api/v1/wechat-role-binding/bridge/inbound",
            bridge_secret="bridge-secret",
            account_id="bot-account",
            account_payload={
                "token": "bot-token",
                "base_url": "https://ilinkai.weixin.qq.com",
            },
            message={
                "from_user_id": "wxid_test_user",
                "message_id": "msg-1",
                "context_token": "ctx-token-1",
                "item_list": [
                    {
                        "type": 1,
                        "text_item": {
                            "text": "你好",
                        },
                    },
                ],
            },
        )
    )

    assert reply == "微信回复测试"
    assert captured["url"].endswith("/api/v1/wechat-role-binding/bridge/inbound")
    assert captured["headers"]["X-WeChat-Bridge-Secret"] == "bridge-secret"
    assert captured["json"]["wechat_channel_user_id"] == "wxid_test_user"
    assert captured["json"]["message_text"] == "你好"
    assert sent_payload["chat_id"] == "wxid_test_user"
    assert sent_payload["message"] == "微信回复测试"
    assert sent_payload["token"] == "bot-token"
