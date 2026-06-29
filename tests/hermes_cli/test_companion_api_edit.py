import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from hermes_cli.web_server import app

client = TestClient(app)

def test_update_message_content_in_local_db(tmp_path, monkeypatch):
    session_id = "test_session_123"
    message_id = "msg_abc"
    
    # 1. Setup profile dir and temporary DB
    profile_dir = tmp_path / session_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    db_path = profile_dir / "state.db"
    
    # Mock get_profile_path
    monkeypatch.setattr(
        "hermes_cli.companion_api.get_profile_path",
        lambda sid: str(tmp_path / sid)
    )
    
    from hermes_state import SessionDB
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(
            session_id=session_id,
            source="test",
            model="test-model",
        )
        
        # Insert a message in the DB
        conn = db._conn
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, platform_message_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", "Original Content", 123456.7, message_id)
        )
        conn.commit()
    finally:
        db.close()
        
    # 2. Call PUT /companion/v1/sessions/{session_id}/messages/{message_id}
    response = client.put(
        "/companion/v1/sessions/{}/messages/{}".format(session_id, message_id),
        json={"content": "Updated Content"}
    )
    
    # 3. Assertions
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True
    
    # Verify DB content was updated
    db = SessionDB(db_path=db_path)
    try:
        conn = db._conn
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content FROM messages WHERE session_id = ? AND platform_message_id = ?",
            (session_id, message_id)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "Updated Content"
    finally:
        db.close()

def test_update_message_not_found(tmp_path, monkeypatch):
    session_id = "test_session_123"
    message_id = "msg_nonexistent"
    
    # Setup profile dir and temporary DB
    profile_dir = tmp_path / session_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    db_path = profile_dir / "state.db"
    
    # Mock get_profile_path
    monkeypatch.setattr(
        "hermes_cli.companion_api.get_profile_path",
        lambda sid: str(tmp_path / sid)
    )
    
    from hermes_state import SessionDB
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(
            session_id=session_id,
            source="test",
            model="test-model",
        )
    finally:
        db.close()
        
    # Call PUT
    response = client.put(
        "/companion/v1/sessions/{}/messages/{}".format(session_id, message_id),
        json={"content": "Some Content"}
    )
    
    # Should return 404
    assert response.status_code == 404
