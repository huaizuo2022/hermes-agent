import os
import shutil
from hermes_cli.companion_api import sync_soul_file

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
