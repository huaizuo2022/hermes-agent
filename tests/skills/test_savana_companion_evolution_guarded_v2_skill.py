import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "savana-companion-evolution-guarded-v2"
    / "SKILL.md"
)


def _skill_text():
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter_lines(text):
    lines = text.splitlines()
    assert lines[0] == "---"
    closing_index = lines.index("---", 1)
    return lines[1:closing_index]


def _frontmatter_value(lines, key):
    prefix = key + ":"
    matches = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    assert len(matches) == 1
    return matches[0].strip('"')


def test_guarded_v2_skill_frontmatter_meets_repository_contract():
    text = _skill_text()
    lines = _frontmatter_lines(text)
    description = _frontmatter_value(lines, "description")
    author = _frontmatter_value(lines, "author")
    platforms = _frontmatter_value(lines, "platforms")

    assert _frontmatter_value(lines, "name") == "savana-companion-evolution-guarded-v2"
    assert 0 < len(description) <= 60
    assert description.startswith("Use when ")
    assert description.endswith(".")
    assert "\n" not in description
    assert _frontmatter_value(lines, "version")
    assert author.startswith("Shanglong Huaizuo (@huaizuo2022)")
    assert "Hermes Agent" in author
    assert _frontmatter_value(lines, "license") == "MIT"
    assert set(platforms.strip("[]").replace(" ", "").split(",")) == {
        "linux",
        "macos",
        "windows",
    }
    assert "metadata:" in lines
    assert "  hermes:" in lines
    assert any(line.strip().startswith("tags:") for line in lines)
    assert any(line.strip().startswith("category:") for line in lines)
    assert any(line.strip().startswith("requires_toolsets:") for line in lines)


def test_guarded_v2_skill_uses_required_section_order():
    text = _skill_text()
    headings = re.findall(r"^#{1,2} .+$", text, re.MULTILINE)

    assert headings == [
        "# Savana Guarded Companion Self-Evolution V2 Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    introduction = text.split(headings[0], 1)[1].split("## When to Use", 1)[0].strip()
    assert 2 <= len(re.findall(r"[.!?](?:\s|$)", introduction)) <= 3


def test_guarded_v2_skill_preserves_evidence_and_output_contracts():
    text = _skill_text()

    for token in (
        "guarded_v2",
        "[evolution_evidence]",
        "[context_only]",
        "[quality_correction_only]",
        "[review unavailable]",
        "`necessary`",
        "`preserves_identity`",
        "`no_unfounded_jump`",
        "`no_error_solidification`",
        "`no_base_override`",
        "`candidate_evolved_persona`",
        "`expected_soul_sha256`",
        "`decision=no_change`",
        "`verdict=reject`",
        "<!-- GUARDED_EVOLUTION_RESULT ",
        " GUARDED_EVOLUTION_RESULT -->",
    ):
        assert token in text
    assert "Only `[evolution_evidence]` user content" in text
    assert "cannot independently justify a persistent change" in text
    assert "Base Persona Snapshot has higher authority" in text
    assert "exactly one single-line JSON object" in text
    assert "Never patch, write, or edit `SOUL.md`" in text
    assert "one physical line" in text
    assert "`\\n`" in text
    assert "`\\r`" in text
    assert "`\\\\`" in text
