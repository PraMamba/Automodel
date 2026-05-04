#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
CLAUDE = ROOT / ".claude"
CODEX = ROOT / ".codex"
MANIFEST = CODEX / "migrations" / "claude-to-codex-map.json"
REQUIRED_MANIFEST_FIELDS = {"source", "target", "kind", "strategy", "status", "notes"}


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def load_manifest(errors: list[str]) -> list[dict]:
    if not MANIFEST.exists():
        fail(f"missing manifest: {MANIFEST}", errors)
        return []
    try:
        data = json.loads(MANIFEST.read_text())
    except Exception as exc:
        fail(f"manifest JSON parse failed: {exc}", errors)
        return []
    if not isinstance(data, list):
        fail("manifest must be a list", errors)
        return []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            fail(f"manifest entry {i} is not an object", errors)
            continue
        missing = REQUIRED_MANIFEST_FIELDS - set(entry)
        if missing:
            fail(f"manifest entry {i} missing fields: {sorted(missing)}", errors)
    return data


def expected_sources():
    patterns = [
        ("agent", CLAUDE / "agents", "*.md"),
        ("command", CLAUDE / "commands", "*.md"),
        ("data", CLAUDE / "data", "*.md"),
        ("rule", CLAUDE / "rules", "*.md"),
        ("skill", CLAUDE / "skills", "*/SKILL.md"),
        ("skill-reference", CLAUDE / "skills", "*/reference.md"),
        ("hook", CLAUDE / "hooks", "*"),
    ]
    out = []
    for kind, base, pattern in patterns:
        out.extend((kind, p) for p in sorted(base.glob(pattern)) if p.is_file())
    for settings in [CLAUDE / "settings.json", CLAUDE / "settings.local.json"]:
        if settings.exists():
            out.append(("settings", settings))
    return out


def has_frontmatter(path: Path, keys=("name", "description")) -> bool:
    text = path.read_text(errors="ignore")
    if not text.startswith("---"):
        return False
    try:
        front = text.split("---", 2)[1]
    except Exception:
        return False
    return all(f"{key}:" in front for key in keys)


def main() -> int:
    errors: list[str] = []
    manifest = load_manifest(errors)
    entries_by_source = {entry.get("source"): entry for entry in manifest if isinstance(entry, dict)}

    for kind, src in expected_sources():
        rel = src.relative_to(ROOT).as_posix()
        entry = entries_by_source.get(rel)
        if not entry:
            fail(f"missing manifest entry for {rel}", errors)
            continue
        if entry.get("kind") != kind:
            fail(f"{rel}: kind {entry.get('kind')!r} != expected {kind!r}", errors)
        if kind in {"hook", "settings"}:
            if (
                entry.get("strategy") != "omitted"
                or entry.get("target") is not None
                or entry.get("status") != "omitted"
            ):
                fail(f"{rel}: hook/settings entries must be omitted with null target", errors)
        else:
            target = entry.get("target")
            if not target:
                fail(f"{rel}: non-hook source has empty target", errors)
            elif not (ROOT / target).exists():
                fail(f"{rel}: target missing: {target}", errors)

    agent_files = sorted((CODEX / "agents").glob("*.toml"))
    source_agents = sorted((CLAUDE / "agents").glob("*.md"))
    if len(agent_files) != len(source_agents):
        fail(f"agent count mismatch: {len(agent_files)} targets vs {len(source_agents)} sources", errors)
    for path in agent_files:
        try:
            data = tomllib.loads(path.read_text())
        except Exception as exc:
            fail(f"{path}: TOML parse failed: {exc}", errors)
            continue
        for field in ["name", "description", "developer_instructions"]:
            if not str(data.get(field, "")).strip():
                fail(f"{path}: missing required field {field}", errors)
        if len(str(data.get("developer_instructions", ""))) < 200:
            fail(f"{path}: developer_instructions looks too short", errors)
        for forbidden in ["tools"]:
            if forbidden in data:
                fail(f"{path}: unsupported Claude field copied as TOML key: {forbidden}", errors)

    expected_commands = sorted((CLAUDE / "commands").glob("*.md"))
    for src in expected_commands:
        target = CODEX / "commands" / src.name
        if not target.exists():
            fail(f"missing command target: {target}", errors)
        else:
            text = target.read_text(errors="ignore")
            if "Codex compatibility note" not in text or "## Usage" not in text:
                fail(f"{target}: missing compatibility note or usage section", errors)
            if ".claude/data" in text:
                fail(f"{target}: stale .claude/data reference", errors)

    for src in sorted((CLAUDE / "data").glob("*.md")):
        if not (CODEX / "data" / src.name).exists():
            fail(f"missing data target: {CODEX / 'data' / src.name}", errors)

    for src in sorted((CLAUDE / "rules").glob("*.md")):
        target = CODEX / "rules" / src.name
        if not target.exists():
            fail(f"missing rule target: {target}", errors)
        elif "Codex compatibility note" not in target.read_text(errors="ignore"):
            fail(f"{target}: missing compatibility note", errors)

    skill_sources = sorted((CLAUDE / "skills").glob("*/SKILL.md"))
    for src in skill_sources:
        target = CODEX / "skills" / src.parent.name / "SKILL.md"
        if not target.exists():
            fail(f"missing skill target: {target}", errors)
        elif not has_frontmatter(target):
            fail(f"{target}: missing name/description frontmatter", errors)
    ref_sources = sorted((CLAUDE / "skills").glob("*/reference.md"))
    for src in ref_sources:
        target = CODEX / "skills" / src.parent.name / "reference.md"
        if not target.exists():
            fail(f"missing skill reference target: {target}", errors)

    if (CODEX / "hooks").exists():
        fail("forbidden executable hook directory exists: .codex/hooks", errors)
    if (CODEX / "hooks.json").exists():
        fail("forbidden executable hook config exists: .codex/hooks.json", errors)
    config = CODEX / "config.toml"
    if config.exists():
        text = config.read_text(errors="ignore")
        if "[hooks]" in text or "features.codex_hooks = true" in text:
            fail(".codex/config.toml contains forbidden hook configuration", errors)
        if "Bash(git push *)" in text or "settings.local" in text:
            fail(".codex/config.toml appears to copy Claude local permissions", errors)
        try:
            tomllib.loads(text)
        except Exception as exc:
            fail(f".codex/config.toml TOML parse failed: {exc}", errors)

    readme = CODEX / "README.md"
    if not readme.exists():
        fail("missing .codex/README.md", errors)
        readme_text = ""
    else:
        readme_text = readme.read_text(errors="ignore")
    required_readme_phrases = [
        "Source-to-target mapping",
        "Native Codex assets",
        "Prompt/template/documentation fallbacks",
        "Omitted hooks and settings",
        "Validation command",
        "Known limitations",
        "skills.config",
    ]
    for phrase in required_readme_phrases:
        if phrase not in readme_text:
            fail(f"README missing required section/phrase: {phrase}", errors)

    if errors:
        print("Validation: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Validation: PASS")
    print(f"Agents: {len(agent_files)}/{len(source_agents)} TOML parsed with required fields")
    print(f"Commands: {len(expected_commands)}/{len(expected_commands)} represented")
    print(f"Data: {len(list((CLAUDE / 'data').glob('*.md')))}/{len(list((CLAUDE / 'data').glob('*.md')))} represented")
    print(
        f"Rules: {len(list((CLAUDE / 'rules').glob('*.md')))}/{len(list((CLAUDE / 'rules').glob('*.md')))} represented"
    )
    print(
        f"Skills: {len(skill_sources)}/{len(skill_sources)} represented; references {len(ref_sources)}/{len(ref_sources)} preserved"
    )
    print("Hooks: 0 executable hook outputs created")
    print("README: required sections present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
