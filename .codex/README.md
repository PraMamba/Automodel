# Codex Migration for AutoModel Claude Workflow

This `.codex` tree is a complete migration of the non-hook assets from the repository's top-level `.claude` workflow. It follows the approved hybrid native-first strategy: use Codex-native assets where reliable, and use prompt templates or documentation where Claude Code concepts do not have a dependable one-to-one Codex equivalent.

## Native Codex assets

- `.codex/agents/*.toml`: Codex custom agents migrated from `.claude/agents/*.md`.
- `.codex/skills/*`: Skill folders containing `SKILL.md`, ready to enable via `skills.config`.
- `.codex/config.toml`: Minimal helper config with agent concurrency defaults and migrated skill entries. It intentionally contains no hooks.

## Prompt/template/documentation fallbacks

- `.codex/commands/*.md`: Claude slash commands migrated as Codex prompt templates/workflow guides.
- `.codex/rules/*.md`: Claude path-scoped rules migrated as project guidance documents. This migration does not claim automatic path-scoped enforcement.
- `.codex/data/*.md`: Support data used by command templates.

## Source-to-target mapping

Full machine-readable mapping: `.codex/migrations/claude-to-codex-map.json`.

### Agents

| Source | Target | Strategy |
|---|---|---|
| `.claude/agents/checkpoint-expert.md` | `.codex/agents/checkpoint-expert.toml` | native |
| `.claude/agents/code-verifier.md` | `.codex/agents/code-verifier.toml` | native |
| `.claude/agents/distributed-expert.md` | `.codex/agents/distributed-expert.toml` | native |
| `.claude/agents/launcher-expert.md` | `.codex/agents/launcher-expert.toml` | native |
| `.claude/agents/model-expert.md` | `.codex/agents/model-expert.toml` | native |
| `.claude/agents/moe-expert.md` | `.codex/agents/moe-expert.toml` | native |
| `.claude/agents/peft-expert.md` | `.codex/agents/peft-expert.toml` | native |
| `.claude/agents/planner.md` | `.codex/agents/planner.toml` | native |
| `.claude/agents/recipe-expert.md` | `.codex/agents/recipe-expert.toml` | native |
| `.claude/agents/simple-code-reviewer.md` | `.codex/agents/simple-code-reviewer.toml` | native |

### Commands

| Source | Target | Strategy |
|---|---|---|
| `.claude/commands/create-pr.md` | `.codex/commands/create-pr.md` | template |
| `.claude/commands/gen-commit-msg.md` | `.codex/commands/gen-commit-msg.md` | template |
| `.claude/commands/review-pr.md` | `.codex/commands/review-pr.md` | template |

### Data

| Source | Target | Strategy |
|---|---|---|
| `.claude/data/review-pr-change-types.md` | `.codex/data/review-pr-change-types.md` | copy-compatible |
| `.claude/data/review-pr-templates.md` | `.codex/data/review-pr-templates.md` | copy-compatible |

### Rules

| Source | Target | Strategy |
|---|---|---|
| `.claude/rules/code-style.md` | `.codex/rules/code-style.md` | guidance |
| `.claude/rules/config-yaml.md` | `.codex/rules/config-yaml.md` | guidance |
| `.claude/rules/distributed.md` | `.codex/rules/distributed.md` | guidance |
| `.claude/rules/testing.md` | `.codex/rules/testing.md` | guidance |

### Skills

| Source | Target | Strategy |
|---|---|---|
| `.claude/skills/automodel-dr/SKILL.md` | `.codex/skills/automodel-dr/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-attention/SKILL.md` | `.codex/skills/automodel-dr-attention/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-autonvtx/SKILL.md` | `.codex/skills/automodel-dr-autonvtx/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-checkpoint/SKILL.md` | `.codex/skills/automodel-dr-checkpoint/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-cli/SKILL.md` | `.codex/skills/automodel-dr-cli/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-config/SKILL.md` | `.codex/skills/automodel-dr-config/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-datasets/SKILL.md` | `.codex/skills/automodel-dr-datasets/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-diffusers/SKILL.md` | `.codex/skills/automodel-dr-diffusers/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-distributed/SKILL.md` | `.codex/skills/automodel-dr-distributed/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-launcher/SKILL.md` | `.codex/skills/automodel-dr-launcher/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-loggers/SKILL.md` | `.codex/skills/automodel-dr-loggers/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-loss/SKILL.md` | `.codex/skills/automodel-dr-loss/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-models/SKILL.md` | `.codex/skills/automodel-dr-models/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-moe/SKILL.md` | `.codex/skills/automodel-dr-moe/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-optim/SKILL.md` | `.codex/skills/automodel-dr-optim/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-peft/SKILL.md` | `.codex/skills/automodel-dr-peft/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-quantization/SKILL.md` | `.codex/skills/automodel-dr-quantization/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-recipes-biencoder/SKILL.md` | `.codex/skills/automodel-dr-recipes-biencoder/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-recipes-llm/SKILL.md` | `.codex/skills/automodel-dr-recipes-llm/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-recipes-vlm/SKILL.md` | `.codex/skills/automodel-dr-recipes-vlm/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-shared/SKILL.md` | `.codex/skills/automodel-dr-shared/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-training/SKILL.md` | `.codex/skills/automodel-dr-training/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-transformers/SKILL.md` | `.codex/skills/automodel-dr-transformers/SKILL.md` | copy-compatible |
| `.claude/skills/automodel-dr-utils/SKILL.md` | `.codex/skills/automodel-dr-utils/SKILL.md` | copy-compatible |

## Omitted hooks and settings

Executable hook migration is intentionally omitted. The source Claude workflow used `.claude/settings.json` to run `.claude/hooks/check-expert-update.sh` after Write/Edit tool use, reminding users to update expert agents for changed code paths. This migration does **not** create `.codex/hooks.json`, `.codex/hooks/`, `[hooks]`, or `features.codex_hooks = true`.

Claude-local permission settings from `.claude/settings.local.json` are also not copied into Codex policy. Codex sandbox and approval settings should be chosen by the user/session, not inferred from Claude's local allowlist.

Omitted source entries:

| Source | Reason |
|---|---|
| `.claude/hooks/check-expert-update.sh` | Executable hooks are explicitly out of scope. Source behavior is documented in .codex/README.md. |
| `.claude/settings.json` | Claude settings are not migrated as executable Codex policy. Relevant omissions are documented in .codex/README.md. |
| `.claude/settings.local.json` | Claude settings are not migrated as executable Codex policy. Relevant omissions are documented in .codex/README.md. |

## Using migrated agents

Ask Codex to spawn the custom agent by its `name` from the TOML file. Example:

```text
Spawn the distributed-expert agent to review this tensor-parallel change, then summarize risks.
```

Subagents inherit the active sandbox/approval policy unless a custom agent config overrides it.

## Using migrated skills

This migration includes active `skills.config` entries in `.codex/config.toml` for all migrated skills. If you want a smaller setup, disable or remove entries you do not need. Example format:

```toml
[[skills.config]]
path = ".codex/skills/automodel-dr"
enabled = true

[[skills.config]]
path = ".codex/skills/automodel-dr-attention"
enabled = true

[[skills.config]]
path = ".codex/skills/automodel-dr-autonvtx"
enabled = true
```

## Using command templates

Open a file in `.codex/commands/`, paste or reference its workflow in Codex, and provide any required arguments manually. For example:

```text
Use .codex/commands/review-pr.md to review PR 123.
```

## Applying rules

Rules in `.codex/rules/` are guidance documents. Use them when working on matching files, and consider promoting selected rules into `AGENTS.md` in a separate user-approved task if automatic project instruction loading is desired.

## Validation command

Run from the repository root:

```bash
python .codex/migrations/validate-claude-to-codex.py
```

Expected success summary:

```text
Validation: PASS
Agents: 10/10 TOML parsed with required fields
Commands: 3/3 represented
Data: 2/2 represented
Rules: 4/4 represented
Skills: 24/24 represented; references 6/6 preserved
Hooks: 0 executable hook outputs created
README: required sections present
```

## Known limitations

- Command templates are not guaranteed project-local slash commands.
- Rules are not guaranteed automatic path-scoped enforcement.
- Hook reminder automation is intentionally absent.
- Model hints from Claude agents are preserved as prose, not mapped to fixed Codex models.
