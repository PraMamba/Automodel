# Codex Prompt Template: gen-commit-msg

> Codex compatibility note: This file was migrated from `.claude/commands/gen-commit-msg.md`. Codex does not currently provide a guaranteed project-local Claude-style slash-command equivalent in this migration, so use this as a prompt template/workflow guide. Referenced support data has been migrated to `.codex/data/`.

## Usage

Paste or reference this template in Codex when you want the `gen-commit-msg` workflow. Preserve repository instructions from `AGENTS.md` and verify any GitHub or git side effects before running commands.

---

---
name: gen-commit-msg
description: Generate intelligent commit messages based on staged changes following Conventional Commits. Invoke with `/gen-commit-msg`.
---

# Generate Commit Message

Generate an intelligent commit message based on staged changes.

## Usage

```
/gen-commit-msg [--amend] [--scope <scope>]
```

**Arguments:**

- `--amend`: Amend the previous commit with new message
- `--scope <scope>`: Override auto-detected scope

## Workflow

### Step 1: Analyze Staged Changes

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached
```

If nothing staged, check for unstaged changes and suggest staging.

### Step 2: Categorize Changes

| Type | When to Use |
| ---- | ----------- |
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without feature/fix |
| `test` | Adding or fixing tests |
| `chore` | Build, deps, config changes |
| `perf` | Performance improvement |

### Step 3: Determine Scope

Auto-detect from changed files:

| Path Pattern | Scope |
| ------------ | ----- |
| `nemo_automodel/components/models/` | `models` |
| `nemo_automodel/components/distributed/` | `distributed` |
| `nemo_automodel/components/checkpoint/` | `checkpoint` |
| `nemo_automodel/components/datasets/` | `datasets` |
| `nemo_automodel/components/config/` | `config` |
| `nemo_automodel/components/loss/` | `loss` |
| `nemo_automodel/components/moe/` | `moe` |
| `nemo_automodel/components/_peft/` | `peft` |
| `nemo_automodel/components/training/` | `training` |
| `nemo_automodel/components/optim/` | `optim` |
| `nemo_automodel/components/loggers/` | `loggers` |
| `nemo_automodel/components/attention/` | `attention` |
| `nemo_automodel/components/quantization/` | `quantization` |
| `nemo_automodel/components/launcher/` | `launcher` |
| `nemo_automodel/recipes/` | `recipes` |
| `nemo_automodel/_transformers/` | `transformers` |
| `nemo_automodel/_cli/` | `cli` |
| `nemo_automodel/shared/` | `shared` |
| `docs/` | `docs` |
| `examples/` | `examples` |
| `tests/` | `tests` |
| Multiple areas | omit scope or use broader term |

### Step 4: Generate Message

Format:

```
<type>(<scope>): <subject>

<body>

Key changes:
- change 1
- change 2
```

**Rules:**

- Subject: ~72 chars, imperative mood, no period
- Body: Explain **why**, not just **what**
- Key changes: List significant modifications
- The `-s` flag on `git commit` auto-adds the `Signed-off-by` trailer; do NOT add it
  manually in the message body

### Step 5: Confirm and Commit

Show the generated message, ask user to confirm, then:

```bash
git commit -s -m "$(cat <<'EOF'
<generated commit message>
EOF
)"
```

Use `--amend` if requested.

## Examples

### Feature Commit

```
feat(models): add Qwen3-MoE model support

Add Qwen3-MoE model implementation with custom MoE layers,
state dict adapter for HF checkpoint conversion, and tensor
parallel plan for distributed training.

Key changes:
- Add Qwen3-MoE model definition in components/models/qwen3_moe/
- Implement state dict adapter for HF ↔ NeMo conversion
- Register model in _transformers/registry.py
- Add TP plan in distributed/optimized_tp_plans.py

# Signed-off-by added automatically by git commit -s
```

### Bug Fix Commit

```
fix(checkpoint): resolve shape mismatch on resume with different TP

Fix tensor shape mismatch when resuming training with a different
tensor parallel size than the original checkpoint. The state dict
adapter now handles resharding transparently.

Key changes:
- Update state_dict_adapter.py to detect TP size change
- Add resharding logic in DCP planner

# Signed-off-by added automatically by git commit -s
```
