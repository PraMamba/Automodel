# Codex Prompt Template: review-pr

> Codex compatibility note: This file was migrated from `.claude/commands/review-pr.md`. Codex does not currently provide a guaranteed project-local Claude-style slash-command equivalent in this migration, so use this as a prompt template/workflow guide. Referenced support data has been migrated to `.codex/data/`.

## Usage

Paste or reference this template in Codex when you want the `review-pr` workflow. Preserve repository instructions from `AGENTS.md` and verify any GitHub or git side effects before running commands.

---

---
name: review-pr
description: Intelligent PR code review with dynamic agent allocation based on change types. Invoke with `/review-pr` <PR_NUMBER> [--quick].
---

@.codex/data/review-pr-change-types.md @.codex/data/review-pr-templates.md

# Review Pull Request

Perform intelligent code review of a PR with dynamic agent allocation based on detected
change types.

## Usage

```
/review-pr <PR_NUMBER> [--quick]
```

**Arguments:**

- `<PR_NUMBER>`: GitHub PR number to review
- `--quick`: Skip deep analysis, quick summary only

## Workflow

### Phase 1: Deep PR Analysis

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,body,files,additions,deletions,state,baseRefName

# Get full diff
gh pr diff <PR_NUMBER>

# Check CI status
gh pr checks <PR_NUMBER>
```

**Analyze:**

1. PR summary and purpose
2. Files changed and categorize by change type (see `.codex/data/review-pr-change-types.md`)
3. Detect framework risks based on file paths
4. Assess overall complexity

### Phase 2: Dynamic Agent Planning

Based on detected change types, generate review tasks:

1. **Map changes to review templates** (see `.codex/data/review-pr-templates.md`)
2. **Assign model by severity**:
   - CRITICAL → Opus (distributed core, parallelizer, MoE layers)
   - HIGH → Opus (checkpoint, DTensor, TP plans)
   - MEDIUM → Sonnet (config, dataset, loss, optimizer)
   - LOW → Haiku (docs, tests only, config only)

### Phase 3: Execute Review Tasks

Run review tasks in parallel where possible:

1. Each task focuses on specific files and concern areas
2. Produce findings with confidence scores
3. Classify findings: CRITICAL / WARNING / INFO / SUGGESTION

### Phase 4: Summary Report

```markdown
## PR Review: #<NUMBER> - <TITLE>

### Overall Assessment
**Confidence**: [0-100] | **Risk Level**: [LOW/MEDIUM/HIGH/CRITICAL]

### Critical Issues
1. **[Issue]** - `file:line` - [description]

### Warnings
1. **[Warning]** - `file:line` - [description]

### Suggestions
1. **[Suggestion]** - `file:line` - [description]

### Positive Observations
- [Good patterns noticed]

### Files Not Reviewed
- [files skipped and why]

### Test Coverage Assessment
- [New code covered by tests?]
- [Existing tests still valid?]
```

## Change Type Detection

Refer to `.codex/data/review-pr-change-types.md` for:

- File path → change type mapping
- Severity levels for each type
- Risk linkage rules (e.g., distributed changes trigger checkpoint review)

## Review Templates

Refer to `.codex/data/review-pr-templates.md` for:

- Framework-specific review checklists
- General review checklists
- Model assignment per template

## False Positive Detection

Before reporting an issue, verify:

1. Is this actually a bug, or intentional design?
2. Is there a comment explaining the rationale?
3. Does the test suite cover this case?
4. Is there a related TODO or known issue?
