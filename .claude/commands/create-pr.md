---
name: create-pr
description: Rebase from the latest `origin/main`, squash the commits from it, and then create a PR on GitHub with intelligent commit messages. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub with an
intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

**Arguments:**

- `--draft`: Create as draft PR
- `--base <branch>`: Target branch (default: `main`)

## Workflow

### Step 1: Verify Prerequisites

```bash
git branch --show-current

if [[ $(git branch --show-current) == "main" || $(git branch --show-current) == "master" ]]; then
  echo "ERROR: Cannot create PR from main/master branch"
  exit 1
fi

git status --short
gh --version
```

**Action:** If there are uncommitted changes, stop and ask user to commit or stash first.

### Step 2: Check for Existing PR

```bash
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

If PR exists, inform user and ask permission to force-update it.

### Step 3: Fetch and Rebase

```bash
git fetch origin main
git log --oneline HEAD ^origin/main
git rebase origin/main
```

If rebase fails, abort and let user handle manually.

### Step 4: Squash Commits into Single Commit

```bash
git rev-list --count origin/main..HEAD
git reset --soft origin/main
```

Generate commit message using `/gen-commit-msg` logic.

### Step 5: Analyze Combined Changes

```bash
git diff origin/main...HEAD --name-only
git diff origin/main...HEAD
```

**Determine Scope:**

- `nemo_automodel/components/models/` → `models`
- `nemo_automodel/components/distributed/` → `distributed`
- `nemo_automodel/components/checkpoint/` → `checkpoint`
- `nemo_automodel/components/datasets/` → `datasets`
- `nemo_automodel/components/config/` → `config`
- `nemo_automodel/components/loss/` → `loss`
- `nemo_automodel/components/moe/` → `moe`
- `nemo_automodel/components/_peft/` → `peft`
- `nemo_automodel/components/training/` → `training`
- `nemo_automodel/components/optim/` → `optim`
- `nemo_automodel/recipes/` → `recipes`
- `nemo_automodel/_transformers/` → `transformers`
- `nemo_automodel/_cli/` → `cli`
- `docs/` → `docs`
- `examples/` → `examples`
- `tests/` → `tests`
- Multiple areas → omit scope or use broader term

### Step 6: Generate PR Title and Description

**PR Title Format:**

```
<type>(<scope>): <brief description>
```

Keep under 70 characters, imperative mood, no period.

**PR Description Format** - follow `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
# What does this PR do ?

[Clear and concise description]

# Changelog

- [Specific line by line changes]

# Before your PR is "Ready for review"

**Pre checks**:

- [x] Make sure you read and followed [Contributor guidelines](CONTRIBUTING.md)
- [ ] Did you write any new necessary tests?
- [ ] Did you add or update any necessary documentation?

# Additional Information

- Related to # (issue)
```

### Step 7: Push and Create/Update PR

Show preview to user, then:

```bash
git push -f -u origin $(git branch --show-current)

if gh pr view &>/dev/null; then
  gh pr edit --title "..." --body "..."
else
  gh pr create --base main --title "..." --body "..."
fi
```

**Important**: Remind user about `--signoff` requirement for commits.

## Safety Checks

- Confirm no uncommitted changes
- Confirm not on main/master branch
- Check for existing PR and get user permission
- Show full preview before push
- Warn about force push rewriting history
