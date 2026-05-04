# AGENTS.md - NeMo AutoModel

This file is the Codex-native project guidance for agents working in this repository.
Codex reads `AGENTS.md` before doing work and layers it with any deeper `AGENTS.md` or
`AGENTS.override.md` files. Follow system, developer, and user instructions first; this file
sets repository defaults.

## WHAT: Project Overview

NeMo AutoModel is a PyTorch DTensor-native SPMD library for training and fine-tuning
LLMs and VLMs with day-0 Hugging Face support, GPU acceleration, and memory efficiency.

**Tech Stack**: Python 3.10+ | PyTorch (DTensor/FSDP2) | HuggingFace Transformers | YAML Config

**Core Directories**:

- `nemo_automodel/` - Core package
  - `components/` - Independent training components
    - `config/` - YAML configuration system with `_target_` instantiation
    - `models/` - 20+ model implementations (Llama, DeepSeek-V3, Qwen3-MoE, etc.)
    - `datasets/` - LLM/VLM dataset loaders, sequence packing, Megatron data pipeline
    - `distributed/` - FSDP2, MegatronFSDP, tensor/context/pipeline parallelism
    - `checkpoint/` - Distributed checkpointing, DCP, SafeTensors, HF format conversion
    - `loss/` - Cross-entropy, knowledge distillation, chunked CE, Triton kernels
    - `optim/` - Optimizer creation, LR schedulers (cosine, warmup, constant)
    - `training/` - Step scheduler, timers, signal handling, RNG state
    - `loggers/` - W&B, MLflow, TensorBoard metric logging
    - `attention/` - Flex attention, context parallel attention
    - `moe/` - Mixture of Experts with DeepEP, expert parallelism
    - `_peft/` - LoRA, QLoRA, MoE-LoRA adapters
    - `quantization/` - FP8, QAT, QLoRA quantization
    - `launcher/` - SLURM job submission
    - `utils/` - Model construction, FLOPS, compile, YAML helpers
  - `recipes/` - Training recipes
    - `llm/` - Pretrain, SFT, KD, sequence classification
    - `vlm/` - Vision-language model fine-tuning
    - `biencoder/` - Contrastive biencoder training
  - `_transformers/` - HuggingFace auto-model/tokenizer registry
  - `_diffusers/` - Diffusion model support
  - `_cli/` - CLI entry point (`automodel` command)
  - `shared/` - Cross-cutting utilities (imports, patches)
- `examples/` - Example YAML configs and training scripts
- `tests/` - Functional and unit tests
- `docs/` - Sphinx documentation

## WHY: Purpose

- Enable efficient LLM/VLM training and fine-tuning with DTensor-native SPMD
- Provide day-0 HuggingFace model support via auto-registration
- Composable parallelism (TP, DP, PP, CP, SP, EP) via DeviceMesh
- Memory-efficient training with FSDP2, activation checkpointing, FP8

## HOW: Core Commands

```bash
# Check environment
python --version              # Requires 3.10+
uv --version                  # Install: https://docs.astral.sh/uv/

# Sync dependencies
uv sync --locked --extra all  # Full install
uv sync --locked              # Minimal install

# Repository pre-commit tooling
pre-commit install            # Set up repository hooks (run once)
pre-commit run --all-files    # Format and lint

# Run tests
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
uv run pytest tests/unit_tests/ -v
uv run pytest tests/functional_tests/ -v  # Many require GPU

# Build docs
uv sync --group docs
cd docs && sphinx-build . _build/html

# Run training
automodel --config examples/llm_pretrain/nanogpt_pretrain.yaml

# Validate Codex project assets
python .codex/migrations/validate-claude-to-codex.py
```

## Boundaries

### Constraints

- Designed for distributed GPU clusters; assume containerized execution.
- Functional tests require GPU hardware; explain skips when unavailable.
- Components must not cross-import; import boundaries are enforced by import-linter.
- `.codex/` contains Codex-native project assets plus documented fallback prompts and rules.
- Codex lifecycle hooks are intentionally not configured for this project. Repository
  `pre-commit` hooks are still normal development tooling and should be used.

### Always Do

- Read relevant files before modifying code.
- Use Codex project guidance from this file first, then consult `.codex/README.md` and
  specific `.codex/*` assets as needed.
- Run `pre-commit run --all-files` before committing when feasible.
- Follow existing code patterns in the same module.
- Add tests for new functionality or behavior changes.
- Sign commits with `--signoff` (`git commit -s`).
- For multi-file features, architectural work, or risky changes, plan first and use a
  specialized Codex subagent when it materially improves quality or speed.

### Ask First

- Modifying the config system in `nemo_automodel/components/config/`.
- Adding new dependencies to `pyproject.toml`.
- Changing distributed parallelism logic.
- Deleting or renaming public APIs.
- Modifying recipe training loops.
- Running GPU/distributed tests after checking GPU availability:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`.

### Never Do

- Hardcode secrets, paths, tokens, or endpoints.
- Skip pre-commit verification when preparing a commit.
- Use wildcard imports (`from x import *`).
- Cross-import between components.
- Guess cluster configs or rebuild CUDA/driver stacks.
- Use underscores in Markdown filenames; use hyphens instead.
- Add `.codex/hooks.json`, inline `[hooks]`, or `features.codex_hooks` unless a future
  task explicitly reintroduces Codex lifecycle hooks.

## Progressive Disclosure: Detailed Guides

| Task                   | Reference                                                        |
| ---------------------- | ---------------------------------------------------------------- |
| Add Model              | `nemo_automodel/components/models/llama/`, `_transformers/registry.py` |
| Add Dataset            | `nemo_automodel/components/datasets/`, `examples/llm_finetune/`  |
| Add Recipe             | `nemo_automodel/recipes/base_recipe.py`, `recipes/llm/train_ft.py` |
| Add Loss Function      | `nemo_automodel/components/loss/masked_ce.py`                    |
| Distributed Patterns   | `nemo_automodel/components/distributed/parallelizer.py`          |
| Checkpoint System      | `nemo_automodel/components/checkpoint/checkpointing.py`          |
| Config System          | `nemo_automodel/components/config/loader.py`                     |
| MoE Integration        | `nemo_automodel/components/moe/layers.py`                        |
| PEFT/LoRA              | `nemo_automodel/components/_peft/lora.py`                        |
| Quickstart             | `README.md`, `examples/`                                         |
| Architecture Overview  | `docs/repository-structure.md`                                   |
| Performance Benchmarks | `docs/performance-summary.md`                                    |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body, signed-off (`-s`).
- **Squash**: Squash WIP commits before opening a PR.
- **PR requirements**: Run pre-commit, document test coverage, and note hardware limitations.
- **Prompt templates**: Use `.codex/commands/gen-commit-msg.md`,
  `.codex/commands/create-pr.md`, and `.codex/commands/review-pr.md` as reusable prompt
  templates. They are repository documents, not automatically registered slash commands.

## Codex Project Configuration

Project-scoped Codex assets live under `.codex/`:

- `.codex/config.toml` - Project-scoped Codex configuration for enabled skills and
  subagent limits. It intentionally does not configure lifecycle hooks.
- `.codex/agents/*.toml` - Project-scoped custom agents. Each file defines a Codex
  custom agent with `name`, `description`, and `developer_instructions`.
- `.codex/skills/*/SKILL.md` - Codex skills for AutoModel domain knowledge. Skills are
  enabled through `.codex/config.toml` and should be loaded only when relevant.
- `.codex/commands/*.md` - Prompt templates for common workflows.
- `.codex/rules/*.md` - Documented code-quality guidance. Treat these as reference
  standards; do not assume path-scoped automatic enforcement.
- `.codex/data/*.md` - Supporting data used by command templates.
- `.codex/migrations/` - Migration manifest and validation script for the Codex asset tree.

### Custom Agents

Codex subagents are useful for independent, bounded tasks such as code exploration,
verification, review, and specialist analysis. Subagents inherit the current sandbox and
approval policy. Spawn them only when explicitly requested or when parallel specialist work
is clearly beneficial.

| Agent                  | Purpose                                | Activation Trigger                                          |
| ---------------------- | -------------------------------------- | ----------------------------------------------------------- |
| `planner`              | Implementation planning                | Before multi-file changes, new features, architectural decisions |
| `simple-code-reviewer` | Quick code quality checks              | After code changes, before committing                       |
| `code-verifier`        | Formatting/linting/tests               | After code changes, before committing                       |
| `distributed-expert`   | FSDP2, MegatronFSDP, parallelizer, TP  | Distributed code changes or questions                       |
| `model-expert`         | Model implementations and registration | Model code changes or questions                             |
| `checkpoint-expert`    | Distributed checkpointing              | Checkpoint code changes or questions                        |
| `moe-expert`           | Mixture of Experts integration         | MoE code changes or questions                               |
| `peft-expert`          | LoRA/DoRA/QLoRA/MoE-LoRA               | PEFT code changes or questions                              |
| `recipe-expert`        | Training recipes and workflows         | Recipe code changes or questions                            |
| `launcher-expert`      | CLI, torchrun, SLURM, multi-node       | Launch config or job submission questions                   |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (before coding): use `planner` for architecture design and
   implementation planning when the task spans multiple files or design boundaries.
2. **Domain Stage** (while designing or editing): use the relevant specialist agent for
   distributed, model, checkpoint, MoE, PEFT, recipe, or launcher work.
3. **Verification Stage** (after coding): use `code-verifier` to run formatting,
   linting, and tests when that can proceed independently.
4. **Review Stage** (after verification): use `simple-code-reviewer` for focused quality
   review before committing.

### Skills: Domain Knowledge

Use the `automodel-dr-*` skills in `.codex/skills/` for focused module knowledge. Prefer the
smallest relevant skill:

- `automodel-dr` for repository-wide orientation.
- `automodel-dr-distributed`, `automodel-dr-checkpoint`, `automodel-dr-models`,
  `automodel-dr-moe`, `automodel-dr-peft`, and other component skills for targeted work.
- Load referenced `reference.md` files only when the skill requires deeper context.

### Rules: Code Quality Standards

- `.codex/rules/code-style.md` - Coding conventions beyond pre-commit tooling.
- `.codex/rules/config-yaml.md` - YAML configuration and `_target_` instantiation patterns.
- `.codex/rules/distributed.md` - Distributed training patterns and constraints.
- `.codex/rules/testing.md` - Testing strategy and coverage requirements.

## Specification and Planning Workflow

- Before creating or changing project specs, inspect existing project context. If LeanSpec
  tooling is available, use `lean-spec board` and `lean-spec search` first.
- Do not manually create LeanSpec spec files when the tooling is available; use the tool's
  create/update/link commands instead.
- If LeanSpec tooling is unavailable in the current environment, state that limitation in the
  work log and proceed with repository inspection plus existing plan/spec artifacts.
- Treat task status as implementation status, not merely documentation status.

## Code Intelligence & Navigation

When navigating and understanding code:

1. **Prefer semantic tools over raw text search for code relationships**:
   - Use LSP or code-intelligence tools to jump to symbol definitions.
   - Use reference-finding tools to locate usages across the codebase.
   - Use implementation and workspace-symbol tools for interfaces and broad symbol searches.

2. **Use Grep/Glob/Read for**:
   - Text or pattern searches in comments and strings.
   - Configuration files (JSON, YAML, TOML, Markdown, etc.).
   - Exploratory fuzzy searches when unsure what to inspect.
   - Finding files by name patterns.

3. **Workflow**:
   - First: use semantic tooling to understand code structure and relationships.
   - Second: use text tools only when semantic tooling cannot help.
   - Avoid reading entire large files just to find references; use focused search instead.

## Completion Checklist

Before claiming completion:

- Confirm changed files are intentional and no unrelated work was modified.
- Run the narrowest useful verification for the change.
- For `.codex/` asset changes, run `python .codex/migrations/validate-claude-to-codex.py`.
- Report tests or checks actually run, skipped checks with reasons, and any remaining risks.
