# CLAUDE.md - NeMo AutoModel

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

# Pre-commit hooks
pre-commit install            # Set up hooks (run once)
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
```

## Boundaries

### Constraints

- Designed for distributed GPU clusters; assume containerized execution
- Functional tests require GPU hardware; explain skips when unavailable
- Components must not cross-import (enforced by import-linter)

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Add tests for new functionality
- Sign commits with `--signoff` (`git commit -s`)

### Ask First

- Modifying the config system in `nemo_automodel/components/config/`
- Adding new dependencies to `pyproject.toml`
- Changing distributed parallelism logic
- Deleting or renaming public APIs
- Modifying recipe training loops
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`)

### Never Do

- Hardcode secrets, paths, or endpoints
- Skip pre-commit hooks
- Use wildcard imports (`from x import *`)
- Cross-import between components (enforced by import-linter)
- Guess cluster configs or rebuild CUDA/driver stacks
- Use underscores in Markdown filenames (use hyphens)

## Progressive Disclosure: Detailed Guides

| Task                    | Reference                                                        |
| ----------------------- | ---------------------------------------------------------------- |
| Add Model               | `nemo_automodel/components/models/llama/`, `_transformers/registry.py` |
| Add Dataset             | `nemo_automodel/components/datasets/`, `examples/llm_finetune/`  |
| Add Recipe              | `nemo_automodel/recipes/base_recipe.py`, `recipes/llm/train_ft.py` |
| Add Loss Function       | `nemo_automodel/components/loss/masked_ce.py`                    |
| Distributed Patterns    | `nemo_automodel/components/distributed/parallelizer.py`          |
| Checkpoint System       | `nemo_automodel/components/checkpoint/checkpointing.py`          |
| Config System           | `nemo_automodel/components/config/loader.py`                     |
| MoE Integration         | `nemo_automodel/components/moe/layers.py`                        |
| PEFT/LoRA               | `nemo_automodel/components/_peft/lora.py`                        |
| Quickstart              | `README.md`, `examples/`                                         |
| Architecture Overview   | `docs/repository-structure.md`                                   |
| Performance Benchmarks  | `docs/performance-summary.md`                                    |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body, signed-off (`-s`)
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent                  | Purpose                                   | Activation Trigger                                          |
| ---------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| `planner`              | Implementation planning                   | Before multi-file changes, new features, architectural decisions |
| `simple-code-reviewer` | Quick code quality checks                 | After code changes, before committing                       |
| `code-verifier`        | Formatting/linting/tests                  | After code changes, before committing                       |
| `distributed-expert`   | FSDP2, MegatronFSDP, parallelizer, TP     | Distributed code changes or questions                       |
| `model-expert`         | Model implementations and registration    | Model code changes or questions                             |
| `checkpoint-expert`    | Distributed checkpointing                 | Checkpoint code changes or questions                        |
| `moe-expert`           | Mixture of Experts integration            | MoE code changes or questions                               |
| `peft-expert`          | LoRA/DoRA/QLoRA/MoE-LoRA                  | PEFT code changes or questions                              |
| `recipe-expert`        | Training recipes and workflows            | Recipe code changes or questions                            |
| `launcher-expert`      | CLI, torchrun, SLURM, multi-node          | Launch config or job submission questions                   |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and
   implementation planning
1. **Code Formatting & Linting** (After coding): Use `code-verifier` to automatically
   run formatting, linting, and tests
1. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for quick code
   quality checks

### Skills (Domain Knowledge)

Skills provide deep module-level knowledge for common development tasks. See
`.claude/skills/` for the full list of `automodel-dr-*` skills covering every component.

### Commands (User-invoked Actions)

- `/create-pr` - Rebase, squash commits, and create/update PR with intelligent messages
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/review-pr` - Intelligent PR code review with dynamic agent allocation

### Rules (Code Quality Standards)

- `code-style.md` - Coding conventions beyond pre-commit hooks
- `config-yaml.md` - YAML configuration and `_target_` instantiation patterns
- `distributed.md` - Distributed training patterns and constraints
- `testing.md` - Testing strategy and coverage requirements

## Code Intelligence & Navigation

When navigating and understanding code:

1. **ALWAYS prefer LSP tools over text search for code relationships**:
   - Use `goToDefinition` to jump to symbol definitions
   - Use `findReferences` to find all usages across the codebase
   - Use `goToImplementation` for interfaces/abstract methods
   - Use `workspaceSymbol` to search symbols across entire project

2. **Use Grep/Glob/Read ONLY for**:
   - Text/pattern searches in comments or strings
   - Searching configuration files (JSON, YAML, etc.)
   - Exploratory "fuzzy" searches when unsure what you're looking for
   - Finding files by name patterns

3. **Workflow**:
   - First: Use LSP to understand code structure and relationships
   - Second: Use text tools only when LSP cannot help (non-code content)
   - NEVER read entire large files to find references; use LSP instead
