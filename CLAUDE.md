# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NeMo AutoModel is a PyTorch DTensor-native SPMD training library for LLMs and VLMs with day-0 Hugging Face support. It enables scalable training using PyTorch-native parallelism (FSDP2, TP, CP, SP, Pipeline) without model-specific rewrites.

## Development Environment Setup

```bash
# Setup and install dependencies
uv venv
uv sync --frozen --all-extras

# Verify installation
uv run python -c "import nemo_automodel; print('AutoModel ready')"
```

## Common Commands

### Linting and Formatting
```bash
# Format code
ruff check --fix .
ruff format .

# Check import dependencies (components must not import each other)
lint-imports --debug --verbose --no-cache
```

### Testing

**Run unit tests:**
```bash
# All unit tests
pytest tests/unit_tests -vs

# Specific test file
pytest tests/unit_tests/path/to/test_file.py -vs

# With coverage
coverage run --source=. -m pytest tests/unit_tests -vs
coverage report -i
```

**Run functional tests:**
```bash
# Specific test suite (requires GPUs)
pytest tests/functional_tests/hf_transformer_llm -vs

# With CUDA visible devices
CUDA_VISIBLE_DEVICES=0,1 pytest tests/functional_tests/hf_peft -vs
```

### Running Recipes

**Single GPU:**
```bash
uv run python examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

**Multi-GPU (torchrun):**
```bash
uv run torchrun --nproc-per-node=8 \
  examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

**Using automodel CLI:**
```bash
# Interactive multi-GPU
uv run automodel finetune llm \
  --nproc-per-node=8 \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml

# SLURM cluster (requires slurm section in YAML)
uv run automodel finetune llm \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

**Override YAML parameters via CLI:**
```bash
uv run python examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
  --step_scheduler.local_batch_size 16 \
  --model.pretrained_model_name_or_path meta-llama/Llama-3.2-3B
```

## Architecture Overview

### Three-Layer Structure

1. **Components** (`nemo_automodel/components/`) - Self-contained, reusable modules
   - Each component is independent with no cross-module imports (enforced by import-linter)
   - Examples: datasets, distributed (FSDP2, MegatronFSDP), checkpoint, loss, optim

2. **Recipes** (`nemo_automodel/recipes/`) - End-to-end training workflows
   - `llm/train_ft.py` - LLM pretraining & fine-tuning (SFT, PEFT)
   - `llm/kd.py` - Knowledge distillation for LLMs
   - `vlm/finetune.py` - VLM fine-tuning (SFT, PEFT)

3. **CLI** (`nemo_automodel/_cli/`) - Job launcher for interactive and SLURM environments

### Key Architectural Principles

**SPMD (Single Program, Multiple Data):**
- Same script runs on 1 GPU or 1000+ GPUs by changing device mesh configuration
- Parallelism is configuration, not code changes
- Compose tensor/sequence/data/pipeline parallelism via placements

**Component Independence:**
- Components must NOT import each other (contract enforced in pyproject.toml)
- Recipes import and compose components
- Breaking component independence will fail CI via import-linter

**DTensor-Native:**
- Uses PyTorch Distributed with `DeviceMesh` + placements (`Shard`, `Replicate`)
- FSDP2 for memory-efficient sharding (including HSDP for multi-node)
- Native pipeline parallelism composable with FSDP2 (3D parallelism)

## Model Integration

### Using HuggingFace Models

Any HF causal LM works out-of-the-box:
```python
# In YAML config
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

### Custom Model Implementations

For optimized models, see `nemo_automodel/components/models/`:
- `llama/` - Optimized Llama implementations
- `deepseek_v3/` - DeepSeek-V3 MoE architecture
- `qwen3_moe/`, `qwen3_omni_moe/`, `qwen3_vl_moe/` - Qwen3 MoE variants
- `gpt_oss/` - GPT-OSS models
- `mistral3/` - Mistral models

Models are registered via `_transformers/auto_model.py` and can override default HF implementations for better performance.

## Configuration System

**YAML-driven with CLI overrides:**
- Base configs in `examples/llm_finetune/` and `examples/vlm_finetune/`
- Override any nested field: `--section.subsection.field value`
- Uses Hydra-style instantiation with `_target_` for Python objects

**Key config sections:**
- `step_scheduler` - Training loop parameters (batch size, checkpointing, validation)
- `model` - Model selection and initialization
- `dataset` - Dataset configuration
- `dist_env` - Distributed environment settings
- `slurm` - SLURM job configuration (optional)

## Testing Structure

**Unit tests** (`tests/unit_tests/`) - Mirror component structure:
- Run on CPU or GPU
- Test individual components in isolation
- Fast, focused tests

**Functional tests** (`tests/functional_tests/`) - End-to-end workflows:
- Require GPUs
- Test recipes and integration scenarios
- Examples: hf_transformer_llm, hf_peft, pretrain_llm

**Coverage:**
- Tests run with coverage tracking
- Omitted from coverage: `tests/`, `checkpoint/_backports/`, `moe/megatron/`

## Important Development Constraints

### Code Style
- Line length: 120 characters
- Quote style: double quotes
- Linting: ruff with Google-style docstrings (D101, D103 for modules/functions)
- Formatting: ruff format (skip magic trailing comma)

### Commit Signing
All commits MUST be signed with `git commit -s` (Developer Certificate of Origin).

### Dependencies
- Managed via `uv` (not pip)
- Add dependencies: `uv add $DEPENDENCY`
- Always commit `uv.lock` and `pyproject.toml` together
- Some deps require source install for CUDA compatibility (see CONTRIBUTING.md):
  - TransformerEngine, flash-attn, grouped_gemm, mamba, DeepEP

### Container Development
Primary development path is via Docker container:
```bash
# Build container
export AUTOMODEL_INSTALL=vlm  # or: fa, moe
export BASE_IMAGE=pytorch
docker build -f docker/Dockerfile \
  --build-arg AUTOMODEL_INSTALL=$AUTOMODEL_INSTALL \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  -t automodel --target=automodel_final .

# Run container
docker run --rm -it --runtime nvidia --gpus all automodel
```

## Key Concepts for Development

### Recipes vs Components
- **Components** = Libraries (datasets, models, optimizers, distributed strategies)
- **Recipes** = Applications (combine components into training workflows)
- Never make components depend on each other - compose them in recipes

### FSDP2 vs MegatronFSDP
- **FSDP2**: PyTorch-native fully sharded data parallelism
- **MegatronFSDP**: Hybrid approach combining Megatron-style model parallelism with FSDP
- Both live in `components/distributed/`

### Checkpoint Format
- Uses PyTorch Distributed Checkpoint (DCP) with SafeTensors output
- Mesh-aware: can merge to HF format or reshard for different topology
- See `components/checkpoint/` for utilities

### Sequence Packing
- Available for LLM training via dataset components
- Significantly improves training throughput
- Configured in dataset YAML section

### FP8 Training
- Supported via torchao integration
- Requires torch.compile-compatible models
- See examples: `llama3_1_8b_hellaswag_fp8.yaml`

## File Locations for Common Tasks

**Adding a new model:**
- Register in `nemo_automodel/_transformers/auto_model.py`
- Custom implementation in `nemo_automodel/components/models/<model_name>/`
- Recipe example in `examples/llm_finetune/<model_name>/`

**Adding a new dataset:**
- Implementation in `nemo_automodel/components/datasets/llm/` or `.../vlm/`
- Example usage in recipe YAML configs

**Modifying distributed strategies:**
- FSDP2/MegatronFSDP in `nemo_automodel/components/distributed/`
- Tensor parallel plans in `components/distributed/optimized_tp_plans.py`

**Adding optimizers/schedulers:**
- `nemo_automodel/components/optim/`

**Custom kernels/loss functions:**
- `nemo_automodel/components/loss/`
- `nemo_automodel/components/moe/` (MoE-specific kernels)
