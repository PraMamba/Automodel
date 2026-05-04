---
name: automodel-dr
description: Use when working with the automodel (NeMo AutoModel) codebase — provides comprehensive module knowledge, design logic, and modification guides (generated from main @ bbc0ed2)
---

# NeMo AutoModel — Deep Read Index

- **Source**: `/home/scbjtfy/Automodel` (local)
- **Version**: `bbc0ed22c17e40386a02a295f621c23ef9adeacb` (main)
- **Generated**: 2026-03-26

## Architecture Overview

NeMo AutoModel is a distributed LLM/VLM training framework built on PyTorch native primitives (FSDP2, DTensor, DeviceMesh). It follows a strict 4-layer dependency hierarchy enforced by import-linter:

```
Layer 4: recipes/        ← Application logic (compose everything)
Layer 3: _transformers/  ← HF integration layer
         _diffusers/
Layer 2: components/*    ← 15 independent component modules (no cross-imports)
Layer 1: shared/         ← Universal utilities
         _cli/           ← CLI entry point
         autonvtx/       ← Profiling annotations
```

## Module Index

### Layer 1 — Leaf Utilities

| Module | Skill | Purpose |
|--------|-------|---------|
| `shared/` | `automodel-dr-shared` | Dtype conversion, safe imports, torch patches, import helpers |
| `_cli/` | `automodel-dr-cli` | CLI launcher for interactive and SLURM with dynamic recipe loading |
| `autonvtx/` | `automodel-dr-autonvtx` | Autonomous NVTX annotation for profiling |

### Layer 2 — Independent Components

| Module | Skill | Purpose |
|--------|-------|---------|
| `components/models/` | `automodel-dr-models` | Custom optimized models (Llama, DeepSeek-V3, Qwen3-MoE, Mistral3, GPT-OSS) with TP plans |
| `components/distributed/` | `automodel-dr-distributed` | FSDP2, MegatronFSDP, TP, CP, PP, device mesh, SPMD strategies |
| `components/datasets/` | `automodel-dr-datasets` | Dataset loading, sequence packing, tokenization, chat templates |
| `components/checkpoint/` | `automodel-dr-checkpoint` | Distributed checkpointing with SafeTensors, DCP, mesh-aware resharding |
| `components/moe/` | `automodel-dr-moe` | Mixture of Experts with DeepEP, expert parallelism, routing |
| `components/_peft/` | `automodel-dr-peft` | LoRA/DoRA adapter integration with FSDP2 and TP compatibility |
| `components/training/` | `automodel-dr-training` | Training loop, step scheduler, gradient accumulation |
| `components/loss/` | `automodel-dr-loss` | Loss functions: cross-entropy, KD, chunked, fused, Triton parallel |
| `components/config/` | `automodel-dr-config` | YAML config system with CLI overrides, _target_ instantiation |
| `components/optim/` | `automodel-dr-optim` | Optimizer creation and LR/WD scheduling |
| `components/utils/` | `automodel-dr-utils` | Compilation, FLOPs, model introspection, YAML serialization |
| `components/attention/` | `automodel-dr-attention` | FlexAttention, TE/SDPA backends, context parallel attention |
| `components/loggers/` | `automodel-dr-loggers` | Metric logging: JSONL, MLflow, W&B noise suppression |
| `components/quantization/` | `automodel-dr-quantization` | FP8 training via torchao, QAT, QLoRA |
| `components/launcher/` | `automodel-dr-launcher` | SLURM job submission with container mounts |

### Layer 3 — Integration

| Module | Skill | Purpose |
|--------|-------|---------|
| `_transformers/` | `automodel-dr-transformers` | HF model/tokenizer auto-registration, NeMoAutoModelForCausalLM |
| `_diffusers/` | `automodel-dr-diffusers` | HF Diffusers model registration and auto-pipeline support |

### Layer 4 — Recipes

| Module | Skill | Purpose |
|--------|-------|---------|
| `recipes/llm/` | `automodel-dr-recipes-llm` | LLM pretraining, SFT, KD, sequence classification, benchmarking |
| `recipes/vlm/` | `automodel-dr-recipes-vlm` | VLM fine-tuning with SFT and PEFT |
| `recipes/biencoder/` | `automodel-dr-recipes-biencoder` | Contrastive biencoder training for embedding models |

## Inter-Module Dependencies

```
shared ──────────────────────────────────────────────────→ ALL modules
                                                           (universal utility)

components/* ───── each imports only shared/ ────────────→ No cross-component imports
                   (enforced by import-linter)

_transformers/ ──→ components/models (registry)
                 → components/config (instantiation)
                 → components/utils (model introspection)
                 → shared/ (imports, patches)

_diffusers/ ────→ components/distributed (FSDP2Manager)
                → shared/ (safe_import)

recipes/llm/ ──→ _transformers/ (model loading)
               → components/distributed (parallelization)
               → components/checkpoint (save/load)
               → components/datasets (data pipeline)
               → components/training (StepScheduler)
               → components/loss (loss functions)
               → components/optim (optimizer/scheduler)
               → components/loggers (metric logging)
               → components/_peft (LoRA adapters)
               → components/quantization (FP8)
               → components/config (YAML loading)
               → components/utils (compile, FLOPs)
               → components/attention (FlexAttention setup)
               → autonvtx/ (profiling)

recipes/vlm/ ──→ Same as llm/ + HF processor for vision

recipes/biencoder/ → Similar to llm/ + contrastive loss

_cli/ ──────────→ components/launcher (SLURM)
                → recipes/* (dynamic import)
```

## Cross-Module Scenario Guides

### 1. Adding a New Model Architecture

1. **components/models/**: Create new directory with model class, config, TP plan, state dict adapter
2. **components/models/__init__.py**: Export `ModelClass` variable for auto-discovery by `_ModelRegistry`
3. **_transformers/registry.py**: Model is auto-registered if it exports `ModelClass`
4. **components/distributed/optimized_tp_plans.py**: Add TP plan if custom parallelization needed
5. **components/checkpoint/**: Add state dict adapter if checkpoint format differs from HF

### 2. Running LLM Fine-Tuning End-to-End

1. **_cli/app.py**: `automodel finetune llm --config path/to/config.yaml`
2. **components/config/**: YAML loaded, CLI overrides applied, `_target_` resolved
3. **_transformers/auto_model.py**: Model loaded via `NeMoAutoModelForCausalLM.from_pretrained()`
4. **components/distributed/**: Model wrapped with FSDP2/TP/CP based on config
5. **components/datasets/**: Dataset tokenized, packed, collated
6. **components/training/**: `StepScheduler` yields micro-batches for gradient accumulation
7. **recipes/llm/train_ft.py**: Training loop drives forward/backward/optimizer steps
8. **components/checkpoint/**: Periodic saves in SafeTensors + DCP format

### 3. Adding a New Loss Function

1. **components/loss/**: Create new loss class inheriting from `nn.Module`
2. **components/loss/__init__.py**: Export the new class
3. **recipes/llm/train_ft.py**: Wire into `build_loss()` or use `_target_` in YAML config

### 4. Enabling LoRA Fine-Tuning

1. **YAML config**: Set `peft.type: lora` with target module patterns
2. **components/_peft/lora.py**: `apply_lora_to_linear_modules()` patches matching `nn.Linear` modules
3. **components/distributed/**: FSDP2 shards both base and LoRA parameters
4. **components/checkpoint/addons.py**: `PeftAddon` saves HF-compatible adapter weights separately

### 5. Adding Expert Parallelism for MoE Models

1. **components/distributed/init_utils.py**: Configure device mesh with EP dimension
2. **components/moe/**: MoE layers use `GroupedExperts` or `GroupedExpertsDeepEP`
3. **components/models/**: MoE model's TP plan includes expert parallel strategies
4. **components/checkpoint/**: MoE tensor merging handles grouped-expert checkpoint format

### 6. Adding a New Dataset Format

1. **components/datasets/**: Create new dataset class (map-style or iterable)
2. **components/datasets/__init__.py**: Export via `__init__.py`
3. **YAML config**: Point `dataset._target_` to the new class
4. **recipes/*/**: No recipe changes needed if the dataset follows the standard interface
