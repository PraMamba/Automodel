# PR Review Change Types

Maps file paths to change types and severity levels for dynamic review agent allocation.

## CRITICAL Level (must use Opus)

| Change Type | File Patterns | Expert Agent |
|-------------|---------------|-------------|
| DISTRIBUTED_CORE | `nemo_automodel/components/distributed/parallelizer.py` | distributed-expert |
| FSDP2_CORE | `nemo_automodel/components/distributed/fsdp2.py` | distributed-expert |
| MEGATRON_FSDP | `nemo_automodel/components/distributed/megatron_fsdp.py` | distributed-expert |
| MOE_LAYERS | `nemo_automodel/components/moe/layers.py` |
| MOE_PARALLELIZER | `nemo_automodel/components/moe/parallelizer.py` |
| PIPELINE_PARALLEL | `nemo_automodel/components/distributed/pipelining/` |

## HIGH Level (recommend Opus)

| Change Type | File Patterns |
|-------------|---------------|
| CHECKPOINT_CORE | `nemo_automodel/components/checkpoint/checkpointing.py` |
| STATE_DICT_ADAPTER | `nemo_automodel/components/checkpoint/state_dict_adapter.py` |
| DTENSOR_OPS | `nemo_automodel/components/distributed/*_utils.py` |
| TP_PLANS | `nemo_automodel/components/distributed/optimized_tp_plans.py` |
| PARALLEL_STYLES | `nemo_automodel/components/distributed/parallel_styles.py` |
| MODEL_CORE | `nemo_automodel/components/models/*/model.py` |
| RECIPE_CORE | `nemo_automodel/recipes/llm/train_ft.py` |
| BASE_RECIPE | `nemo_automodel/recipes/base_recipe.py` |
| PEFT_CORE | `nemo_automodel/components/_peft/lora.py` |
| MOE_FSDP | `nemo_automodel/components/moe/fsdp_mixin.py` |

## MEDIUM Level (use Sonnet)

| Change Type | File Patterns |
|-------------|---------------|
| CONFIG_SYSTEM | `nemo_automodel/components/config/` |
| DATASET_LOADER | `nemo_automodel/components/datasets/` |
| LOSS_FUNCTION | `nemo_automodel/components/loss/` |
| OPTIMIZER | `nemo_automodel/components/optim/` |
| ATTENTION | `nemo_automodel/components/attention/` |
| QUANTIZATION | `nemo_automodel/components/quantization/` |
| TRAINING_UTILS | `nemo_automodel/components/training/` |
| LOGGERS | `nemo_automodel/components/loggers/` |
| TRANSFORMERS_REG | `nemo_automodel/_transformers/` |
| CLI | `nemo_automodel/_cli/` |
| LAUNCHER | `nemo_automodel/components/launcher/` |
| VLM_RECIPE | `nemo_automodel/recipes/vlm/` |
| BIENCODER_RECIPE | `nemo_automodel/recipes/biencoder/` |
| KD_RECIPE | `nemo_automodel/recipes/llm/kd.py` |

## LOW Level (use Haiku)

| Change Type | File Patterns |
|-------------|---------------|
| TESTS | `tests/` |
| DOCS | `docs/` |
| EXAMPLES | `examples/` |
| CONFIG_ONLY | `*.yaml`, `*.json`, `*.toml` (no Python) |
| SHARED_UTILS | `nemo_automodel/shared/` |

## Risk Linkage Rules

When certain change types are detected, automatically add related review tasks:

| Primary Change | Also Review |
|----------------|-------------|
| DISTRIBUTED_CORE | CHECKPOINT_CORE (save/load may be affected) |
| MOE_LAYERS | MOE_PARALLELIZER, MOE_FSDP (parallelism integration) |
| TP_PLANS | MODEL_CORE (model layer names must match) |
| STATE_DICT_ADAPTER | MODEL_CORE (key mapping must match model structure) |
| CONFIG_SYSTEM | CLI (CLI overrides depend on config loader) |
| RECIPE_CORE | TRAINING_UTILS (step scheduler interaction) |
| PEFT_CORE | DISTRIBUTED_CORE (LoRA + FSDP2 interaction) |

## Core Framework Paths (Always Opus)

Any change to these paths should use Opus-level review regardless of other factors:

```
nemo_automodel/components/distributed/parallelizer.py
nemo_automodel/components/distributed/fsdp2.py
nemo_automodel/components/distributed/megatron_fsdp.py
nemo_automodel/components/moe/layers.py
nemo_automodel/components/checkpoint/checkpointing.py
nemo_automodel/recipes/base_recipe.py
```
