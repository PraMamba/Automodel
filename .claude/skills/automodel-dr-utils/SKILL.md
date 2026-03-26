---
name: automodel-dr-utils
description: Use when working with the utils module of automodel — component utilities for memory tracking, performance profiling, and distributed helpers
---

# Module: `nemo_automodel/components/utils/`

## 1. Module Purpose & Capabilities

The `components/utils/` module provides four categories of cross-cutting utility functions consumed by recipes, the `_transformers/auto_model.py` model-loading pipeline, and custom model implementations. The files are:

| File | Lines | Purpose |
|------|-------|---------|
| `compile_utils.py` | 250 | `torch.compile` configuration, Flash Attention patching, model compilation |
| `flops_utils.py` | 861 | Per-architecture FLOPs calculation formulas, MFU computation |
| `model_utils.py` | 333 | Model introspection, parameter freezing, THD format conversion, meta-device initialization |
| `yaml_utils.py` | 171 | Safe YAML serialization of Python objects (functions, enums, torch dtypes, partials) |
| `__init__.py` | 0 | Empty (no re-exports) |

The module follows the component independence principle: it imports only from `nemo_automodel.shared` (specifically `import_utils.safe_import` in `model_utils.py`) and standard library / PyTorch. It never imports from other components, recipes, or the CLI layer.

## 2. Core Design Logic

### 2.1 Why a flat utilities package with no `__init__` re-exports

Each file is imported individually at point of use (e.g., `from nemo_automodel.components.utils.compile_utils import compile_model`). This avoids triggering side effects from unrelated utilities when only one is needed. For instance, importing `flops_utils` does not import torch or trigger the Flash Attention monkey-patch that runs at `compile_utils` module load time (line 250: `_FLASH_ATTENTION_FIX_APPLIED = apply_flash_attention_compile_fix()`).

### 2.2 Why compile_utils applies a patch at import time

The `prepare_fa2_from_position_ids` monkey-patch on `transformers.modeling_flash_attention_utils` (lines 103-148 of `compile_utils.py`) replaces the HuggingFace implementation with one that calls `.item()` on `position_ids.max()`, converting a `FakeTensor` to a Python int during `torch.compile` tracing. This is executed eagerly at import time (line 250) because every code path that eventually calls `compile_model()` already imports from this module, guaranteeing the fix is in place before any model forward pass. A separate `configure_torch_dynamo()` call is deferred to per-compilation time to allow per-model cache size tuning.

### 2.3 Why flops_utils has per-architecture functions instead of a single generic formula

Different model families compute FLOPs differently due to architectural variations:
- **GQA (Grouped Query Attention)** changes the K/V projection ratio (Llama2/3, Nemotron, Qwen3, Mixtral, DeepSeek V3, GLM4-MoE, GPT-OSS).
- **MoE routing** multiplies FFN FLOPs by `top_k` selected experts rather than total experts (Mixtral, Qwen3-MoE, DeepSeek V3, GLM4-MoE, GPT-OSS).
- **Multi-Latent Attention (MLA)** in DeepSeek V3 introduces low-rank KV compression with `q_lora_rank` / `kv_lora_rank`, requiring a fundamentally different parameter count (lines 419-503).
- **Hybrid architectures** like NemotronH mix Mamba (SSM), attention, and MLP-only layers with a pattern string like `"M-*M-*"` (lines 552-583).
- **Sliding window attention** in GPT-OSS alternates full and windowed attention layers, each with different FLOPs (lines 586-717).

The dispatch function `get_flops_formula_for_hf_config()` (lines 804-861) maps HuggingFace config class names to the correct formula, falling back to `transformer_flops` for unknown architectures.

### 2.4 Why model_utils handles both introspection and initialization

The functions serve the model-loading pipeline in `_transformers/auto_model.py`:

1. **Pre-loading**: `resolve_trust_remote_code()` whitelists `nvidia/` models for `trust_remote_code=True`, preventing arbitrary code execution from untrusted model repos.
2. **Meta-device init**: `init_empty_weights()` is a context manager that redirects `nn.Module.register_parameter` to create parameters on `torch.device("meta")`, enabling model instantiation without GPU memory. It handles special cases for `torchao` FP8 tensors (`WeightWithDynamicFloat8CastTensor`) and standard `nn.Parameter` (lines 277-334).
3. **Post-loading**: `_supports_logits_to_keep()` and `_supports_seq_lens()` inspect `model.forward()` signature to determine whether the model accepts optimization kwargs, enabling recipes to conditionally pass them.
4. **VLM training**: `apply_parameter_freezing()` freezes vision/audio/language towers by attribute name and name-pattern matching.

### 2.5 Why yaml_utils uses a context manager for YAML representers

`safe_yaml_representers()` temporarily patches `yaml.SafeDumper.yaml_representers` and restores the original on exit (lines 24-77). This design prevents global representer pollution: the custom representers for `functools.partial`, enums, functions, `torch.dtype`, and `GenerationConfig` exist only within the `with` block. The representers serialize Python objects into the `_target_` / `_call_` / `_partial_` dictionary format that the Hydra-style instantiation system can reconstruct.

## 3. Core Data Structures

### 3.1 `CompileConfig` dataclass
**File**: `/nemo_automodel/components/utils/compile_utils.py`, lines 27-67

```python
@dataclass
class CompileConfig:
    enabled: bool = False
    mode: str = "default"          # torch.compile mode
    fullgraph: bool = False
    dynamic: bool = False
    backend: Optional[str] = None  # e.g., "inductor"
    options: Optional[Dict[str, Any]] = None
    dynamo_cache_size_limit: int = 256
```

Created via `build_compile_config(cfg)` (line 234) or `create_compile_config_from_dict(config_dict)` (line 214). Consumed by `compile_model(model, config)` (line 171) which orchestrates dynamo configuration, FA patching, and `torch.compile()`.

### 3.2 FLOPs Formula Functions
**File**: `/nemo_automodel/components/utils/flops_utils.py`

All FLOPs functions share the same signature pattern:
```python
def <arch>_flops(config, gbs=1, seq_len=None) -> float
```

Where `config` is a HuggingFace `AutoConfig` object. The functions read attributes like `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, and architecture-specific fields (e.g., `q_lora_rank`, `moe_intermediate_size`, `hybrid_override_pattern`).

Available formulas: `gpt3_flops`, `llama2_flops`, `llama3_flops`, `nemotron_flops`, `mixtral_flops`, `qwen3_flops`, `bert_flops`, `transformer_flops`, `deepseekv3_flops`, `nemotronh_flops`, `gpt_oss_flops`, `glm4_moe_flops`, plus vision-specific `clip_vit_l_flops`, `neva_projection_flops`, `flux_flops`.

Component calculators for composable FLOPs: `attention_flops_calculator`, `moe_mlp_flops_calculator`, `loss_flops_calculator` (lines 586-637).

### 3.3 Key Utility Functions in model_utils.py
**File**: `/nemo_automodel/components/utils/model_utils.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `_supports_logits_to_keep(model)` | 30-43 | Checks `model.forward` signature for `logits_to_keep` parameter |
| `_supports_seq_lens(model)` | 46-70 | Checks for `seq_lens` param or `**kwargs` in forward |
| `_get_model_param_stats(model)` | 73-98 | Returns `(total_params, trainable_params, l2_norm)` |
| `print_trainable_parameters(model)` | 117-144 | Logs parameter summary, returns `(trainable, total)` |
| `resolve_trust_remote_code(path)` | 101-114 | Returns `True` only for `nvidia/` HF model IDs |
| `apply_parameter_freezing(model, config)` | 167-193 | Freezes vision/audio/language towers by config dict |
| `squeeze_input_for_thd(...)` | 196-272 | Removes batch dim and filters padding for THD format |
| `init_empty_weights()` | 277-334 | Context manager for meta-device parameter initialization |

### 3.4 YAML Representers
**File**: `/nemo_automodel/components/utils/yaml_utils.py`

| Representer | Lines | Serializes |
|------------|-------|------------|
| `_function_representer` | 80-86 | Functions/classes to `{"_target_": "module.qualname", "_call_": False}` |
| `_torch_dtype_representer` | 89-95 | `torch.dtype` to `{"_target_": "torch.float16", ...}` |
| `_safe_object_representer` | 98-127 | Fallback for arbitrary objects; sets `_call_: True` for instances |
| `_partial_representer` | 130-146 | `functools.partial` to `{"_target_": ..., "_partial_": True, "_args_": [...], **keywords}` |
| `_enum_representer` | 149-159 | Enum members to `{"_target_": "module.EnumClass", "_call_": True, "_args_": [value]}` |
| `_generation_config_representer` | 162-171 | HuggingFace `GenerationConfig` via `.from_dict(config_dict)` |

## 4. State Flow

### 4.1 Model Loading Pipeline (auto_model.py)

```
YAML config
  |
  v
train_ft.py: build_compile_config(cfg_compile) --> CompileConfig
  |
  v
auto_model.py: resolve_trust_remote_code(model_name) --> bool
  |
  v
auto_model.py: init_empty_weights() context manager (meta device)
  |  - Monkey-patches nn.Module.register_parameter
  |  - Handles torchao FP8 WeightWithDynamicFloat8CastTensor
  v
auto_model.py: _supports_logits_to_keep(model) --> bool
  |  - Inspects forward() signature
  v
auto_model.py: _supports_seq_lens(model) --> bool
  |  - Inspects forward() for seq_lens or **kwargs
  v
auto_model.py: compile_model(model, compile_config) --> compiled model
  |  - Calls configure_torch_dynamo(cache_size_limit)
  |  - Calls apply_flash_attention_compile_fix()
  |  - Calls torch.compile(model, **kwargs)
  v
auto_model.py: print_trainable_parameters(model) --> logs summary
  |
  v
auto_model.py: apply_parameter_freezing(model, freeze_config)
     - Freezes vision_tower / audio_tower / language_model
```

### 4.2 Benchmarking Pipeline (benchmark.py)

```
benchmark.py: setup()
  |
  v
get_flops_formula_for_hf_config(model.config) --> formula function
  |  - Maps config class name to architecture-specific formula
  v
formula(config, gbs=global_batch_size, seq_len=seq_len) --> raw FLOPs
  |
  v
tflops = flops / 1e12
  |
  v  (per iteration)
calculate_mfu(tflops, world_size, time_seconds, reference_mfu=peak_tflops) --> MFU%
```

### 4.3 THD Format Conversion (custom models)

```
Custom model forward() (e.g., DeepSeek V3, Qwen3-MoE, GLM4-MoE, etc.)
  |
  v
Check attn_kwargs["qkv_format"] == "thd"
  |
  v
squeeze_input_for_thd(input_ids, position_ids, padding_mask, attn_kwargs)
  |  - Removes batch dim (squeeze(0))
  |  - Filters cu_seqlens padding values (sentinel = -1000)
  |  - Converts max_seqlen tensor to scalar
  v
Proceed with TransformerEngine THD attention
```

Models that call `squeeze_input_for_thd`: DeepSeek V3, Qwen3-MoE, Qwen3-VL-MoE, Qwen3-Omni-MoE, Qwen3-Next, GLM4-MoE, KimiVL, Kimi-K25-VL, Step3P5.

### 4.4 YAML Serialization (config logging)

```
Recipe base_recipe.py: log_config()
  |
  v
(Currently uses yaml.safe_dump directly)
  |
  v
yaml_utils.safe_yaml_representers() context manager
  |  - Temporarily registers representers for:
  |    functools.partial, enums, functions, torch.dtype,
  |    GenerationConfig, and arbitrary objects
  |  - Restores originals on exit
  v
yaml.safe_dump(config_dict) --> YAML string
```

## 5. Common Modification Scenarios

### 5.1 Adding a FLOPs Formula for a New Model Architecture

**When**: A new model family (e.g., a new MoE variant) is added to `nemo_automodel/components/models/`.

**Steps**:
1. Add a new function to `/nemo_automodel/components/utils/flops_utils.py` following the signature `def new_arch_flops(config, gbs=1, seq_len=None) -> float`. Read architecture-specific fields from the HF config object, guarding optional fields with `hasattr()` checks.
2. Register the HuggingFace config class name in the `class_name_to_formula` dictionary inside `get_flops_formula_for_hf_config()` (line 818).
3. Add a unit test in `/tests/unit_tests/utils/test_flops_utils.py` with a mock config object.

**Key constraint**: The function must handle both HuggingFace `AutoConfig` objects and any normalized config, as indicated by the docstrings. Use `hasattr()` for optional fields and provide sensible defaults.

### 5.2 Adding a New Model Feature Check to model_utils.py

**When**: A recipe needs to conditionally pass a new kwarg to `model.forward()` (similar to how `logits_to_keep` and `seq_lens` are checked today).

**Steps**:
1. Add a new function like `_supports_<feature>(model: nn.Module) -> bool` to `/nemo_automodel/components/utils/model_utils.py`, following the pattern of `_supports_logits_to_keep()` (lines 30-43). Use `inspect.signature(model.forward).parameters` to check for the parameter name.
2. Import and call it in the recipe (e.g., `train_ft.py`) or `auto_model.py` to conditionally include the kwarg in the forward call.
3. If the feature check needs to handle `**kwargs`, follow the pattern in `_supports_seq_lens()` (lines 46-70) which also checks for `VAR_KEYWORD` parameters.

### 5.3 Extending torch.compile Support for a New Attention Backend

**When**: A new attention implementation (beyond Flash Attention 2) needs compatibility patches for `torch.compile`.

**Steps**:
1. Add a new patch function in `/nemo_automodel/components/utils/compile_utils.py`, following the pattern of `patch_prepare_fa2_from_position_ids()` (lines 103-148). The function should monkey-patch the problematic function in the target library to produce `torch.compile`-compatible operations.
2. Call the new patch function from within `apply_flash_attention_compile_fix()` (line 151) or create a parallel `apply_<backend>_compile_fix()` function.
3. If the fix should apply eagerly (before any compilation), add a module-level call similar to line 250. If it should be per-compilation, call it from within `compile_model()` (line 171).
4. If new `CompileConfig` fields are needed, add them to the `CompileConfig` dataclass (line 28) and update `to_dict()`, `create_compile_config_from_dict()`, and `build_compile_config()`.

### 5.4 Adding a New YAML Representer for a Custom Type

**When**: A new type (e.g., a custom training schedule object) needs to be serialized to YAML for config logging.

**Steps**:
1. Add a new representer function `_<type>_representer(dumper, data)` in `/nemo_automodel/components/utils/yaml_utils.py`, following the pattern of `_generation_config_representer()` (lines 162-171). Return a dict with `_target_` pointing to the reconstruction path and `_call_: True` if the target must be called.
2. Register it inside the `try` block of `safe_yaml_representers()` using `yaml.SafeDumper.add_representer(YourType, _your_representer)`.
3. Wrap the import of the custom type in a `try/except ModuleNotFoundError` block to avoid hard dependencies, following the pattern at lines 55-68.

### 5.5 Adding a New Parameter Freezing Strategy for Multimodal Models

**When**: A new multimodal tower (e.g., a video encoder) needs selective freezing support.

**Steps**:
1. Add a new key to the `freeze_config` dictionary handled by `apply_parameter_freezing()` in `/nemo_automodel/components/utils/model_utils.py` (line 167), e.g., `freeze_video_tower`.
2. Add a corresponding call to `_freeze_module_by_attribute_and_patterns()` (line 147) with the attribute name (e.g., `"video_tower"`) and name patterns (e.g., `["video", "visual_temporal", "video_encoder"]`).
3. Update the VLM recipe YAML configs to expose the new freeze option.
