---
name: automodel-dr-quantization
description: Use when working with the quantization module of automodel — FP8 training and quantization via torchao integration
---

# Quantization Module -- FP8, QAT, and QLoRA Integration

## 1. Module Purpose & Capabilities

The quantization module (`/home/scbjtfy/Automodel/nemo_automodel/components/quantization/`) provides three distinct quantization strategies for training and inference, all built on top of third-party libraries:

- **FP8 Training** via `torchao.float8`: Converts `nn.Linear` layers to `Float8Linear` for reduced-precision training on H100+ GPUs. Supports tensorwise scaling, rowwise scaling, and rowwise-with-global-weight-high-precision recipes. Integrates with FSDP2 via float8 all-gather and precomputed dynamic scaling.
- **Quantization-Aware Training (QAT)** via `torchao.quantization.qat`: Prepares models with fake-quantization nodes for INT4/INT8 quantization-aware training. Supports delayed fake-quant (disable initially, enable after N training steps).
- **QLoRA** via `bitsandbytes` + `transformers`: Creates `BitsAndBytesConfig` for 4-bit (NF4) or 8-bit base weight quantization, used in combination with LoRA adapters for parameter-efficient fine-tuning.

The module is consumed by:
- `nemo_automodel/_transformers/auto_model.py` -- `_apply_peft_and_lower_precision()` (line 455) calls `apply_fp8_to_model()` and `prepare_qat_model()` after PEFT application but before FSDP2/TP sharding.
- `nemo_automodel/recipes/llm/train_ft.py` -- `build_fp8_config()` (line 181), `create_bnb_config()` (line 192), and `_setup_qat()` (line 1025) for training recipe configuration.
- `nemo_automodel/recipes/vlm/finetune.py` -- `build_fp8_config()` (line 119) for VLM fine-tuning.

### Files (4 files, 535 lines total)

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 33 | Package exports; conditionally re-exports `Float8LinearConfig` from torchao |
| `fp8.py` | 352 | FP8 config dataclass, model conversion, hardware checks, verification |
| `qat.py` | 101 | QAT quantizer wrappers, fake-quant enable/disable toggle functions |
| `qlora.py` | 49 | BitsAndBytes config creation for 4-bit/8-bit QLoRA |

---

## 2. Core Design Logic

### Why a unified FP8Config dataclass wraps torchao's Float8LinearConfig

The `FP8Config` dataclass (`fp8.py`, line 28) exists as an intermediate layer between the user-facing YAML configuration and torchao's `Float8LinearConfig`. This separation serves three purposes:

1. **Recipe abstraction**: Users can specify a recipe name (`"tensorwise"`, `"rowwise"`, `"rowwise_with_gw_hp"`) and get correct torchao configuration without knowing the underlying `Float8LinearConfig` parameters. Non-tensorwise recipes use `Float8LinearConfig.from_recipe_name()` (line 215), while tensorwise uses manual construction with `enable_fsdp_float8_all_gather` and `force_recompute_fp8_weight_in_bwd` (lines 224-228).
2. **Serialization stability**: `FP8Config.to_dict()` (line 87) produces a flat dictionary with `fp8_`-prefixed keys (e.g., `fp8_recipe_name`, `fp8_filter_fqns`) that is independent of torchao version changes. This allows config to round-trip through YAML/JSON without coupling to torchao's internal representation.
3. **Backward-compatible API**: `apply_fp8_to_model()` (line 143) accepts either an `FP8Config` object or individual keyword parameters, so callers that were written before `FP8Config` existed continue to work.

### Why module filtering uses substring matching and a dimension-divisibility check

The `_module_filter_fn()` (line 109) combines two filters before allowing FP8 conversion:

1. **FQN substring matching**: The `filter_fqns` list is checked via `fqn in name` (line 126), using substring containment rather than exact equality. This lets users exclude broad categories (e.g., `"lm_head"` will match `"model.lm_head"` regardless of prefix).
2. **Dimension divisibility by 16**: FP8 hardware (SM89+) requires tensor dimensions divisible by 16 for efficient FP8 GEMM. Lines 136-137 check both `weight.shape[0]` and `weight.shape[1]`, logging a skip message for non-compliant layers. This prevents silent correctness errors on layers like embedding projections with non-standard vocabulary sizes.

The filter is applied via `functools.partial` (line 241) and passed to `convert_to_float8_training()` as `module_filter_fn`, which torchao calls for every `nn.Linear` in the model.

### Why QAT uses mode-string dispatch tables rather than class inheritance

The QAT module (`qat.py`) maps quantizer classes to string mode identifiers (`_QUANTIZER_TO_MODE`, line 40) and then maps those modes to enable/disable functions (`_ENABLE_FN_BY_MODE`, line 50; `_DISABLE_FN_BY_MODE`, line 45). This indirection exists because:

1. **Decoupling from torchao's class hierarchy**: The training loop (`train_ft.py`, lines 1025-1067) only needs mode strings and callable toggles, not references to torchao quantizer classes. This means the training loop code does not import torchao at all.
2. **Delayed fake-quant pattern**: The `_setup_qat()` method in `train_ft.py` (line 1025) initially disables fake-quant via `_qat_disable_fn(part)` (line 1049), then `_enable_qat_if_delayed()` (line 1055) enables it after `fake_quant_after_n_steps`. The enable/disable functions are stored on the trainer instance, so they must be simple callables -- the mode string acts as the key to look them up.

### Why QLoRA creation is stateless and returns None for no quantization

`create_bnb_config()` (`qlora.py`, line 22) is a pure factory function that returns a `transformers.BitsAndBytesConfig` or `None`. It returns `None` when neither `load_in_4bit` nor `load_in_8bit` is set (line 41). This `None` sentinel is consumed by `_apply_peft_and_lower_precision()` in `auto_model.py` (line 466), where it is passed as `quantization_config` to `apply_lora_to_linear_modules()`. The LoRA application code in `_peft/lora.py` uses this config to determine whether base weights should use BitsAndBytes quantized storage.

### Why FP8 conversion is fail-safe with a try/except

The `apply_fp8_to_model()` function wraps the entire `convert_to_float8_training()` call in a `try/except` block (lines 239-262). On failure, it logs a warning and returns the original unmodified model rather than crashing. This design choice reflects the fact that FP8 is an optimization, not a correctness requirement -- a training run should be able to fall back to BF16/FP16 if FP8 conversion fails (e.g., due to unsupported layer types or torchao version mismatches).

---

## 3. Core Data Structures

### FP8Config (fp8.py, line 28)

Dataclass that encapsulates all FP8 quantization settings. Created from YAML config via `build_fp8_config()` or `create_fp8_config_from_dict()`.

```
@dataclass
class FP8Config:
    enabled: bool = False                                    # Master on/off switch
    recipe_name: Optional["tensorwise"|"rowwise"|"rowwise_with_gw_hp"] = None
                                                             # Recipe shorthand; None defaults to tensorwise
    enable_fsdp_float8_all_gather: bool = False              # FSDP2 float8 all-gather (recommended for tensorwise)
    precompute_float8_dynamic_scale_for_fsdp: bool = False   # Precompute scales for FSDP (tensorwise only)
    force_recompute_fp8_weight_in_bwd: bool = False          # Recompute FP8 weights in backward pass
    filter_fqns: List[str] = []                              # Module FQNs to skip FP8 conversion
    emulate: bool = False                                    # Software emulation (testing on non-H100 GPUs)
```

Construction entry points:
- `FP8Config.from_config_node(config_node)` (line 74): Creates from a Hydra/OmegaConf config node by iterating `__dataclass_fields__` and extracting matching attributes.
- `create_fp8_config_from_dict(config_dict)` (line 318): Creates from a plain dictionary with `dict.get()` defaults.
- `build_fp8_config(cfg)` (line 339): Top-level factory that returns a disabled `FP8Config` for `None` input or delegates to `create_fp8_config_from_dict()`.

Serialization:
- `to_dict()` (line 87): Returns a dictionary with `fp8_`-prefixed keys for fields that would otherwise collide with other config namespaces.

### HAVE_TORCHAO (fp8.py, line 22)

Module-level boolean set at import time via a `try/except` around `from torchao.float8 import Float8LinearConfig, convert_to_float8_training`. Used as a guard throughout the module and re-exported from `__init__.py` (line 2). When `False`, `apply_fp8_to_model()` raises `ImportError` with `MISSING_TORCHAO_MSG` from `nemo_automodel/shared/import_utils.py`.

### HAS_BNB (qlora.py, line 18)

Module-level boolean set via `safe_import("bitsandbytes")` from `nemo_automodel/shared/import_utils.py`. When `False`, `create_bnb_config()` raises `ImportError`. Re-exported from `__init__.py` (line 10).

### QAT Dispatch Tables (qat.py, lines 40-53)

Three dictionaries that form the QAT mode dispatch system:

```
_QUANTIZER_TO_MODE = {
    Int8DynActInt4WeightQATQuantizer: "8da4w-qat",   # 8-bit dynamic activation, 4-bit weight
    Int4WeightOnlyQATQuantizer: "4w-qat",             # 4-bit weight-only
}

_DISABLE_FN_BY_MODE = {
    "8da4w-qat": disable_8da4w_fake_quant,
    "4w-qat": disable_4w_fake_quant,
}

_ENABLE_FN_BY_MODE = {
    "8da4w-qat": enable_8da4w_fake_quant,
    "4w-qat": enable_4w_fake_quant,
}
```

These are consumed via `get_quantizer_mode()` (line 56), `get_disable_fake_quant_fn()` (line 65), and `get_enable_fake_quant_fn()` (line 71).

---

## 4. State Flow

### FP8: Config to Model Conversion Flow

```
YAML config (fp8 section)
    |
    v
build_fp8_config(cfg_fp8)  [train_ft.py:181 or finetune.py:119]
    |  --> create_fp8_config_from_dict() --> FP8Config dataclass
    v
NeMoAutoModelForCausalLM.from_pretrained() / from_config()
    calls apply_model_infrastructure() [auto_model.py]
        |
        v
    _apply_peft_and_lower_precision(model, ..., fp8_config=...) [auto_model.py:455]
        |
        v
    apply_fp8_to_model(model, config=fp8_config) [fp8.py:143]
        |
        |- Check fp8_config.enabled; return early if False [line 198]
        |- Check HAVE_TORCHAO; raise ImportError if False [line 203]
        |- Set model.precompute_float8_dynamic_scale_for_fsdp attribute [line 207]
        |  (only True when tensorwise + fsdp_all_gather + precompute all enabled)
        |- Build torchao Float8LinearConfig:
        |    - Non-tensorwise recipe: Float8LinearConfig.from_recipe_name() [line 215]
        |      - Rowwise also sets torch._inductor.config.emulate_precision_casts = True [line 220]
        |    - Tensorwise: Float8LinearConfig(enable_fsdp_float8_all_gather=..., ...) [line 224]
        |- Check CUDA SM89+ capability; raise ValueError if missing and not emulating [line 233]
        |- Create filter_fn via partial(_module_filter_fn, filter_fqns=...) [line 241]
        |- Call convert_to_float8_training(model, config=torchao_config, module_filter_fn=filter_fn) [line 244]
        |  (torchao replaces eligible nn.Linear with Float8Linear in-place)
        |- Call verify_fp8_conversion(model) [line 255] to log conversion statistics
        |- On exception: log warning, return original model [line 261]
        v
    Model returned with Float8Linear layers
        |
        v
    FSDP2/TP sharding applied (float8 all-gather participates in FSDP communication)
```

### QAT: Prepare and Delayed Fake-Quant Flow

```
YAML config (qat section)
    |
    v
train_ft.py:183 -- instantiate quantizer from config:
    quantizer = cfg_qat.quantizer.instantiate(precision=bfloat16, scales_precision=bfloat16)
    |
    v
NeMoAutoModelForCausalLM.from_pretrained()
    calls _apply_peft_and_lower_precision(..., qat_quantizer=quantizer)
        |
        v
    prepare_qat_model(model, quantizer) [qat.py:77]
        |-- quantizer.prepare(model)  --> inserts fake-quant nodes into model
        |-- get_quantizer_mode(quantizer) --> mode string ("8da4w-qat" or "4w-qat")
        |-- model._qat_mode = mode  [auto_model.py:481]
        v
    Model returned with fake-quant nodes and _qat_mode attribute
        |
        v
Trainer._setup_qat(cfg, model_parts) [train_ft.py:1025]
    |-- Reads fake_quant_after_n_steps from config
    |-- Retrieves _qat_mode from model_parts[0]
    |-- Gets _qat_disable_fn and _qat_enable_fn via dispatch tables
    |-- Initially disables fake-quant on all model parts [line 1049]
    v
Training loop [train_ft.py:1069+]
    |-- Each step: _enable_qat_if_delayed(step) [line 1055]
    |   |-- If step >= fake_quant_after_n_steps:
    |   |     _qat_enable_fn(part) for each model part
    |   |     Sets _qat_enable_after = None (one-shot)
    v
Model trains with fake-quant enabled after delay
```

### QLoRA: Config Creation and Application Flow

```
YAML config (quantization section)
    |
    v
train_ft.py:192 -- create_bnb_config(cfg_quantization) [qlora.py:22]
    |-- Check HAS_BNB and HAS_TRANSFORMERS; raise ImportError if missing
    |-- If load_in_4bit: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
    |     bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_quant_storage="bfloat16")
    |-- If load_in_8bit: BitsAndBytesConfig(load_in_8bit=True)
    |-- Else: None
    v
kwargs["quantization_config"] = bnb_config
    |
    v
NeMoAutoModelForCausalLM.from_pretrained(**kwargs)
    passes quantization_config to HuggingFace's from_pretrained()
    --> Model loaded with quantized base weights
        |
        v
    _apply_peft_and_lower_precision(model, ..., quantization_config=bnb_config)
        |-- apply_lora_to_linear_modules(model, peft_config, quantization_config=bnb_config)
        |   --> LoRA adapters added on top of quantized base weights
        v
    verify_qlora_quantization(model) [qlora.py:44]
        |-- Walks named_modules, checks for quant_state attribute of type QuantState
        |-- Returns True if any quantized module found
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new FP8 recipe

To support a new torchao recipe (e.g., a hypothetical `"blockwise"` scaling):

1. Add the recipe name to the `Literal` type in `FP8Config.recipe_name` (`fp8.py`, line 34):
   ```python
   recipe_name: Optional[Literal["tensorwise", "rowwise", "rowwise_with_gw_hp", "blockwise"]] = None
   ```
2. In `apply_fp8_to_model()` (`fp8.py`, line 214), the `if fp8_config.recipe_name is not None and fp8_config.recipe_name != "tensorwise"` branch already delegates to `Float8LinearConfig.from_recipe_name()`, so if torchao supports the new recipe natively, no additional code changes are needed in that function.
3. If the new recipe requires special inductor flags (like rowwise sets `emulate_precision_casts` at line 220), add a new conditional block after line 221.
4. If the recipe should support FSDP precomputed dynamic scaling, update the `precompute_float8_dynamic_scale_for_fsdp` condition at line 207-211 to include the new recipe name.

### Scenario 2: Adding a new QAT quantizer mode

To support a new torchao QAT quantizer (e.g., `Int8WeightOnlyQATQuantizer`):

1. Import the new quantizer class and its corresponding enable/disable functions at the top of `qat.py` (lines 29-38).
2. Add entries to the three dispatch dictionaries:
   ```python
   _QUANTIZER_TO_MODE[Int8WeightOnlyQATQuantizer] = "8w-qat"
   _DISABLE_FN_BY_MODE["8w-qat"] = disable_8w_fake_quant
   _ENABLE_FN_BY_MODE["8w-qat"] = enable_8w_fake_quant
   ```
3. No changes needed in `prepare_qat_model()` (line 77) or the training loop in `train_ft.py` -- the mode string dispatch handles everything automatically.
4. Add the new quantizer as a configurable `_target_` option in the YAML config for the `qat.quantizer` field so Hydra can instantiate it.

### Scenario 3: Extending QLoRA to support new quantization backends

To add support for a quantization backend beyond BitsAndBytes (e.g., GPTQ or AWQ):

1. In `qlora.py`, add a new `safe_import()` for the backend library:
   ```python
   HAS_GPTQ, auto_gptq = safe_import("auto_gptq")
   ```
2. Create a new factory function (e.g., `create_gptq_config(config)`) that returns the appropriate quantization config object for the backend.
3. Export the new function and availability flag from `__init__.py`.
4. In `train_ft.py` (around line 190), add a conditional branch that selects between `create_bnb_config()` and `create_gptq_config()` based on the config's backend field.
5. Create a corresponding `verify_gptq_quantization()` function following the pattern of `verify_qlora_quantization()` (`qlora.py`, line 44).

### Scenario 4: Excluding specific layers from FP8 conversion

To skip FP8 conversion for specific modules (e.g., the language model head and embedding layers):

1. In the YAML config, set the `filter_fqns` field on the FP8 config:
   ```yaml
   fp8:
     enabled: true
     recipe_name: tensorwise
     filter_fqns:
       - "lm_head"
       - "embed_tokens"
   ```
2. These strings are passed through `FP8Config.filter_fqns` to `_module_filter_fn()` (`fp8.py`, line 109), which uses substring matching (`fqn in name` at line 126). Any module whose fully qualified name contains any of these strings will be skipped.
3. Note that layers with weight dimensions not divisible by 16 are always skipped automatically (line 136), so those do not need to be listed explicitly.

### Scenario 5: Enabling FP8 emulation for testing on non-H100 hardware

To test FP8 training logic on GPUs without SM89+ (e.g., A100):

1. Set `emulate: true` in the FP8 YAML config:
   ```yaml
   fp8:
     enabled: true
     emulate: true
   ```
2. This bypasses the `_has_cuda_capability(8, 9)` check in `apply_fp8_to_model()` (`fp8.py`, line 233). The `emulate` flag is passed through to `Float8LinearConfig(emulate=True)` (line 228) for tensorwise, or checked via `getattr(torchao_config, "emulate", fp8_config.emulate)` (line 232) for recipe-based configs.
3. Emulation uses software floating-point casting instead of hardware FP8 GEMM, so performance will be significantly slower but functional correctness is preserved.
