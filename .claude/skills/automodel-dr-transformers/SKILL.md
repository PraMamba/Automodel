---
name: automodel-dr-transformers
description: Use when working with the _transformers module of automodel — HuggingFace model/tokenizer auto-registration, custom model overrides, and NeMoAutoModelForCausalLM
---

# Module: `nemo_automodel/_transformers/`

## 1. Module Purpose & Capabilities

The `_transformers` module is the bridge between HuggingFace Transformers and NeMo Automodel's custom model/tokenizer implementations. It provides:

- **NeMo Auto-Model classes** (`NeMoAutoModelForCausalLM`, `NeMoAutoModelForImageTextToText`, `NeMoAutoModelForSequenceClassification`, `NeMoAutoModelForTextToWaveform`) that are drop-in replacements for HuggingFace `Auto*` classes, adding Liger kernel patching, SDPA attention patching, PEFT/LoRA, FP8/QAT quantization, distributed parallelism (FSDP2, DDP, pipeline parallelism), and custom model registry lookup. Defined in `auto_model.py`.
- **Custom model registry** (`_ModelRegistry`) that discovers and registers custom model implementations from `nemo_automodel.components.models` by scanning for modules that expose a `ModelClass` attribute. Defined in `registry.py`.
- **NeMo Auto-Tokenizer** (`NeMoAutoTokenizer`) that dispatches to either a custom tokenizer (looked up by `model_type` in the tokenizer registry) or falls back to `NeMoAutoTokenizerWithBosEosEnforced`. Defined in `auto_tokenizer.py`.
- **Tokenizer registry** (`_TokenizerRegistry`) that maps `model_type` strings to tokenizer implementations (classes, callables, or lazy import strings). Defined in `tokenization/registry.py`.
- **BOS/EOS enforcement wrapper** (`NeMoAutoTokenizerWithBosEosEnforced`) that wraps HuggingFace tokenizers to guarantee BOS and EOS tokens are always present. Defined in `tokenization/nemo_auto_tokenizer.py`.
- **Mistral Common backend** (`MistralCommonBackend`) providing a full HuggingFace-compatible interface over `mistral-common` tokenizers for Mistral/Pixtral/Mistral3 models. Defined in `tokenization/tokenization_mistral_common.py`.
- **Utility functions** for sliding-window config workarounds and cache compatibility patches. Defined in `utils.py`.

## 2. Core Design Logic

### Why a Custom Model Registry?

Automodel needs to override standard HuggingFace model implementations with custom, optimized versions (e.g., models with fused kernels, MoE support, or custom checkpoint logic). The `_ModelRegistry` (in `registry.py`) uses a **convention-based discovery pattern**: it walks all submodules under `nemo_automodel.components.models` and registers any module that exposes a `ModelClass` variable. This means adding a new custom model requires only creating a module with `ModelClass = YourModelClass` -- no central manifest to update.

The registry is instantiated as a module-level singleton via `@lru_cache` on `get_registry()` (line 89-91, `registry.py`). A `naming_override` dict handles cases where the internal class name differs from the architecture string in HuggingFace configs (e.g., `Qwen3OmniMoeThinkerForConditionalGeneration` maps to `Qwen3OmniMoeForConditionalGeneration`).

### Why the Three-Tier Model Resolution?

The `_init_model()` function (line 361, `auto_model.py`) implements a three-tier resolution strategy:

1. **`force_hf=True`**: Always use the standard HuggingFace model, wrapped with `HFCheckpointingMixin` via `_get_mixin_wrapped_class()`.
2. **Custom model available**: If `config.architectures[0]` is found in `ModelRegistry.model_arch_name_to_cls`, instantiate the custom model class directly with the HuggingFace config. This path also applies `_consume_config_overrides()` and `_filter_kwargs_for_init()` to avoid passing config-level kwargs into the model constructor.
3. **Fallback to HF**: If no custom model exists, load via HuggingFace's standard `from_pretrained` or `from_config` and wrap with `HFCheckpointingMixin`.

This design ensures custom optimized models are preferred when available but never break the ability to load any HuggingFace model.

### Why Retry-on-Failure for Kernel Patching?

Both `from_pretrained` and `from_config` in `_BaseNeMoAutoModelClass` (lines 800-1005 and 1007-1213, `auto_model.py`) use an internal `_retry()` helper. If Liger kernel patching fails (`RuntimeError`), the model is deleted and reloaded with `use_liger_kernel=False`. If attention implementation is unsupported (`ValueError`), it falls back through `_get_next_fallback_attn()` (line 247). This guarantees users always get a functional model even when optional optimizations are unavailable.

### Why Lazy Imports Everywhere?

The `__init__.py` uses `_LAZY_ATTRS` with `__getattr__` to defer importing `torch` and model code until a specific class is accessed. Similarly, `auto_tokenizer.py` lazily imports `TokenizerRegistry` and `NeMoAutoTokenizerWithBosEosEnforced`. The tokenizer registry stores import strings like `"nemo_automodel._transformers.tokenization.tokenization_mistral_common:MistralCommonBackend"` that are resolved only on first use via `_resolve_tokenizer_impl()`. This keeps `import nemo_automodel._transformers` lightweight.

### Why a Separate Tokenizer Registry?

The tokenizer registry (`_TokenizerRegistry` in `tokenization/registry.py`) exists because some model families (Mistral, Pixtral) need specialized tokenization via `mistral-common` rather than HuggingFace's built-in tokenizers. The registry maps `model_type` (from HF config) to tokenizer classes, with graceful fallback: if a custom tokenizer's optional dependency (`mistral-common`) is missing, the entry is silently removed and the default wrapper is used instead (see `get_custom_tokenizer_cls()`, line 73-87).

### Why `NeMoAutoTokenizerWithBosEosEnforced`?

Some HuggingFace tokenizers (e.g., GPT2Tokenizer) do not reliably add BOS/EOS tokens. `NeMoAutoTokenizerWithBosEosEnforced` (in `tokenization/nemo_auto_tokenizer.py`) patches this by overriding `__call__()`, `encode()`, and `save_pretrained()`. It dynamically creates a new class that inherits from both the wrapper and the actual tokenizer class (line 47), preserving the original class for HuggingFace-compatible serialization.

## 3. Core Data Structures

| Class / Structure | File | Purpose |
|---|---|---|
| `_BaseNeMoAutoModelClass` | `auto_model.py:760` | Base class overriding `from_pretrained` and `from_config` with kernel patching, registry lookup, and infrastructure application |
| `NeMoAutoModelForCausalLM` | `auto_model.py:1216` | Concrete auto-model for causal LM, inherits `_BaseNeMoAutoModelClass` + `AutoModelForCausalLM` |
| `NeMoAutoModelForImageTextToText` | `auto_model.py:1247` | Concrete auto-model for image-text-to-text tasks |
| `NeMoAutoModelForSequenceClassification` | `auto_model.py:1277` | Concrete auto-model for sequence classification |
| `NeMoAutoModelForTextToWaveform` | `auto_model.py:1307` | Concrete auto-model for text-to-waveform tasks |
| `_ModelRegistry` | `registry.py:33` | Dataclass holding `model_arch_name_to_cls` dict mapping architecture names to model classes, with auto-discovery from `MODELING_PATH` |
| `ModelRegistry` | `registry.py:94` | Module-level singleton instance of `_ModelRegistry` |
| `NeMoAutoTokenizer` | `auto_tokenizer.py:50` | Dispatcher class with `from_pretrained()` that routes to custom tokenizer or default wrapper |
| `_TokenizerRegistry` | `tokenization/registry.py:49` | Dataclass mapping `model_type` strings to tokenizer implementations (class, callable, or import string) |
| `TokenizerRegistry` | `tokenization/registry.py:106` | Module-level singleton instance of `_TokenizerRegistry` |
| `NeMoAutoTokenizerWithBosEosEnforced` | `tokenization/nemo_auto_tokenizer.py:19` | AutoTokenizer wrapper that ensures BOS/EOS tokens are always added |
| `MistralCommonBackend` | `tokenization/tokenization_mistral_common.py:169` | Full HuggingFace-compatible tokenizer using `mistral-common` library for Mistral model families |
| `ValidationMode` | `tokenization/tokenization_mistral_common.py:42` | Enum with `serving`, `finetuning`, `test` modes for `MistralCommonBackend` |

## 4. State Flow

### Model Loading Flow (from_pretrained)

```
NeMoAutoModelForCausalLM.from_pretrained(model_name, ...)
  |
  +--> _BaseNeMoAutoModelClass.from_pretrained()           [auto_model.py:800]
       |
       +--> get_hf_config()                                [auto_model.py:275]
       |      Loads AutoConfig to determine architectures
       |
       +--> get_is_hf_model()                              [auto_model.py:292]
       |      Checks if architectures[0] is in ModelRegistry
       |
       +--> _pop_tp_cp_has_packed()                        [auto_model.py:302]
       |      Extracts TP/CP/packed-sequence flags from kwargs
       |
       +--> _apply_preload_overrides()                     [auto_model.py:312]
       |      Adjusts attn_implementation and liger_kernel based on constraints
       |
       +--> Determine is_meta_device based on model_wrapper type and world_size
       |
       +--> _init_model()                                  [auto_model.py:361]
       |      |
       |      +--> [force_hf] cls._from_pretrained_parent_class() -> wrap with HFCheckpointingMixin
       |      |
       |      +--> [custom model found in ModelRegistry]
       |      |      _download_model_weights()
       |      |      _consume_config_overrides()
       |      |      _filter_kwargs_for_init()
       |      |      model_cls(hf_config, **kwargs) inside local_torch_dtype()
       |      |
       |      +--> [fallback] cls._from_pretrained_parent_class() -> wrap with HFCheckpointingMixin
       |
       +--> _patch_liger_kernel(model)                     [auto_model.py:213]
       |      (only for HF models, retries without on failure)
       |
       +--> _patch_attention(model, sdpa_method)           [auto_model.py:148]
       |      (only for HF models, retries without on failure)
       |
       +--> apply_model_infrastructure()                   [auto_model.py:531]
              |
              +--> _verify_sdpa_support()
              +--> _apply_peft_and_lower_precision()       (LoRA, FP8, QAT)
              +--> apply_parameter_freezing()              (if freeze_config)
              +--> _shard_pp() or _shard_ep_fsdp()         (pipeline or EP/FSDP parallelism)
              +--> compile_model()                         (if compile_config)
              +--> Checkpointer.load_base_model()          (if meta device)
              +--> model.to(device)
```

### Tokenizer Loading Flow

```
NeMoAutoTokenizer.from_pretrained(model_name, ...)
  |
  +--> [force_hf=True] -> transformers.AutoTokenizer.from_pretrained()
  |
  +--> _get_model_type()                              [auto_tokenizer.py:21]
  |      Loads AutoConfig to get config.model_type
  |
  +--> _get_tokenizer_registry()                      [auto_tokenizer.py:42]
  |      Lazily imports TokenizerRegistry singleton
  |
  +--> registry.get_custom_tokenizer_cls(model_type)  [tokenization/registry.py:73]
  |      |
  |      +--> _resolve_tokenizer_impl()               (resolves import strings)
  |      |
  |      +--> [found] custom_cls.from_pretrained()
  |           e.g., MistralCommonBackend.from_pretrained() for model_type="mistral"
  |
  +--> [not found or force_default]
       NeMoAutoTokenizerWithBosEosEnforced.from_pretrained()
         |
         +--> AutoTokenizer.from_pretrained()
         +--> Set add_bos_token, add_eos_token
         +--> Dynamically create wrapper class type(cls.__name__, (cls, base_cls), {})
```

### Model Registry Discovery Flow

```
Module import of registry.py
  |
  +--> get_registry() [cached via @lru_cache]           [registry.py:89]
       |
       +--> _ModelRegistry(modeling_path=MODELING_PATH)  [registry.py:33]
            |
            +--> __post_init__()                         [registry.py:39]
                 |
                 +--> For each path in modeling_path:
                      _mapping_model_arch_name_to_cls()  [registry.py:57]
                        |
                        +--> pkgutil.walk_packages() over nemo_automodel.components.models
                        +--> For each module: check for ModelClass attribute
                        +--> Register class.__name__ -> class in model_arch_name_to_cls
                        +--> Apply naming_override if class name has an override entry
```

## 5. Common Modification Scenarios

### Scenario 1: Adding a New Custom Model Implementation

To add a new custom model (e.g., `MyNewModelForCausalLM`):

1. Create a new module under `nemo_automodel/components/models/` (the path scanned by `MODELING_PATH` in `registry.py:29`).
2. In that module, define your model class and expose it as `ModelClass`:
   ```python
   ModelClass = MyNewModelForCausalLM  # or a list for multiple architectures
   ```
3. Ensure the class name matches the HuggingFace `config.architectures[0]` string. If it differs, add an entry to `naming_override` in `_ModelRegistry.__post_init__()` (`registry.py:40`).
4. If the model needs special config compatibility checks, add a case to `_is_config_compatible_with_custom_model()` (`auto_model.py:189`).
5. No changes to `auto_model.py` or `__init__.py` are needed -- the registry discovers the model automatically.

### Scenario 2: Registering a Custom Tokenizer for a New Model Family

To add a custom tokenizer for model type `"my_model"`:

1. Create your tokenizer class with a `from_pretrained(cls, pretrained_model_name_or_path, ...)` classmethod.
2. Register it in `_register_default_tokenizers()` in `tokenization/registry.py:109`:
   ```python
   TokenizerRegistry.register("my_model", "my_package.my_module:MyTokenizerClass")
   ```
   Import strings are resolved lazily, so optional dependencies won't break import time.
3. Alternatively, register at runtime via `NeMoAutoTokenizer.register("my_model", MyTokenizerClass)` (defined in `auto_tokenizer.py:73`).

### Scenario 3: Adding a New Auto-Model Task Type

To add support for a new HuggingFace Auto-Model type (e.g., `AutoModelForTokenClassification`):

1. In `auto_model.py`, create a new class inheriting from `_BaseNeMoAutoModelClass` and the HuggingFace auto class:
   ```python
   class NeMoAutoModelForTokenClassification(_BaseNeMoAutoModelClass, AutoModelForTokenClassification):
       pass
   ```
2. Add the import to `_LAZY_ATTRS` in `__init__.py:20` and to `__all__`.
3. The new class inherits all `from_pretrained`/`from_config` logic including Liger patching, registry lookup, and infrastructure application from `_BaseNeMoAutoModelClass`.

### Scenario 4: Modifying the Model Initialization Pipeline

The model initialization pipeline in `from_pretrained` and `from_config` is decomposed into standalone functions for testability and extensibility:

- **Change attention defaults**: Modify `DEFAULT_ATTN_IMPLEMENTATION` (`auto_model.py:78`) or `_get_next_fallback_attn()` (`auto_model.py:247`).
- **Add a new kernel patching step**: Add a new try/except block after the Liger and SDPA patching sections in `from_pretrained()` (around line 965-981), following the same retry pattern.
- **Change post-init infrastructure**: Modify `apply_model_infrastructure()` (`auto_model.py:531`), which is the single function that handles PEFT, quantization, parallelism, compilation, and checkpoint loading. This function is also usable standalone for models built via custom builder functions.

### Scenario 5: Changing BOS/EOS Token Enforcement Behavior

The `NeMoAutoTokenizerWithBosEosEnforced` class (`tokenization/nemo_auto_tokenizer.py:19`) controls BOS/EOS insertion. Key modification points:

- **Disable enforcement for specific models**: Override `from_pretrained()` with `add_bos_token=False` or `add_eos_token=False`.
- **Change token insertion logic**: Modify the `_add_token()` helper (`tokenization/nemo_auto_tokenizer.py:102`), which handles both single sequences and batched sequences, with an `always_add` flag for non-`input_ids` keys.
- **Change save behavior**: The `save_pretrained()` override (line 86) temporarily swaps `self.__class__` back to the original HuggingFace tokenizer class to ensure HuggingFace-compatible serialization.

## Key Files Summary

| File | Lines | Role |
|---|---|---|
| `__init__.py` | 51 | Lazy-loading package init exposing 5 public classes |
| `registry.py` | 94 | Model registry with auto-discovery from `nemo_automodel.components.models` |
| `auto_model.py` | 1334 | Core auto-model classes with kernel patching, registry lookup, and full infrastructure pipeline |
| `auto_tokenizer.py` | 151 | Auto-tokenizer dispatcher with custom tokenizer registry integration |
| `utils.py` | 50 | Sliding-window config workaround and cache compatibility patches |
| `tokenization/registry.py` | 124 | Tokenizer registry mapping model_type to tokenizer implementations |
| `tokenization/nemo_auto_tokenizer.py` | 126 | BOS/EOS enforcement wrapper around HuggingFace AutoTokenizer |
| `tokenization/tokenization_mistral_common.py` | 2004 | Full HuggingFace-compatible Mistral tokenizer using mistral-common library |
