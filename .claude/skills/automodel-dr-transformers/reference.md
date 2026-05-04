# Reference: `nemo_automodel/_transformers/`

## File: `__init__.py`

### Module-Level Constants
- `_LAZY_ATTRS` (dict, line 20): Maps public class names to `(module_path, attr_name)` tuples for lazy importing. Contains entries for `NeMoAutoModelForCausalLM`, `NeMoAutoModelForImageTextToText`, `NeMoAutoModelForSequenceClassification`, `NeMoAutoModelForTextToWaveform`, and `NeMoAutoTokenizer`.

### Functions
- `__getattr__(name: str)` (line 40): Module-level lazy attribute loader. On first access of a name in `_LAZY_ATTRS`, imports the target module and caches the attribute in `globals()`.
- `__dir__()` (line 50): Returns sorted `__all__` for IDE introspection.

---

## File: `registry.py`

### Constants
- `MODELING_PATH` (list, line 29): Default modeling paths to scan. Value: `["nemo_automodel.components.models"]`.

### Classes

#### `_ModelRegistry` (dataclass, line 33)
Fields:
- `modeling_path: List[str]` -- Paths to scan for model modules.
- `model_arch_name_to_cls: Dict[str, Union[Type[nn.Module], str]]` -- Maps architecture name to model class.
- `naming_override: Dict[str, str]` -- Overrides for architecture name mismatches (e.g., `Qwen3OmniMoeThinkerForConditionalGeneration` -> `Qwen3OmniMoeForConditionalGeneration`).

Methods:
- `__post_init__(self)` (line 39): Populates `naming_override` and calls `_mapping_model_arch_name_to_cls()` for each path.
- `supported_models` (property, line 44): Returns keys of `model_arch_name_to_cls`.
- `get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]` (line 48): Direct dict lookup.
- `register_modeling_path(self, path: str) -> None` (line 51): Adds a new path and registers all models from it. Idempotent (skips if already registered).
- `_mapping_model_arch_name_to_cls(self, modeling_path: str)` (line 57): Walks submodules via `pkgutil.walk_packages()`. For each module with a `ModelClass` attribute, registers the class(es). Supports both single class and list of classes. Applies `naming_override`. Asserts no duplicates.

### Functions
- `get_registry() -> _ModelRegistry` (line 89): `@lru_cache` factory returning singleton `_ModelRegistry(modeling_path=MODELING_PATH)`.

### Module-Level Singletons
- `ModelRegistry` (line 94): The singleton `_ModelRegistry` instance, imported throughout the codebase.

---

## File: `auto_model.py`

### Module-Level Constants
- `HAS_LIGER_KERNEL` (bool, line 76): Whether `liger_kernel.transformers` is importable.
- `HAS_FA` (bool, line 77): Whether `flash_attn` is importable.
- `DEFAULT_ATTN_IMPLEMENTATION` (str, line 78): `"flash_attention_2"` if Flash Attention is available, else `"sdpa"`.

### Standalone Functions

- `_get_mixin_wrapped_class(model_class: type) -> type` (line 83): Creates a new class inheriting from both `HFCheckpointingMixin` and `model_class`. Returns `model_class` unchanged if it already has the mixin.

- `local_torch_dtype(dtype, model_class_name=None, default_dtype=torch.bfloat16)` (line 110): Context manager that temporarily sets `torch.set_default_dtype(dtype)` and restores the original on exit.

- `_assert_same_signature(original, patched)` (line 137): Validates that two callables have identical signatures. Used after attention patching.

- `_patch_attention(obj, sdpa_method=None)` (line 148): Wraps `obj.forward` in an `sdpa_kernel` context manager with the given backends. Default order: `[CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH]`.

- `_is_config_compatible_with_custom_model(arch_name: str, config) -> bool` (line 189): Validates HuggingFace config compatibility with custom model. Currently only checks `NemotronHForCausalLM` (requires `n_routed_experts` for v3 MoE).

- `_patch_liger_kernel(model)` (line 213): Applies `liger_kernel.transformers._apply_liger_kernel_to_instance()` to the model. Raises `RuntimeError` on failure (caught by caller for retry).

- `_get_next_fallback_attn(attn_implementation: str) -> str` (line 247): Returns the next lower-priority attention implementation. Priority order: `eager < sdpa < flash_attention_2 < flash_attention_3`.

- `get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs)` (line 275): Loads `AutoConfig.from_pretrained()` with trust_remote_code resolution.

- `get_is_hf_model(config, force_hf) -> bool` (line 292): Returns `True` if no architecture is found in `ModelRegistry` or if `force_hf=True`.

- `_pop_tp_cp_has_packed(kwargs) -> tuple[int, int, bool]` (line 302): Pops and returns `tp_size`, `cp_size`, `has_packed_sequence` from kwargs.

- `_apply_preload_overrides(is_hf_model, tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel)` (line 312): Adjusts attention and liger kernel settings based on TP/CP/packed-sequence constraints. Disables Liger with TP>1 for HF models. Forces `sdpa` with CP>1. Forces `flash_attention_2` with packed sequences.

- `_verify_sdpa_support(model, is_hf_model, cp_size)` (line 338): Raises `ValueError` if model doesn't support SDPA but CP>1 is requested.

- `_download_model_weights(hf_config, pretrained_model_name_or_path)` (line 347): Downloads model via `snapshot_download()` within `FirstRankPerNode` context to avoid redundant downloads.

- `_init_model(cls, pretrained_model_name_or_path_or_config, attn_implementation, torch_dtype, quantization_config, force_hf, *model_args, **kwargs)` (line 361): Three-tier model resolution: (1) force_hf, (2) custom model from registry, (3) fallback HF. Returns `(is_custom_model: bool, model: PreTrainedModel)`.

- `_apply_peft_and_lower_precision(model, tp_size, autopipeline, peft_config, quantization_config, fp8_config, qat_quantizer)` (line 455): Applies LoRA via `apply_lora_to_linear_modules()`, FP8 via `apply_fp8_to_model()`, and QAT via `prepare_qat_model()`.

- `_shard_pp(autopipeline, model, loss_fn, parallelize_fn)` (line 486): Applies pipeline parallelism via `autopipeline.build()`. Stores trainable/total param counts before splitting.

- `_shard_ep_fsdp(model, model_wrapper, parallelize_fn)` (line 500): Applies Expert Parallelism + FSDP2 via `parallelize_fn()` or `model_wrapper.parallelize()`.

- `apply_model_infrastructure(model, *, is_hf_model, is_meta_device, device, ...)` (line 531): The main post-init orchestrator. Applies SDPA verification, PEFT/FP8/QAT, parameter freezing, loss function setup, pipeline or EP/FSDP sharding, compilation, checkpoint loading, and device placement. Also callable standalone for custom-built models.

- `get_architectures(hf_config) -> list` (line 690): Extracts `architectures` list from HF config, defaulting to empty list.

- `_get_init_param_names(model_cls) -> set[str]` (line 700): Inspects `model_cls.__init__` signature to extract parameter names.

- `_consume_config_overrides(config, kwargs, *, init_param_names=None)` (line 713): Moves config-level kwargs (e.g., `output_hidden_states`) from `kwargs` onto the config object. Skips keys that are explicit model `__init__` parameters.

- `_filter_kwargs_for_init(model_cls, kwargs) -> dict` (line 739): Filters kwargs to only those accepted by `model_cls.__init__`. Passes all kwargs through if constructor has `**kwargs`.

### Classes

#### `_BaseNeMoAutoModelClass` (line 760)
Inherits: `_BaseAutoModelClass` (from `transformers.models.auto.auto_factory`)

Methods:
- `_from_pretrained_parent_class(cls, *args, **kwargs)` (classmethod, line 782): Temporarily strips `"NeMo"` prefix from class name to call parent `from_pretrained()`. This is needed because HuggingFace's auto-factory resolves the class name.
- `_from_config_parent_class(cls, *args, **kwargs)` (classmethod, line 791): Same pattern for `from_config()`.
- `from_pretrained(cls, pretrained_model_name_or_path, *, use_liger_kernel=True, use_sdpa_patching=True, sdpa_method=None, torch_dtype="auto", attn_implementation=DEFAULT_ATTN_IMPLEMENTATION, quantization_config=None, force_hf=False, model_wrapper=None, autopipeline=None, parallelize_fn=None, peft_config=None, fp8_config=None, qat_quantizer=None, loss_fn=None, compile_config=None, **kwargs) -> PreTrainedModel` (classmethod, line 800): Full model loading pipeline. Determines HF vs custom, applies meta device init, calls `_init_model()`, patches kernels, calls `apply_model_infrastructure()`.
- `from_config(cls, config, *, ...)` (classmethod, line 1007): Same pipeline but for config-based initialization (no pretrained weights). Sets `load_base_model=False`.

#### `NeMoAutoModelForCausalLM` (line 1216)
Inherits: `_BaseNeMoAutoModelClass`, `AutoModelForCausalLM`. Body is `pass`.

#### `NeMoAutoModelForImageTextToText` (line 1247)
Inherits: `_BaseNeMoAutoModelClass`, `AutoModelForImageTextToText`. Body is `pass`.

#### `NeMoAutoModelForSequenceClassification` (line 1277)
Inherits: `_BaseNeMoAutoModelClass`, `AutoModelForSequenceClassification`. Body is `pass`.

#### `NeMoAutoModelForTextToWaveform` (line 1307)
Inherits: `_BaseNeMoAutoModelClass`, `AutoModelForTextToWaveform`. Body is `pass`.

---

## File: `auto_tokenizer.py`

### Functions
- `_get_model_type(pretrained_model_name_or_path, trust_remote_code=False) -> Optional[str]` (line 21): Loads `AutoConfig` and returns `config.model_type`.
- `_get_tokenizer_registry()` (line 42): Lazy import of `TokenizerRegistry` singleton from `tokenization/registry.py`.

### Classes

#### `NeMoAutoTokenizer` (line 50)
Class-level attribute:
- `_registry = None`

Methods:
- `register(cls, model_type: str, tokenizer_cls: Union[Type, Callable]) -> None` (classmethod, line 73): Delegates to `TokenizerRegistry.register()`.
- `from_pretrained(cls, pretrained_model_name_or_path, *, force_default=False, force_hf=False, trust_remote_code=False, **kwargs)` (classmethod, line 84): Dispatch logic:
  1. `force_hf=True` -> raw `AutoTokenizer.from_pretrained()`.
  2. Custom tokenizer found for `model_type` -> `custom_cls.from_pretrained()`.
  3. Fallback -> `NeMoAutoTokenizerWithBosEosEnforced.from_pretrained()`.

### Module-Level Lazy Attributes
- `__getattr__` (line 140): Lazily provides `TokenizerRegistry` and `NeMoAutoTokenizerWithBosEosEnforced`.

---

## File: `utils.py`

### Functions
- `sliding_window_overwrite(model_name: str) -> dict[str, Any]` (line 20): Loads HuggingFace config and returns `{"sliding_window": None}` if `use_sliding_window=False` but `sliding_window` is set. Workaround for HuggingFace bug #38002.
- `apply_cache_compatibility_patches()` (line 44): Aliases `DynamicCache.get_usable_length` to `DynamicCache.get_seq_length` if the former is missing.

---

## File: `tokenization/registry.py`

### Type Aliases
- `TokenizerImpl` (line 22): `Union[Type[Any], Callable[..., Any], str]` -- A tokenizer can be a class, callable, or import string.

### Constants
- `_DEFAULT_TOKENIZER_IMPL` (str, line 24): `"nemo_automodel._transformers.tokenization.nemo_auto_tokenizer:NeMoAutoTokenizerWithBosEosEnforced"`.

### Functions
- `_resolve_tokenizer_impl(tokenizer_impl: TokenizerImpl) -> Union[Type, Callable]` (line 29): If `tokenizer_impl` is a string of the form `"module:Class"`, imports and returns the attribute. Otherwise returns the input unchanged.
- `_register_default_tokenizers()` (line 109): Registers `MistralCommonBackend` (as import string) for model types `"mistral"`, `"pixtral"`, and `"mistral3"`.

### Classes

#### `_TokenizerRegistry` (dataclass, line 49)
Fields:
- `model_type_to_tokenizer: Dict[str, TokenizerImpl]` -- Maps model_type to tokenizer impl.
- `default_tokenizer: TokenizerImpl` -- Default tokenizer (the BOS/EOS enforced wrapper).

Methods:
- `register(self, model_type: str, tokenizer_cls: TokenizerImpl)` (line 62): Adds entry to `model_type_to_tokenizer`.
- `get_custom_tokenizer_cls(self, model_type: str) -> Optional[Union[Type, Callable]]` (line 73): Resolves and returns the custom tokenizer, or `None`. On import failure, silently removes the entry to avoid retrying.
- `get_tokenizer_cls(self, model_type: str) -> Union[Type, Callable]` (line 89): Returns custom tokenizer if available, otherwise the resolved default.
- `has_custom_tokenizer(self, model_type: str) -> bool` (line 100): Simple dict membership check.

### Module-Level Singletons
- `TokenizerRegistry` (line 106): The singleton `_TokenizerRegistry` instance.

---

## File: `tokenization/nemo_auto_tokenizer.py`

### Classes

#### `NeMoAutoTokenizerWithBosEosEnforced` (line 19)
Inherits: `AutoTokenizer`

Methods:
- `from_pretrained(cls, pretrained_model_name_or_path, *, add_bos_token=True, add_eos_token=True, **kwargs)` (classmethod, line 28): Loads HF tokenizer, sets `add_bos_token`/`add_eos_token`, stores original class in `_base_class`, dynamically creates wrapper class via `type()`.
- `__call__(self, *args, **kwargs)` (line 50): Calls parent `__call__`, then ensures BOS/EOS in `input_ids`, `attention_mask`, and `assistant_masks` keys of `BatchEncoding`.
- `encode(self, *args, **kwargs)` (line 74): Calls parent `encode`, then prepends BOS and appends EOS if missing.
- `save_pretrained(self, save_directory, push_to_hub=False, **kwargs)` (line 86): Temporarily swaps `self.__class__` to `_base_class` for HF-compatible serialization, then restores.

### Helper Functions
- `_add_token(tokenized, value, position, key)` (line 102): Adds a token value at position 0 (prepend) or -1 (append) in `tokenized[key]`. Handles both single sequences and batched (list of lists). The `always_add` flag is `True` for non-`input_ids` keys (always adds attention mask values even if the token was already present).

---

## File: `tokenization/tokenization_mistral_common.py`

### Enums
- `ValidationMode` (line 42): `serving`, `finetuning`, `test`. Controls how `MistralCommonBackend` validates input and adds special tokens (finetuning adds both BOS+EOS, test adds only BOS).
- `MistralTokenizerType` (line 161): `spm`, `tekken`. Internal enum for tokenizer backend type.

### Classes

#### `MistralCommonBackend` (line 169)
Inherits: `PushToHubMixin`
Wraps the `mistral-common` tokenizer library with a HuggingFace-compatible interface.

Class attributes:
- `model_input_names: list[str]` -- `["input_ids", "attention_mask"]`
- `padding_side: str` -- `"left"` (default)
- `truncation_side: str` -- `"right"` (default)

Constructor `__init__(self, tokenizer_path, mode=ValidationMode.test, model_max_length=VERY_LARGE_INTEGER, padding_side="left", truncation_side="right", model_input_names=None, clean_up_tokenization_spaces=False, **kwargs)` (line 220):
- Creates `MistralTokenizer.from_file()`.
- Detects tokenizer type (spm vs tekken).
- Caches all special token IDs.

Key properties:
- `mode` (line 316), `bos_token_id` (line 326), `eos_token_id` (line 333), `unk_token_id` (line 340), `pad_token_id` (line 347), `bos_token` (line 354), `eos_token` (line 361), `unk_token` (line 368), `pad_token` (line 375), `all_special_ids` (line 382), `all_special_tokens` (line 389), `vocab_size` (line 396).

Key methods:
- `get_vocab(self) -> dict[str, int]` (line 404): Returns vocabulary dict with caching.
- `encode(self, text, ...) -> list[int]` (line 439): Tokenizes text with padding/truncation support.
- `decode(self, token_ids, ...) -> str | list[str]` (line 495): Decodes token IDs, handles both single and batch.
- `batch_decode(self, sequences, ...) -> list[str]` (line 541): Batch decoding (delegates to `_batch_decode`).
- `convert_ids_to_tokens(self, ids, skip_special_tokens=False)` (line 633): ID to token string conversion.
- `convert_tokens_to_ids(self, tokens) -> int | list[int]` (line 691): Token string to ID conversion.
- `_text_to_ids(self, text, add_special_tokens) -> list[int]` (line 717): Core encoding. Adds BOS always if `add_special_tokens=True`, adds EOS only in `finetuning` mode.
- `tokenize(self, text, **kwargs) -> list[str]` (line 725): Returns token strings.
- `_encode_plus(self, text, ...) -> BatchEncoding` (line 746): Single sequence encoding with padding/truncation.
- `_batch_encode_plus(self, batch_text, ...) -> BatchEncoding` (line 791): Batch encoding.
- `get_special_tokens_mask(self, token_ids_0, ...) -> list[int]` (line 847): Returns 1/0 mask for special tokens.
- `prepare_for_model(self, ids, ...) -> BatchEncoding` (line 942): Truncation, padding, and output formatting.
- `_pad(self, encoded_inputs, ...) -> dict` (line 1151): Low-level padding implementation.
- `pad(self, encoded_inputs, ...) -> BatchEncoding` (line 1225): High-level padding supporting single/batch inputs.
- `truncate_sequences(self, ids, ...) -> tuple[list[int], None, list[int]]` (line 1387): Truncation with stride-based overflow.
- `apply_chat_template(self, conversation, ...) -> str | list[int] | BatchEncoding` (line 1462): Converts chat messages to token IDs via `mistral-common`'s `ChatCompletionRequest`. Handles images and audio. Returns pixel_values if images present.
- `__call__(self, text, ...) -> BatchEncoding` (line 1675): Main tokenization entry point. Routes to `_encode_plus` (single) or `_batch_encode_plus` (batch).
- `from_pretrained(cls, pretrained_model_name_or_path, *, mode=ValidationMode.test, ...) -> MistralCommonBackend` (classmethod, line 1791): Downloads tokenizer from HuggingFace Hub or loads from local directory. Prefers `tekken.json` over sentencepiece files when multiple are found.
- `save_pretrained(self, save_directory, push_to_hub=False, ...) -> tuple[str, ...]` (line 1923): Copies tokenizer file to save directory.
- `_get_validation_mode(mode) -> ValidationMode` (staticmethod, line 1984): Parses string or enum mode.

### Module-Level Aliases
- `MistralCommonTokenizer = MistralCommonBackend` (line 2004): Backward compatibility alias.
