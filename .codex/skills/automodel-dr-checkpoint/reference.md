# Checkpoint Module — Reference

Quick lookup of every file, class, function, constant, and data structure in the checkpoint module.

## File Index

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 25 | Version-conditional monkey-patching of PyTorch DCP |
| `checkpointing.py` | 1124 | Central `Checkpointer` class and all save/load orchestration |
| `stateful_wrappers.py` | 334 | DCP `Stateful` wrappers: `ModelState`, `OptimizerState` |
| `addons.py` | 269 | `CheckpointAddon` protocol, `ConsolidatedHFAddon`, `PeftAddon` |
| `state_dict_adapter.py` | 73 | ABC for state dict format conversion |
| `conversion_mapping.py` | 229 | Transformers-based checkpoint key/tensor conversion |
| `utils.py` | 38 | `is_tied_word_embeddings()` utility |
| `_torch_backports.py` | 95 | Monkey-patches for old/new PyTorch versions |
| `_backports/__init__.py` | 0 | Empty package init |
| `_backports/_version.py` | 19 | `_derived_version` global for checkpoint format compat |
| `_backports/hf_utils.py` | 112 | SafeTensors constants, dtype map, file metadata parsing |
| `_backports/hf_storage.py` | 441 | `_HuggingFaceStorageWriter`, `_HuggingFaceStorageReader` |
| `_backports/filesystem.py` | 1081 | `FileSystemWriter`, `FileSystemReader`, `SerializationFormat`, SafeTensors streaming writer |
| `_backports/_fsspec_filesystem.py` | 174 | `FsspecWriter`, `FsspecReader` for remote storage |
| `_backports/default_planner.py` | 649 | `DefaultSavePlanner`, `DefaultLoadPlanner` with caching |
| `_backports/consolidate_hf_safetensors.py` | 859 | Sharded-to-consolidated SafeTensors merging |
| `_backports/planner_helpers.py` | 31 | `_contains_usable_plan()` for plan caching |

## Classes

### `Checkpointer` — `checkpointing.py:130`
High-level checkpoint manager. All save/load goes through this.

| Method | Line | Description |
|--------|------|-------------|
| `__init__(config, dp_rank, tp_rank, pp_rank, moe_mesh)` | 144 | Init with ranks, register addons, create async contexts |
| `save_model(model, weights_path, peft_config, tokenizer)` | 183 | Full model save with sharding + consolidation |
| `save_optimizer(optimizer, model, weights_path, scheduler)` | 264 | Save optimizer + scheduler state via DCP |
| `load_model(model, model_path, is_init_step, ...)` | 299 | Load model from DCP or HF checkpoint |
| `load_optimizer(optimizer, model, weights_path, scheduler)` | 282 | Load optimizer + scheduler state |
| `load_base_model(model, device, root_dir, model_name, ...)` | 361 | Full base model initialization from HF checkpoint |
| `maybe_wait_for_staging()` | 435 | Block until async staging completes |
| `async_wait()` | 446 | Block until async upload completes |
| `save_on_dp_ranks(state, state_name, path)` | 457 | DP-aware auxiliary state save (TP=0, PP=0 only) |
| `load_on_dp_ranks(state, state_name, path)` | 473 | DP-aware auxiliary state load |
| `close()` | 489 | Wait for all async ops, close stagers |
| `_do_save(state_dict, path, storage_writer)` | 529 | Internal: DCP save or PEFT rank-0 save |
| `_do_load(state_dict, path, storage_reader, is_init_step)` | 500 | Internal: DCP load or PEFT rank-0 load |
| `_should_write_consolidated_safetensors()` | 574 | Predicate: consolidation enabled + safetensors + not PEFT |
| `_should_write_hf_metadata()` | 582 | Predicate: safetensors + not PEFT |
| `_maybe_build_consolidated_index(model_state, state_dict)` | 588 | Build fqn-to-shard-index mapping |
| `_get_storage_writer(...)` | 650 | Create `_HuggingFaceStorageWriter` |
| `_get_storage_reader(model_path, key_mapping, is_init_step)` | 678 | Create `_HuggingFaceStorageReader` |
| `_get_original_model_path(model_state)` | 696 | Resolve HF model snapshot path |

### `CheckpointingConfig` — `checkpointing.py:88`
Dataclass with fields: `enabled`, `checkpoint_dir`, `model_save_format`, `model_cache_dir`, `model_repo_id`, `save_consolidated`, `is_peft`, `model_state_dict_keys`, `is_async`, `dequantize_base_checkpoint`, `original_model_root_dir`, `skip_task_head_prefixes_for_base_model`, `single_rank_consolidation`, `staging_dir`.

### `_AsyncSaveContext` — `checkpointing.py:72`
Dataclass: `stager`, `process_group`, `future`, `staging_active`.

### `ModelState` — `stateful_wrappers.py:106`

| Method | Line | Description |
|--------|------|-------------|
| `__init__(model, is_peft, is_init_step, skip_task_head_prefixes)` | 117 | Detect tied embeddings, store config |
| `state_dict()` | 153 | Get model state dict (handles PEFT, quantized, tied heads) |
| `load_state_dict(state_dict, strict)` | 190 | Set model state dict (handles PEFT prefix, DoRA keys) |
| `_get_base_model_state_dict()` | 223 | State dict for init step (strips LoRA, task heads) |
| `_set_base_model_state_dict(state_dict)` | 248 | Load with strict=False for base model |

### `OptimizerState` — `stateful_wrappers.py:253`

| Method | Line | Description |
|--------|------|-------------|
| `__init__(model, optimizer, scheduler)` | 266 | Store references as lists |
| `state_dict()` | 293 | Get flattened optimizer + scheduler state |
| `load_state_dict(state_dict)` | 316 | Restore optimizer + scheduler state |

### `StateDictAdapter` (ABC) — `state_dict_adapter.py:22`

| Method | Description |
|--------|-------------|
| `to_hf(state_dict, **kwargs)` | Convert native to HF format |
| `from_hf(hf_state_dict, device_mesh, **kwargs)` | Convert HF to native format |
| `convert_single_tensor_to_hf(fqn, tensor, **kwargs)` | Convert one tensor (may produce multiple) |

### `CheckpointAddon` (Protocol) — `addons.py:30`

| Method | Description |
|--------|-------------|
| `pre_save(**kwargs)` | Called before DCP save |
| `post_save(**kwargs)` | Called after DCP save |

### `ConsolidatedHFAddon` — `addons.py:40`

| Method | Line | Description |
|--------|------|-------------|
| `pre_save(**kwargs)` | 48 | Rank 0: save config.json, generation_config.json, tokenizer, custom code |
| `post_save(**kwargs)` | 87 | Move HF metadata from temp to consolidated dir |

### `PeftAddon` — `addons.py:118`

| Method | Line | Description |
|--------|------|-------------|
| `pre_save(**kwargs)` | 126 | Rank 0: save adapter_config.json, automodel_peft_config.json, tokenizer |
| `post_save(**kwargs)` | 158 | No-op |

### `_HuggingFaceStorageWriter` — `_backports/hf_storage.py:67`

| Method | Line | Description |
|--------|------|-------------|
| `__init__(path, fqn_to_index_mapping, ...)` | 74 | Init with file index mapping, consolidation options |
| `prepare_global_plan(plans)` | 130 | Attach shard_index to storage_data |
| `write_data(plan, planner)` | 144 | Split items by file index, delegate to base |
| `finish(metadata, results)` | 173 | Consolidate if sharded, or write weight map |

### `_HuggingFaceStorageReader` — `_backports/hf_storage.py:224`

| Method | Line | Description |
|--------|------|-------------|
| `__init__(path, token, key_mapping)` | 231 | Init with optional key remapping |
| `read_data(plan, planner)` | 251 | Read tensor data from SafeTensors files |
| `read_metadata()` | 286 | Build DCP Metadata from SafeTensors headers |

### `DefaultSavePlanner` — `_backports/default_planner.py:83`
Extended save planner with plan caching support.

### `DefaultLoadPlanner` — `_backports/default_planner.py:271`
Extended load planner with partial load and v2.3 compatibility.

### `_EmptyStateDictLoadPlanner` — `_backports/default_planner.py:383`
Load planner that rebuilds state dict from metadata (for offline conversion).

### `FileSystemWriter` / `FileSystemReader` — `_backports/filesystem.py`
Core file-system-based DCP storage implementations with SafeTensors streaming support.

### `FsspecWriter` / `FsspecReader` — `_backports/_fsspec_filesystem.py`
Remote-storage-capable writer/reader using fsspec.

## Free Functions

### checkpointing.py

| Function | Line | Description |
|----------|------|-------------|
| `get_safetensors_index_path(cache_dir, repo_id)` | 722 | Find model.safetensors.index.json in HF cache |
| `to_empty_parameters_only(model, device, recurse, dtype)` | 779 | Move params to device without storage copy |
| `save_config(config, weights_path)` | 798 | Save config dict as YAML |
| `_ensure_dirs(*dirs)` | 810 | Create dirs on all ranks with barrier |
| `_init_peft_adapters(model, peft_init_method)` | 824 | Initialize LoRA weights |
| `_apply(module, fn, recurse)` | 840 | Apply fn to parameters only (skip buffers) |
| `_load_full_state_dict_into_model(model_parts, state_dict)` | 934 | Load full state dict with broadcast_from_rank0 |
| `_convert_checkpoint_with_transformers(model, model_path, key_mapping)` | 964 | Convert checkpoint using transformers WeightConverter |
| `_maybe_adapt_state_dict_to_hf(model_part, state_dict, ...)` | 1077 | Apply state_dict_adapter.to_hf() if available |
| `_maybe_adapt_state_dict_from_hf(model_part, state_dict, moe_mesh)` | 1112 | Apply state_dict_adapter.from_hf() if available |
| `_equally_divide_layers(num_shards, keys)` | 1089 | Distribute keys evenly across shard indices |
| `_is_geq_torch_2_9()` | 59 | Version check |

### stateful_wrappers.py

| Function | Line | Description |
|----------|------|-------------|
| `_is_quantized_param(param)` | 32 | Check for BitsAndBytes quant_state |
| `_has_quantized_params(model)` | 41 | Any quantized params in model? |
| `_get_peft_state_dict(model)` | 46 | Collect trainable params directly |
| `_drop_outer_prefix(sd, prefix)` | 60 | Remove prefix from keys in-place |
| `_add_outer_prefix(sd, prefix, skip_keys)` | 69 | Add prefix to keys in-place |
| `_rename_dora_keys_to_hf(sd)` | 78 | Rename `.lora_magnitude` -> `.lora_magnitude_vector.default.weight` |
| `_rename_dora_keys_from_hf(sd)` | 87 | Reverse DoRA key rename |
| `_get_lm_head_weight_and_name(model)` | 97 | Find lm_head parameter name and tensor |

### addons.py

| Function | Line | Description |
|----------|------|-------------|
| `_get_hf_peft_config(peft_config, model_state)` | 162 | Build minimal HF PEFT config dict |
| `_get_automodel_peft_metadata(peft_config)` | 211 | Build automodel PEFT metadata dict |
| `_extract_target_modules(model)` | 226 | Find LoRA target module names |
| `_maybe_save_custom_model_code(original_model_path, hf_metadata_dir)` | 251 | Copy custom .py files from original model |

### conversion_mapping.py

| Function | Line | Description |
|----------|------|-------------|
| `requires_tensor_merging(model_type)` | 84 | Check if model needs expert tensor merging |
| `get_checkpoint_conversion_mapping(model_type)` | 101 | Get conversion rules from transformers |
| `get_model_conversion_mapping(model, ...)` | 126 | Get all conversion rules for a model |
| `get_combined_key_mapping(model_type, model_key_mapping)` | 168 | Get simple regex key renames only |
| `is_transformers_conversion_available()` | 221 | Check if transformers has conversion_mapping |

### consolidate_hf_safetensors.py

| Function | Line | Description |
|----------|------|-------------|
| `consolidate_safetensors_files(input_dir, output_dir, fqn_to_index_mapping, ...)` | 699 | Single-process consolidation |
| `consolidate_safetensors_files_on_every_rank(...)` | 752 | Multi-rank distributed consolidation |
| `_consolidate_safetensors_files(...)` | 620 | Core consolidation pipeline |
| `_parse_input_metadata(input_files_data, output_files_data)` | 96 | Determine full tensor shapes from shards |
| `_write_metadata(output_files_data)` | 164 | Write SafeTensors headers |
| `_write_data(input_files_data, output_files_data, num_threads)` | 314 | Write tensor data (optional multi-thread) |
| `_process_output_file(output_file, output_data, input_files_data)` | 245 | Process one output file |
| `_read_tensor_data_mmap(file_path, start_offset, end_offset, metadata_size)` | 219 | Memory-mapped tensor read |
| `_write_sub_tensor_to_file_optimized(...)` | 359 | Byte-level sub-tensor reassembly |
| `_calculate_max_contiguous_elements(indices, sub_tensor_shape, tensor_shape)` | 437 | Contiguity calculation for writes |
| `_write_overall_metadata_file(output_dir, output_files_data)` | 505 | Write model.safetensors.index.json |
| `_write_overall_metadata_file_from_shards(input_dir, output_dir, fqn_to_index_mapping)` | 535 | Write index.json from input shard metadata |

### hf_storage.py

| Function | Line | Description |
|----------|------|-------------|
| `get_fqn_to_file_index_mapping(reference_model_path, key_mapping)` | 400 | Build FQN-to-shard-index mapping from reference checkpoint |
| `_extract_file_index(filename)` | 361 | Parse shard index from SafeTensors filename |
| `_get_key_renaming_mapping(key, key_mapping)` | 427 | Apply regex key remapping |

### hf_utils.py

| Function | Line | Description |
|----------|------|-------------|
| `_gen_file_name(index, largest_index, shard_index)` | 72 | Generate SafeTensors filename |
| `_get_safetensors_file_metadata(file_bytes)` | 84 | Parse SafeTensors header JSON |
| `_get_dtype(dtype_str)` | 98 | Convert SafeTensors dtype string to torch.dtype |
| `_get_dcp_custom_metadata(metadata)` | 107 | Extract DCP_SHARDING_INFO from SafeTensors metadata |

### _torch_backports.py

| Function | Line | Description |
|----------|------|-------------|
| `apply_patches()` | 26 | Add `SavePlanner._cached_metadata` for old torch |
| `apply_async_checkpoint_patch()` | 58 | Serialize async process executor creation for torch >= 2.9 |

## Constants

### hf_utils.py
- `_metadata_fn = "model.safetensors.index.json"` (line 25)
- `FILE_NAME = "model-{cpt_idx}-of-{num_files}"` (line 27)
- `SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"` (line 28)
- `SUFFIX = ".safetensors"` (line 29)
- `CUSTOM_METADATA_KEY = "DCP_SHARDING_INFO"` (line 32)
- `DEFAULT_EXTRA_METADATA_KEY = "__metadata__"` (line 33)
- `SAVED_OFFSETS_KEY = "saved_offsets"` (line 34)
- `DTYPE_MAP` (line 40): Maps SafeTensors dtype strings to torch dtypes (F16, F32, BF16, F8_E4M3, etc.)
- `HF_DCP_VERSION = 1.0` (line 53)

### conversion_mapping.py
- `MODELS_REQUIRING_TENSOR_MERGING` (line 60): Set of model types needing expert tensor merging: mixtral, minimax, phimoe, qwen2_moe, qwen3_moe, deepseek_v2, deepseek_v3, jamba, olmoe, lfm2_moe, dots1, ernie4_5_moe, glm4_moe, glm4v_moe, longcat_flash, qwen3_omni_moe, qwen3_next, qwen3_vl_moe, hunyuan_v1_moe, flex_olmo.

### stateful_wrappers.py
- `_PREFIX = "model."` (line 29): Outer prefix for non-PEFT state dict keys.

## Directory Layout of a Saved Checkpoint

```
weights_path/
  config.yaml                              # Training config
  model/
    shard-00000-model-00001-of-00008.safetensors  # Per-rank shards (DCP)
    shard-00001-model-00001-of-00008.safetensors
    ...
    .metadata                              # DCP metadata (pickle)
    consolidated/                          # HF-compatible (optional)
      model-00001-of-00008.safetensors
      model-00002-of-00008.safetensors
      ...
      model.safetensors.index.json
      config.json
      generation_config.json
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
  optim/
    __0_0.distcp                           # DCP optimizer shards
    .metadata
```

For PEFT models:
```
weights_path/
  model/
    adapter_model.safetensors              # Single file, all adapter weights
    adapter_config.json                    # HF PEFT config
    automodel_peft_config.json             # AutoModel PEFT metadata
    tokenizer.json
    tokenizer_config.json
```
