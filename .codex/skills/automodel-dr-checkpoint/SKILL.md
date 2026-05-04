---
name: automodel-dr-checkpoint
description: Use when working with the checkpoint module of automodel — distributed checkpointing with SafeTensors, DCP, mesh-aware resharding, and HuggingFace format conversion
---

# Checkpoint Module Deep Read

## 1. Module Purpose & Capabilities

The checkpoint module (`nemo_automodel/components/checkpoint/`) provides a complete distributed checkpointing system built on PyTorch Distributed Checkpoint (DCP) with native SafeTensors support. It handles:

- **Saving and loading model weights** in sharded SafeTensors format across arbitrary GPU topologies via `Checkpointer` in `checkpointing.py`.
- **Consolidating sharded checkpoints** into HuggingFace-compatible consolidated files (with `model.safetensors.index.json`) via `consolidate_safetensors_files_on_every_rank()` in `_backports/consolidate_hf_safetensors.py`.
- **PEFT/LoRA adapter checkpointing** with rank-0-only save/load and HF-compatible `adapter_config.json` via `PeftAddon` in `addons.py`.
- **State dict format conversion** between custom model implementations and HuggingFace format via `StateDictAdapter` in `state_dict_adapter.py` and the conversion mapping system in `conversion_mapping.py`.
- **Async checkpointing** for torch >= 2.9.0, using a process-based async executor with staging and upload phases.
- **Optimizer and scheduler state** save/load via `OptimizerState` in `stateful_wrappers.py`.
- **MoE tensor merging** during checkpoint loading for models like Mixtral, DeepSeek-V3, and Qwen3 MoE variants, using transformers' `WeightConverter` system.
- **DP-rank-aware auxiliary state** save/load for dataloader and RNG state via `save_on_dp_ranks()`/`load_on_dp_ranks()`.

## 2. Core Design Logic

### Why SafeTensors + DCP Hybrid

The module does not use DCP's default serialization. Instead, it combines DCP's coordination and sharding logic with SafeTensors as the on-disk format. This is achieved through custom storage reader/writer classes (`_HuggingFaceStorageWriter`, `_HuggingFaceStorageReader` in `_backports/hf_storage.py`) that wrap DCP's `FsspecWriter`/`FsspecReader` and emit SafeTensors files with DCP sharding metadata embedded in the `__metadata__` section under the `DCP_SHARDING_INFO` key.

**Why this design**: Checkpoints are always HuggingFace-loadable. You can take a training checkpoint and directly use it with `transformers.AutoModel.from_pretrained()` without any conversion step. The DCP sharding metadata embedded inside each SafeTensors file allows resharding back to different topologies.

### Two-Phase Save: Sharded + Consolidated

Saving writes two representations:

1. **Sharded SafeTensors** (`weights_path/model/`): Each rank writes its own shard file (e.g., `shard-00000-model-00001-of-00008.safetensors`). These contain DCP sharding metadata and can be loaded back by DCP for resuming training on any topology.

2. **Consolidated SafeTensors** (`weights_path/model/consolidated/`): Full HF-compatible files (e.g., `model-00001-of-00008.safetensors`) plus `model.safetensors.index.json`, `config.json`, `generation_config.json`, and tokenizer artifacts. Created by `consolidate_safetensors_files_on_every_rank()` which distributes consolidation work across all ranks.

**Why two representations**: Sharded files support efficient resume with arbitrary topology changes. Consolidated files support direct inference loading with HF and other tools that expect standard HF checkpoint layout.

### Addon Architecture for Extensibility

The `Checkpointer` uses a `CheckpointAddon` protocol (`addons.py:30`) with `pre_save()` and `post_save()` hooks. Two concrete addons are registered:

- `ConsolidatedHFAddon`: Writes HF metadata artifacts (config, tokenizer, generation config) on rank 0 before the main DCP save, then moves them to the consolidated directory after save.
- `PeftAddon`: Writes `adapter_config.json` and `automodel_peft_config.json` for LoRA/DoRA adapters.

This allows adding new checkpoint artifacts without modifying the core save flow.

### PEFT as a Special Case

PEFT models bypass the DCP machinery entirely for model saves/loads. Instead:
- **Save**: Rank 0 calls `safetensors.torch.save_file()` to write a single `adapter_model.safetensors`.
- **Load**: Rank 0 calls `safetensors.torch.load_file()` then broadcasts via `StateDictOptions(broadcast_from_rank0=True)`.
- Quantized PEFT models (QLoRA with BitsAndBytes) additionally bypass `get_model_state_dict()` because it fails on `Params4bit` types; `_get_peft_state_dict()` in `stateful_wrappers.py:46` directly iterates `named_parameters()`.

### State Dict Adapter Pattern

Custom model implementations (DeepSeek-V3, Llama optimized, etc.) register a `state_dict_adapter` attribute on their model class. This is an instance of a `StateDictAdapter` subclass (`state_dict_adapter.py:22`) with `to_hf()`, `from_hf()`, and `convert_single_tensor_to_hf()` methods. The `Checkpointer` automatically detects and uses this adapter at both save time (`_maybe_adapt_state_dict_to_hf()` in `checkpointing.py:1077`) and load time (`_maybe_adapt_state_dict_from_hf()` in `checkpointing.py:1112`).

### MoE Tensor Merging via Transformers

Models listed in `MODELS_REQUIRING_TENSOR_MERGING` (`conversion_mapping.py:60`) store individual expert weights in their HF checkpoints but use grouped 3D tensors at runtime. During init-step loading, the module checks if the model type requires merging and has no custom `state_dict_adapter`. If so, `_convert_checkpoint_with_transformers()` (`checkpointing.py:964`) loads the checkpoint, applies transformers' `WeightConverter` operations to merge experts, then loads the converted state dict via `_load_full_state_dict_into_model()` with `full_state_dict=True` to properly handle FSDP DTensor distribution.

### Backports Layer

The `_backports/` package contains code back-ported from upstream PyTorch to support features not yet available in the pinned torch version. The `__init__.py` conditionally applies monkey-patches:
- For torch <= 2.7.1: patches `SavePlanner._cached_metadata` attribute.
- For torch >= 2.9.0: patches the async checkpoint process executor with a creation lock to prevent race conditions.

These patches are intended to be removed as the project moves to newer PyTorch versions.

## 3. Core Data Structures

### `Checkpointer` (checkpointing.py:130)
Central orchestrator. Holds `CheckpointingConfig`, parallelism rank info (`dp_rank`, `tp_rank`, `pp_rank`), MoE mesh, async contexts, and registered addons. All save/load operations go through this class.

### `CheckpointingConfig` (checkpointing.py:88)
Dataclass configuring the checkpoint system. Key fields:
- `model_save_format`: `SerializationFormat` enum (`SAFETENSORS` or `TORCH_SAVE`)
- `save_consolidated`: Whether to produce HF-compatible consolidated output
- `is_peft`: Triggers PEFT-specific save/load path
- `is_async`: Enable async DCP save (torch >= 2.9.0 only)
- `single_rank_consolidation`: If True, only rank 0 consolidates (for remote storage)
- `staging_dir`: Optional local staging dir for consolidation when system temp is limited
- `dequantize_base_checkpoint`: Whether to dequantize during base model loading
- `skip_task_head_prefixes_for_base_model`: Prefixes to skip when loading base model

### `ModelState` (stateful_wrappers.py:106)
DCP-compliant `Stateful` wrapper for model state. Handles:
- Tied word embeddings detection and lm_head exclusion from state dict
- PEFT vs full model state dict extraction
- Quantized PEFT bypass
- DoRA key renaming (`lora_magnitude` to/from `lora_magnitude_vector.default.weight`)
- `base_model.model.` prefix management for HF PEFT compatibility
- Task head prefix skipping for base model loading

### `OptimizerState` (stateful_wrappers.py:253)
DCP-compliant `Stateful` wrapper for optimizer + scheduler state. Uses `get_optimizer_state_dict()` with `flatten_optimizer_state_dict=True`.

### `StateDictAdapter` (state_dict_adapter.py:22)
Abstract base class with three methods: `to_hf()`, `from_hf()`, `convert_single_tensor_to_hf()`. Custom model implementations subclass this to handle weight format differences (e.g., fused QKV projections, packed gate/up projections).

### `CheckpointAddon` (addons.py:30)
Protocol with `pre_save(**kwargs)` and `post_save(**kwargs)`. Concrete implementations: `ConsolidatedHFAddon` (addons.py:40) and `PeftAddon` (addons.py:118).

### `_HuggingFaceStorageWriter` (_backports/hf_storage.py:67)
Extends `FsspecWriter`. Handles sharded SafeTensors output with DCP metadata, file index mapping, and optional single-rank consolidation.

### `_HuggingFaceStorageReader` (_backports/hf_storage.py:224)
Extends `FsspecReader`. Reads HF SafeTensors checkpoints (with or without DCP sharding info), supports key remapping for VLMs.

### `_AsyncSaveContext` (checkpointing.py:72)
Dataclass holding async save state: stager (`DefaultStager`), process group, future (`AsyncSaveResponse`), and staging status flag. One instance per model save and one per optimizer save.

### `SerializationFormat` (_backports/filesystem.py:86)
Enum: `TORCH_SAVE` or `SAFETENSORS`.

### Consolidation Data Structures (_backports/consolidate_hf_safetensors.py)
- `_FqnData` (line 51): Per-tensor info (offset, shape, dtype) for output files
- `_OutputFileData` (line 69): Per-output-file metadata and tensor assignments
- `_InputFileData` (line 83): Per-input-file metadata cache

## 4. State Flow

### Full Model Save Flow

```
Checkpointer.save_model()
  |
  +--> ModelState(model).state_dict()
  |     |- get_model_state_dict() via PyTorch DCP
  |     |- Pop lm_head for tied embeddings
  |     |- Add "base_model.model." prefix for PEFT
  |
  +--> _maybe_adapt_state_dict_to_hf()
  |     |- If model has state_dict_adapter: adapter.to_hf(state_dict)
  |
  +--> _maybe_build_consolidated_index()
  |     |- Read base checkpoint index from HF cache
  |     |- Build fqn_to_file_index_mapping
  |
  +--> addon.pre_save() for each addon
  |     |- ConsolidatedHFAddon: write config.json, tokenizer, generation_config.json
  |     |- PeftAddon: write adapter_config.json, automodel_peft_config.json
  |
  +--> _do_save(state_dict, model_dir, storage_writer)
  |     |- PEFT path: rank 0 writes adapter_model.safetensors directly
  |     |- Async path: dcp.async_save() with DefaultStager
  |     |- Sync path: dcp.save() with _HuggingFaceStorageWriter
  |
  +--> addon.post_save() for each addon
  |     |- ConsolidatedHFAddon: move HF metadata to consolidated dir
  |
  +--> consolidate_safetensors_files_on_every_rank()  [if enabled & sync mode]
        |- Distribute output files across ranks (index % world_size)
        |- Each rank: parse shard metadata, write consolidated SafeTensors
        |- Rank 0: write model.safetensors.index.json
        |- barrier()
```

### Full Model Load Flow

```
Checkpointer.load_model()
  |
  +--> ModelState(model, is_init_step=..., is_peft=...)
  |
  +--> [If MoE with tensor merging and no adapter]:
  |     _convert_checkpoint_with_transformers()
  |       |- Load all SafeTensors files
  |       |- Apply WeightConverter/WeightRenaming from transformers
  |       |- _load_full_state_dict_into_model() with broadcast_from_rank0
  |       |- RETURN
  |
  +--> model_state.state_dict()  [get empty sharded structure]
  |
  +--> _maybe_adapt_state_dict_to_hf()  [adapt keys for HF reader]
  |
  +--> _do_load(state_dict, path, storage_reader)
  |     |- PEFT non-init: rank 0 loads adapter_model.safetensors
  |     |- Otherwise: dcp.load() with _HuggingFaceStorageReader
  |
  +--> _maybe_adapt_state_dict_from_hf()  [convert back to native format]
  |
  +--> model_state.load_state_dict()
        |- set_model_state_dict() via PyTorch DCP
        |- Inject lm_head reference for tied embeddings
```

### Base Model Initialization Flow

```
Checkpointer.load_base_model()
  |
  +--> to_empty_parameters_only(model, device)
  |     |- Move params to device without copying storage (skip buffers)
  |
  +--> Reset _is_hf_initialized flags on all modules
  +--> model.initialize_weights()  [re-initialize from scratch]
  +--> _init_peft_adapters(model, peft_init_method)
  |
  +--> [If load_base_model=True]:
  |     get_combined_key_mapping()  [merge model + transformers key maps]
  |     load_model(model, model_path, is_init_step=True, key_mapping=...)
  |
  +--> model.tie_weights()  [if tied embeddings]
```

### Consolidation Flow (consolidate_safetensors_files_on_every_rank)

```
1. Each rank determines which output file indices it owns (idx % world_size)
2. Filter fqn_to_index_mapping to only this rank's indices
3. _consolidate_safetensors_files():
   a. Read metadata from all input shard files
   b. _parse_input_metadata(): determine full tensor shapes from shards
   c. _write_metadata(): write SafeTensors headers to output files
   d. _write_data(): copy tensor bytes from shards to correct positions
      - Uses memory-mapped reads (_read_tensor_data_mmap)
      - _write_sub_tensor_to_file_optimized() handles row-wise/column-wise reassembly
      - Optional multi-threading per output file
   e. Optional staging: write to temp dir, then copy to final location
4. Rank 0: write model.safetensors.index.json
5. barrier()
```

### Error Handling Patterns

- `FileNotFoundError` raised in `load_model()` if checkpoint path does not exist.
- `ValueError` raised in `CheckpointingConfig.__post_init__()` for unsupported save formats.
- `_convert_checkpoint_with_transformers()` catches all exceptions and returns `None`, falling back to standard loading.
- Consolidation validates DCP custom metadata presence with `ValueError`.
- Size mismatches during loading produce `ValueError` via `create_default_local_load_plan()`.
- Async mode silently disabled with error log for torch < 2.9.0.

## 5. Common Modification Scenarios

### Scenario 1: Adding a New Checkpoint Format

**Goal**: Support a new serialization format beyond `TORCH_SAVE` and `SAFETENSORS`.

**Steps**:
1. Add a new variant to `SerializationFormat` enum in `_backports/filesystem.py:86`.
2. Update `CheckpointingConfig.__post_init__()` in `checkpointing.py:114` -- the assertion already uses `SerializationFormat` values dynamically.
3. Update `_write_files_from_queue()` in `_backports/filesystem.py:380` to add a new branch alongside the `SAFETENSORS` and `TORCH_SAVE` branches for writing tensor data.
4. Create a new storage reader/writer class (analogous to `_HuggingFaceStorageWriter`/`_HuggingFaceStorageReader` in `_backports/hf_storage.py`).
5. Update `_get_storage_writer()` and `_get_storage_reader()` in `Checkpointer` (`checkpointing.py:650`, `checkpointing.py:678`) to return the new reader/writer based on the format.

### Scenario 2: Adding a New Checkpoint Addon

**Goal**: Write additional metadata alongside checkpoints (e.g., training metrics or evaluation results).

**Steps**:
1. Create a new class implementing the `CheckpointAddon` protocol from `addons.py:30`. Must define `pre_save(**kwargs)` and `post_save(**kwargs)`.
2. Register the addon in `Checkpointer.__init__()` at `checkpointing.py:177` by appending to `self._addons`. Use a condition similar to the existing `_should_write_hf_metadata()` check.
3. The kwargs passed to hooks include: `model_state`, `model_path`, `consolidated_path`, `hf_metadata_dir`, `tokenizer`, `peft_config`, `fqn_to_file_index_mapping`, `original_model_path`. Add new kwargs in `save_model()` at `checkpointing.py:234` if needed.
4. Use rank-0-only writes with a `torch.distributed.barrier()` at the end for consistency.

### Scenario 3: Adding a New Custom Model with State Dict Conversion

**Goal**: Add a custom model implementation that stores weights differently from HuggingFace format.

**Steps**:
1. Create a subclass of `StateDictAdapter` from `state_dict_adapter.py:22` with `to_hf()`, `from_hf()`, and `convert_single_tensor_to_hf()` implementations.
2. Attach an instance as the `state_dict_adapter` attribute on the model class. The `Checkpointer` detects this at `checkpointing.py:335` (`has_state_dict_adapter = hasattr(model_state.model[0], "state_dict_adapter")`).
3. The adapter's `to_hf()` is called during save at `checkpointing.py:1084`, and `from_hf()` during load at `checkpointing.py:1122`.
4. For MoE models with expert parallelism, `from_hf()` receives the `device_mesh` kwarg so it can load only the experts needed for the current rank.
5. No changes to `Checkpointer` are needed; the adapter is auto-detected.

### Scenario 4: Changing Consolidation Behavior

**Goal**: Modify how sharded checkpoints are consolidated (e.g., change sharding strategy or add compression).

**Key files and functions**:
- `consolidate_safetensors_files_on_every_rank()` in `_backports/consolidate_hf_safetensors.py:752` controls rank distribution.
- `_consolidate_safetensors_files()` at line 620 controls the core consolidation pipeline.
- `_write_sub_tensor_to_file_optimized()` at line 359 handles the byte-level reassembly.
- The `fqn_to_file_index_mapping` dict (built in `_maybe_build_consolidated_index()` at `checkpointing.py:588`) controls which tensors go to which output file.
- The `use_staging` parameter enables a two-phase write for remote storage systems.
- To add compression, wrap the output streams in `_write_metadata()` and `_process_output_file()`.

### Scenario 5: Supporting a New MoE Model Type for Tensor Merging

**Goal**: Add support for a new MoE architecture that requires expert weight merging during checkpoint loading.

**Steps**:
1. Add the model's `model_type` string to the `MODELS_REQUIRING_TENSOR_MERGING` set in `conversion_mapping.py:60`.
2. Ensure the model has a `_checkpoint_conversion_mapping` defined in the transformers library (HuggingFace side) with appropriate `WeightConverter` entries for merging individual expert weights.
3. If the model has a custom implementation in NeMo AutoModel, implement a `StateDictAdapter` instead (see Scenario 3). Models with a `state_dict_adapter` skip the transformers conversion path entirely (`checkpointing.py:338`).
4. Verify the model's `config.model_type` matches what you added to the set. The check is performed via `requires_tensor_merging()` at `conversion_mapping.py:84`.

### Scenario 6: Modifying Async Checkpoint Behavior

**Goal**: Change how async saves are managed (e.g., different stager, custom process group).

**Key locations**:
- `_AsyncSaveContext` dataclass at `checkpointing.py:72` holds per-save-operation state.
- The stager is initialized as `DefaultStager()` in `Checkpointer.__init__()` at line 172. Replace with a custom stager.
- Process groups created at line 174 use `gloo` backend. Change backend or group configuration there.
- `_do_save()` at line 529 calls `dcp.async_save()` with the stager and process group.
- `maybe_wait_for_staging()` at line 435 blocks on staging completion; `async_wait()` at line 446 blocks on upload completion.
- `close()` at line 489 ensures all async operations complete.
