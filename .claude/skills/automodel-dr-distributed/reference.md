# Distributed Module Reference

Complete file inventory and API reference for `nemo_automodel/components/distributed/`.

## File Inventory (18 files, ~4816 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 14 | Package marker (empty beyond license) |
| `init_utils.py` | 164 | Distributed initialization: `initialize_distributed()`, `DistInfo`, rank helpers |
| `fsdp2.py` | 318 | `FSDP2Manager` dataclass: mesh construction, HSDP, EP mesh, parallelization entry point |
| `megatron_fsdp.py` | 266 | `MegatronFSDPManager` dataclass: Megatron-style FSDP with ZeRO, FP8, NCCL UB support |
| `ddp.py` | 126 | `DDPManager` dataclass: classic DDP wrapper with activation checkpointing |
| `parallelizer.py` | 1180 | Core parallelization engine: strategy pattern, TP plan resolution, FSDP sharding, HF TP plan translation |
| `parallelizer_utils.py` | 136 | `fully_shard_by_dtype()`, `iter_maximal_uniform_dtype_subtrees()` for mixed-dtype FSDP |
| `optimized_tp_plans.py` | 316 | Model-specific TP plans: Llama, Qwen, Gemma3, Phi3 + `PARALLELIZE_FUNCTIONS` registry |
| `parallel_styles.py` | 115 | LoRA-aware TP styles: `ColwiseParallelLora`, `RowwiseParallelLora`, `SequenceParallelLora` |
| `cp_utils.py` | 338 | Context parallelism: `make_cp_batch_and_ctx()`, `create_context_parallel_ctx()`, TE THD support |
| `thd_utils.py` | 242 | THD format conversion: `process_input_for_thd()`, `split_batch_into_thd_chunks()` |
| `utils.py` | 242 | Training utilities: `reduce_loss()`, `get_sync_ctx()`, `FirstRankPerNode`, `barrier_and_log()` |
| `grad_utils.py` | 113 | Gradient utilities: `get_grad_norm()`, `clip_grad_by_total_norm_()` |
| `tensor_utils.py` | 108 | DTensor/Tensor utilities: `to_cpu()`, `to_local_if_dtensor()`, `get_cpu_state_dict()` |
| `pipelining/__init__.py` | 17 | Exports `AutoPipeline` |
| `pipelining/autopipeline.py` | 260 | `AutoPipeline` class: PP orchestrator with debug utilities |
| `pipelining/functional.py` | 597 | PP core: `pipeline_model()`, `split_model_into_stages()`, `build_pipeline_schedule()`, `stage_ids_this_rank()` |
| `pipelining/hf_utils.py` | 279 | HF model patching for PP: `patch_hf_model_for_pp()`, `create_pipeline_forward_inner()`, `validate_hf_model_for_pipeline_support()` |

## Public API Reference

### init_utils.py

- **`initialize_distributed(backend: str, timeout_minutes: int = 1) -> DistInfo`**: Initialize torch.distributed, set CUDA device, return `DistInfo`. Handles single-GPU case with gloo+HashStore.
- **`DistInfo`**: Dataclass with `backend`, `rank`, `world_size`, `device`, `is_main`.
- **`get_rank_safe() -> int`**: Returns rank from torch.distributed or `RANK` env var.
- **`get_world_size_safe() -> int`**: Returns world size from torch.distributed or `WORLD_SIZE` env var.
- **`get_local_rank_preinit() -> int`**: Returns `LOCAL_RANK` env var (pre-init safe).
- **`get_local_world_size_preinit() -> int`**: Returns `LOCAL_WORLD_SIZE` env var.
- **`destroy_global_state()`**: Registered at exit to destroy process group (ignores SIGINT during cleanup).

### fsdp2.py

- **`FSDP2Manager`**: Dataclass managing 5D mesh `(pp, dp_replicate, dp_shard, cp, tp)` and MoE mesh.
  - **`__post_init__()`**: Calls `_setup_distributed()` if world_size > 1.
  - **`_setup_distributed()`**: Infers sizes, builds meshes, creates flattened submeshes.
  - **`_get_device_mesh() -> DeviceMesh`**: Constructs 5D mesh and flattened `dp`, `dp_shard_cp`, `dp_cp` submeshes.
  - **`_get_moe_mesh() -> DeviceMesh`**: Constructs 3D `(pp, ep_shard, ep)` mesh for MoE models.
  - **`parallelize(model) -> nn.Module`**: Resolves TP plan, calls `fsdp2_strategy_parallelize()`.

### megatron_fsdp.py

- **`MegatronFSDPManager`**: Dataclass managing 3D mesh `(dp, cp, tp)` for MegatronFSDP.
  - **`_setup_distributed()`**: Builds mesh, flattens `dp_cp` if CP > 1.
  - **`parallelize(model, optimizer=None) -> tuple[nn.Module, optimizer]`**: Applies TP + MegatronFSDP wrapping. Returns both model and optimizer (MegatronFSDP modifies optimizer).

### ddp.py

- **`DDPManager`**: Dataclass for classic DDP.
  - **`parallelize(model) -> DDP`**: Wraps model with `DistributedDataParallel`. Supports activation checkpointing on `mlp`, `self_attn`, `input_layernorm`, `post_attention_layernorm`.

### parallelizer.py

- **`ParallelizationStrategy`** (ABC): Abstract base with `parallelize()` method.
- **`DefaultParallelizationStrategy`**: Standard flow: TP -> activation checkpointing -> recursive FSDP -> root FSDP.
- **`NemotronHParallelizationStrategy`**: NemotronH-specific: MLP-only TP, per-block-type checkpointing, `fully_shard_by_dtype`.
- **`WanParallelizationStrategy`**: Wan diffusion model: condition embedder TP, FFN TP, proj_out TP, then FSDP.
- **`PARALLELIZATION_STRATEGIES`**: Dict mapping model class names to strategy instances.
- **`get_parallelization_strategy(model) -> ParallelizationStrategy`**: Looks up by `type(model).__name__`.
- **`register_parallel_strategy(name=...) -> decorator`**: Registers out-of-tree strategies.
- **`apply_fsdp2_sharding_recursively(module, mesh, mp_policy, offload_policy)`**: Walks tree, applies `fully_shard()` per layer. Last layer: `reshard_after_forward=False`.
- **`get_hf_tp_shard_plan(model) -> dict`**: Extracts HF `_tp_plan` from model class/instance, translates string styles to DTensor parallel styles. Handles VLM model prefixes.
- **`translate_to_torch_parallel_style(style: str)`**: Converts string ("colwise", "rowwise", "colwise_rep", "rowwise_rep", "sequence_parallel") to DTensor parallel style objects.
- **`validate_tp_mesh(model, tp_mesh)`**: Asserts `num_attention_heads % tp_size == 0` and `num_key_value_heads % tp_size == 0`. Handles VLM model config lookups.
- **`_get_parallel_plan(model, sequence_parallel, tp_shard_plan, use_hf_tp_plan) -> dict`**: 4-level priority TP plan resolution.
- **`_extract_model_layers(model) -> List[nn.Module]`**: Extracts transformer layers from various architectures. Uses `VLM_MODEL_CLS_TO_LAYERS`, `LLM_MODEL_CLS_TO_LAYERS`, and `_find_largest_module_list()` as fallback.
- **`fsdp2_strategy_parallelize(model, device_mesh, ...)`**: Entry point delegating to the appropriate `ParallelizationStrategy`.
- **`megatron_fsdp_strategy_parallelize(model, device_mesh, optimizer, ...)`**: Applies TP then wraps with `megatron_fsdp.fully_shard()`.
- **`unshard_fsdp2_model(model) -> contextmanager`**: Explicitly unshards all FSDP2 modules. Useful for logprob inference.
- **`import_class_from_path(name: str)`**: Imports a class from a dotted path string.

### parallelizer_utils.py

- **`iter_maximal_uniform_dtype_subtrees(module, include_buffers, tensor_pred, return_paths) -> Iterator`**: Yields maximal submodules whose entire subtree has a unified dtype.
- **`fully_shard_by_dtype(module, mesh, mp_policy, offload_policy)`**: Handles mixed-dtype modules by FSDP-sharding dtype-uniform subtrees separately.
- **`_fully_shard(module, mesh, mp_policy, offload_policy)`**: Handles ModuleList by applying fully_shard to each element.

### optimized_tp_plans.py

- **`_parallelize_llama(model, sequence_parallel) -> dict`**: Llama TP plan with optional SP. Includes fused `qkv_proj` and `gate_up_proj`.
- **`_parallelize_qwen(model, sequence_parallel) -> dict`**: Qwen2/Qwen3 TP plan with `Qwen3QKNorm` SP support.
- **`_parallelize_qwen_classification(model, sequence_parallel) -> dict`**: Qwen3 classification variant (no lm_head, has score layer).
- **`_parallelize_gemma3(model, sequence_parallel) -> dict`**: Gemma3 TP plan. Handles both `Gemma3ForCausalLM` and `Gemma3ForConditionalGeneration` prefixes.
- **`_parallelize_phi3(model, sequence_parallel) -> dict`**: Phi3 plan. Fused attention cannot be sharded; only MLP is TP'd.
- **`SequenceParallelAllGatherActivation`**: SP variant that all-gathers activations in `_prepare_output_fn`.
- **`RotaryEmbedParallel`**: SP variant for rotary embeddings that handles tuple inputs (position_ids as `Replicate`).
- **`PARALLELIZE_FUNCTIONS`**: Registry dict mapping model class -> plan function.

### parallel_styles.py

- **`ColwiseParallelLora`**: Shards all LoRA params with `Shard(0)`, adds hook on `lora_A` output to all-gather.
- **`RowwiseParallelLora`**: Shards base weight `Shard(1)`, LoRA A/B with `Shard(1)`, magnitude replicated.
- **`SequenceParallelLora`**: Replicates LayerNorm/RMSNorm params via `DTensor.from_local(..., [Replicate()])`.
- **`translate_to_lora(plan) -> plan`**: Converts parallel style class in-place via `CLS_MAP`.

### cp_utils.py

- **`make_cp_batch_and_ctx(device_mesh, batch, loss_mask, use_te, ...) -> (context_mgr, batch)`**: Main entry point. Returns nullcontext if CP disabled. For TE: calls `make_cp_batch_for_te()`. For standard: creates CP context via `create_context_parallel_ctx()`.
- **`create_context_parallel_ctx(cp_mesh, cp_buffers, cp_seq_dims, cp_no_restore_buffers, cp_rotate_method)`**: Wraps `torch.distributed.tensor.experimental.context_parallel()`.
- **`get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_context)`**: Stacks optional context managers: `loss_parallel()`, `compiled_autograd`, SDPA kernel selection, CP context.
- **`make_cp_batch_for_te(cp_mesh, batch, qkv_format, ...)`**: Converts batch to THD format and shards for TE. Supports multi-chunk processing.
- **`_shard_thd_chunk_for_te(batch, cp_mesh, ...)`**: Uses `transformer_engine_torch.thd_get_partitioned_indices()` to select indices for this CP rank.

### thd_utils.py

- **`process_input_for_thd(batch, seq_lens_padding_value, padding_token_id) -> dict`**: Converts BSHD batch to THD format. Collapses batch dim, computes `cu_seqlens` from filtered `seq_lens_padded`.
- **`split_batch_into_thd_chunks(batch, num_chunks, ...) -> dict`**: Splits batch into chunks, processes each with `process_input_for_thd()`, pads `cu_seqlens` to uniform length, stacks results.

### utils.py

- **`reduce_loss(loss_store, total_num_tokens, per_token_loss, dp_group) -> (loss, denominator)`**: All-reduces loss across DP group. Supports per-token and per-sample normalization.
- **`get_sync_ctx(model, is_optim_step, defer_fsdp_grad_sync) -> context_manager`**: Controls gradient sync for FSDP2 and DDP models during gradient accumulation.
- **`FirstRankPerNode`**: Context manager ensuring rank 0 processes protected code first. Uses Gloo barriers with timeout for safety.
- **`barrier_and_log(string)`**: Distributed barrier + timestamped log on rank 0.
- **`_barrier_with_timeout(timeout, group) -> bool`**: Gloo-based monitored barrier.

### grad_utils.py

- **`get_grad_norm(parameters, dp_cp_group, tp_group, norm_type, dtype) -> float`**: Computes gradient norm across DP+CP and TP groups. Supports L2 and inf norms.
- **`clip_grad_by_total_norm_(parameters, max_grad_norm, total_norm, dtype)`**: In-place gradient clipping using pre-computed total norm.

### tensor_utils.py

- **`to_local_if_dtensor(tensor) -> torch.Tensor`**: Returns local shard if DTensor, otherwise passes through.
- **`to_cpu(v) -> Tensor`**: Moves DTensor (via `full_tensor().cpu()`) or Tensor to CPU.
- **`get_cpu_state_dict(state_generator, pin_memory) -> dict`**: Copies state dict to CPU with pinned memory support and non-blocking transfers.

### pipelining/autopipeline.py

- **`AutoPipeline`**: PP orchestrator.
  - **`__init__(world_mesh, pp_schedule, pp_microbatch_size, pp_batch_size, ...)`**: Stores config, creates pp_mesh.
  - **`build(model, loss_fn, parallelize_fn) -> self`**: Validates model, splits into stages, builds schedule.
  - **`info -> PipelineInfo`**: Access pipeline state.
  - **`parts -> list[nn.Module]`**: Access model parts.
  - **`list_stage_modules() -> list[list[str]]`**: Debug: module names per stage.
  - **`pretty_print_stages() -> str`**: Debug: formatted stage summary.
  - **`debug_summary() -> str`**: Debug: PP degree, schedule type, param counts.
  - **`visualize_current_schedule(filename)`**: Uses PyTorch's schedule visualizer.
- **`PipelineInfo`**: Dataclass with `enabled`, `schedule`, `has_first_stage`, `has_last_stage`, `model_parts`, `stages`.

### pipelining/functional.py

- **`pipeline_model(model, world_mesh, ...) -> (schedule, model_parts, has_first, has_last, stages)`**: Main PP entry point. Splits model, applies parallelization, builds schedule.
- **`split_model_into_stages(model, pp_mesh, ...) -> (stages, models)`**: Deep-copies model per stage, removes unneeded modules, creates `PipelineStage` objects.
- **`generate_hf_model_fqn_per_model_part(num_stages, num_layers, ...) -> list[list[str]]`**: Auto-generates module FQN lists. Includes multimodal encoders on first stage.
- **`calculate_virtual_stages(num_layers, layers_per_stage, pp_size, ...) -> (num_virtual_stages, stages_per_rank)`**: Computes virtual stage count with optional rounding.
- **`build_pipeline_schedule(schedule_csv, schedule_name, ...) -> _PipelineSchedule`**: Creates schedule from class name or CSV.
- **`stage_ids_this_rank(pp_rank, pp_size, num_stages, style) -> tuple[int]`**: Returns stage indices for "loop" or "v" styles.
- **`scale_grads_by_divisor(stages, divisor)`**: Scales gradients across pipeline stages.
- **`ParallelizeFnProtocol`**: Protocol for the parallelization callback passed to `pipeline_model`.

### pipelining/hf_utils.py

- **`patch_hf_model_for_pp(model, patch_inner_model, patch_causal_lm_model)`**: Replaces forward methods with PP-compatible versions.
- **`create_pipeline_forward_inner(model_class_name) -> Callable`**: Creates forward for inner model (handles embeddings, rotary, decoder layers, norm). Returns hidden states directly for pipeline stages.
- **`create_pipeline_forward_causal_lm() -> Callable`**: Creates forward for CausalLM wrapper (delegates to inner model, applies lm_head if present).
- **`validate_hf_model_for_pipeline_support(model)`**: Checks `tie_word_embeddings=False` and `is_encoder_decoder=False`.
- **`get_text_module(model) -> nn.Module`**: Finds nested text/LLM module by checking `language_model`, `text_model`, `text_decoder` attributes.
- **`init_hf_model_buffers(model, device)`**: Initializes rotary embedding buffers on device.
- **`TEXT_MODULE_ATTRS`**: Tuple of attribute names to check for nested text models.
- **`MULTIMODAL_SUFFIXES`**: Tuple of multimodal encoder/projector attribute names to include in first PP stage.

## Device Mesh Topology

### FSDP2Manager 5D Mesh

```
Shape: (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
Names: ("pp", "dp_replicate", "dp_shard", "cp", "tp")

Flattened submeshes:
  "dp"          = flatten("dp_replicate", "dp_shard")
  "dp_shard_cp" = flatten("dp_shard", "cp")
  "dp_cp"       = flatten("dp_replicate", "dp_shard", "cp")
```

### MegatronFSDPManager 3D Mesh

```
Shape: (dp_size, cp_size, tp_size)
Names: ("dp", "cp", "tp")

Flattened submesh (if cp > 1):
  "dp_cp" = flatten("dp", "cp")
```

### MoE Mesh (FSDP2Manager)

```
Shape: (pp_size, ep_shard_size, ep_size)
Names: ("pp", "ep_shard", "ep")

Where: ep_shard_size = (dp_size * cp_size) / ep_size
```

## Supported Model Architectures

### Optimized TP Plans (optimized_tp_plans.py)

- `LlamaForCausalLM` (HF + Custom)
- `Qwen2ForCausalLM` (HF + Custom)
- `Qwen3ForCausalLM`
- `Qwen3ForSequenceClassification`
- `Gemma3ForCausalLM`
- `Gemma3ForConditionalGeneration`
- `Phi3ForCausalLM`

### Specialized Strategies (parallelizer.py)

- `NemotronHForCausalLM`: Hybrid Mamba-Transformer with per-block-type handling
- `WanTransformer3DModel`: Diffusion model with condition embedder TP

### VLM Layer Extraction (parallelizer.py)

- `Gemma3ForConditionalGeneration`
- `Qwen2VLForConditionalGeneration`
- `Qwen2_5_VLForConditionalGeneration`
- `SmolVLMForConditionalGeneration`
- `LlavaForConditionalGeneration`
- `LlavaNextForConditionalGeneration`
- `LlavaNextVideoForConditionalGeneration`
- `LlavaOnevisionForConditionalGeneration`
- `Mistral3ForConditionalGeneration`
- `Llama4ForConditionalGeneration`

### HF TP Plan Support (parallelizer.py)

Any model with a `_tp_plan` attribute on the class or instance is supported via `get_hf_tp_shard_plan()`. HF MoE styles (`ep_router`, `local_colwise`, `local_rowwise`, `gather`) are intentionally skipped.
