---
name: automodel-dr-distributed
description: Use when working with the distributed module of automodel — FSDP2, MegatronFSDP, tensor/context/pipeline parallelism, device mesh management, and SPMD training strategies
---

# Distributed Module Skill Document

## 1. Module Purpose & Capabilities

The `nemo_automodel/components/distributed/` module (18 files, ~4816 lines) implements the entire distributed training infrastructure for NeMo AutoModel. It provides composable, DTensor-native parallelism strategies that allow the same training script to scale from 1 GPU to 1000+ GPUs purely through configuration changes.

### Core Capabilities

- **FSDP2 (Fully Sharded Data Parallelism v2)**: PyTorch-native parameter sharding via `FSDP2Manager` in `fsdp2.py`. Supports mixed precision, CPU offloading, HSDP (Hybrid Sharded Data Parallelism), activation checkpointing, and deferred gradient synchronization.

- **MegatronFSDP**: Hybrid approach combining Megatron-LM's overlapped communication with FSDP sharding via `MegatronFSDPManager` in `megatron_fsdp.py`. Supports ZeRO stages, FP8 transpose cache, NCCL user buffers, and double buffering.

- **Tensor Parallelism (TP)**: Shards individual weight matrices across GPUs using `ColwiseParallel` and `RowwiseParallel` placements. Model-specific plans are registered in `optimized_tp_plans.py` via `PARALLELIZE_FUNCTIONS`. The `parallelizer.py` module resolves plans through a 4-level priority system (custom dict > HF plan > optimized plan > default plan).

- **Sequence Parallelism (SP)**: Extends TP by sharding activations along the sequence dimension between TP-sharded layers. Implemented via `SequenceParallel`, `SequenceParallelAllGatherActivation`, and `RotaryEmbedParallel` in `optimized_tp_plans.py`.

- **Context Parallelism (CP)**: Splits input sequences across ranks for long-context training. Implemented in `cp_utils.py` via `create_context_parallel_ctx()` which wraps PyTorch's `torch.distributed.tensor.experimental.context_parallel`. Supports both standard SDPA and Transformer Engine THD format.

- **Pipeline Parallelism (PP)**: Splits model layers across pipeline stages via the `pipelining/` subpackage. `AutoPipeline` orchestrates stage splitting, schedule construction (1F1B, ZBV Zero Bubble, CSV-defined), and composability with TP/FSDP.

- **Expert Parallelism (EP)**: Supported through a separate MoE mesh (`_get_moe_mesh()` in `fsdp2.py`) with `ep_size` and `ep_shard_size` dimensions.

- **DDP**: Classic DistributedDataParallel wrapper via `DDPManager` in `ddp.py` for simpler use cases.

- **Device Mesh Management**: Multi-dimensional mesh construction with named axes (`pp`, `dp_replicate`, `dp_shard`, `cp`, `tp`) and flattened submeshes (`dp`, `dp_shard_cp`, `dp_cp`) for composing parallelism strategies.

- **LoRA-aware TP**: Custom parallel styles (`ColwiseParallelLora`, `RowwiseParallelLora`, `SequenceParallelLora`) in `parallel_styles.py` that correctly shard LoRA adapter weights during tensor parallelism.

## 2. Core Design Logic

### Why SPMD (Single Program, Multiple Data)

The module is designed so that a single Python script runs on every GPU. The parallelism strategy is entirely determined by the `DeviceMesh` configuration and the parallel plan, not by code branches. This means:

- **Configuration, not code changes**: Switching from 1-GPU to 8-GPU TP + FSDP requires only changing `tp_size` and `dp_size` in the YAML config. The same `FSDP2Manager.parallelize()` call handles both.

- **Composable parallelism**: The 5D mesh `(pp, dp_replicate, dp_shard, cp, tp)` in `FSDP2Manager._get_device_mesh()` allows any combination of parallelism dimensions. PP splits the model vertically, TP splits horizontally, FSDP shards parameters, CP splits sequences, and HSDP replicates across node boundaries.

- **DTensor-native**: All sharding is expressed through PyTorch's DTensor placements (`Shard`, `Replicate`). Parameters are never manually split; instead, `parallelize_module()` and `fully_shard()` handle redistribution automatically based on placement specs.

### Strategy Pattern for Model Diversity

The `parallelizer.py` uses a strategy pattern (`ParallelizationStrategy` ABC) to handle different model architectures. `PARALLELIZATION_STRATEGIES` maps model class names to specialized strategies (e.g., `NemotronHParallelizationStrategy`, `WanParallelizationStrategy`). Unknown models fall through to `DefaultParallelizationStrategy`. New strategies can be registered via the `@register_parallel_strategy(name="...")` decorator.

### TP Plan Resolution Priority

The `_get_parallel_plan()` function in `parallelizer.py` resolves tensor parallel plans with this priority:
1. Custom dict or import path (`tp_shard_plan` parameter)
2. HuggingFace model's `_tp_plan` attribute (when `use_hf_tp_plan=True`)
3. Optimized model-specific plans from `PARALLELIZE_FUNCTIONS` in `optimized_tp_plans.py`
4. A default base plan compatible with Llama-style architectures

### Recursive FSDP Sharding

`apply_fsdp2_sharding_recursively()` in `parallelizer.py` walks the module tree and applies `fully_shard()` to each transformer layer individually, with an optimization where the last layer does not reshard after forward (since FSDP will prefetch it immediately for backward).

### Mixed-Dtype FSDP Handling

`parallelizer_utils.py` provides `fully_shard_by_dtype()` which detects modules containing multiple parameter dtypes (common in MoE models with fp8 experts) and applies FSDP sharding to dtype-uniform subtrees separately before wrapping the parent.

## 3. Core Data Structures

### FSDP2Manager (`fsdp2.py`)

```
@dataclass
class FSDP2Manager:
    dp_size, dp_replicate_size, tp_size, cp_size, pp_size, ep_size: int
    sequence_parallel, use_hf_tp_plan, activation_checkpointing: bool
    custom_tp_plan: Optional[dict]
    mp_policy: MixedPrecisionPolicy
    offload_policy: Optional[CPUOffloadPolicy]
    defer_fsdp_grad_sync: bool
    device_mesh: DeviceMesh  # 5D: (pp, dp_replicate, dp_shard, cp, tp)
    moe_mesh: Optional[DeviceMesh]  # 3D: (pp, ep_shard, ep)
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/fsdp2.py`, lines 34-135

### MegatronFSDPManager (`megatron_fsdp.py`)

```
@dataclass
class MegatronFSDPManager:
    dp_size, tp_size, cp_size: int
    sequence_parallel, use_hf_tp_plan: bool
    megatron_fsdp_unit_modules: List[str]
    zero_dp_strategy: int  # 1, 2, or 3
    overlap_grad_reduce, overlap_param_gather: bool
    # ... 10+ MegatronFSDP-specific config fields
    device_mesh: DeviceMesh  # 3D: (dp, cp, tp)
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/megatron_fsdp.py`, lines 32-127

### DDPManager (`ddp.py`)

```
@dataclass
class DDPManager:
    backend: str
    world_size: int
    rank: int  # set in _setup_distributed
    activation_checkpointing: bool
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/ddp.py`, lines 31-48

### DistInfo (`init_utils.py`)

```
@dataclass
class DistInfo:
    backend: str
    rank: int
    world_size: int
    device: torch.device
    is_main: bool
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/init_utils.py`, lines 71-87

### PipelineInfo (`pipelining/autopipeline.py`)

```
@dataclass
class PipelineInfo:
    enabled: bool
    schedule: Optional[_PipelineSchedule]
    has_first_stage: bool
    has_last_stage: bool
    model_parts: Optional[list[nn.Module]]
    stages: Optional[list[PipelineStage]]
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/pipelining/autopipeline.py`, lines 37-43

### AutoPipeline (`pipelining/autopipeline.py`)

Orchestrator class storing world_mesh, pp_mesh, schedule config, stage splitting config, and patching flags. Created with mesh + schedule parameters, then `.build(model, loss_fn=..., parallelize_fn=...)` produces the `PipelineInfo`.
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/pipelining/autopipeline.py`, lines 46-260

### PARALLELIZE_FUNCTIONS (`optimized_tp_plans.py`)

```python
PARALLELIZE_FUNCTIONS: Dict[type, Callable[..., Dict[str, ParallelStyle]]] = {
    Qwen2ForCausalLM: _parallelize_qwen,
    Qwen3ForCausalLM: _parallelize_qwen,
    Qwen3ForSequenceClassification: _parallelize_qwen_classification,
    LlamaForCausalLM: _parallelize_llama,
    Gemma3ForCausalLM: _parallelize_gemma3,
    Gemma3ForConditionalGeneration: _parallelize_gemma3,
    Phi3ForCausalLM: _parallelize_phi3,
    CustomLlamaForCausalLM: _parallelize_llama,
    CustomQwen2ForCausalLM: _parallelize_qwen,
}
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/optimized_tp_plans.py`, lines 303-315

### PARALLELIZATION_STRATEGIES (`parallelizer.py`)

```python
PARALLELIZATION_STRATEGIES: Dict[str, ParallelizationStrategy] = {
    "NemotronHForCausalLM": NemotronHParallelizationStrategy(),
    "WanTransformer3DModel": WanParallelizationStrategy(),
}
```
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/parallelizer.py`, lines 380-383

### LoRA Parallel Styles (`parallel_styles.py`)

- `ColwiseParallelLora`: Extends `ColwiseParallel` to shard `lora_A.weight` and `lora_B.weight` with `Shard(0)` and adds a forward hook on `lora_A` to all-gather its output.
- `RowwiseParallelLora`: Extends `RowwiseParallel` to shard base weight with `Shard(1)` and LoRA weights with `Shard(1)`.
- `SequenceParallelLora`: Extends `SequenceParallel` for LoRA replicated parameters.
- `translate_to_lora(plan)`: Converts standard parallel styles to their LoRA variants via `CLS_MAP`.

**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/parallel_styles.py`, lines 40-114

## 4. State Flow

### Initialization Flow

```
1. Recipe calls initialize_distributed(backend="nccl")
   -> init_utils.py: Sets CUDA device, calls torch.distributed.init_process_group()
   -> Returns DistInfo(backend, rank, world_size, device, is_main)

2. Recipe creates FSDP2Manager(tp_size=X, cp_size=Y, ...)
   -> fsdp2.py __post_init__() -> _setup_distributed()
   -> Infers dp_size = world_size / (tp_size * cp_size * pp_size)
   -> Computes dp_shard_size = dp_size / dp_replicate_size
   -> Calls _get_device_mesh():
      -> init_device_mesh("cuda", (pp, dp_replicate, dp_shard, cp, tp))
      -> Creates flattened submeshes: "dp", "dp_shard_cp", "dp_cp"
   -> If ep_size > 1: _get_moe_mesh() creates (pp, ep_shard, ep) mesh
```

### Model Parallelization Flow (FSDP2)

```
3. FSDP2Manager.parallelize(model)
   -> If tp_size > 1: _get_parallel_plan(model, ...) resolves TP plan
      -> Priority: custom_tp_plan > use_hf_tp_plan > PARALLELIZE_FUNCTIONS > default
   -> Calls fsdp2_strategy_parallelize(model, device_mesh, tp_shard_plan, ...)
      -> get_parallelization_strategy(model) selects strategy by class name
      -> DefaultParallelizationStrategy.parallelize():
         a. Extract layers via _extract_model_layers(model)
         b. If TP > 1: validate_tp_mesh(), then parallelize_module(model, tp_mesh, plan)
            - For LoRA: translate_to_lora() converts styles to Lora variants
         c. If activation_checkpointing: wrap mlp + self_attn with checkpoint_wrapper
         d. apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy):
            - Walks ModuleList, applies fully_shard() per layer
            - Last layer: reshard_after_forward=False (optimization)
         e. fully_shard(model, mesh=dp_mesh, reshard_after_forward=False)
   -> Returns parallelized model
```

### Pipeline Parallelism Flow

```
4. AutoPipeline(world_mesh, pp_schedule="1f1b", ...).build(model, loss_fn=loss_fn)
   -> validate_hf_model_for_pipeline_support(model)
      - Checks: no tie_word_embeddings, no encoder-decoder
   -> pipeline_model() in functional.py:
      a. split_model_into_stages(model, pp_mesh, ...):
         - calculate_virtual_stages(num_layers, layers_per_stage, pp_size)
         - generate_hf_model_fqn_per_model_part(): assigns layer FQNs to stages
         - For each stage on this rank (stage_ids_this_rank):
           - Deep-copy model, patch forward with pipeline_forward
           - Remove modules not belonging to this stage
           - Create PipelineStage(stage_model, stage_idx, ...)
      b. Apply parallelize_fn to each model_part (TP + FSDP)
      c. build_pipeline_schedule():
         - get_schedule_class(pp_schedule) -> e.g., PipelineScheduleSingle
         - Create schedule with n_microbatches = batch_size / microbatch_size
   -> Stores PipelineInfo(schedule, model_parts, stages, ...)
```

### Context Parallelism Flow

```
5. make_cp_batch_and_ctx(device_mesh, batch)  [cp_utils.py]
   -> Extracts cp_mesh = device_mesh["cp"]
   -> If cp_mesh.size() <= 1: returns nullcontext + unmodified batch
   -> Otherwise:
      a. Builds position_ids if missing
      b. Removes attention_mask (CP doesn't support it)
      c. Registers input_ids, labels, position_ids as CP buffers
      d. create_context_parallel_ctx(cp_mesh, cp_buffers, cp_seq_dims, ...)
         -> Uses torch.distributed.tensor.experimental.context_parallel
         -> Sets rotation method (e.g., "allgather")
      e. Wraps in get_train_context() with optional loss_parallel + compiled_autograd
   -> Returns (context_manager, sharded_batch)
```

### Context Parallelism with Transformer Engine (THD format)

```
6. make_cp_batch_for_te(cp_mesh, batch, qkv_format="thd", num_chunks=N)
   -> split_batch_into_thd_chunks(batch, num_chunks)  [thd_utils.py]
      -> process_input_for_thd(): BSHD -> THD conversion
         - Collapse batch dim: [B, S] -> [B*S]
         - Filter padding from seq_lens, compute cu_seqlens
   -> For each chunk: _shard_thd_chunk_for_te()
      -> Uses transformer_engine_torch.thd_get_partitioned_indices()
      -> Index-selects input_ids, labels, position_ids, padding_mask
   -> Returns stacked chunked batch with cu_seqlens + max_seqlen
```

### Training Loop Synchronization

```
7. get_sync_ctx(model, is_optim_step, defer_fsdp_grad_sync)  [utils.py]
   -> For FSDP2 models: model.set_requires_gradient_sync(is_optim_step)
      - If defer_fsdp_grad_sync=True: sync only on final micro-batch
      - If defer_fsdp_grad_sync=False: always sync
   -> For DDP models: model.no_sync() on non-final micro-batches
```

### Gradient Clipping Flow

```
8. get_grad_norm(parameters, dp_cp_group, tp_group)  [grad_utils.py]
   -> Converts DTensor grads to local tensors
   -> Computes local norm, all-reduces across dp_cp_group then tp_group

9. clip_grad_by_total_norm_(parameters, max_grad_norm, total_norm)
   -> Computes clip_coeff = max_grad_norm / (total_norm + 1e-6)
   -> Scales all grads in-place if clip_coeff < 1.0
```

## 5. Common Modification Scenarios

### Scenario 1: Adding a New Optimized TP Plan for a New Model Architecture

**When**: You need to support a new HuggingFace model (e.g., `NewModelForCausalLM`) with an optimized tensor parallel plan instead of relying on the default fallback.

**Steps**:

1. **Create the plan function** in `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/optimized_tp_plans.py`:

```python
from transformers.models.new_model.modeling_new_model import NewModelForCausalLM

def _parallelize_new_model(
    model: NewModelForCausalLM,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    base_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }
    if sequence_parallel:
        base_plan.update({
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            # ... add SP entries
        })
    return base_plan
```

2. **Register it** in the `PARALLELIZE_FUNCTIONS` dict at the bottom of the same file:

```python
PARALLELIZE_FUNCTIONS[NewModelForCausalLM] = _parallelize_new_model
```

3. If the model has a non-standard layer structure (e.g., not `model.layers`), also add it to `_extract_model_layers()` in `parallelizer.py` (around line 829):

```python
VLM_MODEL_CLS_TO_LAYERS[NewModelForCausalLM] = ["model.decoder.layers"]
```

4. If the model needs special TP head validation, add a branch in `validate_tp_mesh()` in `parallelizer.py` (around line 664).

**Key files**: `optimized_tp_plans.py`, `parallelizer.py`

### Scenario 2: Adding a Custom Parallelization Strategy for a Non-Standard Model

**When**: A model architecture (e.g., a hybrid Mamba-Transformer) requires a fundamentally different parallelization approach that cannot be expressed as a simple TP plan dictionary.

**Steps**:

1. **Create a new strategy class** in `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/parallelizer.py` by subclassing `ParallelizationStrategy`:

```python
class MambaHybridParallelizationStrategy(ParallelizationStrategy):
    def parallelize(self, model, device_mesh, mp_policy=None, offload_policy=None,
                    sequence_parallel=False, activation_checkpointing=False,
                    tp_shard_plan=None, use_hf_tp_plan=False,
                    dp_replicate_mesh_name="dp_replicate",
                    dp_shard_cp_mesh_name="dp_shard_cp",
                    tp_mesh_name="tp") -> nn.Module:
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh = device_mesh[(dp_replicate_mesh_name, dp_shard_cp_mesh_name)]

        # Custom TP for attention layers only
        if tp_mesh.size() > 1:
            attn_plan = {...}
            parallelize_module(model, tp_mesh, attn_plan)

        # Custom FSDP wrapping per block type
        for layer in model.layers:
            if layer.block_type == "mamba":
                parallelizer_utils.fully_shard_by_dtype(layer, mesh=dp_mesh, ...)
            else:
                fully_shard(layer, mesh=dp_mesh, ...)

        return fully_shard(model, mesh=dp_mesh, reshard_after_forward=False, ...)
```

2. **Register it** in the `PARALLELIZATION_STRATEGIES` dict (line 380):

```python
PARALLELIZATION_STRATEGIES["MambaHybridForCausalLM"] = MambaHybridParallelizationStrategy()
```

Alternatively, use the decorator from an external module:

```python
@register_parallel_strategy(name="MambaHybridForCausalLM")
class MambaHybridParallelizationStrategy(ParallelizationStrategy):
    ...
```

**Key files**: `parallelizer.py`

### Scenario 3: Changing the FSDP Sharding Strategy (e.g., Enabling HSDP)

**When**: You want to switch from pure FSDP to HSDP (Hybrid Sharded Data Parallelism) where parameters are sharded within a node but replicated across nodes.

**Steps**:

1. **Set `dp_replicate_size`** in your YAML config or `FSDP2Manager` instantiation. For example, with 4 nodes of 8 GPUs each (32 total), `tp_size=2`:

```yaml
dist_env:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  tp_size: 2
  dp_replicate_size: 4  # Replicate across 4 groups
  # dp_size is inferred as 16 (32 / 2), dp_shard_size = 16 / 4 = 4
```

2. The mesh is automatically constructed in `_get_device_mesh()` (fsdp2.py, line 216) with shape `(1, 4, 4, 1, 2)` for `(pp, dp_replicate, dp_shard, cp, tp)`. The flattened submesh `device_mesh[("dp_replicate", "dp_shard_cp")]` is used as the FSDP mesh, which automatically creates HSDP behavior.

3. **Constraints to verify** (enforced in `_setup_distributed()`, lines 197-199):
   - `dp_size % dp_replicate_size == 0`
   - `dp_replicate_size < dp_size` (pure DDP is not supported through FSDP2Manager)

**Key files**: `fsdp2.py`

### Scenario 4: Enabling Pipeline Parallelism with Custom Stage Splitting

**When**: The automatic equal-layer splitting does not suit your model (e.g., first/last stages are much heavier due to embeddings and lm_head).

**Steps**:

1. **Option A** -- Control layers per virtual stage via `layers_per_stage`:

```yaml
pipeline_parallel:
  pp_size: 4
  layers_per_stage: 4  # For a 32-layer model -> 8 virtual stages -> 2 per rank
  pp_schedule: "interleaved_1f1b"
```

2. **Option B** -- Provide explicit module FQNs per stage:

```python
auto_pp = AutoPipeline(
    world_mesh=mesh,
    pp_schedule="1f1b",
    module_fqns_per_model_part=[
        ["model.embed_tokens", "model.layers.0", ..., "model.layers.9"],
        ["model.layers.10", ..., "model.layers.19"],
        ["model.layers.20", ..., "model.layers.29"],
        ["model.layers.30", "model.layers.31", "model.norm", "lm_head"],
    ],
)
```

3. The `generate_hf_model_fqn_per_model_part()` function in `pipelining/functional.py` (line 82) handles automatic splitting with ceiling division for uneven layer counts. It always places `embed_tokens` on the first stage, `norm` and `lm_head` on the last stage, and `rotary_emb` on all stages.

4. For V-schedule (ZBV Zero Bubble), `stage_ids_this_rank()` (line 70) uses "v" style assignment instead of "loop" style.

**Key files**: `pipelining/autopipeline.py`, `pipelining/functional.py`

### Scenario 5: Adding Context Parallelism Support for a Custom Attention Implementation

**When**: Your model uses a custom attention mechanism and you need to integrate it with the CP infrastructure.

**Steps**:

1. Ensure your model's forward pass accepts the CP context manager. The CP context is created by `make_cp_batch_and_ctx()` in `cp_utils.py` (line 104) and wraps PyTorch's `context_parallel()`.

2. For standard PyTorch SDPA-based attention:
   - CP automatically handles sequence splitting via registered buffers
   - The `get_train_context()` function (line 36) sets the SDPA kernel to FLASH_ATTENTION + EFFICIENT_ATTENTION (MATH backend is incompatible with DTensor)
   - Attention mask is automatically removed (line 156) since CP doesn't support it

3. For Transformer Engine THD format:
   - Use `make_cp_batch_for_te()` (line 187) which converts BSHD to THD format
   - Set `use_te=True` when calling `make_cp_batch_and_ctx()`
   - Ensure your batch has `seq_lens` and `seq_lens_padded` fields
   - The THD sharding uses `transformer_engine_torch.thd_get_partitioned_indices()`

4. CP rotation method can be changed (default: "allgather") at `cp_utils.py` line 179. Options include "allgather" and "addtoall" (set via `set_rotate_method()`).

**Key files**: `cp_utils.py`, `thd_utils.py`
