---
name: automodel-dr-moe
description: Use when working with the MoE module of automodel — Mixture of Experts with DeepEP integration, expert parallelism, token routing, and Megatron-style MoE kernels
---

# MoE Module Deep Read

Module path: `nemo_automodel/components/moe/`
11 files, 4143 lines.

---

## 1. Module Purpose & Capabilities

The MoE module provides a complete Mixture-of-Experts implementation for sparse expert routing in large language models. It spans five capabilities:

**A. Expert Layer Implementations** (`layers.py`, 1120 lines)

Two expert backends with identical interfaces:

- `GroupedExperts` (line 238): All-gather/reduce-scatter based MoE. Iterates over local experts one-by-one, uses DTensor `Shard(0)`/`Partial()` for expert-parallel all-reduce. Works at any scale but does not fuse communication with compute.
- `GroupedExpertsDeepEP` (line 450): DeepEP-backed MoE using `grouped_gemm.ops.gmm()` for batched GEMM across experts and fused all-to-all dispatch/combine via `deep_ep.Buffer`. Requires EP > 1 and the `deep_ep` package.

Both support three activation functions: SwiGLU (`swiglu`), Quick-GEGLU (`quick_geglu`), and ReLU-squared (`relu2`). Gated activations (SwiGLU, Quick-GEGLU) store `gate_and_up_projs` with shape `[n_experts, dim, 2*inter_dim]`; non-gated (ReLU-squared) uses `[n_experts, dim, inter_dim]`, saving 50% memory on that parameter.

**B. Token Routing** (`layers.py` Gate class, line 675)

The `Gate` class implements a configurable router with:
- Two scoring functions: `softmax` and `sigmoid` (configurable via `score_func`)
- Group-based routing: tokens are first assigned to expert groups, then to individual experts within groups (controlled by `n_expert_groups` and `n_limited_groups`)
- DeepSeek-V3-style correction bias (`e_score_correction_bias`): updated each optimizer step via `update_bias()` to rebalance expert utilization
- Auxiliary load-balancing loss scaled by `MoEAuxLossAutoScaler` (in `megatron/moe_utils.py`)
- `FakeBalancedGate` (line 623) for performance profiling with uniform expert assignment

**C. DeepEP All-to-All Communication** (`megatron/fused_a2a.py`, 276 lines)

Custom `torch.autograd.Function` wrappers around `deep_ep.Buffer`:
- `FusedDispatch` (line 80): forward computes dispatch layout + all-to-all send in one kernel; backward calls `buffer.combine()` for the gradient reverse path
- `FusedCombine` (line 179): forward calls `buffer.combine()` to gather expert outputs; backward calls `buffer.dispatch()` for gradient

Both support `async_finish` and `allocate_on_comm_stream` flags for overlapping communication with compute. A singleton `_buffer` is lazily allocated by `get_buffer()` and reused across layers.

**D. Token Dispatch Orchestration** (`megatron/token_dispatcher.py`, 571 lines)

- `MoEFlexTokenDispatcher` (line 339): High-level dispatcher coordinating preprocess -> all-to-all -> postprocess pipeline. Uses a shared `_DeepepManager` singleton across all transformer layers (controlled by `SHARING_DEEPEP_MANAGER` flag).
- `_DeepepManager` (line 90): Manages the lifecycle of a single dispatch-combine cycle: `setup_metadata()` -> `dispatch()` -> `get_permuted_hidden_states_by_experts()` -> expert compute -> `get_restored_hidden_states_by_experts()` -> `combine()`.

**E. Expert Parallelism, FSDP Integration, and Checkpoint Conversion** (`parallelizer.py`, `fsdp_mixin.py`, `state_dict_mixin.py`, `state_dict_utils.py`)

- `ExpertParallel` style (parallelizer.py line 51): Shards expert parameters on dim=0 via DTensor `Shard(0)` across the EP mesh.
- `MoEFSDPSyncMixin` (fsdp_mixin.py line 95): Manages gradient sync and resharding during gradient accumulation for both standard FSDP and pipeline-parallel FSDP.
- `MoESplitExpertsStateDictMixin` (state_dict_mixin.py line 31): Converts between HuggingFace per-expert format (`layers.L.mlp.experts.E.gate_proj.weight`) and grouped format (`layers.L.mlp.experts.gate_and_up_projs`), handling DTensor-aware splitting and EP rank filtering.

---

## 2. Core Design Logic

### Why custom MoE instead of HuggingFace MoE

HuggingFace MoE implementations use individual `nn.Linear` modules per expert. This module replaces them with:

1. **Grouped weight tensors** (`gate_and_up_projs`, `down_projs`) that stack all experts into a single 3D parameter, enabling DTensor sharding on the expert dimension (`Shard(0)`) and efficient grouped GEMM via `grouped_gemm.ops.gmm()`.
2. **Fused all-to-all dispatch/combine** via DeepEP that combines token permutation and inter-GPU communication into single CUDA kernels, avoiding separate permute + alltoall steps.
3. **Shared expert stream overlap**: When shared experts exist (`n_shared_experts > 0`), `MoE.forward()` (layers.py line 1072-1086) runs shared experts on a separate CUDA stream to overlap their compute with the routed expert communication.

### GroupedExperts vs GroupedExpertsDeepEP

`GroupedExperts` (line 238) operates by gathering all tokens to every EP rank, iterating over local experts, and using DTensor partial reduction. This is straightforward but communication-heavy.

`GroupedExpertsDeepEP` (line 450) uses the DeepEP library for NVLink-aware and RDMA-aware all-to-all that only sends tokens to the ranks hosting their assigned experts. It uses `grouped_gemm.ops.gmm()` to process all local experts in a single batched matrix multiply, guided by `tokens_per_expert` counts. This avoids the per-expert loop and reduces communication volume.

The selection is automatic in `MoE.__init__()` (line 1009-1020): DeepEP is used when `backend.enable_deepep=True` AND `world_size > 1`; otherwise falls back to `GroupedExperts`.

### Megatron heritage

The `megatron/` subdirectory contains code derived from Megatron-Core's MoE utilities but adapted for AutoModel's DTensor-native design. Key adaptations:
- `permute()`/`unpermute()` in `moe_utils.py` support optional TransformerEngine fused kernels (`moe_permute`, `moe_unpermute`)
- `MoEAuxLossAutoScaler` (moe_utils.py line 469) preserves auxiliary loss through autograd by storing it in context and scaling its gradient by `main_loss_backward_scale`
- Triton kernels in `fused_indices_converter.py` convert between topk-index and multihot routing representations on GPU

---

## 3. Core Data Structures

### MoEConfig (layers.py line 48)

Dataclass holding all MoE hyperparameters. Key fields:
- `n_routed_experts`, `n_shared_experts`, `n_activated_experts`: expert counts and top-k
- `score_func`: `"softmax"` or `"sigmoid"` routing
- `expert_activation`: `"swiglu"`, `"quick_geglu"`, or `"relu2"`
- `route_scale`, `norm_topk_prob`: weight normalization controls
- `gate_bias_update_factor`: DeepSeek-V3 correction bias learning rate
- `aux_loss_coeff`: load-balancing auxiliary loss coefficient
- `n_expert_groups`, `n_limited_groups`: group-level routing parameters
- `expert_bias`: whether experts have bias terms
- `shared_expert_gate`: whether shared experts have a gating sigmoid

### MoE (layers.py line 978)

Top-level nn.Module composing Gate + experts + optional shared experts.
- `self.gate`: `Gate` or `FakeBalancedGate`
- `self.experts`: `GroupedExperts` or `GroupedExpertsDeepEP`
- `self.shared_experts`: `MLP` instance (or None)
- `self.shared_expert_gate`: optional sigmoid gate for shared experts

### Gate (layers.py line 675)

Router module. Key attributes:
- `self.weight`: `nn.Parameter` shape `[n_routed_experts, dim]`
- `self.bias`: optional `nn.Parameter` shape `[n_routed_experts]`
- `self.e_score_correction_bias`: buffer shape `[n_experts]` (float32), updated by `update_bias()`
- `self._cumulative_expert_load`: tracks token counts per expert across gradient accumulation

### GroupedExperts (layers.py line 238)

- `self.gate_and_up_projs`: `nn.Parameter` shape `[n_routed_experts, dim, 2*moe_inter_dim]` (gated) or `[n_routed_experts, dim, moe_inter_dim]` (non-gated)
- `self.down_projs`: `nn.Parameter` shape `[n_routed_experts, moe_inter_dim, dim]`
- `self.gate_up_proj_bias`: optional bias `[n_routed_experts, up_proj_dim]`
- `self.down_proj_bias`: optional bias `[n_routed_experts, dim]`
- `self.expert_activation`: compiled activation function (swiglu/quick_geglu/relu2)

### GroupedExpertsDeepEP (layers.py line 450)

Same parameter shapes as `GroupedExperts`, plus:
- `self.token_dispatcher`: `MoEFlexTokenDispatcher` instance (initialized by `init_token_dispatcher()` during EP application)
- `self.ep_size`, `self.ep_rank`: set during `init_token_dispatcher()`

### MoEFlexTokenDispatcher (megatron/token_dispatcher.py line 339)

- `shared_comm_manager`: class-level `_DeepepManager` singleton shared across all dispatcher instances
- `self.group`: EP process group
- `self.ep_size`: expert parallel world size

### _DeepepManager (megatron/token_dispatcher.py line 90)

Manages dispatch/combine lifecycle:
- `self.token_indices`: `[num_tokens, topk]` expert assignments
- `self.token_probs`: `[num_tokens, topk]` routing weights
- `self.handle`: opaque DeepEP handle linking dispatch to combine
- `self.tokens_per_expert`: tensor of token counts per local expert
- `self.dispatched_routing_map`: multihot format routing after dispatch

### MoEConfig (megatron) (megatron/token_dispatcher.py line 312)

Separate dataclass for Megatron-style dispatcher configuration. Fields: `moe_enable_deepep`, `moe_permute_fusion`, `moe_expert_capacity_factor`, `moe_router_topk`, `num_moe_experts`, `moe_router_dtype`, `moe_router_expert_pad_multiple`.

### ExpertParallel (parallelizer.py line 51)

`ParallelStyle` subclass for DTensor-based expert sharding. Shards all parameters on dim=0 via `distribute_tensor(param, device_mesh, [Shard(0)])`. For `GroupedExpertsDeepEP`, also calls `init_token_dispatcher()` to set up the DeepEP communication.

### MoEFSDPSyncMixin (fsdp_mixin.py line 95)

Mixin for gradient sync control during gradient accumulation:
- `prepare_for_grad_accumulation()`: disables gradient sync and resharding for all FSDP modules
- `prepare_for_final_backward()`: re-enables sync and resharding before the last backward pass

### MoESplitExpertsStateDictMixin (state_dict_mixin.py line 31)

Mixin providing checkpoint format conversion. Key methods:
- `_from_hf_w_merged_experts()`: HF per-expert format -> grouped tensors with DTensor wrapping
- `_to_hf_w_split_experts()`: grouped tensors -> HF per-expert format
- `_validate_expert_availability()`: ensures required expert weights exist before loading

### FusedDispatch / FusedCombine (megatron/fused_a2a.py lines 80, 179)

`torch.autograd.Function` subclasses wrapping DeepEP buffer operations. Forward saves the communication handle in `ctx.handle` for the backward pass.

### IndicesToMultihot (megatron/fused_indices_converter.py line 190)

Triton-accelerated conversion between `[num_tokens, topk]` index representation and `[num_tokens, num_local_experts]` multihot representation. Uses `_indices_to_multihot_kernel` (forward) and `_multihot_to_indices_kernel` (backward).

---

## 4. State Flow

### Complete MoE Forward Pass

```
Input: x [batch*seq, dim], padding_mask [batch, seq]
                     |
                     v
        MoE.forward() (layers.py:1039)
        Flatten to [num_tokens, dim]
        Create token_mask from padding_mask
                     |
                     v
        Gate.forward() (layers.py:747)
        scores = F.linear(x, weight, bias)
        Apply score_func (softmax or sigmoid)
        Apply e_score_correction_bias (sigmoid only)
        Apply group routing if n_groups > 1
        topk selection -> weights, indices
        Compute aux_loss if training
                     |
                     v
  +------ Is shared_experts present? ------+
  |Yes                                     |No
  v                                        |
  Launch shared experts on                 |
  _shared_experts_stream                   |
  z = shared_experts(x)                    |
  Apply shared_expert_gate if present      |
  |                                        |
  +------ experts.forward() <--------------+
          |
          +-- GroupedExperts path (all-gather) --+
          |   Replicate x,weights,indices via    |
          |   DTensor Shard(0) -> full_tensor()  |
          |   For each local expert:             |
          |     gather tokens, apply activation, |
          |     weighted scatter-add to output   |
          |   DTensor Partial() -> Shard(0)      |
          |   reduce-scatter result              |
          |                                      |
          +-- GroupedExpertsDeepEP path ---------+
              |
              v
        token_dispatcher.token_permutation2()
        (token_dispatcher.py:503)
              |
              v
        dispatch_preprocess2()
        Set token_probs and token_indices on _comm_manager
              |
              v
        dispatch_all_to_all()
        _comm_manager.dispatch() ->
          FusedDispatch.apply() (fused_a2a.py:80)
            buffer.get_dispatch_layout()
            buffer.dispatch() -- NVLink/RDMA all-to-all
            Returns: recv_x, recv_indices, recv_probs,
                     tokens_per_expert, handle
              |
              v
        dispatch_postprocess()
        _comm_manager.get_permuted_hidden_states_by_experts()
          Convert indices -> multihot (Triton or torch)
          permute() to group tokens by expert
          -> permuted_local_hidden_states, tokens_per_expert,
             permuted_probs
              |
              v
        Expert Compute (layers.py:593-616)
        output1 = gmm(hidden, gate_and_up_projs, tokens_per_expert)
        output1 = activation(output1, permuted_probs)
        output2 = gmm(output1, down_projs, tokens_per_expert)
              |
              v
        token_dispatcher.token_unpermutation()
          combine_preprocess() -> unpermute() to restore order
          combine_all_to_all() ->
            FusedCombine.apply() (fused_a2a.py:179)
              buffer.combine() -- reverse all-to-all
          combine_postprocess() -> reshape to original shape
              |
              v
        Wait for shared_experts_stream
        output = y + z (if shared experts)
        Reshape to [batch, seq, dim]
```

### Checkpoint Load Flow (HF -> Native)

```
HF state_dict with per-expert keys:
  layers.L.mlp.experts.E.gate_proj.weight  [inter_dim, dim]
  layers.L.mlp.experts.E.up_proj.weight    [inter_dim, dim]
  layers.L.mlp.experts.E.down_proj.weight  [dim, inter_dim]
          |
          v
  _from_hf_w_merged_experts() (state_dict_mixin.py:214)
    For each expert on this EP rank:
      - Transpose gate_proj and up_proj: [inter_dim, dim] -> [dim, inter_dim]
      - Concatenate: gate_proj_T | up_proj_T -> [dim, 2*inter_dim]
    Stack all experts: -> [n_local_experts, dim, 2*inter_dim]
    Wrap as DTensor with Shard(0) on EP mesh
          |
          v
  Native grouped format:
    gate_and_up_projs  DTensor [n_experts, dim, 2*inter_dim] Shard(0)
    down_projs         DTensor [n_experts, inter_dim, dim]   Shard(0)
```

### Expert Parallelism Application Flow

```
  parallelize_model() (parallelizer.py:278)
    |
    v
  apply_ep() (parallelizer.py:77)
    For each transformer block:
      If block.mlp is MoE:
        parallelize_module(moe.experts, ep_mesh, ExpertParallel())
          ExpertParallel._partition_fn():
            distribute_tensor(param, ep_mesh, [Shard(0)])  for all params
            If GroupedExpertsDeepEP: init_token_dispatcher(ep_mesh)
    |
    v
  apply_fsdp() (parallelizer.py:154)
    For each transformer block:
      If MoE and ep_shard_enabled:
        fully_shard(moe.experts, mesh=ep_shard_mesh, shard_placement_fn=Shard(1))
      If MoE and ep_enabled:
        ignored_params = moe.experts.parameters()  # exclude from block FSDP
      fully_shard(block, ignored_params=ignored_params)
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a New Activation Function

To add a new expert activation (e.g., GeGLU):

1. In `layers.py`, add the `@torch.compile` kernel alongside `swiglu` (line 166), `quick_geglu` (line 180), and `relu2` (line 207). If gated, it should accept `gate_and_up_proj` and `down_proj`; if non-gated, just `gate_and_up_proj` and `down_proj` but with `gate_and_up_proj` having shape `[dim, inter_dim]` instead of `[dim, 2*inter_dim]`.

2. Update `is_gated_activation()` (line 81) to classify the new activation.

3. Update `get_expert_activation()` (line 227) to return the new kernel for `GroupedExperts`.

4. For DeepEP: add a DeepEP-specific compiled variant similar to `quick_geglu_deepep` (line 406) or `relu2_deepep` (line 423), and update `get_expert_activation_for_deepep()` (line 434).

5. Update `MoEConfig.expert_activation` Literal type (line 66) to include the new option.

6. In `megatron/moe_utils.py`, if the new activation needs weighted variants (like `WeightedSwiGLUFunction` at line 235 or `WeightedQuickGeGLUFunction` at line 374), add the corresponding `torch.autograd.Function` with FP8 input store support.

### Scenario 2: Adding a New Token Dispatch Strategy

To add an alternative to DeepEP (e.g., NCCL-based all-to-all):

1. In `megatron/token_dispatcher.py`, create a new class inheriting from `_DispatchManager` (line 47). Implement all abstract methods: `setup_metadata()`, `dispatch()`, `combine()`, `get_dispached_metadata()`, `get_permuted_hidden_states_by_experts()`, `get_restored_hidden_states_by_experts()`.

2. In `MoEFlexTokenDispatcher.__init__()` (line 346), add a condition to instantiate the new manager based on a config flag (add the flag to `MoEConfig` at line 312).

3. If the new dispatch does not need `grouped_gemm`, you may also create a new expert class in `layers.py` alongside `GroupedExpertsDeepEP`, or modify `GroupedExpertsDeepEP.forward()` to support both dispatch backends.

4. Update `MoE.__init__()` (line 992) to select the new expert class based on backend configuration.

### Scenario 3: Integrating a New MoE Model Architecture

To add support for a new MoE model (like a new architecture with different expert structures):

1. Create a model directory under `nemo_automodel/components/models/<model_name>/`.

2. In `model.py`, inherit from `MoEFSDPSyncMixin` (fsdp_mixin.py line 95) to get gradient accumulation optimization.

3. Construct `MoEConfig` from the HuggingFace config, mapping fields appropriately. See `nemo_automodel/components/models/deepseek_v3/model.py` or `nemo_automodel/components/models/qwen3_moe/model.py` for examples.

4. In `state_dict_adapter.py`, inherit from `MoESplitExpertsStateDictMixin` (state_dict_mixin.py line 31) and override `_hf_prefix` and `_expert_path_segment` if the model uses a non-standard expert path (default is `"mlp.experts"`).

5. Call `_from_hf_w_merged_experts()` and `_to_hf_w_split_experts()` in the adapter's `from_hf()` / `to_hf()` methods.

6. If the model has non-standard shared experts or routing, customize the `MoEConfig` fields (`shared_expert_gate`, `shared_expert_inter_dim`, `shared_expert_activation`).

### Scenario 4: Modifying FSDP Behavior for MoE Layers

The MoE module has special FSDP handling because expert parameters are already sharded by EP:

1. `apply_fsdp()` in `parallelizer.py` (line 154) wraps expert modules with `fully_shard()` on `ep_shard_mesh` using `Shard(1)` (sharding on the non-expert dimension), only when `ep_shard_enabled=True`.

2. When EP is enabled, expert parameters are excluded from the block-level FSDP via `ignored_params` (line 210-211).

3. For pipeline parallelism, `patched_backward_maybe_with_nosync()` (fsdp_mixin.py line 194) detects `MoEFSDPSyncMixin` submodules and calls `_disable_fsdp_for_moe_module()` / `_run_post_backward_for_moe_module()` specifically for MoE modules. Modify these functions to change gradient sync timing.

4. The `IS_OPTIM_STEP` global flag (fsdp_mixin.py line 23) is set by `set_is_optim_step()` from the training loop to signal when the final backward before optimizer step occurs.

### Scenario 5: Tuning Expert Load Balancing

The module provides two load-balancing mechanisms:

1. **Auxiliary loss** (Gate._compute_aux_loss, layers.py line 927): Standard MoE load-balancing loss comparing actual expert load fractions (`f_i`) with average routing probabilities (`P_i`). Controlled by `MoEConfig.aux_loss_coeff`. The loss is scaled by token count and applied via `MoEAuxLossAutoScaler` (moe_utils.py line 469).

2. **Correction bias** (Gate.update_bias, layers.py line 842): DeepSeek-V3-style post-hoc bias adjustment. After each optimizer step, `update_bias()` computes `sign(average_load - expert_load)` and applies it scaled by `gate_bias_update_factor`. The bias accumulates across DP ranks using DTensor `Partial()` reduction. The master copy is kept in float32 to avoid quantization drift (`e_score_correction_bias_master`, line 900).

To tune: adjust `aux_loss_coeff` for gradient-based balancing, or `gate_bias_update_factor` for the bias correction rate. Set `force_e_score_correction_bias=True` in `MoEConfig` to load pre-existing bias from checkpoints without updating it during training.
