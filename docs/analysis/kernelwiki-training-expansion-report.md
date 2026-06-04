# KernelWiki Training Library Expansion Report: NeMo AutoModel

**Framework**: NVIDIA NeMo AutoModel  
**GitHub URL**: `NVIDIA-NeMo/Automodel`  
**Analysis Date**: 2026-05-28  
**Source Path**: `/root/Automodel/.worktrees/source_code_analysis/`  
**Library Type**: **Training Orchestration Framework** (full-stack, zero raw CUDA kernels)

---

## Dimension 1: Compute Kernels

### Kernel File Census

Total kernel files: 10
- CUDA/C++: 1 file (`nemo_automodel/components/datasets/llm/megatron/helpers.cpp`) — CPU-only data preprocessing via pybind11, **not** a GPU compute kernel
- Triton: 3 files with 14 `@triton.jit` / `@triton.autotune` decorated kernels
  - `nemo_automodel/components/loss/triton/te_cross_entropy.py` (3 kernels)
  - `nemo_automodel/components/_peft/lora_kernel.py` (9 kernels, including autotuned variants)
  - `nemo_automodel/components/moe/megatron/fused_indices_converter.py` (2 kernels)
- TileLang: 6 files with 8 `@tl.jit` / `@T.prim_func` kernel definitions (vendored from Miles/DeepSeek V4)
  - `nemo_automodel/components/models/deepseek_v4/kernels/tilelang_indexer_bwd.py` (2 kernels)
  - `nemo_automodel/components/models/deepseek_v4/kernels/tilelang_indexer_fwd.py` (2 kernels)
  - `nemo_automodel/components/models/deepseek_v4/kernels/tilelang_sparse_mla_bwd.py` (3 kernels)
  - `nemo_automodel/components/models/deepseek_v4/kernels/tilelang_sparse_mla_fwd.py` (1 kernel)
- Extension entry points: 1 (`helpers.cpp` via `PYBIND11_MODULE(helpers_cpp, m)` — CPU data indexing only)
- `torch.compile`-generated kernels: ~18 annotated callsites across MoE utils and common utils

### Training-Specific Kernels

| Kernel | File Path | Proposed Tag | Description |
|--------|-----------|--------------|-------------|
| `cross_entropy_kernel` (Triton) | `components/loss/triton/te_cross_entropy.py:112` | `triton-cross-entropy` | Fused cross-entropy forward+gradient computation in a single pass. Supports TP-sharded vocab, label smoothing, and ignore_index |
| `element_mul_kernel` (Triton) | `components/loss/triton/te_cross_entropy.py:256` | `grad-scale-kernel` | Backward gradient scaling: in-place element-wise multiply of grad_output into pre-computed gradients |
| `online_softmax_kernel` (Triton) | `components/loss/triton/te_cross_entropy.py:48` | `online-softmax` | First-pass online softmax (max + sum) per TP rank for distributed cross-entropy |
| `lora_da_dx_kernel` (Triton) | `components/_peft/lora_kernel.py:336` | `lora-backward` | Training-only: fused LoRA backward computing both dX and intermediate dY*B for dlora_A |
| `lora_db_kernel` (Triton) | `components/_peft/lora_kernel.py:496` | `lora-backward` | Training-only: LoRA backward for dlora_B weight gradient |
| `_multihot_to_indices_kernel` (Triton) | `components/moe/megatron/fused_indices_converter.py:122` | `moe-index-converter` | Backward pass of fused MoE index conversion |
| `tl_indexer_bwd_impl` (TileLang) | `components/models/deepseek_v4/kernels/tilelang_indexer_bwd.py:44` | `sparse-mla-backward` | DSV4 C4 indexer backward pass computing grad_q, grad_w, grad_k |
| `tilelang_sparse_mla_bwd` (TileLang, 3 kernels) | `components/models/deepseek_v4/kernels/tilelang_sparse_mla_bwd.py` | `sparse-mla-backward` | Backward kernels for DeepSeek V4 sparse MLA attention (dQ, dKV, dSink gradients) |
| `swiglu_back` / `weighted_swiglu_back` (torch.compile) | `components/moe/megatron/moe_utils.py:216,229` | `activation-backward` | torch.compile'd backward for SwiGLU activation in MoE experts |
| `geglu_back` / `weighted_geglu_back` (torch.compile) | `components/moe/megatron/moe_utils.py:263,289` | `activation-backward` | torch.compile'd backward for GEGLU activation in MoE experts |
| `FusedAdam` (TE) | `shared/te_patches.py:56` | `fused-optimizer` | TE FusedAdam optimizer with FP8 quantized tensor support |

### Inference-Shared Kernels (with training behavior differences)

| Kernel | File Path | Training Behavior Difference |
|--------|-----------|------------------------------|
| `lora_forward_kernel` (Triton) | `components/_peft/lora_kernel.py:182` | Shared fwd/infer: fused X * lora_A * lora_B GEMM. In training, autograd triggers backward kernels |
| `_indices_to_multihot_kernel` (Triton) | `components/moe/megatron/fused_indices_converter.py:42` | Forward shared: MoE index conversion. In training, triggers backward kernel |
| `tl_indexer_fwd_impl` (TileLang) | `components/models/deepseek_v4/kernels/tilelang_indexer_fwd.py` | Forward shared. In training, triggers `tl_indexer_bwd_impl` backward |
| `tilelang_sparse_mla_fwd` (TileLang) | `components/models/deepseek_v4/kernels/tilelang_sparse_mla_fwd.py` | Forward shared. In training, saves LSE for backward recompute |
| `swiglu` / `geglu` (torch.compile) | `components/moe/megatron/moe_utils.py:195-260` | Forward shared; explicit backward companions compiled separately |
| `_float32_rms_norm_fwd` (torch.compile) | `components/models/common/utils.py:250` | Forward shared: fp32 RMSNorm. Autograd generates backward automatically |
| TE `DotProductAttention` | `components/attention/utils.py:37` | Both fwd and bwd dispatched through TE. FP8 recipe enables FP8 quantized attention in training only |
| TE `Linear` / `GroupedLinear` | `components/models/common/utils.py:344-350` | In training, TE Linear handles FP8 weight quantization+caching per microbatch via `is_first_microbatch` flag |
| PyTorch `F.scaled_dot_product_attention` | `components/attention/utils.py:77` | In training with FlashAttention backend, saves softmax LSE for backward recompute |

### Fused Kernels

| Kernel | Operations Fused | Estimated Memory Savings |
|--------|-----------------|------------------------|
| `cross_entropy_kernel` (Triton) | Softmax + log-loss + gradient in single pass | ~V tokens saved per sample (no softmax output stored) |
| `lora_forward_kernel` (Triton) | X * lora_A * lora_B triple matmul | Avoids materializing [M, N] intermediate |
| `FusedLinearCrossEntropy` | Linear projection + cross-entropy via `cut_cross_entropy` | Avoids materializing [B, S, V] logit tensor |
| `ChunkedCrossEntropy` | `torch.compile`'d cross-entropy in sequence chunks | Reduces peak memory by seq_len / chunk_len factor |
| `SwiGLU` / `GEGLU` fwd+bwd (torch.compile) | Activation + element-wise multiply + routing weight | Eliminates intermediate activation storage |
| TE `DotProductAttention` | Q*K^T + scaling + masking + softmax + dropout + V matmul | O(N) memory instead of O(N^2) for attention |
| DSV4 Sparse MLA (TileLang) | Sparse attention with top-k indices + softmax + sink | Only computes attention for selected key positions |

### Kernel Dependency Graph

| Provider Library | Kernel Types Provided | Import/Include Evidence |
|-----------------|----------------------|------------------------|
| **NVIDIA Transformer Engine (TE)** | Attention (FlashAttention-3, FP8 DPA), Linear (FP8 GEMM), GroupedLinear (MoE FP8), RMSNorm, RoPE, FusedAdam | `attention/utils.py:37`, `common/utils.py:344`, `te_patches.py:56` |
| **Triton (OpenAI)** | Cross-entropy loss, LoRA fused matmul, MoE indices conversion | `loss/triton/te_cross_entropy.py:33`, `_peft/lora_kernel.py:23` |
| **TileLang** | DeepSeek V4 sparse MLA attention (fwd+bwd), C4 indexer (fwd+bwd) | `deepseek_v4/kernels/tilelang_indexer_bwd.py:23` |
| **DeepSeek TileKernels** | Sinkhorn normalization for HyperConnection | `deepseek_v4/optimized_kernels.py:48` |
| **PyTorch SDPA** | Scaled dot-product attention (FlashAttention-2, CuDNN, Efficient, Math) | `attention/utils.py:77` |
| **PyTorch FlexAttention** | Block-causal and custom mask attention | `attention/flex_attention.py:18-22` |
| **Flash Attention (flash_attn)** | FlashAttention-2 via HF `attn_implementation` | `kernel_patches.py:40` |
| **Liger Kernel** | Fused RMSNorm, SwiGLU, CrossEntropy, RoPE | `kernel_patches.py:142` |
| **cut_cross_entropy** | Fused linear+cross-entropy (Apple) | `loss/linear_ce.py:73` |
| **grouped_gemm** | Grouped GEMM for MoE experts | `moe/experts.py:30` |
| **torch._grouped_mm** | PyTorch native grouped GEMM for MoE | `moe/experts.py:491` |
| **DeepEP** | Token dispatch all-to-all for expert parallelism | `moe/megatron/fused_a2a.py:23` |
| **UCCL-EP** | Token dispatch across heterogeneous GPUs/NICs | `moe/megatron/fused_a2a.py:36` |
| **torchao** | FP8 training via Float8LinearConfig | `quantization/fp8.py:20` |
| **Dion/Muon** | Matrix-aware optimizers | `components/optim/utils.py:26` |

### Backend System

`BackendConfig` (`components/models/common/utils.py:140`) is the central kernel dispatch configuration:

| Field | Options | Default | Kernel Dispatched |
|-------|---------|---------|-------------------|
| `attn` | `"te"`, `"sdpa"`, `"flex"`, `"eager"`, `"tilelang"` | `"te"` | TE DotProductAttention / PyTorch SDPA / FlexAttention / TileLang sparse MLA |
| `linear` | `"torch"`, `"te"` | `"te"` | `nn.Linear` vs TE `Linear` (FP8-capable) |
| `rms_norm` | `"torch"`, `"torch_fp32"`, `"te"` | `"torch_fp32"` | `nn.RMSNorm` / `Float32RMSNorm` / TE `RMSNorm` |
| `rope_fusion` | `True`/`False` | `True` | TE fused RoPE vs PyTorch elementwise RoPE |
| `experts` | `"torch"`, `"te"`, `"gmm"`, `"torch_mm"` | `"torch_mm"` | Per-expert loop / TE GroupedLinear / grouped_gemm / `torch._grouped_mm` |
| `dispatcher` | `"torch"`, `"deepep"`, `"hybridep"`, `"uccl_ep"` | `"deepep"` | DTensor vs DeepEP/UCCL-EP token dispatch |
| `te_fp8` | `TEFp8Config` or `None` | `None` | FP8 quantization across all TE modules |

### Proposed New kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| `triton-cross-entropy` | `components/loss/triton/te_cross_entropy.py` | Triton fused cross-entropy with online softmax, TP-parallel vocab, in-place gradient |
| `fused-linear-ce` | `components/loss/linear_ce.py` | Fused linear projection + cross-entropy avoiding full logit materialization |
| `chunked-ce` | `components/loss/chunked_ce.py` | torch.compile'd chunked cross-entropy for memory-efficient long-sequence training |
| `lora-fused-matmul` | `components/_peft/lora_kernel.py` | Triton fused triple-matmul (X * A * B) for LoRA forward |
| `lora-backward` | `components/_peft/lora_kernel.py` | Triton LoRA backward: fused dX/dA and dB gradient kernels |
| `moe-index-converter` | `components/moe/megatron/fused_indices_converter.py` | Triton fused MoE topk-indices to multihot conversion |
| `moe-grouped-gemm` | `components/moe/experts.py` | MoE expert GEMM: `torch._grouped_mm` / `grouped_gemm` / TE GroupedLinear |
| `moe-token-dispatch` | `components/moe/megatron/fused_a2a.py` | DeepEP / UCCL-EP all-to-all token dispatch |
| `sparse-mla` | `components/models/deepseek_v4/kernels/` | TileLang sparse MLA attention for DeepSeek V4 (fwd+bwd) |
| `fused-optimizer` | `shared/te_patches.py` | TE FusedAdam optimizer with FP8 support |

### Key Finding: Orchestration-First Architecture

NeMo AutoModel contains **zero raw CUDA/C++ GPU kernels** (`.cu`/`.cuh` files). All GPU compute is delegated to external libraries (primarily TE, PyTorch native ops, Triton, TileLang, torch.compile). The `BackendConfig` is the single point of control for kernel selection, enabling runtime switching between "fast path" (TE) and "portable path" (PyTorch native).

---

## Dimension 2: Communication Kernels and Strategies

### Collective Operations

| Operation | Algorithm/Mechanism | SM Usage | File Path |
|-----------|-------------------|----------|-----------|
| AllReduce | NCCL via `dist.all_reduce` | 0 (NCCL) | `grad_utils.py:95-109`, `kd_loss.py:70-89` |
| AllGather | NCCL via `dist.all_gather`, FSDP2 internal | 0 (NCCL) | `experts.py:52-68`, `parallelizer.py:297-317` |
| ReduceScatter | FSDP2 gradient reduction; async TP row-parallel | 0 (NCCL) | `config.py:89-90,126-131` |
| AllToAll | `dist.all_to_all_single` (Mamba CP); DeepEP/UCCL-EP (MoE) | DeepEP: configurable `num_sms` (default 20) | `mamba_cp.py:40-65`, `fused_a2a.py:117-260` |
| AllToAll (HybridEP) | `HybridEPBuffer.dispatch_with_permute` | configurable: `num_sms_dispatch_api` (default 24) | `fused_a2a.py:375-498` |
| Send/Recv (P2P) | TE CP p2p mode; PyTorch pipeline parallelism | 0 (NCCL P2P) | `parallelizer.py:388`, `pipelining/functional.py:28-34` |
| Barrier | `dist.barrier` | 0 | `utils.py:73,89` |

### Communication-Compute Overlap Patterns

| Pattern | Mechanism | Evidence |
|---------|-----------|----------|
| FSDP2 Backward Prefetch | `set_modules_to_backward_prefetch()` chains FSDP units; AllGather for layer N-1 overlaps with backward of layer N. Depth 2 default | `parallelizer.py:862-868` |
| FSDP2 Forward Prefetch | `set_modules_to_forward_prefetch()` chains; depth 1 default | `parallelizer.py:856-861` |
| Async Tensor Parallel | `torch._inductor.config._micro_pipeline_tp = True` enables Inductor's fused AllGather-Matmul and Matmul-ReduceScatter. Requires `sequence_parallel=True` + `torch.compile` | `parallelizer.py:173-219` |
| Symmetric Memory for TP | `enable_symm_mem_for_group()` enables `SymmetricMemory` allocator for TP group | `parallelizer.py:209-219` |
| MegatronFSDP Grad Reduce Overlap | `overlap_grad_reduce=True` overlaps ReduceScatter with backward compute | `config.py:156-157` |
| MegatronFSDP Param Gather Overlap | `overlap_param_gather=True` overlaps AllGather with forward compute | `config.py:157,178` |
| MegatronFSDP Double Buffer | `fsdp_double_buffer=True` for double-buffered parameter storage | `config.py:185` |
| MoE Shared Expert Overlap | Shared expert MLP runs on side CUDA stream concurrently with grouped-expert dispatch | `layers.py:731-750` |
| DeepEP/UCCL-EP Async Dispatch | `async_finish=True` allows fused dispatch/combine to overlap with compute via CUDA events | `fused_a2a.py:133-176` |
| FSDP Deferred Grad Sync | `defer_fsdp_grad_sync=True` defers to final micro-batch | `config.py:86,113` |
| FP8 AllGather | `enable_fsdp_float8_all_gather=True` reduces communication volume by 2x | `fp8.py:37,60,197,212` |
| PP No-Reshard | `reshard_after_forward=False` keeps weights gathered across PP microbatches | `parallelizer.py:836-839` |

### Advanced Communication Features Checklist

- [x] NCCL Integration: via `dist.init_process_group(backend="nccl")`
- [x] PyTorch SymmetricMemory: enabled for async TP (`parallelizer.py:209-219`)
- [x] Fused AllGather-Matmul: via Inductor `_micro_pipeline_tp`
- [x] Fused Matmul-ReduceScatter: via Inductor `_micro_pipeline_tp`
- [x] Device API / GPU-initiated Communication: via UCCL-EP
- [ ] Copy Engine / Zero-CTA: not present
- [ ] NCCL Inspector: not present
- [ ] MSCCL++ / MSCCL: not present
- [x] NVSHMEM: via DeepEP for inter-node RDMA (`fused_a2a.py:53-61`)
- [x] UCCL-EP: vendored wrapper (`uccl_ep/`)
- [x] DeepEP: external dependency (`fused_a2a.py:23-24`)
- [x] HybridEP: external dependency (`fused_a2a.py:322-498`)
- [x] Transformer Engine CP: p2p and all_gather modes
- [x] NCCL User Buffers: MegatronFSDP option (`config.py:163,184`)
- [x] FP8 AllGather: via torchao Float8LinearConfig
- [x] DTensor-based Communication: core paradigm for all TP sharding

### Proposed New Communication kernel_types

| Tag | Description |
|-----|-------------|
| `fsdp2-allgather` | FSDP2 parameter unshard AllGather (forward/backward prefetch) |
| `fsdp2-reduce-scatter` | FSDP2 gradient ReduceScatter (with mixed-precision reduce_dtype) |
| `tp-allreduce` | Tensor-parallel AllReduce for row-parallel output (non-SP mode) |
| `async-tp-fused-ag-mm` | Inductor fused AllGather-Matmul (async TP micro-pipeline) |
| `async-tp-fused-mm-rs` | Inductor fused Matmul-ReduceScatter (async TP micro-pipeline) |
| `moe-dispatch-a2a` | MoE fused permute + AllToAll dispatch (DeepEP/UCCL/HybridEP) |
| `moe-combine-a2a` | MoE fused AllToAll combine + unpermute |
| `cp-allgather` | Context-parallel AllGather for attention KV |
| `cp-p2p` | Context-parallel P2P send/recv for TE attention |
| `cp-alltoall` | Mamba CP hidden-parallel AllToAll |
| `pp-p2p` | Pipeline stage send/recv |
| `fp8-allgather` | FSDP2 FP8-quantized AllGather (2x bandwidth reduction) |

### Proposed New Communication techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `compute-comm-overlap` | `parallelizer.py:856-868` | FSDP2 forward/backward prefetch chains for AllGather-compute overlap |
| `async-tensor-parallel` | `parallelizer.py:173-219` | Inductor micro-pipeline TP with fused AG-MM and MM-RS via SymmetricMemory |
| `deferred-grad-sync` | `config.py:86,113` | Defer FSDP gradient sync to final micro-batch during gradient accumulation |
| `fp8-comm-reduction` | `fp8.py:37,212` | FP8 AllGather reducing FSDP communication volume by 2x |
| `moe-shared-expert-overlap` | `layers.py:731-750` | Shared expert compute on side CUDA stream concurrent with EP dispatch |

### Key Observation

NeMo AutoModel contains **zero custom CUDA communication kernels**. All communication is delegated to PyTorch (NCCL backend), DTensor/DeviceMesh abstractions, and external libraries (DeepEP, UCCL-EP, MegatronFSDP, TE).

---

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension | Supported | Strategy Availability | Implementation File | Communication Pattern Triggered |
|-----------|-----------|----------------------|--------------------|---------------------------------|
| Data Parallel (FSDP2) | Yes | FSDP2Config | `distributed/fsdp2.py:58-150` | AllGather (forward) + ReduceScatter (backward) |
| Data Parallel (DDP) | Yes | DDPConfig | `distributed/ddp.py:30-127` | AllReduce (gradient sync) |
| Hybrid Sharded DP (HSDP) | Yes | FSDP2Config | `distributed/device_mesh.py:173-187` | AllGather within shard + AllReduce within replicate |
| MegatronFSDP | Yes | MegatronFSDPConfig | `distributed/megatron_fsdp.py:43-178` | Custom NCCL UB, overlapped grad reduce/param gather |
| Tensor Parallel | Yes | FSDP2Config, MegatronFSDPConfig | `distributed/parallelizer.py:182-219` | AllReduce (RowwiseParallel) or ReduceScatter (with SP) |
| Pipeline Parallel | Yes | FSDP2Config only | `distributed/pipelining/functional.py` | P2P Send/Recv between stages |
| Context Parallel | Yes | FSDP2Config, MegatronFSDPConfig | `distributed/cp_utils.py` | AllGather (DTensor SDPA) or P2P (TE DotProductAttention) |
| Expert Parallel | Yes | FSDP2Config only | `moe/parallelizer.py:83-157` | AllToAll (DeepEP/UCCL-EP dispatch+combine) |
| Sequence Parallel | Yes | FSDP2Config | `distributed/optimized_tp_plans.py:53-69` | ReduceScatter (after RowwiseParallel) + AllGather (before ColwiseParallel) |

### DeviceMesh Topology

**FSDP2 Canonical 5D Mesh** (`device_mesh.py:189-222`):

```
mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
```

Flattened submeshes:
- `"dp"` = flatten(dp_replicate, dp_shard) — for data loading
- `"dp_shard_cp"` = flatten(dp_shard, cp) — for FSDP parameter sharding
- `"dp_cp"` = flatten(dp_replicate, dp_shard, cp) — for loss AllReduce

**MoE Mesh** (separate): `("ep_shard", "ep")` derived from non-PP dims.

**MegatronFSDP 3D Mesh**: `("dp", "cp", "tp")` — does not support PP or EP.

### Pipeline Scheduling Strategies

| Schedule | PyTorch Class | Type | Bubble Efficiency |
|----------|--------------|------|-------------------|
| `1f1b` | `PipelineScheduleSingle` | Single-stage | Standard 1F1B bubble |
| `interleaved1f1b` | `PipelineScheduleMulti` | Multi-stage | Reduced bubble via interleaving |
| `gpipe` | `PipelineScheduleSingle` | Single-stage | Full forward then full backward |
| `v_schedule` / `zero_bubble` | `ScheduleZBVZeroBubble` | V-style | Near-zero bubble |
| `looped_bfs` | `PipelineScheduleMulti` | Multi-stage | BFS-ordered looped |
| `dfs` | `PipelineScheduleMulti` | Multi-stage | DFS-ordered |
| CSV-defined | `_PipelineScheduleRuntime` | Custom | User-defined |

### Per-Model Parallelization Strategies

| Strategy Class | Target Model | Special Behavior |
|---------------|-------------|------------------|
| `DefaultParallelizationStrategy` | Most LLM/VLMs | Standard TP + AC + FSDP2 flow |
| `NemotronHParallelizationStrategy` | NemotronH (Mamba hybrid) | Mamba CP + TP for MLP-only blocks + TE CP for attention |
| `Qwen3_5ParallelizationStrategy` | Qwen3.5 (GatedDeltaNet) | Mixed-dtype FSDP via `fully_shard_by_dtype` + CP for linear attention |
| `DeepseekV4ParallelizationStrategy` | DeepSeek-V4 | Custom `fully_shard_deepseek_v4` for fp32-sensitive params |
| `WanParallelizationStrategy` | Wan diffusion model | Custom TP plan for diffusion blocks |
| `HunyuanParallelizationStrategy` | HunyuanVideo | NO_REENTRANT AC for transformer blocks |

### Communication Kernels per Parallelism Dimension

| Parallelism | Forward Communication | Backward Communication |
|-------------|----------------------|----------------------|
| **FSDP2** | AllGather (weight unshard) | ReduceScatter (gradient shard) |
| **HSDP** | AllGather (shard group) | ReduceScatter + AllReduce (replicate group) |
| **DDP** | None | AllReduce (gradient sync) |
| **TP (ColwiseParallel)** | None (sharded compute) | AllReduce (partial sum) |
| **TP (RowwiseParallel)** | None (sharded compute) | AllReduce (partial sum) |
| **SP** | AllGather (before ColwiseParallel) | ReduceScatter (after RowwiseParallel) |
| **Async TP** | Fused AllGather+Matmul | Fused Matmul+ReduceScatter |
| **PP** | P2P Send (activations) | P2P Recv (gradients) |
| **CP (PyTorch)** | AllGather on K/V sequence dim | ReduceScatter on Q gradients |
| **CP (TE p2p)** | Ring-attention P2P | Ring-attention P2P |
| **CP (Mamba)** | AllToAll (seq→head split) | AllToAll (head→seq split) |
| **EP (DeepEP)** | AllToAll (token dispatch) | AllToAll (token combine) |

### Proposed New Parallelism techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `4d-parallelism` | `device_mesh.py:189-222` | DP + TP + PP + CP orthogonal composition via 5D DeviceMesh |
| `zero-bubble-schedule` | `pipelining/functional.py:30` | V-schedule eliminating pipeline bubble |
| `mamba-context-parallel` | `mamba_cp.py` | Hidden-parallel CP strategy for Mamba/SSM via AllToAll |
| `expert-parallel` | `moe/parallelizer.py:83-157` | Expert weight sharding with DeepEP/UCCL-EP token dispatch |
| `hybrid-ep` | `fused_a2a.py:375-498` | Hybrid intra+inter node expert dispatch |

---

## Dimension 4: Memory Management

### Memory Component Analysis

| Component | Storage Format | Sharding Strategy | Communication Kernel Triggered |
|-----------|---------------|-------------------|---------------------------------|
| Parameters | BF16 (or FP8 via TorchAO) | FSDP2 (ZeRO-3) on `dp_shard_cp` | AllGather before each layer's forward |
| Gradients | BF16 compute, FP32 reduce | FSDP2 ReduceScatter | ReduceScatter after each layer's backward |
| Optimizer States | FP32 (Adam: momentum + variance + master copy) | FSDP2 (ZeRO-3) | None (local optimizer step) |
| Activations | BF16 | Selective recomputation or full checkpoint | None (stored or recomputed locally) |

### Sharding Strategies

| Strategy | Config Class | Sharding Level | Equivalent |
|----------|-------------|----------------|------------|
| FSDP2 (default) | `FSDP2Config` | Full shard | ZeRO-3 |
| MegatronFSDP | `MegatronFSDPConfig` | Configurable `zero_dp_strategy` (1/2/3) | ZeRO-1/2/3 |
| DDP | `DDPConfig` | No sharding | ZeRO-0 |
| HSDP | `FSDP2Config` with `dp_replicate_size > 1` | Shard within node, replicate across | ZeRO-3 + ZeRO-0 |

### Activation Checkpointing Strategies

| Strategy | Condition | Granularity | Overhead | Evidence |
|----------|-----------|-------------|----------|----------|
| HF Native Gradient Checkpointing | `activation_checkpointing=True`, no compile | Full-layer via `GradientCheckpointingLayer` | ~33% compute | `parallelizer.py:222-286` |
| Sub-module Wrapping (fallback) | Older transformers versions | Per mlp/self_attn/norms | ~33% compute | `parallelizer.py:269-286` |
| NO_REENTRANT (compile-compatible) | `enable_compile=True` | self_attn + mlp | ~33% compute | `parallelizer.py:237-246` |
| MoE Selective AC | MoE models | Router outputs preserved, experts recomputed | ~25% compute (experts only) | `moe/parallelizer.py:159-233` |
| No AC | `activation_checkpointing=False` | None | 0% | Default |

### Gradient Accumulation

- Computed as: `grad_acc_steps = global_batch_size / (local_batch_size * dp_size)`
- With `defer_fsdp_grad_sync=True` (default): only 1 ReduceScatter per optimizer step
- Non-final micro-batches: `requires_gradient_sync=False`, `reshard_after_backward=False`
- Final micro-batch: enables sync and resharding

### Memory-Efficient Loss Computation

| Loss Function | Memory for Logits | Technique |
|---------------|-------------------|-----------|
| `MaskedCrossEntropy` | O(B * S * V) | Full logit materialization |
| `ChunkedCrossEntropy` | O(B * chunk_len * V) | Process in chunks of 32 |
| `FusedLinearCrossEntropy` | O(B * S * H) | Never materializes logits; fused lm_head + CE |

### Memory Estimation: Llama-3.1-8B on 8 GPUs with FSDP2

| Component | Per-GPU (No AC) | Per-GPU (With AC) |
|-----------|-----------------|-------------------|
| Parameters (BF16, sharded) | 2.0 GB | 2.0 GB |
| Gradients (BF16, sharded) | 2.0 GB | 2.0 GB |
| Optimizer (Adam FP32, sharded) | 12.0 GB | 12.0 GB |
| AllGather buffer (1 layer) | ~0.4 GB | ~0.4 GB |
| Activations (S=4096, B=1) | ~8.0 GB | ~1.5 GB |
| Logit tensor (FusedLinearCE) | ~0.1 GB | ~0.1 GB |
| **Total estimate** | **~24.5 GB** | **~18.0 GB** |

### CPU Offload Support

- FSDP2 CPU Offload via `CPUOffloadPolicy` (`config.py:79,110`)
- `ScopedModuleOffloading` context manager for temporary CPU→GPU transfer (`training/utils.py:372-387`)
- Checkpoint offload: `ShardedStateDictConfig(offload_to_cpu=True)` for diffusion generation

---

## Dimension 5: Precision Management

### FP8 Scaling Strategies Found

| Strategy | Backend | Granularity | Data Formats | Scale Factor Type | GPU Support | Evidence |
|----------|---------|-------------|--------------|-------------------|-------------|----------|
| Tensorwise Dynamic | TorchAO | Per-tensor | E4M3 (fwd) + E5M2 (bwd) | FP32 dynamic | SM89+ (H100) | `fp8.py:34` |
| Rowwise Dynamic | TorchAO | Per-row | E4M3 (fwd) + E5M2 (bwd) | FP32 dynamic | SM89+ | `fp8.py:34` |
| Rowwise + HP Grad Weight | TorchAO | Per-row | E4M3 (fwd) + E5M2 (bwd) | FP32 dynamic + HP grad | SM89+ | `fp8.py:34` |
| Float8CurrentScaling | TE | Per-tensor | E4M3 (fwd) + E5M2 (bwd) | FP32 current-step | SM89+ | `utils.py:128` |
| Float8BlockScaling | TE | Block-wise | E4M3 (fwd) + E5M2 (bwd) | FP32 per block | SM100 (Blackwell) | `utils.py:127` |
| Per-token blockwise (MoE) | Custom | 128-element blocks | E4M3 | FP32, amax/448 | Any FP8 GPU | `uccl_ep/_utils.py:625-630` |
| UE8M0 scales | UCCL-EP | 128-element blocks | E4M3 | Unsigned E8M0 (power-of-2) | SM100+ | `uccl_ep/_buffer.py:282` |

### Precision per Training Component

| Component | Forward Pass | Backward Pass | Optimizer Step | Evidence |
|-----------|-------------|---------------|----------------|----------|
| Linear GEMM inputs | FP8 E4M3 (when enabled) | FP8 E5M2 | N/A | `fp8.py:202-211` |
| Weights | BF16 (or FP8 E4M3 quantized from BF16) | N/A | FP32 master copy | `config.py:126` |
| Gradients | N/A | BF16 compute | FP32 reduce | `config.py:128` |
| Adam momentum | N/A | N/A | FP32 | `mixed-precision-training.md:31` |
| Adam variance | N/A | N/A | FP32 | `mixed-precision-training.md:31` |
| RMSNorm | FP32 (torch_fp32 default) | BF16 grad | N/A | `utils.py:259-276` |
| Loss computation | FP32 (upcast logits) | FP32 | N/A | `masked_ce.py:75-76` |
| MoE Gate | FP32 override available | FP32 | N/A | `layers.py:233,313-316` |
| lm_head | FP32 (custom MixedPrecisionPolicy) | FP32 | N/A | `parallelizer.py:310` |
| MoE Expert Dispatch | E4M3 (UCCL-EP/DeepEP) | E4M3 | N/A | `uccl_ep/_buffer.py:319` |

### FP8 Communication Integration

- [x] FP8 AllGather in FSDP2 (parameters communicated in FP8, 2x volume reduction): `fp8.py:37,212`
- [x] Precompute dynamic scales for FSDP: `fp8.py:40,194-198`
- [x] FP8 MoE token dispatch (DeepEP/UCCL-EP): `_buffer.py:280`
- [x] FP8 Expert GEMM (TE GroupedLinear): `experts.py:1240-1267`
- [ ] FP8 ReduceScatter (gradients always reduced in FP32): `config.py:128`
- [ ] NVLink-SHARP FP8 in-switch reduction: not present
- Estimated communication volume reduction vs BF16: ~50% for AllGather, 0% for ReduceScatter

### Dual-Stack FP8 Architecture: TE vs TorchAO

| Aspect | TransformerEngine (TE) | TorchAO |
|--------|----------------------|---------|
| Config | `TEFp8Config` in `BackendConfig` | `FP8Config` (YAML `fp8:` section) |
| Scope | Replaces `nn.Linear` with `TELinear` | Converts `nn.Linear` to `Float8Linear` |
| Activation | `te_autocast()` context manager | `convert_to_float8_training()` in-place |
| Expert GEMM | Yes (`GroupedLinear`) | No |
| FP8 Attention | Yes (`fp8_dpa=True`) | No |
| FSDP AllGather | No (TE manages own quantization) | Yes (`enable_fsdp_float8_all_gather`) |
| Microbatch caching | Yes (`is_first_microbatch` flag) | No |
| Model types | MoE, custom models | Dense LLM/VLM |
| Coexistence | Can coexist with TorchAO | Can coexist with TE |

### Loss Scaling Strategy

NeMo AutoModel does **not** use `torch.cuda.amp.GradScaler` or any explicit loss scaling. BF16 training with FP32 gradient reduction does not suffer from the underflow issues of FP16, making gradient scaling unnecessary.

### Proposed New Precision techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `fp8-tensorwise` | `fp8.py:34` | Per-tensor dynamic FP8 scaling via TorchAO |
| `fp8-rowwise` | `fp8.py:34` | Per-row FP8 scaling for finer granularity |
| `fp8-block-scaling` | `utils.py:127` | TE block-wise FP8 scaling (MXFP8-like, Blackwell-native) |
| `fp8-current-scaling` | `utils.py:128` | TE per-tensor current-step FP8 scaling |
| `fp8-allgather` | `fp8.py:37,212` | FP8-quantized FSDP2 AllGather for 2x bandwidth reduction |
| `fp8-weight-caching` | `training/utils.py:241-267` | FP8 weight quantize-once-reuse during gradient accumulation |
| `dual-stack-fp8` | `fp8.py`, `utils.py` | TE + TorchAO coexistence for MoE + dense FP8 |
| `fp32-rms-norm` | `utils.py:259-276` | torch.compile'd FP32 upcast RMSNorm for training stability |
| `moe-fp8-dispatch` | `uccl_ep/_utils.py:625-630` | Per-token blockwise FP8 quantization for MoE AllToAll |

---

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Capability | Implementation | File | Activation |
|-----------|---------------|------|------------|
| AutoNVTX model annotation | Recursive hook-based NVTX range injection on all `nn.Module` nodes (forward + backward) | `autonvtx/__init__.py:33-93` | YAML: `nvtx: true` |
| Nsight Systems integration | CUDA profiler start/stop + `emit_nvtx` at configurable step windows, per-rank filtering | `recipes/llm/benchmark.py:258-261` | YAML: `benchmark.nsys_start/nsys_end/nsys_ranks` |
| Megatron-derived Timers | `Timer`/`Timers` with CUDA sync, barrier, min/max across ranks, TensorBoard/WandB export | `components/training/timers.py:1-559` | Benchmark recipe |
| Peak memory tracking | `torch.cuda.max_memory_allocated()` / `reset_peak_memory_stats()` per step | `recipes/llm/train_ft.py:915,1618` | Always active |
| MFU calculation | Architecture-specific FLOP formulas for 30+ model families, auto device detection | `components/utils/flops_utils.py:1-1581` | Always computed when model is supported |
| MoE load balance metrics | Expert utilization, diversity, dead experts, per-layer CV | `components/moe/load_balance_metrics.py` | YAML: `moe_metrics.enabled: true` |

### AutoNVTX System

The `autonvtx` module provides **zero-code-change NVTX instrumentation** for any PyTorch `nn.Module` hierarchy:

1. Single entry point: `patch(model, name=None, add_backward_hooks=True)`
2. Recursive instrumentation via 4 PyTorch hooks per module (forward pre/post, backward pre/post)
3. Hierarchical naming: `"child_name: ClassName"`
4. Activation checkpointing safety: thread-local `set` prevents corrupted timelines during AC replay
5. Idempotent: `_nvtx_patched` sentinel prevents double-hooking

### Logger Backends

| Backend | File | Rank Filtering | Key Features |
|---------|------|---------------|--------------|
| JSONL (MetricLogger) | `loggers/metric_logger.py:88-183` | Rank 0 only | Thread-safe, buffered (100 samples), batched GPU-to-CPU transfer |
| Weights & Biases | `loggers/wandb_utils.py` | Rank 0 only | Full Settings integration, config logging |
| MLflow | `loggers/mlflow_utils.py:28-250` | Rank 0 only | Run resume, FAILED/KILLED status on crash/SIGTERM |
| Comet ML | `loggers/comet_utils.py:24-159` | Rank 0 only | Full Experiment lifecycle, auto-tags |
| TensorBoard | `training/timers.py:504-536` | Rank 0 only | Timer max-times via `add_scalar` |
| Console (stdlib) | `loggers/log_utils.py:167-209` | RankFilter | Color-coded, configurable log level |

### Performance Metrics Collected

| Metric | Key | Unit | Source |
|--------|-----|------|--------|
| Training loss | `loss` | scalar | DP-allreduced sum / num_label_tokens |
| Gradient norm | `grad_norm` | L2 norm | `scale_grads_and_clip_grad_norm()` |
| Learning rate | `lr` | scalar | `optimizer.param_groups[0]["lr"]` |
| Peak GPU memory | `mem` | GiB | `torch.cuda.max_memory_allocated() / 1024^3` |
| Tokens per second (global) | `tps` | tokens/s | `num_tokens_in_batch / time_delta` |
| Tokens per second per GPU | `tps_per_gpu` | tokens/s/GPU | `tps / cp_size / dp_size` |
| Model FLOPs Utilization | `mfu` | percentage | `AutoMFU` + `calculate_mfu()` |
| Tokens per step | `num_tokens_per_step` | count | DP-allreduced |
| Label tokens per step | `num_label_tokens` | count | DP-allreduced |

### MoE Load Balance Metrics

| Mode | Metrics |
|------|---------|
| Brief | `moe/cv_*`, `moe/expert_utilization_*`, `moe/dead_expert_frac_mean`, `moe/expert_diversity_mean` |
| Detailed | All brief + per-layer `moe/layer_{i}/cv`, utilization, dead_expert_frac, diversity |

### Flight Recorder Status

NeMo AutoModel does **not** include a built-in flight recorder. Post-failure diagnosis relies on:
- JSONL metric log with complete time-series
- MLflow crash detection (FAILED/KILLED status)
- Checkpoint `losses.json`
- Library version logging at start
- NCCL debug env vars (`NCCL_DEBUG=INFO`)

### Notable Gaps

1. No built-in PyTorch Profiler integration (relies entirely on nsys externally)
2. No per-kernel timing aggregation (NVTX provides ranges but no in-framework summary)
3. No NCCL bandwidth/latency metrics collected in-process
4. No structured flight recorder for automated failure root-cause analysis
5. No activation memory breakdown (peak only, not per-layer)

---

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property | Value |
|----------|-------|
| Library | NeMo AutoModel |
| GitHub URL | `NVIDIA-NeMo/Automodel` |
| Type | **Training Orchestration Framework** (full-stack) |
| Contains CUDA Kernels | No (zero `.cu`/`.cuh` files; uses Triton/TileLang/torch.compile for in-tree GPU kernels) |
| Primary Knowledge Dimensions | Dim 3 (Parallelism), Dim 4 (Memory), Dim 5 (Precision) |
| Secondary Knowledge Dimensions | Dim 2 (Communication), Dim 1 (Compute — orchestration-focused), Dim 6 (Profiling) |
| Recommended KernelWiki Priority | **P0** (core training orchestration framework — the reference for composable multi-dimensional parallelism) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # New from NeMo AutoModel
  - triton-cross-entropy       # Triton fused cross-entropy with online softmax and TP-parallel vocab
  - fused-linear-ce            # Fused lm_head + cross-entropy avoiding logit materialization
  - chunked-ce                 # torch.compile'd chunked cross-entropy for memory efficiency
  - lora-fused-matmul          # Triton fused triple-matmul (X * A * B) for LoRA
  - lora-backward              # Triton LoRA backward fused dX/dA and dB kernels
  - moe-index-converter        # Triton MoE topk-indices to multihot conversion
  - moe-grouped-gemm           # MoE expert GEMM via torch._grouped_mm / grouped_gemm / TE
  - moe-token-dispatch         # DeepEP/UCCL-EP fused all-to-all token dispatch
  - sparse-mla                 # TileLang DeepSeek V4 sparse multi-latent attention
  - fused-optimizer            # TE FusedAdam with FP8 quantized tensor support
  - fsdp2-allgather            # FSDP2 parameter unshard AllGather
  - fsdp2-reduce-scatter       # FSDP2 gradient ReduceScatter
  - fp8-allgather              # FSDP2 FP8-quantized AllGather (2x bandwidth reduction)
  - async-tp-fused-ag-mm       # Inductor fused AllGather-Matmul (async TP)
  - async-tp-fused-mm-rs       # Inductor fused Matmul-ReduceScatter (async TP)
  - moe-dispatch-a2a           # MoE fused permute + AllToAll dispatch
  - moe-combine-a2a            # MoE fused AllToAll combine + unpermute
  - cp-allgather               # Context-parallel AllGather for attention KV
  - cp-p2p                     # Context-parallel P2P ring-attention
  - cp-alltoall                # Mamba CP hidden-parallel AllToAll
  - pp-p2p                     # Pipeline parallel stage send/recv

techniques:
  # New from NeMo AutoModel
  - 4d-parallelism             # DP + TP + PP + CP orthogonal composition via 5D DeviceMesh
  - zero-bubble-schedule       # V-schedule pipeline parallelism with ~0 bubble
  - compute-comm-overlap       # FSDP2 forward/backward prefetch for AllGather-compute overlap
  - async-tensor-parallel      # Inductor micro-pipeline TP with fused AG-MM/MM-RS via SymmetricMemory
  - deferred-grad-sync         # Defer FSDP gradient sync to final micro-batch
  - fp8-comm-reduction         # FP8 AllGather reducing FSDP communication volume by 2x
  - fp8-tensorwise             # Per-tensor dynamic FP8 scaling
  - fp8-rowwise                # Per-row FP8 scaling for finer granularity
  - fp8-block-scaling          # TE block-wise FP8 scaling (MXFP8-like, Blackwell-native)
  - fp8-weight-caching         # FP8 weight quantize-once-reuse during gradient accumulation
  - dual-stack-fp8             # TE + TorchAO FP8 coexistence for MoE + dense
  - fp32-rms-norm              # torch.compile'd FP32 upcast RMSNorm for training stability
  - moe-fp8-dispatch           # Per-token blockwise FP8 quantization for MoE AllToAll
  - moe-shared-expert-overlap  # Shared expert compute on side CUDA stream concurrent with EP dispatch
  - expert-parallel            # Expert weight sharding with DeepEP/UCCL-EP token dispatch
  - hybrid-ep                  # Hybrid intra+inter node expert dispatch
  - mamba-context-parallel     # Hidden-parallel CP strategy for Mamba/SSM via AllToAll
  - selective-activation-ckpt  # MoE selective AC preserving router outputs
  - fused-linear-ce-technique  # Avoid logit materialization via fused linear+CE
  - autonvtx                   # Zero-code-change recursive NVTX annotation for profiling
  - backend-config-dispatch    # Runtime kernel selection via BackendConfig dataclass

hardware_features:
  # New from NeMo AutoModel (hardware features the framework can leverage)
  - symmetric-memory           # PyTorch SymmetricMemory for async TP fused kernels
  - nvshmem                    # NVSHMEM for DeepEP inter-node RDMA expert dispatch
  - nccl-user-buffers          # NCCL UB for MegatronFSDP reduced latency

source_categories:
  # New
  - training-orchestration     # Framework that orchestrates kernels from upstream libraries
```

### S.3 Wiki Page Topics

| # | Wiki Subdirectory | Proposed Page ID | Title | Source Evidence | Related Existing Pages |
|---|-------------------|------------------|-------|----------------|------------------------|
| 1 | training/ | training-backend-config | BackendConfig: Runtime Kernel Dispatch for Training Frameworks | `components/models/common/utils.py:140` | — |
| 2 | training/ | training-fused-cross-entropy | Fused Cross-Entropy: Triton, Chunked, and Linear-CE Techniques | `components/loss/` (3 loss implementations) | — |
| 3 | training/ | training-lora-triton-kernels | Triton LoRA Fused Matmul and Backward Kernels | `components/_peft/lora_kernel.py` | — |
| 4 | training/ | training-fp8-dual-stack | Dual-Stack FP8: TransformerEngine + TorchAO Coexistence | `quantization/fp8.py`, `common/utils.py` | — |
| 5 | training/ | training-fp8-weight-caching | FP8 Weight Caching During Gradient Accumulation | `training/utils.py:241-267` | — |
| 6 | training/ | training-moe-selective-ac | MoE Selective Activation Checkpointing with Router Preservation | `moe/parallelizer.py:159-233` | — |
| 7 | communication/ | comm-fsdp2-prefetch | FSDP2 Forward/Backward Prefetch for Communication-Compute Overlap | `distributed/parallelizer.py:853-868` | — |
| 8 | communication/ | comm-async-tp | Async Tensor Parallel via Inductor Micro-Pipeline and SymmetricMemory | `distributed/parallelizer.py:173-219` | — |
| 9 | communication/ | comm-deepep-uccl | DeepEP / UCCL-EP / HybridEP: MoE Expert Dispatch Communication | `moe/megatron/fused_a2a.py` | — |
| 10 | communication/ | comm-fp8-allgather | FP8 AllGather: Halving FSDP2 Communication Volume | `quantization/fp8.py:37,212` | — |
| 11 | parallelism/ | parallel-5d-device-mesh | 5D DeviceMesh: Composable Orthogonal Parallelism in NeMo AutoModel | `distributed/device_mesh.py:189-222` | — |
| 12 | parallelism/ | parallel-pipeline-schedules | Pipeline Parallelism Schedules: 1F1B, Interleaved, Zero-Bubble, CSV | `pipelining/functional.py` | — |
| 13 | parallelism/ | parallel-mamba-cp | Mamba Context Parallelism: Hidden-Parallel AllToAll Strategy | `distributed/mamba_cp.py` | — |
| 14 | parallelism/ | parallel-parallelization-strategy | Per-Model Parallelization Strategies Pattern | `distributed/parallelizer.py:115-670` | — |
| 15 | training/ | training-autonvtx | AutoNVTX: Zero-Code-Change NVTX Profiling for Training | `autonvtx/__init__.py` | — |
| 16 | training/ | training-mfu-calculation | AutoMFU: Architecture-Aware Model FLOPs Utilization for 30+ Models | `_transformers/mfu.py`, `utils/flops_utils.py` | — |
| 17 | training/ | training-sparse-mla | DeepSeek V4 Sparse MLA: TileLang Attention Kernels | `models/deepseek_v4/kernels/` | — |

### S.4 Repository Mappings (slug → org/repo)

```python
# For the PR candidate search script
"nemo-automodel": "NVIDIA-NeMo/Automodel",

# For the PR page generation script
"nemo-automodel": "NVIDIA-NeMo/Automodel",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag
"cross_entropy": "triton-cross-entropy",
"fused_linear_ce": "fused-linear-ce",
"linear_cross_entropy": "fused-linear-ce",
"chunked_ce": "chunked-ce",
"lora_kernel": "lora-fused-matmul",
"lora_forward_kernel": "lora-fused-matmul",
"lora_backward": "lora-backward",
"fused_indices": "moe-index-converter",
"grouped_gemm": "moe-grouped-gemm",
"grouped_mm": "moe-grouped-gemm",
"GroupedLinear": "moe-grouped-gemm",
"token_dispatch": "moe-token-dispatch",
"fused_dispatch": "moe-token-dispatch",
"fused_combine": "moe-token-dispatch",
"sparse_mla": "sparse-mla",
"tilelang_sparse": "sparse-mla",
"FusedAdam": "fused-optimizer",
"fully_shard": "fsdp2-allgather",
"float8_all_gather": "fp8-allgather",
"fused_all_gather_matmul": "async-tp-fused-ag-mm",
"fused_matmul_reduce_scatter": "async-tp-fused-mm-rs",
"context_parallel": "cp-allgather",
"ring_attention": "cp-p2p",
"PipelineSchedule": "pp-p2p",

# keyword -> technique tag
"4d_parallel": "4d-parallelism",
"device_mesh": "4d-parallelism",
"DeviceMesh": "4d-parallelism",
"zero_bubble": "zero-bubble-schedule",
"ZBV": "zero-bubble-schedule",
"backward_prefetch": "compute-comm-overlap",
"forward_prefetch": "compute-comm-overlap",
"async_tensor_parallel": "async-tensor-parallel",
"_micro_pipeline_tp": "async-tensor-parallel",
"defer_fsdp_grad_sync": "deferred-grad-sync",
"fp8_allgather": "fp8-comm-reduction",
"Float8LinearConfig": "fp8-tensorwise",
"Float8CurrentScaling": "fp8-block-scaling",
"Float8BlockScaling": "fp8-block-scaling",
"is_first_microbatch": "fp8-weight-caching",
"BackendConfig": "backend-config-dispatch",
"autonvtx": "autonvtx",
"selective_recompute": "selective-activation-ckpt",
"expert_parallel": "expert-parallel",
"DeepEP": "expert-parallel",
"UCCL": "expert-parallel",
"HybridEP": "hybrid-ep",
"mamba_cp": "mamba-context-parallel",

# keyword -> hardware_feature tag
"symm_mem": "symmetric-memory",
"SymmetricMemory": "symmetric-memory",
"enable_symm_mem_for_group": "symmetric-memory",
"nvshmem": "nvshmem",
"NVSHMEM": "nvshmem",
"nccl_ub": "nccl-user-buffers",
```

### S.6 PR Search Keywords (for candidate ledger)

```yaml
keywords_used:
  - cross_entropy
  - fused_linear_ce
  - lora_kernel
  - triton.jit
  - tilelang
  - sparse_mla
  - grouped_gemm
  - grouped_mm
  - token_dispatch
  - fused_dispatch
  - FusedAdam
  - fully_shard
  - FSDP2
  - float8
  - fp8
  - async_tensor_parallel
  - micro_pipeline_tp
  - SymmetricMemory
  - DeviceMesh
  - device_mesh
  - pipeline_schedule
  - zero_bubble
  - context_parallel
  - ring_attention
  - expert_parallel
  - DeepEP
  - UCCL
  - HybridEP
  - mamba_cp
  - autonvtx
  - nvtx
  - BackendConfig
  - activation_checkpoint
  - gradient_accumulation
  - MixedPrecisionPolicy
  - moe_metrics
  - load_balance
```

### S.7 Inclusion Policy Lane

```yaml
training-orchestration:
  description: |
    NeMo AutoModel is a training orchestration framework. Capture PRs that modify
    parallelism strategies, communication patterns, kernel dispatch via BackendConfig,
    precision management (FP8), memory optimization, in-tree Triton/TileLang kernels,
    or profiling infrastructure. Skip pure doc/test/example changes.
  capture_criteria:
    - changed_paths_match:
        - "nemo_automodel/components/distributed/**"
        - "nemo_automodel/components/moe/**"
        - "nemo_automodel/components/loss/**"
        - "nemo_automodel/components/_peft/lora_kernel.py"
        - "nemo_automodel/components/quantization/**"
        - "nemo_automodel/components/attention/**"
        - "nemo_automodel/components/models/common/utils.py"
        - "nemo_automodel/components/models/deepseek_v4/kernels/**"
        - "nemo_automodel/components/training/**"
        - "nemo_automodel/components/utils/flops_utils.py"
        - "nemo_automodel/autonvtx/**"
        - "nemo_automodel/_transformers/infrastructure.py"
        - "nemo_automodel/_transformers/te_attention.py"
        - "nemo_automodel/_transformers/mfu.py"
        - "nemo_automodel/shared/te_patches.py"
    - title_contains_any:
        - fsdp
        - fsdp2
        - tensor_parallel
        - pipeline_parallel
        - context_parallel
        - expert_parallel
        - sequence_parallel
        - DeviceMesh
        - fp8
        - float8
        - triton
        - tilelang
        - BackendConfig
        - async_tp
        - DeepEP
        - UCCL
        - HybridEP
        - grouped_gemm
        - cross_entropy
        - lora_kernel
        - activation_checkpoint
        - nvtx
        - autonvtx
        - mfu
        - MoE
        - sparse_mla
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - "tests/**"
        - "examples/**"
        - "tutorials/**"
        - "*.md"
        - "fern/**"
    - pure_config_only: true
```

### S.8 Schema Extensions

```yaml
# New optional frontmatter fields for Wiki pages from this library:
scope: training                          # "training" | "inference" | "both"
library_type: orchestration              # "orchestration" | "compute-kernel" | "communication" | "full-stack"
kernel_provider: [te, triton, tilelang]  # list of upstream kernel providers used
communication_pattern: collective        # "collective" | "p2p" | "multicast" | "none"
sm_utilization: zero                     # "full" | "partial" | "zero" | "configurable"
parallelism_dimensions: [dp, tp, pp, cp, ep, sp]  # which dimensions this page covers
```

### S.9 Hardware Features Relevant to This Library's Training Workloads

| Hardware Feature | Inference Relevance | Training Relevance | Specific Impact on NeMo AutoModel |
|-----------------|--------------------|--------------------|----------------------------------|
| NVLink 5 (1.8 TB/s) | Partial | Core | Doubles gradient ReduceScatter and AllGather bandwidth for FSDP2 |
| NVSwitch 4 (NVL72, 130 TB/s) | Partial | Core | Enables efficient 72-GPU all-to-all for MoE expert dispatch |
| NVLink-SHARP FP8 | No | Core | Not yet integrated — would reduce AllReduce bandwidth by 4x |
| Symmetric Memory (9x latency) | Partial | Core | **Integrated** — enables async TP fused AG-MM / MM-RS kernels (`parallelizer.py:209-219`) |
| Copy Engine (zero-SM) | Partial | Core | Not yet integrated — could free SMs during FSDP2 AllGather |
| MXFP8 hardware (Blackwell) | Yes | Core | Partially integrated via TE `Float8BlockScaling` recipe |
| 192 MB L2 Cache | Beneficial | Beneficial | Improves activation recomputation and attention kernel locality |
| 192 GB HBM3e @ 8 TB/s | Core | Core | Enables larger models per GPU, faster memory access |
| GPU Direct RDMA | Important | Core | Used by DeepEP for inter-node MoE expert dispatch via NVSHMEM |
| NCCL User Buffers | Partial | Important | **Integrated** — MegatronFSDP option for reduced latency (`config.py:163`) |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug | GitHub URL | Relationship | Justification |
|------|-----------|-------------|---------------|
| `transformer-engine` | `NVIDIA/TransformerEngine` | kernel-provider | Primary kernel provider for attention, linear, norm, RoPE, FP8, optimizer |
| `pytorch` | `pytorch/pytorch` | runtime-dependency | FSDP2, DTensor, DeviceMesh, SDPA, FlexAttention, torch.compile |
| `deep-ep` | `deepseek-ai/DeepEP` | communication-backend | MoE token dispatch all-to-all via NVLink + NVSHMEM |
| `torchao` | `pytorch/ao` | kernel-provider | FP8 training via Float8LinearConfig + convert_to_float8_training |
| `triton` | `triton-lang/triton` | kernel-provider | In-tree Triton kernels for cross-entropy, LoRA, MoE indices |
| `tilelang` | `tile-ai/tilelang` | kernel-provider | DeepSeek V4 sparse MLA attention kernels |
| `grouped-gemm` | `tgale96/grouped_gemm` | kernel-provider | MoE grouped GEMM via CUTLASS |
| `cut-cross-entropy` | `apple/ml-cross-entropy` | kernel-provider | Fused linear+cross-entropy |
| `liger-kernel` | `linkedin/Liger-Kernel` | kernel-provider | Optional fused transformers ops (RMSNorm, SwiGLU, CE, RoPE) |
| `flash-attention` | `Dao-AILab/flash-attention` | kernel-provider | FlashAttention-2 via HF attn_implementation |
| `megatron-fsdp` | `NVIDIA/Megatron-FSDP` | communication-backend | Alternative FSDP with NCCL UB and overlap options |
| `uccl` | `uccl-project/uccl` | communication-backend | GPU-initiated expert dispatch for heterogeneous hardware |
| `dion` | (internal/external) | kernel-provider | Matrix-aware optimizers (Muon, NorMuon, Dion, Dion2) |

---

## Appendix: Library Type Adaptation Rationale

NeMo AutoModel is classified as a **training orchestration framework** based on the Dimension 1 kernel census:

- **Zero raw CUDA kernel files** (`.cu`/`.cuh`)
- **14 in-tree Triton kernels** (cross-entropy, LoRA, MoE indices) — custom but not CUDA
- **8 TileLang kernels** (DeepSeek V4 sparse MLA) — vendored from Miles project
- **~18 torch.compile callsites** — JIT-generated kernels

**Dimension emphasis adjustment:**
- **Dim 1 (Compute)**: Shifted from kernel-by-kernel analysis to **kernel dependency graph** and **BackendConfig dispatch architecture**
- **Dim 2 (Communication)**: Full analysis — communication orchestration is a primary concern
- **Dim 3 (Parallelism)**: **Deep analysis** — the framework's primary value proposition is composable 6-dimension parallelism
- **Dim 4 (Memory)**: **Deep analysis** — FSDP2 sharding, AC strategies, and loss memory optimization are central
- **Dim 5 (Precision)**: **Deep analysis** — dual-stack FP8 (TE + TorchAO) is a distinguishing feature
- **Dim 6 (Profiling)**: Full analysis — AutoNVTX and 30+ model MFU calculation are unique contributions
