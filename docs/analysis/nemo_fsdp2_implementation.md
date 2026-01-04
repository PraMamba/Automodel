# NeMo AutoModel FSDP2 Implementation Deep Dive

## Executive Summary

This document provides a comprehensive source code analysis of how NeMo AutoModel implements FSDP2 (Fully Sharded Data Parallelism 2) using PyTorch's native distributed primitives. The implementation is production-grade, designed for large-scale training (100B+ parameters) with support for N-dimensional parallelism.

**Core Architecture**:
- **Manager-based pattern**: `FSDP2Manager` dataclass centralizes configuration and initialization
- **5D DeviceMesh**: `(pp, dp_replicate, dp_shard, cp, tp)` for fine-grained control
- **Strategy pattern**: Pluggable `ParallelizationStrategy` for model-specific logic
- **Direct PyTorch APIs**: No abstraction layers, uses `torch.distributed.fsdp.fully_shard` directly

**Key Files**:
- `nemo_automodel/components/distributed/fsdp2.py` - FSDP2Manager and DeviceMesh setup
- `nemo_automodel/components/distributed/parallelizer.py` - Parallelization strategies
- `nemo_automodel/components/distributed/optimized_tp_plans.py` - Model-specific TP plans

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [FSDP2Manager Design](#fsdp2manager-design)
3. [DeviceMesh Construction](#devicemesh-construction)
4. [Parallelization Strategy Pattern](#parallelization-strategy-pattern)
5. [FSDP2 Sharding Logic](#fsdp2-sharding-logic)
6. [Mixed Precision and Offloading](#mixed-precision-and-offloading)
7. [Submesh Creation](#submesh-creation)
8. [Integration Flow](#integration-flow)
9. [Advanced Features](#advanced-features)
10. [Production Considerations](#production-considerations)

---

## Architecture Overview

NeMo AutoModel's FSDP2 implementation follows a **manager-based architecture** where a central `FSDP2Manager` dataclass encapsulates all distributed training configuration and orchestrates the setup process.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FSDP2Manager                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Configuration Attributes                                 │  │
│  │  - dp_size, dp_replicate_size, dp_shard_size             │  │
│  │  - tp_size, cp_size, pp_size, ep_size                    │  │
│  │  - sequence_parallel, use_hf_tp_plan                     │  │
│  │  - mp_policy, offload_policy, activation_checkpointing   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  _setup_distributed()                                     │  │
│  │  1. Validate and infer parallelism dimensions            │  │
│  │  2. Create 5D DeviceMesh                                 │  │
│  │  3. Create submeshes (dp, dp_shard_cp, dp_cp)            │  │
│  │  4. Create MoE mesh (if ep_size > 1)                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  parallelize(model)                                       │  │
│  │  1. Select TP plan (_get_parallel_plan)                  │  │
│  │  2. Delegate to fsdp2_strategy_parallelize               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              fsdp2_strategy_parallelize                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  get_parallelization_strategy(model)                     │  │
│  │  → Returns strategy based on model type                  │  │
│  │    - DefaultParallelizationStrategy (most models)        │  │
│  │    - NemotronHParallelizationStrategy (Mamba+Attn)       │  │
│  │    - WanParallelizationStrategy (Diffusion)              │  │
│  │    - Custom strategies via @register_parallel_strategy   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  strategy.parallelize(model, device_mesh, ...)           │  │
│  │  1. Apply TP sharding (parallelize_module)               │  │
│  │  2. Apply activation checkpointing                       │  │
│  │  3. Apply FSDP2 sharding recursively                     │  │
│  │  4. Apply FSDP2 to root module                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Design Philosophy

**Separation of Concerns**:
- **FSDP2Manager**: Configuration, DeviceMesh creation, lifecycle management
- **ParallelizationStrategy**: Model-specific parallelization logic (TP plans, FSDP wrapping)
- **Parallelizer utilities**: Reusable helpers (recursive sharding, validation, plan selection)

**Composability**:
- Each parallelism dimension (DP, TP, CP, PP, EP) is independently configurable
- Strategies are pluggable via registry pattern
- TP plans have 4-level customization hierarchy

**Production-Ready**:
- Validates configuration before execution (attention heads divisible by TP, mesh shape consistency)
- Optimizes communication (last-layer reshard, submesh specialization)
- Supports advanced features (HSDP, CPU offloading, activation checkpointing)

---

## FSDP2Manager Design

### Core Implementation

**File**: `nemo_automodel/components/distributed/fsdp2.py:34-318`

```python
@dataclass
class FSDP2Manager:
    """
    Manager for setting up and parallelizing models using FSDP2 with TP, DP, CP sharding.
    """

    # Parallelism dimensions
    dp_size: Optional[int] = None                  # Data parallelism size
    dp_replicate_size: Optional[int] = None        # HSDP replicate dimension
    tp_size: Optional[int] = 1                     # Tensor parallelism size
    cp_size: Optional[int] = 1                     # Context parallelism size
    pp_size: Optional[int] = 1                     # Pipeline parallelism size
    ep_size: Optional[int] = 1                     # Expert parallelism size (MoE)

    # Parallelization configuration
    sequence_parallel: Optional[bool] = False      # Enable sequence parallelism
    use_hf_tp_plan: Optional[bool] = False         # Use HuggingFace TP plan
    custom_tp_plan: Optional[dict] = None          # Custom TP plan

    # Precision and offloading
    mp_policy: Optional[MixedPrecisionPolicy] = field(
        default=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    )
    offload_policy: Optional[CPUOffloadPolicy] = None

    # Training optimization
    activation_checkpointing: Optional[bool] = False
    defer_fsdp_grad_sync: Optional[bool] = True    # Defer grad sync to last micro-batch

    # Infrastructure
    backend: Optional[str] = "nccl"
    world_size: Optional[int] = None

    def __post_init__(self):
        """Post-initialization hook that sets up the distributed environment."""
        if get_world_size_safe() == 1:
            return None
        return self._setup_distributed()
```

### Key Design Patterns

#### 1. Dataclass-Based Configuration

**Benefit**: Type-safe, self-documenting, IDE-friendly configuration
**Example**:
```python
# User configuration (YAML)
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none      # Auto-infer
  tp_size: 4
  cp_size: 8
  sequence_parallel: true

# Automatically validated and converted to typed dataclass
```

#### 2. Automatic Dimension Inference

**Implementation**: `fsdp2.py:179-208`
```python
def _setup_distributed(self):
    # Auto-infer dp_size if not provided
    if self.dp_size is None or self.dp_size <= 0:
        total_parallel_ranks = self.tp_size * self.cp_size * self.pp_size
        if self.world_size % total_parallel_ranks != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"(tp_size * cp_size * pp_size) = {total_parallel_ranks}"
            )
        self.dp_size = self.world_size // total_parallel_ranks

    # Auto-infer dp_replicate_size for HSDP
    if self.dp_replicate_size is None or self.dp_replicate_size <= 0:
        self.dp_replicate_size = 1  # Pure FSDP by default

    # Compute dp_shard_size for HSDP
    self.dp_shard_size = self.dp_size // self.dp_replicate_size
```

**Validation Logic**:
```python
# HSDP constraints
assert self.dp_size % self.dp_replicate_size == 0, \
    "dp_size must be a multiple of dp_replicate_size"

assert self.dp_replicate_size < self.dp_size or self.dp_replicate_size == 1, \
    "dp_replicate_size must be less than dp_size (pure DDP not supported)"

# Expert parallelism constraints
dp_cp_size = self.dp_size * self.cp_size
assert dp_cp_size % self.ep_size == 0, \
    f"{dp_cp_size=} must be a multiple of {self.ep_size=}"
```

**Why This Matters**:
- **User-friendly**: Users specify `world_size` and key dimensions (TP, CP), system infers DP
- **Fail-fast**: Configuration errors detected before expensive initialization
- **HSDP support**: Automatically computes `dp_shard_size` for hybrid sharding

#### 3. Lifecycle Management via `__post_init__`

**Pattern**: Automatic setup triggered by dataclass instantiation
```python
def __post_init__(self):
    """Automatically sets up distributed environment after initialization."""
    if get_world_size_safe() == 1:
        return None  # Skip distributed setup for single-GPU
    return self._setup_distributed()
```

**Benefit**: Users don't need to manually call `setup()` - distributed environment is ready after construction

---

## DeviceMesh Construction

### 5-Dimensional Mesh Architecture

NeMo AutoModel uses a **5D DeviceMesh** to represent N-dimensional parallelism:

**Dimensions**: `(pp, dp_replicate, dp_shard, cp, tp)`

**File**: `fsdp2.py:215-256`

```python
def _get_device_mesh(self):
    # Define mesh shape and dimension names
    mesh_shape = (
        self.pp_size,           # Pipeline parallel (vertical model split)
        self.dp_replicate_size, # Data parallel replicate (HSDP outer)
        self.dp_shard_size,     # Data parallel shard (HSDP inner)
        self.cp_size,           # Context parallel (sequence split)
        self.tp_size            # Tensor parallel (horizontal model split)
    )
    mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")

    # Validate mesh dimensions
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), f"Expected {name} to be int, got {type(shape)}"
        assert shape > 0, f"Expected {name} > 0, got {shape}"

    # Create the 5D device mesh
    self.device_mesh = init_device_mesh(
        device_type="cuda" if self.backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    # Create submeshes for different operations
    self._create_submeshes()

    return self.device_mesh
```

### Mesh Topology Example

**Configuration**:
```yaml
world_size: 128
pp_size: 1
dp_replicate_size: 2
dp_shard_size: 8    # Inferred: dp_size=16, dp_shard_size=16/2=8
cp_size: 4
tp_size: 2
```

**Resulting 5D Mesh**:
```
mesh_shape = (1, 2, 8, 4, 2)  # Total: 1 × 2 × 8 × 4 × 2 = 128 GPUs

GPU Rank Layout:
┌─────────────────────────────────────────────────────────────┐
│  PP=0 (All GPUs in same pipeline stage)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  DP_replicate=0 (First replicate group)             │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  DP_shard=0..7 (8-way FSDP sharding)        │    │   │
│  │  │  ┌─────────────────────────────────────┐    │    │   │
│  │  │  │  CP=0..3 (4-way context parallel)   │    │    │   │
│  │  │  │  ┌─────────────────────────────┐    │    │    │   │
│  │  │  │  │  TP=0..1 (2-way tensor par.)│    │    │    │   │
│  │  │  │  │  Ranks: 0-1, 2-3, 4-5, ...  │    │    │    │   │
│  │  │  │  └─────────────────────────────┘    │    │    │   │
│  │  │  └─────────────────────────────────────┘    │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  DP_replicate=1 (Second replicate group)    │    │   │
│  │  │  ... (same structure for ranks 64-127)      │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Submesh Creation

**Purpose**: Different operations require different process group topologies

**File**: `fsdp2.py:228-255`

```python
def _create_submeshes(self):
    """Create specialized submeshes for different distributed operations."""

    # Build dimension name lists for each submesh type
    dp_mesh_dim_names = []           # For data loading (no communication)
    dp_shard_cp_mesh_dim_names = []  # For FSDP param sharding
    dp_cp_mesh_dim_names = []        # For loss all-reduce

    # DP replicate dimension
    dp_mesh_dim_names.append("dp_replicate")
    dp_cp_mesh_dim_names.append("dp_replicate")

    # DP shard dimension
    dp_mesh_dim_names.append("dp_shard")
    dp_shard_cp_mesh_dim_names.append("dp_shard")
    dp_cp_mesh_dim_names.append("dp_shard")

    # CP dimension
    dp_shard_cp_mesh_dim_names.append("cp")
    dp_cp_mesh_dim_names.append("cp")

    # Create flattened submeshes
    # 1. dp_mesh: (dp_replicate, dp_shard) → "dp" (for DataLoader)
    self.device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

    # 2. dp_shard_cp_mesh: (dp_shard, cp) → "dp_shard_cp" (for FSDP sharding)
    self.device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")

    # 3. dp_cp_mesh: (dp_replicate, dp_shard, cp) → "dp_cp" (for loss reduction)
    self.device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
```

**Submesh Usage**:

| Submesh | Dimensions | Usage | Communication Pattern |
|---------|------------|-------|----------------------|
| `dp_mesh` | `(dp_replicate, dp_shard)` | DataLoader distribution | No inter-GPU communication (data parallel replicas load different batches) |
| `dp_shard_cp_mesh` | `(dp_shard, cp)` | FSDP parameter sharding | All-gather for forward/backward, reduce-scatter for gradients |
| `dp_cp_mesh` | `(dp_replicate, dp_shard, cp)` | Loss reduction | All-reduce across all DP+CP ranks |
| `tp_mesh` | `(tp)` | Tensor parallelism | All-reduce/all-gather for TP operations |

**Example: Submesh for dp_shard_cp with dp_replicate_size=2, dp_shard_size=8, cp_size=4**:
```python
# Original 5D mesh has shape (1, 2, 8, 4, 2)
# dp_shard_cp submesh selects dimensions: (dp_shard=8, cp=4)
# Each DP replicate group has 8×4=32 ranks that shard parameters together

# Rank 0's dp_shard_cp group: [0, 2, 4, 6, ..., 62] (32 ranks)
# Rank 64's dp_shard_cp group: [64, 66, 68, 70, ..., 126] (32 ranks)
```

### MoE Mesh for Expert Parallelism

**File**: `fsdp2.py:257-270`

```python
def _get_moe_mesh(self):
    """Create separate mesh for expert parallelism (MoE models)."""
    mesh_shape = (
        self.pp_size,      # Pipeline parallel
        self.ep_shard_size,  # Expert parallel shard dimension
        self.ep_size         # Expert parallel (distribute experts across GPUs)
    )
    mesh_names = ("pp", "ep_shard", "ep")

    self.moe_mesh = init_device_mesh(
        device_type="cuda" if self.backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )
    return self.moe_mesh
```

**EP Shard Size Calculation**: `fsdp2.py:201-206`
```python
dp_cp_size = self.dp_size * self.cp_size
assert dp_cp_size % self.ep_size == 0, f"{dp_cp_size=} must be a multiple of {self.ep_size=}"

if self.ep_size < dp_cp_size:
    self.ep_shard_size = dp_cp_size // self.ep_size
else:
    self.ep_shard_size = 1
```

**Example**: Mixtral 8x7B with `ep_size=8`
- Each GPU gets 1 expert (8 experts distributed across 8 GPUs)
- Experts are sharded via FSDP within their `ep_shard` group
- MoE routing dynamically sends tokens to appropriate expert GPUs

---

## Parallelization Strategy Pattern

### Strategy Abstraction

**File**: `parallelizer.py:87-106`

```python
class ParallelizationStrategy(ABC):
    """Abstract base class for model parallelization strategies."""

    @abstractmethod
    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        use_hf_tp_plan: bool = False,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        """Apply parallelization strategy to the model."""
        pass
```

### Strategy Registry

**File**: `parallelizer.py:369-383`

```python
# Built-in strategies
PARALLELIZATION_STRATEGIES: Dict[str, ParallelizationStrategy] = {
    "NemotronHForCausalLM": NemotronHParallelizationStrategy(),
    "WanTransformer3DModel": WanParallelizationStrategy(),
}

_DEFAULT_STRATEGY = DefaultParallelizationStrategy()

def get_parallelization_strategy(model: nn.Module) -> ParallelizationStrategy:
    """Get the appropriate parallelization strategy for the given model."""
    model_name = type(model).__name__
    return PARALLELIZATION_STRATEGIES.get(model_name, _DEFAULT_STRATEGY)
```

### DefaultParallelizationStrategy

**Most common strategy** - used for standard transformer models (Llama, GPT, Qwen, etc.)

**File**: `parallelizer.py:109-206`

```python
class DefaultParallelizationStrategy(ParallelizationStrategy):
    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        use_hf_tp_plan: bool = False,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        """Apply the default parallelization flow."""

        # 1. Extract mesh dimensions
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh_dim_names = (dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        dp_mesh = device_mesh[dp_mesh_dim_names]

        # 2. Extract transformer layers
        layers = _extract_model_layers(model)

        # 3. Apply Tensor Parallelism (if tp_size > 1)
        if tp_mesh.size() > 1:
            # Validate attention heads divisible by TP size
            validate_tp_mesh(model, tp_mesh)

            # Get TP plan (custom, optimized, HF, or default)
            model_parallel_plan = {
                k: translate_to_lora(v)
                for k, v in _get_parallel_plan(
                    model,
                    sequence_parallel,
                    tp_shard_plan,
                    use_hf_tp_plan=use_hf_tp_plan,
                ).items()
            }

            # Apply TP sharding
            if model_parallel_plan:
                parallelize_module(model, tp_mesh, model_parallel_plan)

        # 4. Apply activation checkpointing (if enabled)
        if activation_checkpointing:
            # Disable KV caching for deterministic recomputation
            if hasattr(model, "config"):
                model.config.use_cache = False

            # Checkpoint MLP and attention layers
            for i, layer in enumerate(layers):
                if hasattr(layer, "mlp"):
                    layers[i].mlp = checkpoint_wrapper(layer.mlp)
                if hasattr(layer, "self_attn"):
                    layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)
                if hasattr(layer, "input_layernorm"):
                    layers[i].input_layernorm = checkpoint_wrapper(layers[i].input_layernorm)
                if hasattr(layer, "post_attention_layernorm"):
                    layers[i].post_attention_layernorm = checkpoint_wrapper(layers[i].post_attention_layernorm)

        # 5. Set default mixed precision policy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,  # More stable than bf16 for gradients
                output_dtype=torch.float32,
            )

        # 6. Apply FSDP2 sharding recursively to layers
        apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)

        # 7. Apply FSDP2 to root module
        model = fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,  # Optimization: no reshard for root
            offload_policy=offload_policy,
        )

        return model
```

**Parallelization Flow**:
```
Input: Model + DeviceMesh
  ↓
1. Extract TP and DP meshes from 5D DeviceMesh
  ↓
2. Find transformer layers (model.model.layers, vision_tower.layers, etc.)
  ↓
3. Apply Tensor Parallelism (if tp_size > 1)
   - Validate: num_attention_heads % tp_size == 0
   - Get TP plan: custom → optimized → HF → default base
   - Shard: Q/K/V colwise, O/down_proj rowwise
  ↓
4. Apply Activation Checkpointing (if enabled)
   - Wrap MLP, self_attn, layernorms with checkpoint_wrapper
  ↓
5. Apply FSDP2 Sharding Recursively
   - Traverse model tree, wrap each layer with fully_shard
   - Last layer optimization: reshard_after_forward=False
  ↓
6. Apply FSDP2 to Root Module
   - Wrap entire model with fully_shard (reshard_after_forward=False)
  ↓
Output: Parallelized Model
```

### Custom Strategies

**NemotronHParallelizationStrategy**: Hybrid Mamba+Transformer model
**File**: `parallelizer.py:208-271`

```python
class NemotronHParallelizationStrategy(ParallelizationStrategy):
    def parallelize(self, model, device_mesh, ...) -> nn.Module:
        # NemotronH has mixed block types: "mlp" and "mamba"
        layers: torch.nn.ModuleList = model.backbone.layers
        tp_mesh = device_mesh["tp"]

        if tp_mesh.size() > 1:
            # TP plan for lm_head and MLP blocks only
            model_tp_plan = {
                "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            }
            mlp_tp_plan = {
                "mixer.up_proj": ColwiseParallel(),
                "mixer.down_proj": RowwiseParallel(),
            }

            parallelize_module(model, tp_mesh, model_tp_plan)

            # Only parallelize MLP blocks (skip Mamba blocks)
            for layer in layers:
                if layer.block_type == "mlp":
                    parallelize_module(layer, tp_mesh, mlp_tp_plan)

        # Checkpoint both MLP and Mamba blocks
        if activation_checkpointing:
            for i in range(len(layers)):
                if layers[i].block_type in ["mlp", "mamba"]:
                    layers[i] = checkpoint_wrapper(layers[i])

        # Apply FSDP by dtype (some blocks may be int8/fp8)
        for layer in layers:
            parallelizer_utils.fully_shard_by_dtype(
                layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
            )

        return fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, ...)
```

**WanParallelizationStrategy**: Diffusion transformer (WAN architecture)
**File**: `parallelizer.py:273-367`

```python
class WanParallelizationStrategy(ParallelizationStrategy):
    def parallelize(self, model, device_mesh, ...) -> nn.Module:
        tp_mesh = device_mesh["tp"]

        if tp_mesh.size() > 1:
            # TP for condition embedders
            if hasattr(model, "condition_embedder"):
                cond = model.condition_embedder
                if hasattr(cond, "text_embedder"):
                    cond.text_embedder = parallelize_module(
                        cond.text_embedder,
                        tp_mesh,
                        {"linear_1": ColwiseParallel(), "linear_2": RowwiseParallel()},
                    )

            # TP for FFN blocks
            if hasattr(model, "blocks"):
                for block in model.blocks:
                    if hasattr(block, "ffn"):
                        block.ffn = parallelize_module(
                            block.ffn,
                            tp_mesh,
                            {"net.0.proj": ColwiseParallel(), "net.2": RowwiseParallel()},
                        )

        # Apply FSDP recursively and to root
        apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)
        return fully_shard(model, mesh=dp_mesh, ...)
```

### Registering Custom Strategies

**Decorator Pattern**: `parallelizer.py:385-406`

```python
@register_parallel_strategy(name="CustomModelName")
class CustomParallelizationStrategy(ParallelizationStrategy):
    def parallelize(self, model, device_mesh, ...) -> nn.Module:
        # Custom parallelization logic
        ...
        return model
```

**Usage**:
```python
# In user code or plugin
from nemo_automodel.components.distributed.parallelizer import (
    register_parallel_strategy,
    ParallelizationStrategy
)

@register_parallel_strategy(name="MyCustomModel")
class MyCustomStrategy(ParallelizationStrategy):
    def parallelize(self, model, device_mesh, **kwargs):
        # Implement custom TP plans, FSDP wrapping, etc.
        ...
        return model
```

---

## FSDP2 Sharding Logic

### Recursive Sharding Algorithm

**Core Function**: `apply_fsdp2_sharding_recursively`
**File**: `parallelizer.py:408-458`

```python
def apply_fsdp2_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[OffloadPolicy] = None,
) -> None:
    """
    Recursively apply FSDP2 sharding to modules, with optimizations for ModuleList.

    For ModuleList instances (transformer layers), applies last-layer optimization:
    the last layer doesn't reshard after forward since FSDP2 will prefetch it immediately.
    """
    if isinstance(module, nn.ModuleList):
        for layer_id, child_module in enumerate(module):
            # Handle nested ModuleList (recurse instead of wrapping)
            if isinstance(child_module, nn.ModuleList):
                apply_fsdp2_sharding_recursively(child_module, mesh, mp_policy, offload_policy)
            else:
                # Last layer optimization: no reshard after forward
                reshard_after_forward = int(layer_id) < len(module) - 1

                fully_shard(
                    child_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    offload_policy=offload_policy,
                )
                module[layer_id] = child_module
    else:
        # Non-ModuleList: recurse into children
        for name, sub_module in module.named_children():
            apply_fsdp2_sharding_recursively(sub_module, mesh, mp_policy, offload_policy)
```

### Last-Layer Reshard Optimization

**Rationale**: FSDP2 uses **prefetching** - it all-gathers parameters for layer N+1 during layer N's forward pass. For the last layer, there's no layer N+1, so resharding (freeing parameters) after forward is wasteful since backward will immediately all-gather them again.

**Implementation**:
```python
for layer_id, child_module in enumerate(module):
    # Last layer: reshard_after_forward=False
    # Other layers: reshard_after_forward=True (free memory ASAP)
    reshard_after_forward = int(layer_id) < len(module) - 1

    fully_shard(
        child_module,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        offload_policy=offload_policy,
    )
```

**Memory Timeline**:
```
Forward Pass (with reshard_after_forward=True for all but last layer):
  Layer 0:  all-gather params → forward → reshard (free memory)
  Layer 1:  all-gather params → forward → reshard (free memory)
  ...
  Layer N-1: all-gather params → forward → reshard (free memory)
  Layer N:  all-gather params → forward → NO RESHARD (keep params)

Backward Pass:
  Layer N:  backward (params already present) → reduce-scatter grads
  Layer N-1: all-gather params → backward → reduce-scatter grads
  ...
  Layer 0:  all-gather params → backward → reduce-scatter grads
```

**Benefit**: Saves one all-gather operation for the last layer during backward pass

### Root Module Sharding

**Always sets `reshard_after_forward=False` for root**:

**File**: `parallelizer.py:197-203`
```python
# Apply FSDP to the root model
model = fully_shard(
    model,
    mesh=dp_mesh,
    mp_policy=mp_policy,
    reshard_after_forward=False,  # Root params used immediately in backward
    offload_policy=offload_policy,
)
```

**Why**: Root-level parameters (embeddings, final layer norms, lm_head) are used immediately after forward completes, so resharding would just cause an extra all-gather in backward.

---

## Mixed Precision and Offloading

### Mixed Precision Policy

**Default Policy**: `fsdp2.py:107-114`
```python
mp_policy: Optional[MixedPrecisionPolicy] = field(
    default=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,      # Parameters stored in bf16
        reduce_dtype=torch.bfloat16,     # Gradients reduced in bf16
        output_dtype=torch.bfloat16,     # Activations in bf16
        cast_forward_inputs=True,        # Auto-cast inputs to bf16
    )
)
```

**Alternative (More Stable)**: `parallelizer.py:184-189`
```python
if not mp_policy:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,    # Gradients reduced in fp32 for stability
        output_dtype=torch.float32,
    )
```

**Trade-off**:
- **bf16 reduction**: Faster (less communication), slightly less accurate
- **fp32 reduction**: Slower (more communication), more numerically stable

**When to use fp32 reduction**:
- Large models (>70B) where gradient accumulation errors matter
- Training instability observed with bf16 reduction
- Tasks requiring high precision (scientific computing, financial models)

### CPU Offloading

**Configuration**: `fsdp2.py:116-119`
```python
offload_policy: Optional[CPUOffloadPolicy] = field(
    default=None,  # No offloading by default
    metadata={"help": "CPUOffloadPolicy to offload parameters/optim states to CPU."}
)
```

**Usage**:
```python
from torch.distributed.fsdp import CPUOffloadPolicy

# Offload parameters to CPU (extreme memory saving)
manager = FSDP2Manager(
    tp_size=2,
    cp_size=4,
    offload_policy=CPUOffloadPolicy(offload_params=True),
)
```

**Offloading Modes**:
```python
# Option 1: Offload parameters only
CPUOffloadPolicy(offload_params=True)

# Option 2: Offload gradients only
CPUOffloadPolicy(offload_grads=True)

# Option 3: Offload both (maximum memory saving, slowest)
CPUOffloadPolicy(offload_params=True, offload_grads=True)
```

**Performance Impact**:
- **Memory savings**: 40-60% GPU memory reduction
- **Speed cost**: 20-40% slower due to CPU↔GPU transfers
- **Use case**: Training very large models on limited GPUs (e.g., 70B on 4×A100 40GB)

---

## Submesh Creation

### Three Specialized Submeshes

**Purpose**: Different operations require different communication patterns

**File**: `fsdp2.py:228-255`

```python
def _create_submeshes(self):
    # 1. dp_mesh: (dp_replicate, dp_shard) → "dp"
    #    For: DataLoader distribution
    #    Communication: None (each DP rank loads different data)
    self.device_mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")

    # 2. dp_shard_cp_mesh: (dp_shard, cp) → "dp_shard_cp"
    #    For: FSDP parameter sharding
    #    Communication: All-gather (forward), reduce-scatter (backward)
    self.device_mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp_shard_cp")

    # 3. dp_cp_mesh: (dp_replicate, dp_shard, cp) → "dp_cp"
    #    For: Loss reduction
    #    Communication: All-reduce across all DP+CP ranks
    self.device_mesh[("dp_replicate", "dp_shard", "cp")]._flatten(mesh_dim_name="dp_cp")
```

### Submesh Usage in Training Loop

**Example Configuration**:
```
world_size = 64
dp_replicate_size = 2
dp_shard_size = 8  (dp_size = 16)
cp_size = 4
tp_size = 1  (no TP for simplicity)
```

**Training Step**:
```python
# 1. Data Loading (dp_mesh)
#    - Rank 0-15: Load batch A
#    - Rank 16-31: Load batch B (different data)
#    - Rank 32-47: Load batch C
#    - Rank 48-63: Load batch D
#    Communication: None

# 2. Forward Pass (dp_shard_cp_mesh for FSDP)
#    Each layer's forward:
#    - All-gather parameters across dp_shard_cp group (32 ranks: 8 dp_shard × 4 cp)
#    - Compute forward
#    - Reshard (free memory) if not last layer
#    Communication: All-gather (32 ranks)

# 3. Loss Computation (dp_cp_mesh)
#    - Compute local loss
#    - All-reduce loss across dp_cp group (all 64 ranks)
#    Communication: All-reduce (64 ranks)

# 4. Backward Pass (dp_shard_cp_mesh for FSDP)
#    Each layer's backward:
#    - All-gather parameters (if resharded)
#    - Compute gradients
#    - Reduce-scatter gradients across dp_shard_cp group
#    Communication: All-gather + reduce-scatter (32 ranks)
```

### Why Separate Submeshes?

**Efficiency**: Avoids unnecessary communication
- **dp_mesh**: DataLoader doesn't need to communicate (each rank loads different data)
- **dp_shard_cp_mesh**: FSDP only shards within DP+CP, not across DP replicate groups
- **dp_cp_mesh**: Loss must be averaged across all DP+CP ranks (not TP, since TP already computes consistent outputs)

**Correctness**: Ensures proper gradient synchronization
- HSDP (dp_replicate > 1): Gradients first reduce-scattered within dp_shard_cp, then all-reduced across dp_replicate
- CP (cp_size > 1): Sequences split across CP ranks must have gradients synchronized

---

## Integration Flow

### End-to-End Parallelization Flow

**User Code**:
```python
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

# 1. Initialize distributed training
torch.distributed.init_process_group(backend="nccl")

# 2. Create FSDP2 manager (automatically sets up DeviceMesh)
manager = FSDP2Manager(
    dp_size=None,  # Auto-infer
    tp_size=4,
    cp_size=8,
    sequence_parallel=True,
    activation_checkpointing=True,
)

# 3. Load model (preferably on meta device)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    torch_dtype=torch.bfloat16,
    device_map="meta",  # Delay materialization
)

# 4. Parallelize model
model = manager.parallelize(model)

# 5. Train as normal
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
trainer = Trainer(model, optimizer, ...)
trainer.train()
```

### Internal Parallelization Steps

**Triggered by `manager.parallelize(model)`**:

**File**: `fsdp2.py:272-317`
```python
def parallelize(self, model):
    # 1. Skip parallelization if single GPU
    if get_world_size_safe() == 1:
        if self.activation_checkpointing:
            model.gradient_checkpointing_enable()
        return model

    # 2. Get TP plan (if tp_size > 1)
    if self.device_mesh["tp"].size() > 1:
        tp_shard_plan = _get_parallel_plan(
            model,
            sequence_parallel=bool(self.sequence_parallel),
            tp_shard_plan=self.custom_tp_plan,
            use_hf_tp_plan=self.use_hf_tp_plan,
        )
    else:
        tp_shard_plan = None

    # 3. Delegate to strategy-based parallelization
    fsdp2_strategy_parallelize(
        model,
        device_mesh=self.device_mesh,
        mp_policy=self.mp_policy,
        tp_shard_plan=tp_shard_plan,
        offload_policy=self.offload_policy,
        activation_checkpointing=self.activation_checkpointing,
    )

    return model
```

**File**: `parallelizer.py:920-984`
```python
def fsdp2_strategy_parallelize(model, device_mesh, ...):
    # 1. Select parallelization strategy based on model type
    strategy = get_parallelization_strategy(model)
    #   → DefaultParallelizationStrategy for most models
    #   → NemotronHParallelizationStrategy for NemotronH
    #   → WanParallelizationStrategy for WAN
    #   → Custom strategies via registry

    # 2. Delegate to strategy's parallelize method
    return strategy.parallelize(
        model=model,
        device_mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        sequence_parallel=sequence_parallel,
        activation_checkpointing=activation_checkpointing,
        tp_shard_plan=tp_shard_plan,
        dp_replicate_mesh_name="dp_replicate",
        dp_shard_cp_mesh_name="dp_shard_cp",
        tp_mesh_name="tp",
    )
```

**DefaultParallelizationStrategy.parallelize**:
```
1. Extract meshes
   - tp_mesh = device_mesh["tp"]
   - dp_mesh = device_mesh[("dp_replicate", "dp_shard_cp")]

2. Find transformer layers
   - _extract_model_layers(model)
   - Returns: model.model.layers, vision_tower.layers, etc.

3. Apply TP (if tp_size > 1)
   - Validate: attention heads divisible by tp_size
   - Get plan: custom → optimized → HF → default
   - Apply: parallelize_module(model, tp_mesh, tp_plan)

4. Apply activation checkpointing (if enabled)
   - Wrap: mlp, self_attn, layernorms with checkpoint_wrapper

5. Apply FSDP2 recursively
   - apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)
   - Wraps each layer with fully_shard
   - Last layer: reshard_after_forward=False

6. Apply FSDP2 to root
   - fully_shard(model, mesh=dp_mesh, reshard_after_forward=False, ...)

7. Return parallelized model
```

---

## Advanced Features

### 1. Hybrid Sharded Data Parallelism (HSDP)

**Configuration**:
```python
manager = FSDP2Manager(
    dp_size=16,
    dp_replicate_size=2,  # HSDP: 2 replicate groups × 8 shard groups
    tp_size=2,
)
# Automatically computes: dp_shard_size = 16 / 2 = 8
```

**Benefit**: Balances communication cost vs memory savings
- **Pure FSDP** (`dp_replicate_size=1`): Maximum memory savings, high communication cost
- **Pure DDP** (not supported in FSDP2): Minimum memory savings, low communication cost
- **HSDP**: Middle ground - replicate across `dp_replicate_size` groups, shard within each group

**Communication Pattern**:
```
Gradient Reduction:
1. Reduce-scatter within each dp_shard group (8 GPUs)
2. All-reduce across dp_replicate groups (2 groups)

vs Pure FSDP (dp_replicate_size=1):
1. Reduce-scatter across all 16 GPUs (slower due to larger process group)
```

### 2. Deferred Gradient Synchronization

**Configuration**: `fsdp2.py:131-134`
```python
defer_fsdp_grad_sync: Optional[bool] = field(
    default=True,
    metadata={"help": "Defer FSDP gradient sync to only the final micro-batch before the optimizer step"}
)
```

**Use Case**: Gradient accumulation with micro-batches
```python
# Without defer: gradients synced after every micro-batch (wasteful)
for micro_batch in micro_batches:
    loss = model(micro_batch)
    loss.backward()  # ← Gradient sync happens here (expensive!)

optimizer.step()

# With defer: gradients synced only before optimizer.step()
with model.no_sync():  # Disable auto sync
    for micro_batch in micro_batches[:-1]:
        loss = model(micro_batch)
        loss.backward()  # ← No sync

# Last micro-batch: sync happens
loss = model(micro_batches[-1])
loss.backward()  # ← Sync only here

optimizer.step()
```

**Benefit**: 3-5× speedup for gradient accumulation

### 3. Sequence Parallelism Integration

**When enabled** (`sequence_parallel=True`):
- TP plan includes `SequenceParallel` styles for layernorms and projections
- Output layouts set to `Shard(1)` (shard along sequence dimension)
- Requires `AllGather` before operations that need full sequence

**Example TP Plan with SP**: `parallelizer.py:901-911`
```python
if sequence_parallel:
    base_model_sp_plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1)  # Output sharded on seq dim
        ),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1)  # Keep sequence sharded
        ),
        "model.layers.*.post_attention_layernorm": SequenceParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.lm_head": ColwiseParallel(
            input_layouts=Shard(1),      # Expect sharded input
            output_layouts=Replicate()   # All-gather for logits
        ),
    }
```

**Benefit**: Reduces activation memory by `tp_size` (each GPU stores 1/tp_size of sequence)

### 4. Expert Parallelism for MoE

**Configuration**:
```python
manager = FSDP2Manager(
    dp_size=8,
    cp_size=2,
    ep_size=8,  # 8-way expert parallelism (e.g., Mixtral 8×7B)
)
```

**MoE Mesh**: `fsdp2.py:257-270`
```python
moe_mesh_shape = (pp_size, ep_shard_size, ep_size)
#                  (1,      2,            8)
# ep_shard_size = (dp_size * cp_size) / ep_size = (8 * 2) / 8 = 2
```

**Usage**: Passed to MoE layers for expert distribution
```python
# In MoE layer forward
expert_outputs = []
for expert_id, expert in enumerate(self.experts):
    # Expert is sharded via FSDP on moe_mesh["ep_shard"]
    # Only activated when routing sends tokens to this GPU's expert
    if expert_id in active_experts:
        expert_outputs.append(expert(tokens))
```

### 5. Activation Checkpointing

**Configuration**: `fsdp2.py:126-129`
```python
activation_checkpointing: Optional[bool] = field(
    default=False,
    metadata={"help": "Enable activation checkpointing if True. Applies to linear layers."}
)
```

**Implementation**: `parallelizer.py:158-181`
```python
if activation_checkpointing:
    # Disable KV caching for deterministic recomputation
    model.config.use_cache = False

    for i, layer in enumerate(layers):
        # Checkpoint MLP and attention (most memory-intensive)
        if hasattr(layer, "mlp"):
            layers[i].mlp = checkpoint_wrapper(layer.mlp)
        if hasattr(layer, "self_attn"):
            layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)

        # Checkpoint layernorms (cheap but consistent with checkpointing all layer ops)
        if hasattr(layer, "input_layernorm"):
            layers[i].input_layernorm = checkpoint_wrapper(layers[i].input_layernorm)
        if hasattr(layer, "post_attention_layernorm"):
            layers[i].post_attention_layernorm = checkpoint_wrapper(layers[i].post_attention_layernorm)
```

**Trade-off**:
- **Memory savings**: 40-50% activation memory reduction
- **Speed cost**: 20-30% slower (recompute activations in backward)
- **When to use**: Training very large models where activation memory is bottleneck

---

## Production Considerations

### 1. Model Validation

**Attention Head Validation**: `parallelizer.py:629-699`
```python
def validate_tp_mesh(model, tp_mesh):
    """Validate that attention heads and key value heads are divisible by TP size."""
    if tp_mesh.size() == 1:
        return  # No validation needed for TP=1

    # Extract num_attention_heads and num_key_value_heads
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    # Validate divisibility
    assert num_key_value_heads % tp_mesh.size() == 0, \
        f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
    assert num_attention_heads % tp_mesh.size() == 0, \
        f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
```

**Why**: TP splits attention heads across GPUs. If `num_heads % tp_size != 0`, some GPUs get fractional heads (invalid).

**Example Error**:
```python
# Llama-3.1-8B: num_attention_heads=32, num_key_value_heads=8
manager = FSDP2Manager(tp_size=6)  # INVALID: 32 % 6 != 0, 8 % 6 != 0

# Valid TP sizes: 1, 2, 4, 8 (divisors of both 32 and 8)
```

### 2. Layer Extraction Robustness

**Heuristic Fallback**: `parallelizer.py:701-743`
```python
def _extract_model_layers(model: nn.Module) -> List[nn.Module]:
    # 1. Try known model types (VLMs, LLMs)
    if type(model) in MODEL_CLS_TO_LAYERS:
        return _reduce_attrs(model, MODEL_CLS_TO_LAYERS[type(model)])

    # 2. Try common patterns (model.model.layers, model.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)

    # 3. Fallback: find largest ModuleList heuristically
    else:
        logger.warning(f"Unknown model type: {type(model)}. Using heuristic.")
        largest_module_list = _find_largest_module_list(model)
        if largest_module_list is None:
            raise ValueError(f"Unknown model type and no ModuleList found")
        return list(largest_module_list)
```

**Benefit**: Works with custom models without requiring code changes

### 3. TP Plan Selection Priority

**Priority Order**: `parallelizer.py:825-915`
```
1. Custom TP plan (user-provided dict or import path)
   ↓ (if not provided)
2. use_hf_tp_plan=True (use HuggingFace _tp_plan attribute)
   ↓ (if not requested)
3. Optimized plan (PARALLELIZE_FUNCTIONS registry, model-specific)
   ↓ (if not in registry OR fails)
4. HuggingFace plan (fallback from optimized)
   ↓ (if HF plan not available)
5. Default base plan (generic Llama-style TP plan)
```

**Code**: `parallelizer.py:825-915`
```python
def _get_parallel_plan(model, sequence_parallel, tp_shard_plan, use_hf_tp_plan):
    # 1. Custom plan
    if isinstance(tp_shard_plan, dict):
        return tp_shard_plan
    elif tp_shard_plan is not None:
        return import_class_from_path(tp_shard_plan)

    # 2. Explicit HF plan
    elif use_hf_tp_plan:
        return get_hf_tp_shard_plan(model)

    # 3. Optimized plan (try, fallback to HF on failure)
    elif type(model) in PARALLELIZE_FUNCTIONS:
        try:
            func = PARALLELIZE_FUNCTIONS[type(model)]
            return func(model, sequence_parallel)
        except Exception as e:
            logger.info(f"Optimized plan failed: {e}. Falling back to HF plan.")
            return get_hf_tp_shard_plan(model)

    # 4. Default base plan
    else:
        base_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        }
        if sequence_parallel:
            base_plan.update(base_model_sp_plan)
        return base_plan
```

### 4. Meta Device Support

**Recommended Pattern**:
```python
# Load on meta device to avoid materialization spike
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-405B",
    torch_dtype=torch.bfloat16,
    device_map="meta",  # Parameters are placeholders (no memory allocated)
)

# Parallelize (FSDP materializes sharded parameters on GPU)
model = manager.parallelize(model)
# Each GPU only materializes its shard (405B / 128 GPUs = ~3B per GPU)
```

**Alternative (Not Recommended for Large Models)**:
```python
# Loads full model on CPU/GPU (405B * 2 bytes = 810GB!)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-405B",
    torch_dtype=torch.bfloat16,
)
# Then shards, but causes massive memory spike
model = manager.parallelize(model)
```

### 5. Unshard Utility for Inference

**Use Case**: Logprob inference requires unsharded parameters

**File**: `parallelizer.py:1108-1120`
```python
@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Generator[None, None, None]:
    """Explicitly unshard and then reshard the FSDP2 modules."""
    try:
        # All-gather all FSDP modules
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        # Reshard after use
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
```

**Usage**:
```python
# During training: parameters are sharded
model.train()
loss = model(batch).loss
loss.backward()

# During inference: temporarily unshard
model.eval()
with unshard_fsdp2_model(model):
    # All parameters now available on each GPU (high memory!)
    logprobs = model.compute_logprobs(batch)

# Parameters automatically resharded after exiting context
```

---

## Summary

NeMo AutoModel's FSDP2 implementation is **production-grade, highly composable, and optimized for large-scale training**. Key strengths:

### Architectural Strengths

1. **Manager Pattern**: Centralized configuration with automatic setup
2. **5D DeviceMesh**: Fine-grained control over PP, HSDP, CP, TP dimensions
3. **Strategy Pattern**: Pluggable parallelization logic per model type
4. **Direct PyTorch APIs**: No abstraction overhead, full FSDP2 feature access

### Technical Optimizations

1. **Last-layer reshard optimization**: Saves one all-gather per backward pass
2. **Submesh specialization**: Separate process groups for data loading, FSDP sharding, loss reduction
3. **HSDP support**: Balances communication cost vs memory savings
4. **Deferred gradient sync**: 3-5× speedup for gradient accumulation
5. **Meta device support**: Avoids materialization spikes for 100B+ models

### Production Features

1. **Validation**: Attention head divisibility, mesh shape consistency
2. **Robustness**: Heuristic layer extraction for unknown models
3. **Flexibility**: 4-level TP plan customization hierarchy
4. **Extensibility**: Registry for custom parallelization strategies
5. **Expert Parallelism**: Native MoE support with separate MoE mesh

### When to Use NeMo AutoModel FSDP2

- **Scale**: Training models >100B parameters
- **Complexity**: Need TP + CP + SP + EP in single run
- **Control**: Require fine-grained DeviceMesh topology control
- **Production**: Building training infrastructure for long-running jobs
- **Customization**: Implementing model-specific parallelization strategies

**Comparison to Alternatives**:
- **vs Axolotl/Accelerate**: More complex, but far more powerful for N-D parallelism
- **vs DeepSpeed**: Native PyTorch (no framework lock-in), better TP/CP integration
- **vs Megatron-LM**: Comparable features, but Megatron has more optimized kernels

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Analysis Based On**: NeMo AutoModel source code (latest commits)
**Source Files**:
- `nemo_automodel/components/distributed/fsdp2.py:34-318`
- `nemo_automodel/components/distributed/parallelizer.py:1-1120`
- `nemo_automodel/components/distributed/optimized_tp_plans.py`
