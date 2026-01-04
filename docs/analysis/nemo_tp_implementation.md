# NeMo AutoModel Tensor Parallelism (TP) Implementation Deep Dive

## Executive Summary

This document provides a comprehensive source code analysis of how NeMo AutoModel implements Tensor Parallelism (TP) using PyTorch's native DTensor and `parallelize_module` APIs. The implementation features a **4-level customization hierarchy** that balances flexibility with ease of use.

**Core Architecture**:
- **Direct PyTorch DTensor**: Uses PyTorch's native `torch.distributed.tensor` and `ParallelStyle` classes
- **4-Level Hierarchy**: Custom ParallelStyle → Model-specific functions → Layer-specific overrides → Per-parameter annotations
- **Model-Specific Plans**: Optimized TP plans for Llama, Qwen, Gemma3, Phi3 models
- **LoRA Integration**: Specialized ParallelStyle subclasses for LoRA/PEFT compatibility

**Key Features**:
- Colwise/Rowwise parallelism with configurable input/output layouts
- Sequence Parallelism with all-gather optimization
- Custom ParallelStyle classes for special cases (rotary embeddings, QK norms)
- HuggingFace TP plan fallback
- LoRA-aware sharding

**Key Files**:
- `nemo_automodel/components/distributed/optimized_tp_plans.py` - Model-specific TP plans
- `nemo_automodel/components/distributed/parallel_styles.py` - LoRA-compatible ParallelStyle
- `nemo_automodel/components/distributed/parallelizer.py` - TP plan selection logic

---

## Table of Contents

1. [TP Architecture Overview](#tp-architecture-overview)
2. [PyTorch ParallelStyle Classes](#pytorch-parallelstyle-classes)
3. [4-Level TP Customization Hierarchy](#4-level-tp-customization-hierarchy)
4. [Model-Specific TP Plans](#model-specific-tp-plans)
5. [SequenceParallel Optimizations](#sequenceparallel-optimizations)
6. [LoRA Integration](#lora-integration)
7. [TP Plan Selection Logic](#tp-plan-selection-logic)
8. [HuggingFace TP Plan Integration](#huggingface-tp-plan-integration)
9. [DTensor Sharding Mechanics](#dtensor-sharding-mechanics)
10. [Production Considerations](#production-considerations)

---

## TP Architecture Overview

### What is Tensor Parallelism?

**Tensor Parallelism (TP)** splits model weights **horizontally** across multiple GPUs. Each GPU holds a portion of each layer's parameters and computes on a portion of the activations.

**Example**: Linear layer with TP=4
```python
# Original (single GPU)
x = [batch, seq_len, hidden_size]  # [2, 1024, 4096]
weight = [out_features, in_features]  # [4096, 4096]
y = x @ weight.T  # [2, 1024, 4096]

# TP=4 (4 GPUs, Colwise sharding)
# GPU 0: weight[0:1024, :]   → computes y[:, :, 0:1024]
# GPU 1: weight[1024:2048, :] → computes y[:, :, 1024:2048]
# GPU 2: weight[2048:3072, :] → computes y[:, :, 2048:3072]
# GPU 3: weight[3072:4096, :] → computes y[:, :, 3072:4096]
# All GPUs: all-gather to get full y
```

### NeMo's TP Implementation Strategy

NeMo AutoModel uses **PyTorch DTensor** (Distributed Tensor) for TP, which provides:
- **Transparent sharding**: Sharded tensors look like regular tensors to user code
- **Automatic communication**: All-gather, reduce-scatter inserted automatically
- **Flexible layouts**: Configure input/output sharding patterns per layer

**High-Level Flow**:
```
User specifies TP size (e.g., tp_size=4)
  ↓
Select TP plan (custom → optimized → HF → default)
  ↓
Apply TP plan using parallelize_module()
  ↓
For each module in plan:
  - Shard weight/bias parameters
  - Insert communication collectives (all-gather, reduce-scatter)
  - Configure input/output layouts (Shard vs Replicate)
```

### Key PyTorch Primitives

**DTensor**: Distributed tensor with sharding metadata
```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# Create DTensor from local shard
local_tensor = torch.randn(1024, 4096)  # GPU 0's portion
dtensor = DTensor.from_local(
    local_tensor,
    device_mesh,
    placements=[Shard(0)]  # Sharded on dimension 0
)
```

**ParallelStyle**: Defines how to shard a module
```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

# Colwise: shard weight on output dimension (columns)
ColwiseParallel(
    input_layouts=Replicate(),  # Expect replicated input
    output_layouts=Shard(-1),   # Output sharded on last dim
)

# Rowwise: shard weight on input dimension (rows)
RowwiseParallel(
    input_layouts=Shard(-1),    # Expect sharded input
    output_layouts=Replicate(), # Output replicated (all-reduce)
)
```

**parallelize_module**: Apply TP plan to model
```python
from torch.distributed.tensor.parallel import parallelize_module

tp_plan = {
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
}

parallelize_module(model, tp_mesh, tp_plan)
```

---

## PyTorch ParallelStyle Classes

PyTorch provides three core `ParallelStyle` classes for TP:

### 1. ColwiseParallel

**File**: `torch.distributed.tensor.parallel`

**Purpose**: Shard weight matrix **column-wise** (along output dimension)

**Implementation**:
```python
class ColwiseParallel(ParallelStyle):
    """Partition a linear layer column-wise (output features)."""

    def __init__(
        self,
        input_layouts: Placement = Replicate(),
        output_layouts: Placement = Shard(-1),
        use_local_output: bool = True,
    ):
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.use_local_output = use_local_output
```

**Example**: Q/K/V projections in attention
```python
# Original weight shape: [num_heads * head_dim, hidden_size]
# TP=4: Each GPU gets [num_heads/4 * head_dim, hidden_size]

ColwiseParallel(
    input_layouts=Replicate(),  # All GPUs have same input
    output_layouts=Shard(-1),   # Output sharded on feature dim
)

# Forward:
# 1. Input: Replicated across all GPUs
# 2. Each GPU computes: local_output = input @ local_weight.T
# 3. Output: Sharded DTensor (each GPU has different columns)
```

**Use Cases**:
- Q/K/V projections: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`
- MLP up projections: `mlp.up_proj`, `mlp.gate_proj`
- Embedding table: `embed_tokens` (shard vocabulary dimension)

### 2. RowwiseParallel

**Purpose**: Shard weight matrix **row-wise** (along input dimension)

**Implementation**:
```python
class RowwiseParallel(ParallelStyle):
    """Partition a linear layer row-wise (input features)."""

    def __init__(
        self,
        input_layouts: Placement = Shard(-1),
        output_layouts: Placement = Replicate(),
        use_local_output: bool = True,
    ):
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.use_local_output = use_local_output
```

**Example**: Output projection in attention
```python
# Original weight shape: [hidden_size, num_heads * head_dim]
# TP=4: Each GPU gets [hidden_size, num_heads/4 * head_dim]

RowwiseParallel(
    input_layouts=Shard(-1),    # Expect sharded input (from colwise projection)
    output_layouts=Replicate(), # All-reduce output across GPUs
)

# Forward:
# 1. Input: Sharded DTensor (from previous colwise layer)
# 2. Each GPU computes: local_output = local_input @ local_weight.T
# 3. All-reduce outputs to get final result
# 4. Output: Replicated across all GPUs
```

**Use Cases**:
- Attention output projection: `self_attn.o_proj`
- MLP down projection: `mlp.down_proj`
- Embedding: `embed_tokens` (shard embedding dimension)

### 3. SequenceParallel

**Purpose**: Shard tensors along **sequence dimension** to save activation memory

**Implementation**:
```python
class SequenceParallel(ParallelStyle):
    """Replicate parameters, shard activations on sequence dimension."""

    def __init__(self, sequence_dim: int = 1, use_local_output: bool = False):
        self.sequence_dim = sequence_dim
        self.use_local_output = use_local_output
```

**Example**: LayerNorm with sequence parallelism
```python
SequenceParallel()

# Forward:
# 1. Input: Sharded on seq dim [batch, seq_len/tp_size, hidden]
# 2. Parameters: Replicated (same on all GPUs)
# 3. Each GPU normalizes its seq portion
# 4. Output: Sharded on seq dim (no communication needed)
```

**Use Cases**:
- LayerNorms: `input_layernorm`, `post_attention_layernorm`
- RMSNorm: `model.norm`
- Rotary embeddings: `rotary_emb`

**Benefit**: Reduces activation memory by `tp_size` (each GPU stores 1/tp_size of sequence)

---

## 4-Level TP Customization Hierarchy

NeMo AutoModel provides **4 levels of TP customization**, from most specific to most general:

### Hierarchy Overview

```
Level 1 (Most Specific): Custom ParallelStyle Classes
  ↓ (if not provided)
Level 2: Model-Specific Parallelization Functions
  ↓ (if not in registry)
Level 3: Layer-Specific Overrides
  ↓ (if not matched)
Level 4 (Most General): Default Base TP Plan
```

### Level 1: Custom ParallelStyle Classes

**Purpose**: Handle special cases requiring custom input/output processing

**File**: `optimized_tp_plans.py:47-101`

#### Example 1: SequenceParallelAllGatherActivation

**Use Case**: LayerNorm in Llama with sequence parallelism needs to all-gather before next layer

```python
class SequenceParallelAllGatherActivation(SequenceParallel):
    """SequenceParallel that all-gathers activations after processing."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        """Redistribute sharded output to replicated (all-gather)."""
        if isinstance(outputs, DTensor):
            if any(isinstance(p, Shard) for p in outputs.placements):
                # All-gather sharded output
                outputs = outputs.redistribute(
                    device_mesh=device_mesh,
                    placements=[Replicate()]
                )

        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)
```

**Why Needed**:
- LayerNorm outputs are sequence-sharded
- Attention layers expect replicated input
- Need all-gather to transition from sharded → replicated

**Usage in TP Plan**:
```python
# Llama TP plan with SP: optimized_tp_plans.py:168-170
"model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
"model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
```

#### Example 2: RotaryEmbedParallel

**Use Case**: Qwen/Gemma3 rotary embeddings take tuple input (cos, sin)

```python
class RotaryEmbedParallel(SequenceParallel):
    """Custom SequenceParallel for rotary embeddings with tuple inputs."""

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        new_inputs = list(inputs)

        # Shard first input (position embeddings) on sequence dim
        if not isinstance(inputs[0], DTensor):
            new_inputs[0] = DTensor.from_local(
                local_tensor=inputs[0],
                device_mesh=device_mesh,
                placements=sequence_sharding,
                run_check=True,
            )

        # Replicate second input (frequencies)
        if not isinstance(inputs[1], DTensor):
            new_inputs[1] = DTensor.from_local(
                local_tensor=inputs[1],
                device_mesh=device_mesh,
                placements=(Replicate(),),
                run_check=False,
            )

        return type(inputs)(new_inputs)
```

**Why Needed**: Default SequenceParallel expects single tensor input, but rotary embeddings take `(cos_freqs, sin_freqs)` tuple

#### Example 3: Qwen3QKNorm

**Use Case**: Qwen3's Q/K normalization expects sharded input on dimension 2

```python
class Qwen3QKNorm(SequenceParallel):
    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        input_tensor = inputs[0]

        if isinstance(input_tensor, DTensor):
            # Verify already sharded on correct dimension
            assert input_tensor.placements == (Shard(dim=2),)
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # Create DTensor with sequence sharding
            return DTensor.from_local(input_tensor, device_mesh, sequence_sharding, run_check=False)
```

**Why Needed**: QKNorm is applied after Q/K projection (which outputs dim=2 sharded), needs custom input handling

### Level 2: Model-Specific Parallelization Functions

**Purpose**: Provide optimized TP plans for specific model architectures

**File**: `optimized_tp_plans.py:103-315`

#### Llama TP Plan

**Function**: `_parallelize_llama(model, sequence_parallel=False)`

**File**: `optimized_tp_plans.py:146-179`

```python
def _parallelize_llama(model, sequence_parallel=False):
    """Parallelizes a LlamaForCausalLM model."""

    # Base TP plan (TP only, no SP)
    base_model_tp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),

        # Attention projections
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Fused QKV
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),

        # MLP projections
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate+up
        "model.layers.*.mlp.down_proj": RowwiseParallel(),

        # LM head (shard output, keep local)
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    # Sequence Parallel additions (if enabled)
    if sequence_parallel:
        base_model_sp_plan = {
            # Embeddings output sharded on seq dim
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1)  # Shard seq dimension
            ),

            # LayerNorms with all-gather
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),

            # Projections output sharded on seq dim
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),

            # LM head expects sharded input
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False
            ),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return base_model_tp_plan
```

**Key Features**:
- Supports both separate projections (q_proj, k_proj, v_proj) and fused (qkv_proj)
- Supports both separate MLP (up_proj, gate_proj) and fused (gate_up_proj)
- Sequence Parallelism: all-gather after layernorms, keep seq-sharded between layers
- LM head optimization: `use_local_output=False` (keep DTensor for loss computation)

#### Qwen TP Plan

**Function**: `_parallelize_qwen(model, sequence_parallel=False)`

**File**: `optimized_tp_plans.py:182-246`

```python
def _parallelize_qwen(model, sequence_parallel=False):
    """Parallelizes Qwen2/Qwen3 models."""

    if sequence_parallel:
        # Qwen with SP includes Q/K normalization
        base_model_tp_plan = {
            # Standard projections
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),

            # Attention with QK norm
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),

            # QK normalization (Qwen3 only)
            "model.layers.*.self_attn.q_norm": Qwen3QKNorm(),
            "model.layers.*.self_attn.k_norm": Qwen3QKNorm(),

            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(),

            # MLP
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),

            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
        }
    else:
        # Qwen without SP (simpler)
        base_model_tp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        }

    return base_model_tp_plan
```

**Unique Features**:
- QK normalization: Qwen3 normalizes Q/K projections before attention
- Conditional SP: Different plans for `sequence_parallel=True` vs `False`

#### Gemma3 TP Plan

**Function**: `_parallelize_gemma3(model, sequence_parallel=False)`

**File**: `optimized_tp_plans.py:103-143`

```python
def _parallelize_gemma3(model, sequence_parallel=False):
    """Parallelizes Gemma3ForCausalLM and Gemma3ForConditionalGeneration."""

    # Determine model prefix (text-only vs multimodal)
    if isinstance(model, Gemma3ForConditionalGeneration):
        model_prefix = "model.language_model"  # Multimodal: language model is nested
    else:
        model_prefix = "model"  # Text-only

    # Base TP plan
    base_model_tp_plan = {
        f"{model_prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(),
        f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    # SP additions
    if sequence_parallel:
        base_model_sp_plan = {
            f"{model_prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),

            # Rotary embeddings (Gemma3 has both global and local)
            f"{model_prefix}.rotary_emb": RotaryEmbedParallel(use_local_output=True),
            f"{model_prefix}.rotary_emb_local": RotaryEmbedParallel(use_local_output=True),

            # LayerNorms (Gemma3 has pre/post feedforward norms)
            f"{model_prefix}.layers.*.input_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.post_attention_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.pre_feedforward_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.post_feedforward_layernorm": SequenceParallel(),

            f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            f"{model_prefix}.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return base_model_tp_plan
```

**Unique Features**:
- Dual rotary embeddings: `rotary_emb` (global) and `rotary_emb_local` (local attention)
- Extra layernorms: `pre_feedforward_layernorm`, `post_feedforward_layernorm`
- Multimodal support: Different prefix for language model in VLM variant

#### Phi3 TP Plan

**Function**: `_parallelize_phi3(model, sequence_parallel=False)`

**File**: `optimized_tp_plans.py:261-299`

```python
def _parallelize_phi3(model, sequence_parallel=False):
    """Parallelizes Phi3ForCausalLM model."""

    base_model_tp_plan = {
        # Embeddings: replicated (no sharding)
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),

        # FUSED ATTENTION CANNOT BE SHARDED
        # Keep QKV and O projections replicated
        "model.layers.*.self_attn.qkv_proj": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "model.layers.*.self_attn.o_proj": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),

        # SHARD MLP LAYERS ONLY
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
        "model.layers.*.mlp.down_proj": RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=Replicate(),
        ),

        # LM head: shard output
        "lm_head": ColwiseParallel(
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }

    return base_model_tp_plan
```

**Unique Features**:
- **No attention sharding**: Phi3 uses fused attention kernel that cannot be sharded
- **MLP-only TP**: Only shard MLP layers (gate_up_proj, down_proj)
- **No sequence parallelism**: Not supported due to fused attention

**Why Fused Attention Cannot Be Sharded**: Phi3 uses `flash_attn_qkvpacked_func` which requires full Q/K/V tensors on each GPU (no partial attention computation supported)

### Level 3: Layer-Specific Overrides

**Purpose**: Override default plan for specific layers (e.g., lm_head speedup)

**File**: `parallelizer.py:537-541`

```python
# Special handling for lm_head with HF TP plan
for k, v in hf_tp_plan.items():
    # Speed up lm_head: output sharded on vocab dim, keep DTensor
    if (k == "lm_head" or k == "language_model.lm_head") and v == "colwise_rep":
        hf_tp_plan[k] = ColwiseParallel(
            output_layouts=Shard(-1),      # Shard output on vocab dim
            use_local_output=False         # Keep as DTensor for loss computation
        )
    else:
        hf_tp_plan[k] = translate_to_torch_parallel_style(v)
```

**Optimization**: LM head outputs [batch, seq_len, vocab_size]. With TP, each GPU computes a portion of vocab. Keeping output sharded avoids expensive all-gather before cross-entropy loss (which can handle sharded logits).

### Level 4: Default Base TP Plan

**Purpose**: Fallback plan for unknown models (Llama-style architecture)

**File**: `parallelizer.py:888-913`

```python
# Default base plan (Llama-style)
base_model_tp_plan = {
    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(),
    "lm_head": ColwiseParallel(output_layouts=Replicate()),
}

if sequence_parallel:
    base_model_sp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "model.layers.*.post_attention_layernorm": SequenceParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    }
    base_model_tp_plan.update(base_model_sp_plan)
```

**Coverage**: Works for any transformer with Llama-style architecture (separate or fused projections)

### Model Registry

**File**: `optimized_tp_plans.py:303-315`

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

**Usage**: `_get_parallel_plan()` checks if `type(model)` is in this registry

---

## SequenceParallel Optimizations

### AllGather Optimization

**Problem**: Sequence-sharded activations need all-gathering before layers that expect replicated input

**Solution**: `SequenceParallelAllGatherActivation` class

**File**: `optimized_tp_plans.py:47-62`

```python
class SequenceParallelAllGatherActivation(SequenceParallel):
    """SequenceParallel that all-gathers activations after processing."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        """Redistribute sharded DTensor to replicated placement."""
        if isinstance(outputs, DTensor):
            if any(isinstance(p, Shard) for p in outputs.placements):
                # All-gather across TP group
                outputs = outputs.redistribute(
                    device_mesh=device_mesh,
                    placements=[Replicate()]
                )

        # Call parent's prepare_output_fn
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)
```

### Why All-Gather is Needed

**Llama Layer Sequence** (with SP):
```
Input: Replicated [batch, seq_len, hidden]
  ↓
embed_tokens (output_layouts=Shard(1))
  ↓
Output: Sharded [batch, seq_len/tp_size, hidden]
  ↓
input_layernorm (SequenceParallelAllGatherActivation)
  - Input: Sharded on seq dim
  - Normalize each shard
  - Output: All-gather → Replicated
  ↓
Output: Replicated [batch, seq_len, hidden]
  ↓
self_attn (expects replicated input)
  - q_proj/k_proj/v_proj (ColwiseParallel)
  - attention computation
  - o_proj (RowwiseParallel, output_layouts=Shard(1))
  ↓
Output: Sharded [batch, seq_len/tp_size, hidden]
  ↓
post_attention_layernorm (SequenceParallelAllGatherActivation)
  - All-gather → Replicated
  ↓
Output: Replicated [batch, seq_len, hidden]
  ↓
mlp (expects replicated input)
  - gate_proj/up_proj (ColwiseParallel)
  - down_proj (RowwiseParallel, output_layouts=Shard(1))
  ↓
Output: Sharded [batch, seq_len/tp_size, hidden]
```

**Pattern**:
- Embeddings output seq-sharded
- LayerNorm all-gathers → replicated
- Attention/MLP process replicated input, output seq-sharded
- Repeat for each layer

### Memory Savings

**Without SP** (TP=4):
```
Activations per layer: [batch, seq_len, hidden]
Memory per GPU: batch × seq_len × hidden × 4 bytes (fp32) = Full activation
```

**With SP** (TP=4):
```
Activations per layer: [batch, seq_len/4, hidden] (sharded)
Memory per GPU: batch × (seq_len/4) × hidden × 4 bytes = 1/4 activation
All-gather communication: batch × seq_len × hidden × 4 bytes (once per layer)
```

**Trade-off**:
- Memory savings: 4× reduction in activation memory
- Communication cost: All-gather before each attention/MLP block
- Net benefit: For long sequences (>4K tokens), memory savings outweigh communication cost

---

## LoRA Integration

### Why LoRA Needs Special Handling

**Problem**: PEFT LoRA adds low-rank adapters to frozen base weights

**Architecture**:
```
# Base layer (frozen)
output = input @ base_weight.T + base_bias

# LoRA layer (trainable)
lora_output = input @ lora_A @ lora_B
output = base_output + lora_scale * lora_output
```

**TP Challenge**: LoRA_A and LoRA_B must be sharded differently than base weights

### LoRA-Compatible ParallelStyle Classes

**File**: `parallel_styles.py:40-112`

#### ColwiseParallelLora

```python
class ColwiseParallelLora(ColwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # Shard base weight on dimension 0 (colwise)
        for name, param in module.named_parameters():
            if name.endswith("lora_A.weight"):
                # LoRA_A: shard on output dimension (same as base)
                _distribute_param(module.lora_A, "weight", device_mesh, self.src_data_rank, [Shard(0)])
            elif name.endswith("lora_B.weight"):
                # LoRA_B: shard on output dimension (same as base)
                _distribute_param(module.lora_B, "weight", device_mesh, self.src_data_rank, [Shard(0)])
            else:
                # Base weight
                _distribute_param(module, name, device_mesh, self.src_data_rank, [Shard(0)])

        # All-gather LoRA_A output before LoRA_B
        def lora_a_output_hook(module, input, output):
            if isinstance(output, DTensor):
                if any(isinstance(p, Shard) for p in output.placements):
                    output = output.redistribute(device_mesh=output.device_mesh, placements=[Replicate()])
            return output

        if hasattr(module, "lora_A"):
            module.lora_A.register_forward_hook(lora_a_output_hook)
```

**Why All-Gather After LoRA_A**:
```
# With TP=4, ColwiseParallel
Input: Replicated [batch, seq, hidden]
  ↓
lora_A @ input
  - lora_A sharded: [rank/4, hidden]
  - Output: Sharded [batch, seq, rank/4]
  ↓
All-gather (hook)
  - Output: Replicated [batch, seq, rank]
  ↓
lora_B @ lora_a_output
  - lora_B sharded: [hidden/4, rank]
  - Input: Replicated [batch, seq, rank]
  - Output: Sharded [batch, seq, hidden/4]
  ↓
All-gather (standard colwise behavior)
  - Output: Replicated [batch, seq, hidden]
```

#### RowwiseParallelLora

```python
class RowwiseParallelLora(RowwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # Base weight: shard on dimension 1 (rowwise)
        _distribute_param(module, "weight", device_mesh, self.src_data_rank, [Shard(1)])

        # Bias: replicated (all-reduce after forward)
        if getattr(module, "bias", None) is not None:
            _distribute_param(module, "bias", device_mesh, self.src_data_rank, [Replicate()])

        # LoRA adapters: both sharded on dimension 1
        if hasattr(module, "lora_A"):
            _distribute_param(module.lora_A, "weight", device_mesh, self.src_data_rank, [Shard(1)])
            _distribute_param(module.lora_B, "weight", device_mesh, self.src_data_rank, [Shard(1)])
```

**Rowwise LoRA**:
```
# With TP=4, RowwiseParallel
Input: Sharded [batch, seq, hidden/4]
  ↓
lora_A @ local_input
  - lora_A sharded: [rank, hidden/4]
  - Output: Local [batch, seq, rank]
  ↓
All-reduce (implicit in rowwise)
  - Output: Replicated [batch, seq, rank]
  ↓
lora_B @ lora_a_output
  - lora_B sharded: [hidden/4, rank]
  - Output: Local [batch, seq, hidden/4]
  ↓
All-reduce (standard rowwise behavior)
  - Output: Replicated [batch, seq, hidden]
```

### translate_to_lora Function

**File**: `parallel_styles.py:105-112`

```python
def translate_to_lora(plan):
    """Convert ParallelStyle to LoRA-compatible version."""
    CLS_MAP = {
        ColwiseParallel: ColwiseParallelLora,
        RowwiseParallel: RowwiseParallelLora,
        SequenceParallel: SequenceParallelLora,
    }
    plan.__class__ = CLS_MAP.get(type(plan), plan.__class__)
    return plan
```

**Usage**: `parallelizer.py:144-150`
```python
# In DefaultParallelizationStrategy.parallelize()
model_parallel_plan = {
    k: translate_to_lora(v)  # Convert to LoRA-compatible
    for k, v in _get_parallel_plan(model, sequence_parallel, tp_shard_plan, use_hf_tp_plan).items()
}
```

**Benefit**: Automatically handles LoRA layers if present, works for both base models and PEFT models

---

## TP Plan Selection Logic

### Selection Priority

**File**: `parallelizer.py:825-915`

```python
def _get_parallel_plan(model, sequence_parallel, tp_shard_plan, use_hf_tp_plan):
    """Select TP plan with 4-level priority."""

    # 1. Custom plan (user-provided dict or import path)
    if isinstance(tp_shard_plan, dict):
        return tp_shard_plan
    elif tp_shard_plan is not None:
        plan_obj = import_class_from_path(tp_shard_plan)
        if isinstance(plan_obj, FunctionType):
            return plan_obj()  # Call function to get plan
        return plan_obj  # Use plan directly

    # 2. Explicit HuggingFace plan
    elif use_hf_tp_plan:
        assert not sequence_parallel, "SP not supported in HF TP plan"
        return get_hf_tp_shard_plan(model)

    # 3. Optimized plan (model-specific, with fallback to HF)
    elif type(model) in PARALLELIZE_FUNCTIONS:
        try:
            func = PARALLELIZE_FUNCTIONS[type(model)]
            return func(model, sequence_parallel)
        except Exception as e:
            logger.info(f"Optimized plan failed: {e}. Falling back to HF plan.")
            assert not sequence_parallel, "SP not supported in HF TP plan"
            return get_hf_tp_shard_plan(model)

    # 4. Default base plan (Llama-style)
    else:
        base_model_tp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            # ... (full Llama plan)
        }
        if sequence_parallel:
            base_model_sp_plan = {
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                # ... (SP additions)
            }
            base_model_tp_plan.update(base_model_sp_plan)
        return base_model_tp_plan
```

### Priority Breakdown

**Priority 1: Custom Plan** (Highest)
```python
# Option A: Dict
manager = FSDP2Manager(
    tp_size=4,
    custom_tp_plan={
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    }
)

# Option B: Import path to dict
manager = FSDP2Manager(
    tp_size=4,
    custom_tp_plan="myproject.tp_plans.custom_llama_plan"
)

# Option C: Import path to function
manager = FSDP2Manager(
    tp_size=4,
    custom_tp_plan="myproject.tp_plans.generate_custom_plan"
)
```

**Priority 2: Explicit HF Plan**
```python
manager = FSDP2Manager(
    tp_size=4,
    use_hf_tp_plan=True  # Use model._tp_plan attribute
)
```

**Priority 3: Optimized Plan**
```python
# Automatic: if type(model) in PARALLELIZE_FUNCTIONS
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
manager = FSDP2Manager(tp_size=4)
# Uses _parallelize_llama() automatically
```

**Priority 4: Default Base Plan** (Fallback)
```python
# For unknown model types
model = CustomTransformer()  # Not in PARALLELIZE_FUNCTIONS
manager = FSDP2Manager(tp_size=4)
# Uses default Llama-style plan
```

### Plan Application

**File**: `parallelizer.py:138-155`

```python
# In DefaultParallelizationStrategy.parallelize()
if tp_mesh.size() > 1:
    # Validate attention heads divisible by TP size
    validate_tp_mesh(model, tp_mesh)

    # Get TP plan (4-level priority)
    model_parallel_plan = {
        k: translate_to_lora(v)  # Convert to LoRA-compatible
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
```

---

## HuggingFace TP Plan Integration

### HF _tp_plan Attribute

HuggingFace models (transformers >= 4.51) include a `_tp_plan` attribute with TP sharding information.

**Example**: Llama model
```python
model._tp_plan = {
    "model.embed_tokens": "rowwise_rep",
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
    "lm_head": "colwise_rep",
}
```

### Translation to PyTorch ParallelStyle

**File**: `parallelizer.py:460-603`

```python
def get_hf_tp_shard_plan(model):
    """Extract and translate HuggingFace TP plan."""

    # Handle VLM models (nested language model)
    if type(model) in [Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"
    elif type(model) == Gemma3ForConditionalGeneration:
        inner_model = model.language_model
        model_prefix = "language_model"
    else:
        inner_model = model.model
        model_prefix = "model"

    # Collect TP plans from class, instance, inner model
    hf_tp_plan = {}
    if hasattr(type(model), "_tp_plan") and type(model)._tp_plan is not None:
        hf_tp_plan.update(type(model)._tp_plan)
    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)
    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update({f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()})

    assert len(hf_tp_plan) > 0, "HF TP plan not supported for this model"

    # Add embed_tokens if missing (HF omits it)
    if f"{model_prefix}.embed_tokens" not in hf_tp_plan:
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    # Translate string styles to ParallelStyle objects
    for k, v in hf_tp_plan.items():
        # Special optimization for lm_head
        if (k == "lm_head" or k == "language_model.lm_head") and v == "colwise_rep":
            hf_tp_plan[k] = ColwiseParallel(
                output_layouts=Shard(-1),
                use_local_output=False
            )
        else:
            hf_tp_plan[k] = translate_to_torch_parallel_style(v)

    return hf_tp_plan
```

### String Style Translation

**File**: `parallelizer.py:582-603`

```python
def translate_to_torch_parallel_style(style: str):
    """Translate HF string style to PyTorch ParallelStyle."""

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        # Colwise with replicated output (all-gather)
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        # Rowwise with replicated input (no communication needed)
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")
```

**Mapping**:
| HF String | PyTorch ParallelStyle |
|-----------|----------------------|
| `"colwise"` | `ColwiseParallel()` |
| `"rowwise"` | `RowwiseParallel()` |
| `"colwise_rep"` | `ColwiseParallel(output_layouts=Replicate())` |
| `"rowwise_rep"` | `RowwiseParallel(input_layouts=Replicate())` |
| `"sequence_parallel"` | `SequenceParallel()` |

---

## DTensor Sharding Mechanics

### How DTensor Works

**DTensor** is PyTorch's distributed tensor abstraction that makes sharded tensors look like regular tensors.

**Key Concepts**:
- **DeviceMesh**: Logical topology of GPUs
- **Placement**: How tensor is distributed (`Shard(dim)` or `Replicate()`)
- **Local Tensor**: Portion of data on current GPU
- **Global Tensor**: Full logical tensor across all GPUs

**Example**:
```python
# 4 GPUs in 1D mesh
device_mesh = init_device_mesh("cuda", mesh_shape=(4,))

# Full tensor (logical)
global_tensor = torch.randn(1024, 4096)

# Shard on dimension 0
dtensor = DTensor.from_local(
    local_tensor=global_tensor[rank*256:(rank+1)*256, :],  # Each GPU gets 256 rows
    device_mesh=device_mesh,
    placements=[Shard(0)]
)

# DTensor behaves like full tensor in user code
output = model(dtensor)  # Model sees [1024, 4096], but actually sharded
```

### Colwise Sharding Mechanics

**Goal**: Shard linear layer weight on output dimension (columns)

**Weight Sharding**:
```python
# Original weight: [out_features, in_features] = [4096, 4096]
# TP=4: Each GPU gets [1024, 4096]

# GPU 0: weight[0:1024, :]
# GPU 1: weight[1024:2048, :]
# GPU 2: weight[2048:3072, :]
# GPU 3: weight[3072:4096, :]
```

**Forward Pass**:
```python
# Input: Replicated [batch, seq, in_features]
# Weight: Sharded [out_features/4, in_features]

# 1. Local computation (each GPU)
local_output = input @ local_weight.T  # [batch, seq, out_features/4]

# 2. Create sharded DTensor
output = DTensor.from_local(local_output, device_mesh, [Shard(-1)])
# Output: Sharded DTensor [batch, seq, out_features]
```

**Backward Pass**:
```python
# Grad output: Sharded [batch, seq, out_features/4]

# 1. Gradient w.r.t. input (all-reduce)
grad_input = grad_output @ local_weight  # [batch, seq, in_features]
# All-reduce across TP group to get full grad_input

# 2. Gradient w.r.t. weight (local)
grad_weight = grad_output.T @ input  # [out_features/4, in_features]
# No communication needed (already sharded correctly)
```

### Rowwise Sharding Mechanics

**Goal**: Shard linear layer weight on input dimension (rows)

**Weight Sharding**:
```python
# Original weight: [out_features, in_features] = [4096, 4096]
# TP=4: Each GPU gets [4096, 1024]

# GPU 0: weight[:, 0:1024]
# GPU 1: weight[:, 1024:2048]
# GPU 2: weight[:, 2048:3072]
# GPU 3: weight[:, 3072:4096]
```

**Forward Pass**:
```python
# Input: Sharded [batch, seq, in_features/4]
# Weight: Sharded [out_features, in_features/4]

# 1. Local computation
local_output = input @ local_weight.T  # [batch, seq, out_features]

# 2. All-reduce across TP group
output = all_reduce(local_output)  # [batch, seq, out_features]
# Output: Replicated [batch, seq, out_features]
```

**Backward Pass**:
```python
# Grad output: Replicated [batch, seq, out_features]

# 1. Gradient w.r.t. input (local)
grad_input = grad_output @ local_weight  # [batch, seq, in_features/4]
# Already sharded correctly

# 2. Gradient w.r.t. weight (reduce-scatter)
grad_weight = grad_output.T @ input  # [out_features, in_features/4]
# Each GPU computes its shard
```

### Communication Costs

**Colwise Parallelism**:
```
Forward:  No communication (output stays sharded)
Backward: All-reduce grad_input (tensor size: batch × seq × hidden)
```

**Rowwise Parallelism**:
```
Forward:  All-reduce output (tensor size: batch × seq × hidden)
Backward: No communication (grad_input stays sharded)
```

**TP Pattern** (Colwise → Rowwise):
```
Input (replicated)
  ↓ (no communication)
Colwise layer
  ↓ (output sharded)
  ↓ (no communication needed - rowwise expects sharded input)
Rowwise layer
  ↓ (all-reduce in forward)
Output (replicated)

Total communication: 1 all-reduce per layer pair
```

**Why This is Efficient**: Colwise output sharding matches rowwise input sharding → no intermediate all-gather needed

---

## Production Considerations

### 1. Attention Head Validation

**Requirement**: `num_attention_heads % tp_size == 0` and `num_key_value_heads % tp_size == 0`

**File**: `parallelizer.py:629-699`

```python
def validate_tp_mesh(model, tp_mesh):
    """Validate attention heads divisible by TP size."""

    if tp_mesh.size() == 1:
        return  # No validation for TP=1

    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    assert num_key_value_heads % tp_mesh.size() == 0, \
        f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"

    assert num_attention_heads % tp_mesh.size() == 0, \
        f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
```

**Why**: Q/K/V projections are sharded on head dimension. If heads aren't divisible by TP size, some GPUs would get fractional heads (invalid).

**Example**:
```python
# Llama-3.1-8B: num_attention_heads=32, num_key_value_heads=8
# Valid TP sizes: 1, 2, 4, 8 (divisors of 8)
# Invalid TP sizes: 3, 5, 6, 16 (not divisors)

manager = FSDP2Manager(tp_size=6)  # ERROR: 8 % 6 != 0
```

### 2. Fused Attention Constraints

**Phi3 Example**: Flash Attention with fused QKV cannot be sharded

**File**: `optimized_tp_plans.py:261-299`

```python
# Phi3: fused attention cannot be sharded
base_model_tp_plan = {
    # Keep attention replicated (no TP)
    "model.layers.*.self_attn.qkv_proj": RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=Replicate(),
    ),
    "model.layers.*.self_attn.o_proj": ColwiseParallel(
        input_layouts=Replicate(),
        output_layouts=Replicate(),
    ),

    # Only shard MLP
    "model.layers.*.mlp.gate_up_proj": ColwiseParallel(...),
    "model.layers.*.mlp.down_proj": RowwiseParallel(...),
}
```

**Why**: `flash_attn_qkvpacked_func` requires full Q/K/V tensors on each GPU (doesn't support partial attention)

### 3. LM Head Optimization

**Standard Approach** (wasteful):
```python
# lm_head outputs [batch, seq_len, vocab_size=32000]
# With TP=4, each GPU computes vocab_size/4=8000 logits
# All-gather to get full [batch, seq_len, 32000]
# Compute cross-entropy loss
```

**Optimized Approach** (NeMo):
```python
# lm_head: ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
# Output: Sharded DTensor [batch, seq_len, vocab_size] (no all-gather)
# Cross-entropy: Handles sharded logits directly (each GPU computes loss for its vocab shard)
# All-reduce loss scalar (much cheaper than all-gather logits)
```

**Benefit**: For large vocabularies (100K+), saves massive communication (GB of logits → single scalar)

### 4. Sequence Parallelism Trade-offs

**When to Enable SP**:
```python
# Long sequences (>4K tokens)
manager = FSDP2Manager(
    tp_size=4,
    sequence_parallel=True  # Save activation memory
)
```

**When to Disable SP**:
```python
# Short sequences (<2K tokens)
manager = FSDP2Manager(
    tp_size=4,
    sequence_parallel=False  # Avoid all-gather overhead
)
```

**Memory vs Communication**:
```
SP enabled:
- Activation memory: 1/tp_size
- Communication: All-gather per layer (expensive)
- Net benefit: For seq_len > 4K

SP disabled:
- Activation memory: Full (no reduction)
- Communication: Only TP all-reduces (cheaper)
- Net benefit: For seq_len < 2K
```

### 5. LoRA Compatibility

**Automatic Handling**:
```python
# Base model or PEFT model both work
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# OR
model = PeftModel.from_pretrained(base_model, "llama-lora-adapter")

# TP plan automatically converts to LoRA-compatible
manager = FSDP2Manager(tp_size=4)
model = manager.parallelize(model)
# translate_to_lora() applied automatically
```

**No User Code Changes**: LoRA sharding handled transparently

### 6. Custom TP Plans for Out-of-Tree Models

**Example**: Custom transformer with novel architecture

```python
# Define custom TP plan
def my_custom_tp_plan(model, sequence_parallel=False):
    return {
        "backbone.embed": RowwiseParallel(input_layouts=Replicate()),
        "backbone.blocks.*.custom_attn.proj_qkv": ColwiseParallel(),
        "backbone.blocks.*.custom_attn.proj_out": RowwiseParallel(),
        "head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

# Use custom plan
manager = FSDP2Manager(
    tp_size=4,
    custom_tp_plan=my_custom_tp_plan
)
```

**Benefit**: Works for any model architecture without modifying NeMo code

---

## Summary

NeMo AutoModel's TP implementation is **production-grade, highly customizable, and deeply integrated with PyTorch DTensor**.

### Key Strengths

1. **4-Level Customization Hierarchy**
   - Custom ParallelStyle classes for special cases
   - Model-specific optimized plans (Llama, Qwen, Gemma3, Phi3)
   - Layer-specific overrides (lm_head optimization)
   - Default base plan (Llama-style fallback)

2. **Native PyTorch Integration**
   - Direct DTensor usage (no abstraction overhead)
   - Leverages `parallelize_module` API
   - Transparent sharding (model code unchanged)

3. **Advanced Optimizations**
   - SequenceParallel with AllGather optimization
   - LM head sharded output (avoid logits all-gather)
   - LoRA-compatible sharding (automatic PEFT support)
   - Fused projection support (qkv_proj, gate_up_proj)

4. **Production Features**
   - Attention head validation (fail-fast)
   - HuggingFace TP plan fallback
   - Custom plan support (dict or import path)
   - Model-agnostic default plan

### Comparison to Alternatives

**vs HuggingFace Accelerate**:
- ✅ More control (4-level hierarchy vs auto-only)
- ✅ Optimized plans (model-specific vs generic)
- ✅ SequenceParallel AllGather optimization
- ❌ More complex (steeper learning curve)

**vs Megatron-LM**:
- ✅ PyTorch-native (no framework lock-in)
- ✅ DTensor transparency (cleaner code)
- ❌ Megatron has more kernel optimizations (fused kernels)

**vs DeepSpeed**:
- ✅ Simpler (PyTorch API vs custom abstractions)
- ✅ Better documentation
- ❌ DeepSpeed has ZeRO integration

### When to Use NeMo TP

**Ideal For**:
- Training large models (>70B) with TP+DP+CP
- Custom model architectures needing custom TP plans
- Production training infrastructure
- Models with complex attention patterns (rotary, QK norm, etc.)

**Not Ideal For**:
- Single-GPU training (TP overhead unnecessary)
- Small models (<7B) where DP is sufficient
- Rapid prototyping (HF Accelerate simpler)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Analysis Based On**: NeMo AutoModel source code (latest commits)
**Source Files**:
- `nemo_automodel/components/distributed/optimized_tp_plans.py:1-316`
- `nemo_automodel/components/distributed/parallel_styles.py:1-113`
- `nemo_automodel/components/distributed/parallelizer.py:460-915`
