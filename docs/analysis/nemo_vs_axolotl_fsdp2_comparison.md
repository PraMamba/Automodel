# FSDP2 Implementation Comparison: NeMo AutoModel vs Axolotl

## Executive Summary

Both NeMo AutoModel and Axolotl have adopted PyTorch's FSDP2 (Fully Sharded Data Parallelism 2) for distributed training, but with distinctly different architectural approaches and integration strategies.

**Key Differences:**
- **NeMo AutoModel**: Direct PyTorch FSDP2 implementation with custom parallelization strategies
- **Axolotl**: FSDP2 via HuggingFace Accelerate with monkeypatching for bug fixes and optimizations

## Architecture Comparison

### NeMo AutoModel FSDP2 Architecture

**File**: `nemo_automodel/components/distributed/fsdp2.py`

#### Core Design

NeMo AutoModel implements FSDP2 with a **manager-based architecture**:

```python
@dataclass
class FSDP2Manager:
    dp_size: Optional[int] = None
    dp_replicate_size: Optional[int] = None  # For HSDP
    tp_size: Optional[int] = 1
    cp_size: Optional[int] = 1
    pp_size: Optional[int] = 1
    ep_size: Optional[int] = 1  # Expert parallelism
    sequence_parallel: Optional[bool] = False

    def parallelize(self, model):
        # Direct FSDP2 parallelization
        fsdp2_strategy_parallelize(...)
```

**Key Features:**

1. **N-Dimensional DeviceMesh**
   ```python
   mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
   mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
   device_mesh = init_device_mesh(
       device_type="cuda",
       mesh_shape=mesh_shape,
       mesh_dim_names=mesh_names,
   )
   ```
   - Supports 5-dimensional parallelism: PP (pipeline), DP (data), CP (context), TP (tensor), EP (expert)
   - Creates submeshes for different operations (data loading, param sharding, loss all-reduce)

2. **Strategy Pattern for Parallelization**
   ```python
   class ParallelizationStrategy(ABC):
       @abstractmethod
       def parallelize(self, model, device_mesh, ...):
           pass

   class DefaultParallelizationStrategy(ParallelizationStrategy):
       # For most transformer models

   class NemotronHParallelizationStrategy(ParallelizationStrategy):
       # For Nemotron hybrid models (Mamba + Transformer)

   class WanParallelizationStrategy(ParallelizationStrategy):
       # For diffusion transformer models
   ```
   - Pluggable parallelization strategies per model type
   - Clean separation of concerns
   - Easy to add custom strategies via `register_parallel_strategy` decorator

3. **HSDP (Hybrid Sharded Data Parallelism) Support**
   ```python
   dp_shard_size = dp_size // dp_replicate_size
   # dp_replicate_size = 1 → Pure FSDP
   # dp_replicate_size < dp_size → HSDP with replication + sharding
   ```

4. **Recursive FSDP Sharding**
   ```python
   def apply_fsdp2_sharding_recursively(module, mesh, mp_policy, offload_policy):
       if isinstance(module, nn.ModuleList):
           for layer_id, child_module in enumerate(module):
               # Last layer optimization: don't reshard after forward
               reshard_after_forward = (layer_id < len(module) - 1)
               fully_shard(child_module, reshard_after_forward=reshard_after_forward, ...)
   ```
   - Optimized for transformer layers (ModuleList handling)
   - Last layer doesn't reshard after forward (FSDP prefetch optimization)

### Axolotl FSDP2 Architecture

**File**: `src/axolotl/monkeypatch/accelerate/fsdp2.py`

#### Core Design

Axolotl leverages **HuggingFace Accelerate** with extensive monkeypatching:

```python
def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    fsdp2_plugin = accelerator.state.fsdp_plugin

    # CPU RAM-efficient loading
    if fsdp2_plugin.cpu_ram_efficient_loading:
        original_sd = model.state_dict()
        model = model.to(torch.device("meta"))  # Move to meta device
        model.tie_weights()

    # Apply FSDP wrapping
    auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    for module in get_module_children_bottom_up(model)[:-1]:
        if auto_wrap_policy(module) and not isinstance(module, FSDPModule):
            fully_shard(module, **fsdp2_kwargs)

    fully_shard(model, **fsdp2_kwargs)

    # Broadcast full state dict from rank 0
    if fsdp2_plugin.cpu_ram_efficient_loading:
        fsdp2_load_full_state_dict(accelerator, model, original_sd, ...)
```

**Key Features:**

1. **CPU RAM-Efficient Loading**
   ```python
   if fsdp2_plugin.cpu_ram_efficient_loading:
       # 1. Save state dict (only on CPU initially)
       original_sd = model.state_dict()

       # 2. Move to meta device to avoid GPU memory spike
       model = model.to(torch.device("meta"))

       # 3. Apply FSDP sharding on meta device
       fully_shard(model, ...)

       # 4. Broadcast from rank 0 to all ranks
       fsdp2_load_full_state_dict(accelerator, model, original_sd)
   ```
   - Prevents VRAM spike during model initialization
   - Critical for large models (>70B parameters)
   - Uses `distribute_tensor` to efficiently broadcast sharded parameters

2. **LoRA/QLoRA Integration**
   ```python
   def _process_lora_module_for_fsdp(module, fsdp2_kwargs):
       # Fix dtype mismatch: Linear4Bit keeps bias in fp32
       if module.base_layer.bias is not None:
           if module.base_layer.weight.dtype != module.base_layer.bias.dtype:
               module.base_layer.bias.data = module.base_layer.bias.data.to(
                   module.base_layer.weight.dtype
               )

       # Wrap each LoRA adapter separately
       for active_adapter in module.active_adapters:
           fully_shard(module.lora_A[active_adapter], **fsdp2_kwargs)
           fully_shard(module.lora_B[active_adapter], **fsdp2_kwargs)
   ```
   - Specialized handling for PEFT (Parameter-Efficient Fine-Tuning)
   - Addresses dtype mismatches in quantized base layers

3. **State Dict Broadcasting**
   ```python
   def fsdp2_load_full_state_dict(_accelerator, model, full_sd, offload_to_cpu=False):
       meta_sharded_sd = model.state_dict()  # Get sharded placeholders
       sharded_sd = {}

       for param_name, sharded_meta_param in meta_sharded_sd.items():
           if _accelerator.is_main_process:
               full_tensor = full_sd[param_name]

           # Distribute tensor from rank 0 to all ranks
           sharded_param = distribute_tensor(
               full_tensor,
               device_mesh,
               sharded_meta_param.placements,
               src_data_rank=0,
           )
           sharded_sd[param_name] = nn.Parameter(sharded_param)

       model.load_state_dict(sharded_sd, assign=True)
   ```
   - Broadcasts full checkpoint from rank 0
   - Each rank receives only its sharded portion

4. **Auto-Wrap Policy Adaptation**
   - Uses Accelerate's `fsdp2_prepare_auto_wrap_policy` to convert FSDP1-style policies
   - Supports both `transformer_auto_wrap_policy` and `size_based_auto_wrap_policy`
   - Bottom-up module traversal for consistent wrapping

## Comparison Matrix

| Feature | NeMo AutoModel | Axolotl |
|---------|----------------|---------|
| **Integration** | Direct PyTorch FSDP2 | Via HuggingFace Accelerate |
| **DeviceMesh Dimensions** | 5D (PP, DP_replicate, DP_shard, CP, TP) | 2-3D (typically DP + TP) |
| **HSDP Support** | Native via dp_replicate_size | Via Accelerate |
| **Expert Parallelism** | Built-in (ep_size, ep_shard_size) | Not explicitly mentioned |
| **CPU RAM Efficiency** | Standard loading | Specialized meta device loading |
| **LoRA/PEFT Support** | Not specialized | Deep integration with PEFT library |
| **Parallelization Strategy** | Pluggable strategies per model | Accelerate auto-wrap policies |
| **Initialization** | Manager-based (`FSDP2Manager`) | Function-based (`fsdp2_prepare_model`) |
| **Mixed Precision** | MixedPrecisionPolicy with param/reduce/output dtypes | Same via Accelerate |
| **CPU Offloading** | CPUOffloadPolicy | Same via Accelerate |
| **Activation Checkpointing** | Per-module checkpointing | Apply before FSDP wrapping |

## Implementation Philosophy

### NeMo AutoModel

**Philosophy**: **Composable, modular parallelism framework**

- Designed for production-scale training (100B+ models)
- DeviceMesh as first-class citizen for N-D parallelism
- Clear separation: Strategy pattern for model-specific logic
- Assumes advanced users who configure parallelism explicitly

**Code Example**:
```python
# nemo_automodel/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none  # Auto-infer
  dp_replicate_size: 1
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
```

### Axolotl

**Philosophy**: **User-friendly, configuration-driven training**

- Designed for accessibility and ease of use
- Leverages HuggingFace ecosystem (Accelerate + Transformers)
- Aggressive optimizations for resource-constrained environments
- Assumes users want sensible defaults with minimal configuration

**Code Example**:
```yaml
# axolotl config
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_cpu_ram_efficient_loading: true
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

## Technical Deep Dive

### 1. Mixed Precision Implementation

**NeMo AutoModel**:
```python
# fsdp2.py:107-113
mp_policy: Optional[MixedPrecisionPolicy] = field(
    default=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,  # Gradient reduction in bf16
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    ),
)
```
- Default: All operations in bfloat16
- Configurable per-stage precision (param, reduce, output)

**Axolotl**:
```python
# Via Accelerate config
mixed_precision_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # More stable gradient reduction
    output_dtype=torch.float32,
)
```
- Defaults to fp32 for gradient reduction (numerical stability)
- User can override via Accelerate config

### 2. Submesh Creation for Different Operations

**NeMo AutoModel** creates specialized submeshes:

```python
# fsdp2.py:231-254
# Mesh for data loading (no communication)
dp_mesh = device_mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")

# Mesh for param sharding
dp_shard_cp_mesh = device_mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp_shard_cp")

# Mesh for loss all-reduce
dp_cp_mesh = device_mesh[("dp_replicate", "dp_shard", "cp")]._flatten(mesh_dim_name="dp_cp")
```

**Purpose**:
- `dp_mesh`: DataLoader distribution (no cross-rank comm)
- `dp_shard_cp_mesh`: FSDP parameter sharding
- `dp_cp_mesh`: Loss reduction across DP+CP ranks

**Axolotl**:
- Relies on Accelerate's default submesh handling
- Simpler but less control over specific operations

### 3. Reshard After Forward Optimization

**NeMo AutoModel**:
```python
# parallelizer.py:445-446
for layer_id, child_module in enumerate(module):
    reshard_after_forward = int(layer_id) < len(module) - 1
    fully_shard(child_module, reshard_after_forward=reshard_after_forward, ...)
```
- Last transformer layer: `reshard_after_forward=False`
- Reason: FSDP prefetches the last layer's parameters during backward pass
- Avoiding reshard saves one extra all-gather

**Axolotl**:
- Uses default `reshard_after_forward` from Accelerate
- No explicit last-layer optimization in codebase

## Performance Implications

### Memory Efficiency

1. **HSDP (Hybrid Sharding)**
   - NeMo: Native support via `dp_replicate_size`
   - Axolotl: Via Accelerate configuration
   - Benefit: Balance communication cost vs memory savings

2. **CPU Offloading**
   - Both support via `CPUOffloadPolicy`
   - Axolotl: Additional `pin_memory=False` option for even lower memory

3. **Meta Device Initialization**
   - NeMo: Standard approach (model on GPU, then shard)
   - Axolotl: Meta device trick to avoid VRAM spike
   - **Winner**: Axolotl for very large models on limited GPUs

### Communication Efficiency

1. **Gradient Reduction Precision**
   - NeMo default: bf16 (faster, less accurate)
   - Axolotl default: fp32 (slower, more stable)
   - Trade-off: Speed vs numerical stability

2. **Reshard Optimization**
   - NeMo: Explicit last-layer optimization
   - Axolotl: Relies on FSDP defaults
   - **Winner**: NeMo (slightly)

### Ease of Use

1. **Configuration Complexity**
   - NeMo: Explicit DeviceMesh dimensions (steeper learning curve)
   - Axolotl: Simple YAML flags (easier for beginners)
   - **Winner**: Axolotl

2. **Ecosystem Integration**
   - NeMo: Standalone framework
   - Axolotl: Deep HuggingFace integration
   - **Winner**: Axolotl for HF users, NeMo for custom pipelines

## Recommendations

### Use NeMo AutoModel FSDP2 When:
- Training 100B+ models requiring complex N-D parallelism
- Need fine-grained control over DeviceMesh topology
- Implementing custom parallelization strategies
- Combining DP + TP + CP + PP + EP in same run
- Building production training infrastructure

### Use Axolotl FSDP2 When:
- Fine-tuning pre-trained HuggingFace models
- Limited GPU memory (meta device loading is critical)
- Using PEFT/LoRA/QLoRA (better integration)
- Want minimal configuration (Accelerate defaults)
- Rapid prototyping and experimentation

## Source Code References

### NeMo AutoModel
- FSDP2 Manager: `nemo_automodel/components/distributed/fsdp2.py:34-317`
- Parallelization Strategies: `nemo_automodel/components/distributed/parallelizer.py:87-383`
- DeviceMesh Setup: `nemo_automodel/components/distributed/fsdp2.py:215-255`

### Axolotl
- FSDP2 Preparation: `src/axolotl/monkeypatch/accelerate/fsdp2.py:214-376`
- State Dict Loading: `src/axolotl/monkeypatch/accelerate/fsdp2.py:20-90`
- LoRA Integration: `src/axolotl/monkeypatch/accelerate/fsdp2.py:185-211`

## Conclusion

Both frameworks implement FSDP2 effectively but target different use cases:

- **NeMo AutoModel**: Power and flexibility for large-scale production training
- **Axolotl**: Accessibility and optimization for resource-constrained fine-tuning

Neither is universally "better" - the choice depends on project requirements, team expertise, and training scale.
