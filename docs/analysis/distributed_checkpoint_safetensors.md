# NeMo AutoModel Distributed Checkpoint with SafeTensors Output

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [DCP Integration](#dcp-integration)
5. [SafeTensors Format](#safetensors-format)
6. [HuggingFace Storage Integration](#huggingface-storage-integration)
7. [Consolidation Mechanism](#consolidation-mechanism)
8. [PEFT Checkpointing](#peft-checkpointing)
9. [Async Checkpointing](#async-checkpointing)
10. [Mesh-Aware Checkpointing](#mesh-aware-checkpointing)
11. [Complete Workflow](#complete-workflow)

---

## Overview

NeMo AutoModel implements a sophisticated distributed checkpointing system that combines:

1. **PyTorch Distributed Checkpoint Protocol (DCP)**: For sharded, mesh-aware state management
2. **SafeTensors Format**: For secure, fast tensor serialization
3. **HuggingFace Compatibility**: For seamless integration with HF ecosystem
4. **Automatic Consolidation**: For creating single-file checkpoints from sharded outputs

**Key Characteristics**:
- Fully mesh-aware (respects DP/TP/PP/CP/EP dimensions)
- Supports both sharded and consolidated checkpoint formats
- Async checkpoint support (PyTorch >= 2.9.0)
- Special handling for PEFT adapters
- Automatic state dict adaptation between custom and HF formats

**Source Files**:
- Core: `nemo_automodel/components/checkpoint/checkpointing.py`
- Stateful Wrappers: `stateful_wrappers.py`
- HF Storage: `_backports/hf_storage.py`
- Consolidation: `_backports/consolidate_hf_safetensors.py`
- Addons: `addons.py`

---

## Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────┐
│  Training Recipe (train_ft.py)                     │
│  - Calls checkpointer.save_model() at intervals    │
└─────────────────────┬──────────────────────────────┘
                      │
                      ↓
┌────────────────────────────────────────────────────┐
│  Checkpointer (checkpointing.py)                   │
│  - Orchestrates save/load operations               │
│  - Manages addons (PEFT, consolidated HF)          │
│  - Handles async operations                        │
└─────────────────────┬──────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ↓           ↓           ↓
   ┌──────────┐  ┌─────────┐  ┌────────────┐
   │ModelState│  │OptimizerState│  │Addons│
   │Wrapper   │  │Wrapper   │  │(PEFT,HF) │
   └─────┬────┘  └────┬────┘  └────┬───────┘
         │            │            │
         ↓            ↓            ↓
┌────────────────────────────────────────────────────┐
│  PyTorch DCP (torch.distributed.checkpoint)        │
│  - dcp.save() / dcp.load()                         │
│  - dcp.async_save() (torch >= 2.9.0)               │
└─────────────────────┬──────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
┌──────────────┐  ┌────────┐  ┌─────────────────┐
│HFStorageWriter│  │Metadata│  │HFStorageReader  │
│(sharded .st) │  │Manager │  │(load sharded .st)│
└──────┬───────┘  └────────┘  └─────────────────┘
       │
       ↓
┌────────────────────────────────────────────────────┐
│  Consolidation (consolidate_hf_safetensors.py)     │
│  - Merges sharded .safetensors → single file       │
│  - Writes model.safetensors.index.json             │
└────────────────────────────────────────────────────┘
```

### Checkpoint Directory Structure

**Sharded Checkpoint** (model parallelism):
```
checkpoint/epoch_0_step_100/
├── model/
│   ├── shard-00001-model-00001-of-00001.safetensors  # Rank 0 shard
│   ├── shard-00002-model-00001-of-00001.safetensors  # Rank 1 shard
│   ├── .hf_metadata/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── special_tokens_map.json
│   │   └── fqn_to_file_index_mapping.json
│   └── .metadata                                     # DCP metadata
├── optim/
│   ├── __0_0.distcp                                  # Optimizer shard rank 0
│   ├── __1_0.distcp                                  # Optimizer shard rank 1
│   └── .metadata                                     # DCP metadata
├── step_scheduler.pt
├── dataloader/
│   ├── dataloader_dp_rank_0.pt
│   └── dataloader_dp_rank_1.pt
├── rng/
│   ├── rng_dp_rank_0.pt
│   └── rng_dp_rank_1.pt
├── config.yaml
└── losses.json
```

**Consolidated Checkpoint** (HuggingFace format):
```
checkpoint/epoch_0_step_100/model/consolidated/
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json                     # Weight map
├── config.json
├── generation_config.json
└── tokenizer_config.json
```

**PEFT Checkpoint**:
```
checkpoint/epoch_0_step_100/model/
├── adapter_model.safetensors                        # LoRA adapters (rank 0 only)
├── adapter_config.json                              # HF PEFT config
├── automodel_peft_config.json                       # AutoModel PEFT config
├── tokenizer_config.json
└── tokenizer.json
```

---

## Core Components

### 1. CheckpointingConfig

**Source**: `checkpointing.py:78-111`

```python
@dataclass
class CheckpointingConfig:
    """Configuration for checkpointing behavior."""

    enabled: bool                                    # Enable checkpointing
    checkpoint_dir: str | Path                       # Output directory
    model_save_format: str                           # "safetensors" or "torch"
    model_cache_dir: str | Path                      # HF cache directory
    model_repo_id: str                               # HF model ID
    save_consolidated: bool                          # Create consolidated checkpoint
    is_peft: bool                                    # PEFT adapter checkpointing
    model_state_dict_keys: list[str] = None          # Keys to save
    is_async: bool = False                           # Async save (torch >= 2.9.0)
    dequantize_base_checkpoint: bool | None = None   # Dequantize on load
    original_model_root_dir: str | None = None       # Original HF model path
    skip_task_head_prefixes_for_base_model: list[str] | None = None  # Skip task heads
```

**Key Fields**:
- `model_save_format`: Converted to `SerializationFormat.SAFETENSORS` internally
- `save_consolidated`: If `True`, runs consolidation after sharded save
- `is_peft`: Enables special PEFT handling (rank-0 only save)
- `is_async`: Enables async checkpoint for faster saves (requires PyTorch >= 2.9.0)

### 2. Checkpointer Class

**Source**: `checkpointing.py:127-164`

```python
class Checkpointer:
    """Main checkpoint manager."""

    def __init__(
        self,
        config: CheckpointingConfig,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        moe_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        self.config = config
        self.moe_mesh = moe_mesh
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank

        # Async checkpointing context
        self._model_ctx = _AsyncSaveContext(...)
        self._optim_ctx = _AsyncSaveContext(...)

        if self.config.is_async:
            # Setup async stager and process group for async saves
            self._model_ctx.stager = DefaultStager()
            self._optim_ctx.stager = DefaultStager()
            self._model_ctx.process_group = torch.distributed.new_group(backend="gloo")
            self._optim_ctx.process_group = torch.distributed.new_group(backend="gloo")

        # Setup addons
        self._addons = []
        if self._should_write_hf_metadata():
            self._addons.append(ConsolidatedHFAddon())
        if self.config.is_peft:
            self._addons.append(PeftAddon())
```

**Key Responsibilities**:
- Orchestrate save/load operations
- Manage async checkpointing contexts
- Coordinate addon execution (pre-save, post-save hooks)
- Track distributed ranks for mesh-aware operations

**Mesh Awareness**:
```python
# Checkpointer is instantiated with rank information
checkpointer = Checkpointer(
    config=cfg,
    dp_rank=dist_env.dp_rank,
    tp_rank=dist_env.tp_rank,
    pp_rank=dist_env.pp_rank,
    moe_mesh=fsdp2_manager.moe_mesh if hasattr(fsdp2_manager, 'moe_mesh') else None,
)
```

### 3. ModelState Wrapper

**Source**: `stateful_wrappers.py:59-193`

```python
class ModelState:
    """Stateful wrapper for model checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        is_peft: bool = False,
        is_init_step: bool = False,
        skip_task_head_prefixes: list[str] | None = None,
    ):
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.is_tied_lm_head = is_tied_word_embeddings(self.model[0])

        if self.is_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            self.lm_head_param_name = lm_head_param_name

        self.is_peft = is_peft
        self.is_init_step = is_init_step
        self.skip_task_head_prefixes = skip_task_head_prefixes or []

    def state_dict(self) -> dict[str, Any]:
        """Get model state dict with proper handling for PEFT and PP models."""
        if self.is_init_step:
            return self._get_base_model_state_dict()

        options = None
        if self.is_peft:
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True
            )

        # Use PyTorch DCP state dict APIs
        func = partial(get_model_state_dict, options=options)
        model_state_dict = {k: v for sd in map(func, self.model) for k, v in sd.items()}

        # Remove tied lm_head weight (it's the same as embeddings)
        if self.is_tied_lm_head:
            model_state_dict.pop(self.lm_head_param_name, None)

        # Add PEFT prefix for HF compatibility
        if self.is_peft:
            _add_outer_prefix(model_state_dict, "base_model.model.")

        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load state dict with proper handling for PEFT and PP models."""
        if self.is_init_step:
            self._set_base_model_state_dict(state_dict)
            return

        options = StateDictOptions(strict=strict)
        if self.is_peft:
            _drop_outer_prefix(state_dict, "base_model.model.")
            options = StateDictOptions(strict=False, broadcast_from_rank0=True, full_state_dict=True)

        # Inject tied lm_head reference if needed
        if self.is_tied_lm_head and not self.is_peft:
            lm_head_weight, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            if lm_head_weight is not None and lm_head_param_name not in state_dict:
                state_dict[lm_head_param_name] = lm_head_weight.detach()

        func = partial(set_model_state_dict, model_state_dict=state_dict, options=options)
        list(map(func, self.model))
```

**Key Features**:
- **Tied embeddings handling**: Automatically removes `lm_head.weight` if tied to embeddings
- **PEFT prefix management**: Adds/removes `base_model.model.` prefix for HF compatibility
- **Pipeline parallelism support**: Accepts list of model parts for PP stages
- **Task head filtering**: Can skip loading task-specific heads via `skip_task_head_prefixes`

### 4. OptimizerState Wrapper

**Source**: `stateful_wrappers.py:195-276`

```python
class OptimizerState:
    """Stateful wrapper for optimizer and scheduler checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ):
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.optimizer = [optimizer] if isinstance(optimizer, torch.optim.Optimizer) else optimizer
        self.scheduler = [scheduler] if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler) else scheduler

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer and scheduler state dicts."""
        func = partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        optimizer_state_dict = {k: v for sd in map(func, self.model, self.optimizer) for k, v in sd.items()}

        state_dict = {"optim": optimizer_state_dict}
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler[0].state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer and scheduler state dicts."""
        func = partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optimizer))

        if "sched" in state_dict and self.scheduler is not None:
            list(map(lambda x: x.load_state_dict(state_dict["sched"]), self.scheduler))
```

**Key Features**:
- **Flattened state dict**: Uses `flatten_optimizer_state_dict=True` for DCP compatibility
- **Scheduler support**: Saves/loads LR scheduler state alongside optimizer
- **Pipeline parallelism support**: Accepts lists for PP stages

---

## DCP Integration

PyTorch's **Distributed Checkpoint Protocol (DCP)** provides the foundation for sharded, mesh-aware checkpointing.

### Save Path

**Source**: `checkpointing.py:482-525`

```python
def _do_save(
    self,
    state_dict: dict[str, torch.Tensor],
    path: str,
    storage_writer: Optional[_HuggingFaceStorageWriter] = None
) -> Optional["AsyncSaveResponse"]:
    """Core DCP save implementation."""
    is_model = True if "/model" in path else False

    # PEFT special case: only rank 0 saves adapter weights
    if self.config.is_peft and is_model:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_file(state_dict, os.path.join(path, "adapter_model.safetensors"))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # DCP save (async or sync)
    planner = dcp.DefaultSavePlanner(enable_plan_caching=True)
    if self.config.is_async:
        ctx = self._model_ctx if is_model else self._optim_ctx
        ret = dcp.async_save(
            state_dict,
            checkpoint_id=path,
            storage_writer=storage_writer,
            process_group=ctx.process_group,
            async_stager=ctx.stager,
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,
            planner=planner,
        )
        ctx.staging_active = True
    else:
        dcp.save(
            state_dict,
            checkpoint_id=path,
            storage_writer=storage_writer,
            planner=planner,
        )
    return ret
```

**Key Points**:
- **PEFT bypass**: PEFT models skip DCP and use direct `safetensors.torch.save_file()`
- **Async support**: Uses `dcp.async_save()` with background stager for non-blocking saves
- **Custom storage writer**: `_HuggingFaceStorageWriter` for SafeTensors output
- **Plan caching**: `enable_plan_caching=True` speeds up repeated saves

### Load Path

**Source**: `checkpointing.py:323-397`

```python
def _do_load(
    self,
    state_dict: dict[str, torch.Tensor],
    path: str,
    storage_reader: Optional[_HuggingFaceStorageReader] = None,
    is_init_step: bool = False,
) -> dict[str, torch.Tensor]:
    """Core DCP load implementation."""
    is_model = True if "/model" in path else False

    # PEFT special case: rank 0 loads adapter, broadcasts to others
    if self.config.is_peft and is_model:
        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                state_dict_loaded = load_file(adapter_path)
            else:
                state_dict_loaded = {k: torch.empty_like(v) for k, v in state_dict.items()}

            # Broadcast from rank 0
            for k in state_dict.keys():
                torch.distributed.broadcast(state_dict_loaded[k], src=0)
            return state_dict_loaded

    # DCP load
    dcp.load(
        state_dict,
        checkpoint_id=path,
        storage_reader=storage_reader,
        planner=dcp.DefaultLoadPlanner(),
    )
    return state_dict
```

**Key Points**:
- **PEFT bypass**: PEFT models use direct `safetensors.torch.load_file()` + broadcast
- **Custom storage reader**: `_HuggingFaceStorageReader` for SafeTensors input
- **Automatic resharding**: DCP handles mesh topology changes between save/load

### DCP APIs Used

```python
# Save APIs
from torch.distributed.checkpoint import (
    save,                     # Synchronous save
    async_save,               # Asynchronous save (torch >= 2.9.0)
    DefaultSavePlanner,       # Default save planning
)

# Load APIs
from torch.distributed.checkpoint import (
    load,                     # Load checkpoint
    DefaultLoadPlanner,       # Default load planning
)

# State dict APIs
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,     # Extract model state dict (FSDP-aware)
    set_model_state_dict,     # Load model state dict (FSDP-aware)
    get_optimizer_state_dict, # Extract optimizer state dict
    set_optimizer_state_dict, # Load optimizer state dict
    StateDictOptions,         # Options for state dict extraction
)
```

---

## SafeTensors Format

**SafeTensors** is a simple, safe tensor serialization format developed by HuggingFace.

### Format Structure

```
┌───────────────────────────────────────┐
│  Header (8 bytes)                     │
│  - JSON metadata size (little-endian) │
├───────────────────────────────────────┤
│  JSON Metadata                        │
│  {                                    │
│    "tensor_name": {                   │
│      "dtype": "F32",                  │
│      "shape": [1024, 768],            │
│      "data_offsets": [0, 3145728]     │
│    },                                 │
│    "__metadata__": {                  │
│      "dcp_custom_metadata": "{...}"   │
│    }                                  │
│  }                                    │
├───────────────────────────────────────┤
│  Tensor Data (raw bytes)              │
│  - Contiguous tensor data             │
│  - Aligned for efficient loading      │
└───────────────────────────────────────┘
```

### DCP Custom Metadata

NeMo AutoModel adds custom metadata to track tensor sharding information:

```python
{
  "tensor_name": {
    "dtype": "BF16",
    "shape": [512, 1024],           # Local shard shape
    "data_offsets": [0, 1048576]    # Byte offsets in file
  },
  "__metadata__": {
    "dcp_custom_metadata": json.dumps({
      "tensor_name": {
        "saved_offsets": [0, 512]   # Offset of this shard in full tensor
      }
    })
  }
}
```

**Purpose of `saved_offsets`**:
- Tracks where this shard belongs in the full tensor
- Enables consolidation (reassembly) of sharded tensors
- Used for mesh-aware loading (automatic resharding)

### SafeTensors APIs Used

```python
from safetensors.torch import (
    save_file,    # Save dict[str, Tensor] to .safetensors
    load_file,    # Load .safetensors to dict[str, Tensor]
)

# Example usage
save_file(state_dict, "model.safetensors")
state_dict = load_file("model.safetensors")
```

**Advantages of SafeTensors**:
1. **Security**: No arbitrary code execution (unlike pickle)
2. **Speed**: Zero-copy loading via memory mapping
3. **Portability**: Works across frameworks (PyTorch, TensorFlow, JAX)
4. **Validation**: Built-in integrity checks

---

## HuggingFace Storage Integration

NeMo AutoModel implements custom DCP storage readers/writers for SafeTensors compatibility.

### _HuggingFaceStorageWriter

**Source**: `_backports/hf_storage.py:67-215`

```python
class _HuggingFaceStorageWriter(FsspecWriter):
    """Custom DCP storage writer for HuggingFace SafeTensors format."""

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: Optional[dict[str, int]] = None,
        thread_count: int = 1,
        token: Optional[str] = None,
        save_sharded: bool = False,
        consolidated_output_path: Optional[str] = None,
        num_threads_consolidation: Optional[int] = None,
    ) -> None:
        super().__init__(path=path, serialization_format=SerializationFormat.SAFETENSORS)
        self._fqn_to_index_mapping = fqn_to_index_mapping
        self._save_sharded = save_sharded
        self._consolidated_output_path = consolidated_output_path
        self._num_threads_consolidation = num_threads_consolidation or 1

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """Add storage metadata to save plans."""
        new_plans = []
        for i, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            if self._save_sharded:
                storage_data["shard_index"] = i  # Rank index for filename
            new_plans.append(dataclasses.replace(plan, storage_data=storage_data))
        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        """Write tensors to SafeTensors files."""
        if len(plan.items) == 0:
            fut: Future = Future()
            fut.set_result([])
            return fut

        storage_data: dict[str, Any] = plan.storage_data
        shard_index: Optional[int] = storage_data.get("shard_index")

        # Split tensors by target file based on fqn_to_index_mapping
        buckets = self._split_by_storage_plan(self._fqn_to_index_mapping, plan.items)
        highest_index = max(self._fqn_to_index_mapping.values()) if self._fqn_to_index_mapping else 1

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = _gen_file_name(file_index, highest_index, shard_index)
            file_queue.put((self.fs.concat_path(self.path, file_name), file_name, write_items))

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        """Finalize write: optionally trigger consolidation."""
        if self._save_sharded and not self._consolidated_output_path:
            return

        if self._save_sharded:
            # Consolidate sharded files into HF format
            return consolidate_safetensors_files(
                input_dir=self.path,
                output_dir=self._consolidated_output_path,
                num_threads=self._num_threads_consolidation,
                fqn_to_index_mapping=self._fqn_to_index_mapping,
            )

        # Write model.safetensors.index.json
        metadata_to_write = {}
        storage_md = {}
        total_size = 0
        for wr_list in results:
            storage_md.update({wr.index.fqn: wr.storage_data.relative_path for wr in wr_list})
            total_size += sum([wr.storage_data.length for wr in wr_list])

        metadata_to_write["metadata"] = {"total_size": total_size}
        metadata_to_write["weight_map"] = storage_md

        metadata_path = self.fs.concat_path(self.path, "model.safetensors.index.json")
        with self.fs.create_stream(metadata_path, "w") as metadata_file:
            json.dump(metadata_to_write, metadata_file, indent=2)
```

**Filename Generation**:

```python
def _gen_file_name(file_index: int, highest_index: int, shard_index: Optional[int] = None) -> str:
    """Generate HuggingFace-style filenames."""
    if highest_index == 1:
        # Single file: model.safetensors
        filename = "model.safetensors"
    else:
        # Multi-file: model-00001-of-00004.safetensors
        num_digits = len(str(highest_index))
        filename = f"model-{str(file_index).zfill(num_digits)}-of-{str(highest_index).zfill(num_digits)}.safetensors"

    if shard_index is not None:
        # Sharded checkpoint: shard-00001-model-00001-of-00004.safetensors
        filename = f"shard-{str(shard_index).zfill(5)}-{filename}"

    return filename
```

**Examples**:
- Single file, rank 0: `model.safetensors`
- Multi-file (4 files), rank 0: `model-00001-of-00004.safetensors`
- Sharded, rank 1, file 2: `shard-00001-model-00002-of-00004.safetensors`

### _HuggingFaceStorageReader

**Source**: `_backports/hf_storage.py:217-434`

```python
class _HuggingFaceStorageReader(FsspecReader):
    """Custom DCP storage reader for HuggingFace SafeTensors format."""

    def __init__(self, path: str, token: Optional[str] = None, key_mapping: Optional[dict[str, str]] = None) -> None:
        super().__init__(path=path)
        self.key_mapping = key_mapping  # For VLM FQN remapping

    def read_metadata(self) -> Metadata:
        """Read metadata from all SafeTensors files in directory."""
        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        # Find all .safetensors files
        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(".safetensors"):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with self.fs.create_stream(safetensor_file, "rb") as f:
                safetensors_metadata, metadata_size = _get_safetensors_file_metadata(f)
                custom_metadata = safetensors_metadata.get("__metadata__")

                dcp_sharding_info = None
                if custom_metadata and custom_metadata.get("dcp_custom_metadata"):
                    dcp_sharding_info = json.loads(custom_metadata.get("dcp_custom_metadata"))

                for key, val in safetensors_metadata.items():
                    if key == "__metadata__":
                        continue

                    key = _get_key_renaming_mapping(key, self.key_mapping)

                    # Construct TensorStorageMetadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key]["saved_offsets"]
                    else:
                        offset = [0] * len(val["shape"])

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=_get_dtype(val["dtype"])),
                            size=torch.Size([saved + offset for saved, offset in zip(val["shape"], offset)]),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(val["shape"]),
                                )
                            ],
                        )
                    else:
                        # Multiple chunks for same tensor (sharded across files)
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(torch.Size(offset), sizes=torch.Size(val["shape"]))
                        )
                        # Update full tensor size
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], val["shape"][i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # Construct storage data
                    metadata_index = MetadataIndex(fqn=key, offset=offset)
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        offset=val["data_offsets"][0] + metadata_size,
                        length=val["data_offsets"][1] - val["data_offsets"][0],
                        shape=torch.Size(val["shape"]),
                        dtype=_get_dtype(val["dtype"]),
                    )

        return Metadata(state_dict_metadata=state_dict_metadata, storage_data=storage_data)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """Read tensor data from SafeTensors files."""
        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            with self.fs.create_stream(file_name, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]

                    # Seek and read tensor bytes
                    stream.seek(item_md.offset)
                    tensor_bytes = bytearray(stream.read(item_md.length))

                    # Reconstruct tensor
                    tensor = torch.frombuffer(tensor_bytes, dtype=item_md.dtype)
                    tensor = tensor.reshape(item_md.shape)
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert target_tensor.size() == tensor.size()
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut
```

**Key Features**:
- **Automatic shard detection**: Finds all `.safetensors` files in directory
- **DCP metadata parsing**: Reads `saved_offsets` from custom metadata
- **Multi-chunk handling**: Combines shards from multiple files into single tensor
- **VLM key remapping**: Supports FQN remapping for vision-language models

### FQN to File Index Mapping

The `fqn_to_file_index_mapping` determines which tensors go into which output files.

**Source**: `checkpointing.py:543-588`

```python
def _maybe_build_consolidated_index(
    self,
    model_state: ModelState,
    state_dict: dict[str, torch.Tensor],
) -> Optional[dict[str, int]]:
    """Build FQN to file index mapping for consolidation."""
    if not self._should_write_hf_metadata():
        return None

    model = model_state.model[0]

    # Find base model index file
    index_path = get_safetensors_index_path(
        self.config.model_cache_dir,
        self.config.model_repo_id,
    )

    if index_path:
        # Use base model's file structure
        fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
            index_path,
            getattr(model, "_checkpoint_conversion_mapping", None)
        )

        # Remove non-persistent buffers
        keys_to_remove = list(set(fqn_to_file_index_mapping.keys()) - set(self.config.model_state_dict_keys))
        if model_state.is_tied_lm_head:
            keys_to_remove.append(model_state.lm_head_param_name)
        for key in keys_to_remove:
            fqn_to_file_index_mapping.pop(key, None)
    else:
        # Single file fallback
        fqn_to_file_index_mapping = {k: 1 for k in state_dict.keys()}

    # Add any new keys (fine-tuned params) to last file
    default_index = max(fqn_to_file_index_mapping.values())
    for fqn in list(state_dict.keys()):
        fqn_to_file_index_mapping[fqn] = fqn_to_file_index_mapping.get(fqn, default_index)

    return fqn_to_file_index_mapping
```

**Example**:
```python
fqn_to_file_index_mapping = {
    "model.embed_tokens.weight": 1,
    "model.layers.0.self_attn.q_proj.weight": 1,
    "model.layers.0.self_attn.k_proj.weight": 1,
    "model.layers.31.self_attn.q_proj.weight": 4,
    "model.layers.31.self_attn.k_proj.weight": 4,
    "lm_head.weight": 4,
}
# → Creates model-00001-of-00004.safetensors ... model-00004-of-00004.safetensors
```

---

## Consolidation Mechanism

**Consolidation** merges sharded `.safetensors` files from distributed training into HuggingFace-compatible checkpoint files.

### Architecture

```
Input (Sharded):
  shard-00001-model-00001-of-00001.safetensors   # Rank 0 shard
  shard-00002-model-00001-of-00001.safetensors   # Rank 1 shard
  shard-00003-model-00001-of-00001.safetensors   # Rank 2 shard
  shard-00004-model-00001-of-00001.safetensors   # Rank 3 shard

Output (Consolidated):
  model-00001-of-00002.safetensors               # Merged file 1
  model-00002-of-00002.safetensors               # Merged file 2
  model.safetensors.index.json                   # Weight map
```

### Consolidation Workflow

**Source**: `_backports/consolidate_hf_safetensors.py:566-607`

```python
def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
) -> None:
    """Main consolidation function."""

    # Step 1: Setup output file structure
    max_index = max(fqn_to_index_mapping.values())
    fqn_to_file_mapping = {
        fqn: _gen_file_name(idx, max_index)
        for fqn, idx in fqn_to_index_mapping.items()
    }

    output_files_data: dict[str, _OutputFileData] = {}
    for fqn, filename in fqn_to_file_mapping.items():
        output_path = os.path.join(output_dir, filename)
        if output_path not in output_files_data:
            output_files_data[output_path] = _OutputFileData(fqn_data={fqn: _FqnData()})
        else:
            output_files_data[output_path].fqn_data[fqn] = _FqnData()

    # Step 2: Find all sharded SafeTensors files
    safetensors_files = glob.glob(os.path.join(input_dir, "*.safetensors"))

    # Step 3: Read metadata from all input files
    input_files_data: dict[str, _InputFileData] = {}
    for safetensor_file in safetensors_files:
        with open(safetensor_file, "rb") as f:
            metadata, size = _get_safetensors_file_metadata(f)
            input_files_data[safetensor_file] = _InputFileData(metadata_size=size, metadata=metadata)

    # Step 4: Parse metadata to determine full tensor shapes
    _parse_input_metadata(input_files_data, output_files_data)

    # Step 5: Write metadata headers to output files
    _write_metadata(output_files_data)

    # Step 6: Write tensor data from input files to output files
    _write_data(input_files_data, output_files_data, num_threads)

    # Step 7: Write overall model.safetensors.index.json
    _write_overall_metadata_file(output_dir, output_files_data)
```

### Metadata Parsing

**Source**: `_backports/consolidate_hf_safetensors.py:97-163`

```python
def _parse_input_metadata(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """Parse input metadata to determine full tensor shapes."""
    from safetensors.torch import _getdtype

    # Track full tensor size across all shards
    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}

    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)

        if not dcp_sharding_info:
            raise ValueError("No DCP custom metadata found. File must be saved with DCP.")

        for key, val in safetensors_metadata.items():
            if key == "__metadata__":
                continue

            # Get shard shape and offset
            sizes = val["shape"]
            offsets = dcp_sharding_info[key]["saved_offsets"]

            if key not in fqn_to_size_mapping:
                # First shard: calculate full size
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val["dtype"])
            else:
                # Update max dimension for each axis
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    # Populate output file data with full tensor info
    for fqn, tensor_info in fqn_to_size_mapping.items():
        tensor_size = tensor_info[0]
        dtype_str = tensor_info[1]
        for output_data in output_files_data.values():
            if fqn in output_data.fqn_data:
                dtype = _getdtype(dtype_str)
                try:
                    dtype_size = torch.finfo(dtype).bits // 8
                except TypeError:
                    dtype_size = torch.tensor([], dtype=dtype).element_size()
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=dtype_size,
                    dtype_str=dtype_str,
                )
```

**Example**:

Input shards:
```python
# shard-00001 (rank 0): model.embed_tokens.weight
{
  "model.embed_tokens.weight": {
    "shape": [16000, 256],         # Local shard shape
    "dtype": "BF16",
    "data_offsets": [0, 8192000]
  },
  "__metadata__": {
    "dcp_custom_metadata": {
      "model.embed_tokens.weight": {
        "saved_offsets": [0, 0]    # This is shard at offset [0, 0]
      }
    }
  }
}

# shard-00002 (rank 1): model.embed_tokens.weight
{
  "model.embed_tokens.weight": {
    "shape": [16000, 256],
    "dtype": "BF16",
    "data_offsets": [0, 8192000]
  },
  "__metadata__": {
    "dcp_custom_metadata": {
      "model.embed_tokens.weight": {
        "saved_offsets": [0, 256]  # This shard starts at column 256
      }
    }
  }
}
```

Parsed output:
```python
fqn_to_size_mapping = {
  "model.embed_tokens.weight": (
    [16000, 512],  # Full tensor size = max([16000, 256] + [0, 0], [16000, 256] + [0, 256])
    "BF16"
  )
}
```

### Tensor Data Writing

**Source**: `_backports/consolidate_hf_safetensors.py:246-307`

```python
def _process_output_file(
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:
    """Process single output file by merging shards."""

    sorted_tensors = sorted(output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file)

    with open(output_file, "r+b") as output_stream:
        output_stream.seek(0, os.SEEK_END)

        # Process each tensor in output file
        for tensor_fqn, tensor_fqn_data in sorted_tensors:
            # Allocate buffer for full tensor
            full_tensor_mv = memoryview(
                bytearray(math.prod(tensor_fqn_data.shape_in_file) * tensor_fqn_data.dtype_size)
            )

            # Process each input shard file
            for safetensors_file in input_files_data.keys():
                file_metadata = input_files_data[safetensors_file].metadata
                input_metadata_size = input_files_data[safetensors_file].metadata_size

                if tensor_fqn not in file_metadata.keys():
                    continue

                metadata = file_metadata[tensor_fqn]
                data_offsets = metadata["data_offsets"]

                # Read shard data using mmap
                data_to_write = _read_tensor_data_mmap(
                    safetensors_file,
                    data_offsets[0],
                    data_offsets[1],
                    input_metadata_size,
                )

                # Get shard offset in full tensor
                fqn_custom_metadata = _get_dcp_custom_metadata(file_metadata)[tensor_fqn]
                offsets_of_tensor_being_read = fqn_custom_metadata["saved_offsets"]

                # Write shard to appropriate position in full tensor buffer
                _write_sub_tensor_to_file_optimized(
                    full_tensor_mv,
                    data_to_write,
                    tensor_fqn_data.dtype_size,
                    tensor_fqn_data.shape_in_file,
                    offsets_of_tensor_being_read,
                    metadata["shape"],
                )

            # Write merged tensor to output file
            output_stream.write(full_tensor_mv)
```

**Optimized Sub-Tensor Writing**:

**Source**: `_backports/consolidate_hf_safetensors.py:354-497`

```python
def _write_sub_tensor_to_file_optimized(
    full_tensor_mv: memoryview,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
) -> None:
    """Optimized sub-tensor writing with maximum contiguous byte writes."""

    # Calculate strides for both full tensor and sub-tensor
    tensor_strides = [1]
    for i in range(len(tensor_shape) - 1, 0, -1):
        tensor_strides.insert(0, tensor_strides[0] * tensor_shape[i])

    sub_tensor_strides = [1]
    for i in range(len(sub_tensor_shape) - 1, 0, -1):
        sub_tensor_strides.insert(0, sub_tensor_strides[0] * sub_tensor_shape[i])

    total_elements = math.prod(sub_tensor_shape)
    elements_written = 0

    while elements_written < total_elements:
        # Convert linear index to multi-dimensional indices
        temp_idx = elements_written
        indices = []
        for dim_size in reversed(sub_tensor_shape):
            indices.append(temp_idx % dim_size)
            temp_idx //= dim_size
        indices.reverse()

        # Calculate maximum contiguous elements we can write
        max_contiguous = _calculate_max_contiguous_elements(indices, sub_tensor_shape, tensor_shape)

        # Calculate source byte offset in sub_tensor_bytes
        src_pos = sum(idx * stride for idx, stride in zip(indices, sub_tensor_strides))
        src_byte_offset = src_pos * element_size

        # Calculate destination byte offset in full_tensor_mv
        dest_indices = [idx + offset for idx, offset in zip(indices, sub_tensor_offsets)]
        dest_pos = sum(idx * stride for idx, stride in zip(dest_indices, tensor_strides))
        dest_byte_offset = dest_pos * element_size

        # Write contiguous chunk
        bytes_to_write = max_contiguous * element_size
        chunk_data = sub_tensor_bytes[src_byte_offset : src_byte_offset + bytes_to_write]
        full_tensor_mv[dest_byte_offset : dest_byte_offset + bytes_to_write] = chunk_data

        elements_written += max_contiguous
```

**Key Optimizations**:
1. **Memory mapping**: Uses `mmap` for efficient shard reading
2. **Contiguous writes**: Maximizes write size per iteration
3. **Row-wise optimization**: For row-sharded tensors, writes entire rows at once
4. **Multi-threading**: Processes different output files in parallel

### Parallel Consolidation on All Ranks

**Source**: `_backports/consolidate_hf_safetensors.py:609-721`

```python
def consolidate_safetensors_files_on_every_rank(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Consolidate across multiple ranks with work distribution."""

    # Get rank info
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
    else:
        rank = 0
        world_size = 1

    # Distribute output files across ranks
    unique_indices = set(fqn_to_index_mapping.values())
    indices_for_this_rank = []
    for idx in unique_indices:
        if idx % world_size == rank:  # Simple round-robin distribution
            indices_for_this_rank.append(idx)

    # Filter mapping to only this rank's files
    filtered_mapping = {fqn: idx for fqn, idx in fqn_to_index_mapping.items() if idx in indices_for_this_rank}

    if filtered_mapping:
        # Consolidate this rank's subset
        output_files_data = _consolidate_safetensors_files(
            input_dir=input_dir,
            output_dir=output_dir,
            fqn_to_file_mapping={fqn: _gen_file_name(idx, max(unique_indices)) for fqn, idx in filtered_mapping.items()},
            num_threads=num_threads,
        )
    else:
        output_files_data = {}

    # Gather results from all ranks
    global GLOBAL_OUTPUT_FILES_DATA
    if GLOBAL_OUTPUT_FILES_DATA is None:
        GLOBAL_OUTPUT_FILES_DATA = {}
        global_output_files_data_list = [None] * world_size
        dist.all_gather_object(global_output_files_data_list, output_files_data)
        for item in global_output_files_data_list:
            if item:
                GLOBAL_OUTPUT_FILES_DATA.update(item)

    # Rank 0 writes overall index file
    if GLOBAL_OUTPUT_FILES_DATA:
        if rank == 0:
            _write_overall_metadata_file(output_dir, GLOBAL_OUTPUT_FILES_DATA)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
```

**Example** (4 ranks, 8 output files):
- Rank 0: Creates files 1, 5 (indices % 4 == 0)
- Rank 1: Creates files 2, 6 (indices % 4 == 1)
- Rank 2: Creates files 3, 7 (indices % 4 == 2)
- Rank 3: Creates files 4, 8 (indices % 4 == 3)

**Benefits**:
- Parallelizes I/O across ranks
- Reduces consolidation time for large models
- Each rank processes independent subset (no conflicts)

---

## PEFT Checkpointing

**PEFT (Parameter-Efficient Fine-Tuning)** checkpoints require special handling because only adapter parameters are trainable.

### PEFT Save Workflow

**Source**: `checkpointing.py:166-238`

```python
def save_model(
    self,
    model: nn.Module,
    weights_path: str,
    peft_config: Optional["PeftConfig"] = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> None:
    """Save model checkpoint."""

    # ... directory setup ...

    model_state = ModelState(model, self.config.is_peft)
    state_dict = model_state.state_dict()

    # Adapt to HF format
    state_dict = _maybe_adapt_state_dict_to_hf(model_state.model[0], state_dict, ...)

    # Build consolidated index (if needed)
    fqn_to_file_index_mapping = self._maybe_build_consolidated_index(model_state, state_dict)

    # Run addon pre-saves
    for addon in self._addons:
        addon.pre_save(
            model_state=model_state,
            model_path=model_dir,
            hf_metadata_dir=hf_metadata_dir,
            consolidated_path=consolidated_dir,
            fqn_to_file_index_mapping=fqn_to_file_index_mapping,
            tokenizer=tokenizer,
            peft_config=peft_config,
            original_model_path=self._get_original_model_path(model_state),
        )

    # Save using DCP
    storage_writer = self._get_storage_writer(...)
    self._model_ctx.future = self._do_save(state_dict, model_dir, storage_writer)

    # Run addon post-saves
    for addon in self._addons:
        addon.post_save(
            consolidated_path=consolidated_dir,
            hf_metadata_path=hf_metadata_dir,
        )
```

**PEFT-specific `_do_save`**:

**Source**: `checkpointing.py:482-525`

```python
def _do_save(...) -> Optional["AsyncSaveResponse"]:
    is_model = True if "/model" in path else False

    # PEFT special case: only rank 0 writes
    if self.config.is_peft and is_model:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_file(state_dict, os.path.join(path, "adapter_model.safetensors"))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # ... normal DCP save for non-PEFT ...
```

**Why rank-0 only?**
- PEFT adapters are small (typically < 100MB)
- All ranks have identical adapter weights (replicated, not sharded)
- No need for DCP's distributed save machinery
- Simpler to load (standard HF PEFT format)

### PeftAddon

**Source**: `addons.py:118-160`

```python
class PeftAddon:
    """Addon that writes PEFT-specific metadata."""

    def pre_save(self, **kwargs) -> None:
        model_path = kwargs["model_path"]
        tokenizer = kwargs.get("tokenizer", None)
        model_state = kwargs["model_state"]
        peft_config = kwargs["peft_config"]
        original_model_path = kwargs["original_model_path"]

        hf_peft_config = _get_hf_peft_config(peft_config, model_state)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # Save custom model code (if exists)
            _maybe_save_custom_model_code(original_model_path, model_path)

            # Save tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)

            # Save HF PEFT config
            with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
                json.dump(hf_peft_config, f, indent=2, sort_keys=True)

            # Save AutoModel PEFT config
            with open(os.path.join(model_path, "automodel_peft_config.json"), "w") as f:
                json.dump(automodel_peft_metadata, f, indent=2, sort_keys=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
```

**HF PEFT Config**:

```python
def _get_hf_peft_config(peft_config, model_state) -> dict:
    """Generate minimal HF PEFT config."""
    model_part = model_state.model[0]
    target_modules = _extract_target_modules(model_part)  # Find LoRA modules

    try:
        model_task = model_part.config.architectures[0].split("For")[-1]
        task_type = MODEL_TYPE_TO_PEFT_TASK_TYPE[model_task]
    except (AttributeError, IndexError, TypeError, KeyError):
        task_type = "CAUSAL_LM"

    return {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": model_part.config.name_or_path,
    }
```

**Example Output** (`adapter_config.json`):
```json
{
  "task_type": "CAUSAL_LM",
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "target_modules": [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj"
  ],
  "bias": "none",
  "base_model_name_or_path": "meta-llama/Llama-3.2-1B"
}
```

### PEFT Load Workflow

**Source**: `checkpointing.py:323-397`

```python
def _do_load(...) -> dict[str, torch.Tensor]:
    is_model = True if "/model" in path else False

    # PEFT special case: rank 0 loads, broadcasts to others
    if self.config.is_peft and is_model:
        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                state_dict_loaded = load_file(adapter_path)
            else:
                # Other ranks: create empty tensors
                state_dict_loaded = {k: torch.empty_like(v) for k, v in state_dict.items()}

            # Broadcast from rank 0
            for k in state_dict.keys():
                torch.distributed.broadcast(state_dict_loaded[k], src=0)
            return state_dict_loaded

    # ... normal DCP load for non-PEFT ...
```

**PEFT Load Pattern**:
1. Rank 0 loads `adapter_model.safetensors`
2. Other ranks allocate empty tensors with same shape/dtype
3. Rank 0 broadcasts each tensor to all ranks
4. All ranks end up with identical adapter weights

---

## Async Checkpointing

**Async checkpointing** (PyTorch >= 2.9.0) enables non-blocking saves for faster training.

### Async Architecture

**Source**: `checkpointing.py:50-76`

```python
@dataclass
class _AsyncSaveContext:
    """Context for async checkpoint operations."""
    future: Optional[Any] = None              # AsyncSaveResponse
    process_group: Optional[Any] = None       # Gloo process group for async ops
    stager: Optional[Any] = None              # DefaultStager for staging
    staging_active: bool = False              # Is async save in progress?
```

**Initialization**:

```python
def __init__(self, config, ...):
    # ... other init ...

    self._model_ctx = _AsyncSaveContext(...)
    self._optim_ctx = _AsyncSaveContext(...)

    if self.config.is_async:
        # Setup async stager
        self._model_ctx.stager = DefaultStager()
        self._optim_ctx.stager = DefaultStager()

        # Create gloo process group (required for async DCP)
        self._model_ctx.process_group = torch.distributed.new_group(backend="gloo")
        self._optim_ctx.process_group = torch.distributed.new_group(backend="gloo")
```

### Async Save

**Source**: `checkpointing.py:482-525`

```python
def _do_save(...) -> Optional["AsyncSaveResponse"]:
    # ... PEFT special case ...

    planner = dcp.DefaultSavePlanner(enable_plan_caching=True)

    if self.config.is_async:
        ctx = self._model_ctx if is_model else self._optim_ctx
        ret = dcp.async_save(
            state_dict,
            checkpoint_id=path,
            storage_writer=storage_writer,
            process_group=ctx.process_group,           # Gloo process group
            async_stager=ctx.stager,                   # DefaultStager
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,  # Process-based async
            planner=planner,
        )
        ctx.staging_active = True
        return ret
    else:
        dcp.save(state_dict, checkpoint_id=path, storage_writer=storage_writer, planner=planner)
        return None
```

**Async Workflow**:
1. `dcp.async_save()` returns immediately with `AsyncSaveResponse`
2. Background process handles actual I/O
3. Training loop continues without blocking
4. Before next checkpoint, call `wait_model()` / `wait_optimizer()` to ensure completion

### Async Wait

**Source**: `checkpointing.py:399-480`

```python
def wait_model(self) -> None:
    """Wait for async model checkpoint to complete."""
    if not self.config.is_async or not self._model_ctx.staging_active:
        return

    try:
        self._model_ctx.future.result()
    except Exception as e:
        logger.error(f"Async model checkpoint failed: {e}")
        raise
    finally:
        self._model_ctx.staging_active = False
        self._model_ctx.future = None

def wait_optimizer(self) -> None:
    """Wait for async optimizer checkpoint to complete."""
    if not self.config.is_async or not self._optim_ctx.staging_active:
        return

    try:
        self._optim_ctx.future.result()
    except Exception as e:
        logger.error(f"Async optimizer checkpoint failed: {e}")
        raise
    finally:
        self._optim_ctx.staging_active = False
        self._optim_ctx.future = None

def wait_all(self) -> None:
    """Wait for all async operations to complete."""
    self.wait_model()
    self.wait_optimizer()
```

**Usage in Training Loop**:

```python
# Step N: Save checkpoint (non-blocking)
checkpointer.save_model(model, f"checkpoint/step_{step}")
checkpointer.save_optimizer(model, optimizer, scheduler, f"checkpoint/step_{step}")

# Training continues immediately
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# Step N+1: Wait for previous checkpoint before next save
checkpointer.wait_all()  # Block until step N checkpoint completes
checkpointer.save_model(model, f"checkpoint/step_{step+1}")
```

**Benefits**:
- Training doesn't block on I/O
- Typical speedup: 10-30% for large models
- No risk of corruption (waits before overwriting)

---

## Mesh-Aware Checkpointing

NeMo AutoModel's checkpointing is fully aware of the 5D device mesh (PP × DP_replicate × DP_shard × CP × TP).

### Mesh Dimensions in Checkpointing

**Mesh Setup**:

```python
mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")

# Example: 64 GPUs, PP=4, DP=4, TP=4
mesh_shape = (4, 1, 4, 1, 4)
```

**Rank Tracking**:

```python
# Checkpointer is instantiated with rank info
checkpointer = Checkpointer(
    config=cfg,
    dp_rank=dist_env.dp_rank,      # Data parallel rank
    tp_rank=dist_env.tp_rank,      # Tensor parallel rank
    pp_rank=dist_env.pp_rank,      # Pipeline parallel rank
    moe_mesh=fsdp2_manager.moe_mesh if hasattr(fsdp2_manager, 'moe_mesh') else None,
)
```

### Sharding Patterns

**Tensor Parallel (TP)**:
- Parameters are sharded across TP dimension
- DCP automatically handles TP sharding via DTensor metadata
- Each TP rank saves its shard

**Data Parallel (FSDP2)**:
- Parameters are sharded across `dp_shard` dimension
- FSDP2 state dict APIs handle sharding automatically
- Each DP rank saves its parameter shard

**Pipeline Parallel (PP)**:
- Each PP stage is a separate model part
- `ModelState` accepts list of model parts
- Each PP rank saves its stage's parameters

**Example** (PP=2, DP=2, TP=2):

```python
# Total 8 GPUs
# Rank 0: PP stage 0, DP rank 0, TP rank 0
# Rank 1: PP stage 0, DP rank 0, TP rank 1
# Rank 2: PP stage 0, DP rank 1, TP rank 0
# Rank 3: PP stage 0, DP rank 1, TP rank 1
# Rank 4: PP stage 1, DP rank 0, TP rank 0
# ... and so on

# Each rank saves a unique shard:
# checkpoint/model/shard-00001-model-00001-of-00001.safetensors  # Rank 0
# checkpoint/model/shard-00002-model-00001-of-00001.safetensors  # Rank 1
# checkpoint/model/shard-00003-model-00001-of-00001.safetensors  # Rank 2
# ...
# checkpoint/model/shard-00008-model-00001-of-00001.safetensors  # Rank 7
```

### Resharding on Load

DCP automatically handles mesh topology changes between save and load.

**Example**:

```python
# Save with mesh (PP=2, DP=4, TP=2) → 16 GPUs
checkpointer_save = Checkpointer(
    config=cfg,
    dp_rank=dist_env.dp_rank,  # 0-3
    tp_rank=dist_env.tp_rank,  # 0-1
    pp_rank=dist_env.pp_rank,  # 0-1
)
checkpointer_save.save_model(model, "checkpoint/")

# Load with mesh (PP=2, DP=2, TP=4) → 16 GPUs (different topology!)
checkpointer_load = Checkpointer(
    config=cfg,
    dp_rank=dist_env_new.dp_rank,  # 0-1
    tp_rank=dist_env_new.tp_rank,  # 0-3
    pp_rank=dist_env_new.pp_rank,  # 0-1
)
checkpointer_load.load_model(model, "checkpoint/")
# DCP automatically reshards: DP=4→DP=2, TP=2→TP=4
```

**How Resharding Works**:
1. DCP reads metadata from all shards
2. Reconstructs full tensor shapes and shard offsets
3. Computes new sharding plan based on current mesh
4. Each rank loads required slices from multiple old shards
5. Automatically handles all-gather, scatter, and redistribution

**Supported Topology Changes**:
- Change DP size (different FSDP sharding)
- Change TP size (different tensor sharding)
- Change PP size (different pipeline stages)
- Mix and match (e.g., DP=8,TP=2 → DP=4,TP=4)

**Limitations**:
- Total world size should remain same (or be compatible)
- Model architecture must be identical
- PP resharding requires careful handling (not fully automatic)

---

## Complete Workflow

### End-to-End Save Workflow

**From training loop to disk**:

```python
# 1. Training recipe setup
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
trainer.setup()

# 2. Checkpointer initialization (inside trainer.setup())
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

checkpointer_config = CheckpointingConfig(
    enabled=True,
    checkpoint_dir="./checkpoints",
    model_save_format="safetensors",
    model_cache_dir=os.environ.get("HF_HOME", "~/.cache/huggingface"),
    model_repo_id="meta-llama/Llama-3.2-1B",
    save_consolidated=True,
    is_peft=False,
    is_async=True,  # Enable async checkpointing
)

checkpointer = Checkpointer(
    config=checkpointer_config,
    dp_rank=dist_env.dp_rank,
    tp_rank=dist_env.tp_rank,
    pp_rank=dist_env.pp_rank,
)

# 3. Training loop with periodic checkpoints
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # Forward pass
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Checkpoint every N steps
        if (step + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_step_{step}")

            # Wait for previous async checkpoint (if any)
            checkpointer.wait_all()

            # Save model (async, returns immediately)
            checkpointer.save_model(
                model=model,
                weights_path=checkpoint_path,
                tokenizer=tokenizer,
            )

            # Save optimizer (async, returns immediately)
            checkpointer.save_optimizer(
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                weights_path=checkpoint_path,
            )

            # Training continues without blocking...
```

### Save Model Detailed Flow

**Source**: `checkpointing.py:166-238`

```python
def save_model(
    self,
    model: nn.Module,
    weights_path: str,
    peft_config: Optional["PeftConfig"] = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> None:
    """Save model checkpoint with full workflow."""

    # Step 1: Create directory structure
    model_dir = os.path.join(weights_path, "model")
    consolidated_dir = os.path.join(model_dir, "consolidated") if self._should_write_consolidated_safetensors() else None
    hf_metadata_dir = os.path.join(model_dir, ".hf_metadata") if self._should_write_hf_metadata() else None

    _ensure_dirs(model_dir, consolidated_dir, hf_metadata_dir)

    # Step 2: Get model state dict
    model_state = ModelState(model, self.config.is_peft)
    state_dict = model_state.state_dict()

    # Step 3: Adapt to HF format (if custom model)
    state_dict = _maybe_adapt_state_dict_to_hf(model_state.model[0], state_dict)

    # Step 4: Build consolidated index (for HF multi-file models)
    fqn_to_file_index_mapping = self._maybe_build_consolidated_index(model_state, state_dict)

    # Step 5: Run addon pre-saves (PEFT, HF metadata)
    for addon in self._addons:
        addon.pre_save(
            model_state=model_state,
            model_path=model_dir,
            hf_metadata_dir=hf_metadata_dir,
            consolidated_path=consolidated_dir,
            fqn_to_file_index_mapping=fqn_to_file_index_mapping,
            tokenizer=tokenizer,
            peft_config=peft_config,
            original_model_path=self._get_original_model_path(model_state),
        )

    # Step 6: Get storage writer
    consolidate_on_all_ranks = consolidated_dir is not None and self.dp_rank == 0 and self.pp_rank == 0
    storage_writer = self._get_storage_writer(
        consolidated_output_path=consolidated_dir,
        fqn_to_index_mapping=fqn_to_file_index_mapping,
        model_path=model_dir,
        consolidate_on_all_ranks=consolidate_on_all_ranks,
    )

    # Step 7: Save via DCP (async or sync)
    self._model_ctx.future = self._do_save(state_dict, model_dir, storage_writer)

    # Step 8: Run addon post-saves (move HF metadata to consolidated dir)
    for addon in self._addons:
        addon.post_save(
            consolidated_path=consolidated_dir,
            hf_metadata_path=hf_metadata_dir,
        )

    # Step 9: Consolidate sharded files (if enabled)
    if consolidate_on_all_ranks:
        consolidate_safetensors_files_on_every_rank(
            input_dir=model_dir,
            output_dir=consolidated_dir,
            fqn_to_index_mapping=fqn_to_file_index_mapping,
            num_threads=max(fqn_to_file_index_mapping.values()),
        )
```

### Load Model Detailed Flow

**Source**: `checkpointing.py:274-321`

```python
def load_model(
    self,
    model: nn.Module,
    model_path: str,
    is_init_step: bool = False,
    use_checkpoint_id: bool = True,
    key_mapping: Optional[dict[str, str]] = None,
) -> None:
    """Load model checkpoint with full workflow."""

    # Step 1: Validate checkpoint exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    # Step 2: Create model state wrapper
    model_state = ModelState(
        model,
        is_peft=self.config.is_peft,
        is_init_step=is_init_step,
        skip_task_head_prefixes=self.config.skip_task_head_prefixes_for_base_model,
    )
    state_dict = model_state.state_dict()

    # Step 3: Get storage reader
    storage_reader = self._get_storage_reader(model_path, key_mapping, is_init_step=is_init_step)

    # Step 4: Adapt to HF format (for loading into custom models)
    state_dict = _maybe_adapt_state_dict_to_hf(
        model_state.model[0],
        state_dict,
        quantization=self.config.dequantize_base_checkpoint,
    )

    # Step 5: Load via DCP
    state_dict = self._do_load(state_dict, model_path, storage_reader, is_init_step=is_init_step)

    # Step 6: Adapt from HF format (convert back to custom format)
    state_dict = _maybe_adapt_state_dict_from_hf(model_state.model[0], state_dict, moe_mesh=self.moe_mesh)

    # Step 7: Load into model
    model_state.load_state_dict(state_dict, strict=not is_init_step)
```

### Directory Structure After Save

**Complete checkpoint structure**:

```
checkpoints/epoch_0_step_100/
├── model/
│   ├── shard-00001-model-00001-of-00001.safetensors    # Rank 0 model shard
│   ├── shard-00002-model-00001-of-00001.safetensors    # Rank 1 model shard
│   ├── .hf_metadata/
│   │   ├── config.json                                 # Model config
│   │   ├── generation_config.json                      # Generation settings
│   │   ├── tokenizer_config.json                       # Tokenizer config
│   │   ├── tokenizer.json                              # Tokenizer data
│   │   ├── special_tokens_map.json                     # Special tokens
│   │   └── fqn_to_file_index_mapping.json              # Consolidation mapping
│   ├── consolidated/                                   # Consolidated checkpoint
│   │   ├── model-00001-of-00002.safetensors            # Merged file 1
│   │   ├── model-00002-of-00002.safetensors            # Merged file 2
│   │   ├── model.safetensors.index.json                # Weight map
│   │   ├── config.json                                 # Model config (copy)
│   │   ├── generation_config.json                      # Generation config (copy)
│   │   └── tokenizer_config.json                       # Tokenizer config (copy)
│   └── .metadata                                       # DCP metadata
├── optim/
│   ├── __0_0.distcp                                    # Optimizer shard rank 0
│   ├── __1_0.distcp                                    # Optimizer shard rank 1
│   └── .metadata                                       # DCP metadata
├── step_scheduler.pt                                   # Training state
├── dataloader/
│   ├── dataloader_dp_rank_0.pt                         # Dataloader state rank 0
│   └── dataloader_dp_rank_1.pt                         # Dataloader state rank 1
├── rng/
│   ├── rng_dp_rank_0.pt                                # RNG state rank 0
│   └── rng_dp_rank_1.pt                                # RNG state rank 1
├── config.yaml                                         # Training config
└── losses.json                                         # Training metrics
```

---

## Key Takeaways

### 1. PyTorch-Native Checkpointing

NeMo AutoModel uses **PyTorch Distributed Checkpoint Protocol (DCP)** as the foundation:
- No custom serialization code
- Automatic mesh-aware sharding
- Automatic resharding on load
- Built-in FSDP2 and DTensor support

### 2. SafeTensors for Safety and Speed

**SafeTensors format** provides:
- Security (no pickle vulnerabilities)
- Speed (zero-copy loading via mmap)
- Portability (cross-framework compatibility)
- Validation (integrity checks)

### 3. HuggingFace Ecosystem Integration

Custom storage readers/writers enable:
- Direct save to `.safetensors` format
- HuggingFace Hub compatibility
- `model.safetensors.index.json` generation
- Seamless integration with `transformers` library

### 4. Efficient Consolidation

**Parallel consolidation** on all ranks:
- Distributes I/O across GPUs
- Memory-efficient (uses mmap)
- Multi-threaded (processes files in parallel)
- Optimized sub-tensor writing (maximizes contiguous bytes)

### 5. PEFT Special Handling

**PEFT adapters** bypass DCP:
- Rank-0 only save (adapters are replicated)
- Direct SafeTensors save (simpler, faster)
- Broadcast on load (efficient distribution)
- HF PEFT format compatibility

### 6. Async Checkpointing

**Async saves** (PyTorch >= 2.9.0):
- Non-blocking I/O
- Background process handles writes
- 10-30% training speedup
- Safe (waits before overwrite)

### 7. Mesh-Aware Resharding

**Automatic topology changes**:
- Change DP/TP/PP dimensions between save/load
- DCP handles all redistribution
- No manual shard merging required
- Fully automatic for supported changes

### 8. Addon Architecture

**Extensible via addons**:
- `ConsolidatedHFAddon`: Writes HF metadata
- `PeftAddon`: Writes PEFT configs
- Pre-save and post-save hooks
- Clean separation of concerns

### 9. State Dict Adaptation

**Custom models** use state dict adapters:
- `to_hf()`: Convert custom → HuggingFace format
- `from_hf()`: Convert HuggingFace → custom format
- Transparent to checkpointing logic
- Enables custom architectures with HF compatibility

### 10. Production-Ready Features

- **Tied embeddings**: Automatically removes `lm_head.weight` if tied
- **Task head filtering**: Skip loading task-specific heads
- **Async wait**: Ensures checkpoint completion before overwrite
- **Dataloader/RNG state**: Exact training resumption
- **Config saving**: Full reproducibility

---

## Source Code References

All file paths relative to `nemo_automodel/components/checkpoint/`:

| File | Lines | Key Content |
|------|-------|-------------|
| `checkpointing.py` | 879 | Checkpointer class, save/load orchestration |
| `stateful_wrappers.py` | 276 | ModelState, OptimizerState wrappers |
| `_backports/hf_storage.py` | 434 | HF storage reader/writer for DCP |
| `_backports/consolidate_hf_safetensors.py` | 721 | Consolidation logic |
| `addons.py` | 268 | ConsolidatedHFAddon, PeftAddon |
| `_backports/hf_utils.py` | ~200 | Helper functions for HF metadata |
| `_backports/filesystem.py` | ~150 | Fsspec-based filesystem abstraction |
| `utils.py` | ~100 | Utility functions |

**Test Files** (verify behavior):
- `tests/functional_tests/checkpoint/test_hf_sharded.py`
- `tests/functional_tests/checkpoint/test_hf_consolidated_llm.py`
- `tests/functional_tests/checkpoint/test_peft.py`
- `tests/functional_tests/checkpoint/test_dcp.py`

---

## Conclusion

NeMo AutoModel's distributed checkpointing system achieves:

1. **Full PyTorch-native implementation** (DCP, FSDP2, DTensor)
2. **SafeTensors format** for security and performance
3. **HuggingFace compatibility** via custom storage readers/writers
4. **Automatic consolidation** with parallel multi-rank processing
5. **PEFT optimization** with rank-0 only saves
6. **Async checkpointing** for non-blocking I/O
7. **Mesh-aware resharding** for topology flexibility
8. **Extensible addon system** for custom checkpointing logic

This design enables:
- Seamless integration with HuggingFace ecosystem
- Efficient checkpointing at massive scale (thousands of GPUs)
- Safe resumption from failures
- Flexible deployment (change topology between runs)
- Production-ready reliability

The implementation is grounded entirely in source code, with no custom communication primitives or serialization formats - everything is built on PyTorch's stable, tested APIs.
