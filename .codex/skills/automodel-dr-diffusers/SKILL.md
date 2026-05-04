---
name: automodel-dr-diffusers
description: Use when working with the diffusers integration module of automodel — HuggingFace Diffusers model registration and auto-model support
---

# Module: `nemo_automodel/_diffusers/`

## 1. Module Purpose & Capabilities

The `_diffusers` module provides a bridge between HuggingFace's `diffusers` library and NeMo AutoModel's distributed parallelism infrastructure. Its sole public export is `NeMoAutoDiffusionPipeline`, a class that wraps `diffusers.DiffusionPipeline.from_pretrained` to inject per-component FSDP2/TP parallelization during model loading.

**Capabilities:**

- **Drop-in pipeline loading**: Loads any HuggingFace Diffusers pipeline (e.g., Wan2.2, Stable Diffusion) via the standard `from_pretrained` interface, returning a fully functional `DiffusionPipeline` instance.
- **Per-component parallelization**: Accepts a `parallel_scheme` dictionary mapping component names (e.g., `"transformer"`, `"unet"`) to `FSDP2Manager` instances, enabling selective FSDP2/TP sharding of only the heavy compute components while leaving lightweight components (schedulers, tokenizers) untouched.
- **Automatic device placement**: Moves all `nn.Module` components to the correct CUDA device based on `LOCAL_RANK`, with a fallback to CPU when CUDA is unavailable.
- **Flexible dtype handling**: Supports `torch_dtype="auto"` (preserves original dtype), explicit `torch.dtype` objects, and string dtype specifications (e.g., `"torch.bfloat16"`), resolved via `nemo_automodel.shared.utils.dtype_from_str`.
- **Graceful degradation**: The `diffusers` library is an optional dependency. If not installed, `DIFFUSERS_AVAILABLE` is set to `False` and the class falls back to inheriting from `object`, raising `RuntimeError` only if `from_pretrained` is actually called.
- **Fault-tolerant parallelization**: If `FSDP2Manager.parallelize` fails for any component, the error is logged as a warning and the pipeline continues with the unparallelized component rather than crashing.

**Files:**

| File | Role |
|------|------|
| `nemo_automodel/_diffusers/__init__.py` | Package init; exports `NeMoAutoDiffusionPipeline` |
| `nemo_automodel/_diffusers/auto_diffusion_pipeline.py` | Full implementation: class, helpers, parallelization logic |

**Related files:**

| File | Relationship |
|------|-------------|
| `nemo_automodel/components/distributed/fsdp2.py` | Provides `FSDP2Manager` consumed by `parallel_scheme` |
| `nemo_automodel/shared/utils.py` | Provides `dtype_from_str` for string-to-dtype conversion |
| `examples/diffusion/wan2.2/wan_generate.py` | Real-world usage example (Wan2.2 text-to-video with TP/CP/PP/DP) |
| `tests/unit_tests/_diffusers/test_auto_diffusion_pipeline.py` | Comprehensive unit tests for all helpers and the main class |

---

## 2. Core Design Logic

### Why a thin wrapper instead of a custom pipeline?

The module deliberately does NOT subclass or reimplement any diffusion pipeline logic. Instead, `NeMoAutoDiffusionPipeline.from_pretrained` delegates entirely to `DiffusionPipeline.from_pretrained` for model loading and then post-processes the result. This design has three motivations:

1. **Universal compatibility**: Any model that HuggingFace Diffusers supports (Stable Diffusion, Wan, Flux, etc.) works automatically without model-specific code in AutoModel. The `_iter_pipeline_modules` helper generically discovers all `nn.Module` components in the returned pipeline.

2. **Separation of concerns**: Model loading is HuggingFace's responsibility; distributed parallelization is AutoModel's. The `parallel_scheme` dictionary is the single integration point, mapping component names to their parallelization strategy. This mirrors AutoModel's broader architecture where components (like `FSDP2Manager`) are independent and composed at a higher level.

3. **Selective parallelization**: Diffusion pipelines contain heterogeneous components -- some are heavy neural networks (transformers, UNets), others are lightweight (VAE decoders, schedulers, tokenizers). The per-component mapping allows users to parallelize only what benefits from sharding, avoiding the overhead and complexity of wrapping non-parallelizable components. In the Wan2.2 example (`examples/diffusion/wan2.2/wan_generate.py`, lines 113-115), only `"transformer"` and `"transformer_2"` are parallelized while the VAE is loaded separately.

### Why move to device BEFORE parallelization?

Line 118-121 of `auto_diffusion_pipeline.py` moves modules to device before applying `parallel_scheme`. The inline comment explains: this "helps avoid initial OOM during sharding." FSDP2 sharding operations may temporarily increase memory usage during the redistribution phase; having the model already on the correct device ensures the sharding can proceed without needing to hold two copies (CPU + GPU) simultaneously.

### Why `diffusers` is optional

The `try/except` block at module top (lines 26-32) makes `diffusers` an optional dependency. This is because the core AutoModel package targets LLM/VLM training where `diffusers` is not needed. The fallback sets `DiffusionPipeline = object` so the class definition does not fail at import time, and the `DIFFUSERS_AVAILABLE` guard in `from_pretrained` produces a clear error message if someone attempts to use the class without the library installed.

### Why errors in parallelization are warnings, not exceptions

Lines 133-135 catch exceptions from `manager.parallelize` and log them as warnings. This is deliberate: in a multi-component pipeline, a failure to parallelize one component (e.g., due to an unsupported TP plan) should not prevent the entire pipeline from functioning. The component simply runs unsharded, which is correct (just slower), and the user is informed via the log.

---

## 3. Core Data Structures

### `NeMoAutoDiffusionPipeline` (class)

- **File**: `/home/scbjtfy/Automodel/nemo_automodel/_diffusers/auto_diffusion_pipeline.py`, line 79
- **Inherits**: `diffusers.DiffusionPipeline` (or `object` if diffusers unavailable)
- **Purpose**: The sole public API of this module. Provides a `from_pretrained` classmethod that adds distributed parallelization to Diffusers pipeline loading.
- **Key method**: `from_pretrained(pretrained_model_name_or_path, *, parallel_scheme, device, torch_dtype, move_to_device, **kwargs) -> DiffusionPipeline`
- **Note**: Despite inheriting from `DiffusionPipeline`, the `from_pretrained` method returns the upstream `DiffusionPipeline` instance (not an `NeMoAutoDiffusionPipeline`). The class is used purely as a namespace for the factory method.

### `parallel_scheme: Dict[str, FSDP2Manager]`

- **File**: `/home/scbjtfy/Automodel/nemo_automodel/_diffusers/auto_diffusion_pipeline.py`, line 97
- **Type**: `Optional[Dict[str, FSDP2Manager]]`
- **Purpose**: The central integration data structure. Keys are Diffusers component names (strings like `"unet"`, `"transformer"`, `"text_encoder"`). Values are `FSDP2Manager` instances that define the parallelization strategy (TP size, DP size, CP size, etc.) for that component.
- **Consumed at**: Lines 124-135, where each component module is looked up by name and passed to `manager.parallelize(comp_module)`.

### `FSDP2Manager` (external dependency)

- **File**: `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/fsdp2.py`, line 34
- **Purpose**: Manages FSDP2 + TP parallelization. Its `parallelize(model: nn.Module) -> nn.Module` method (line 272) applies tensor-parallel sharding plans and FSDP2 wrapping to a model. Returns either the same model (mutated in place) or a new wrapped model.
- **Used by**: `NeMoAutoDiffusionPipeline.from_pretrained` calls `manager.parallelize(comp_module)` for each component in `parallel_scheme`. If the returned module is a different object, it is set back onto the pipeline via `setattr(pipe, comp_name, new_m)` (line 133).

### `DIFFUSERS_AVAILABLE: bool` (module-level flag)

- **File**: `/home/scbjtfy/Automodel/nemo_automodel/_diffusers/auto_diffusion_pipeline.py`, line 29
- **Purpose**: Guards against missing `diffusers` dependency. Checked at line 103 inside `from_pretrained` to produce a clear error message.

### Helper Functions

| Function | Location (line) | Signature | Purpose |
|----------|-----------------|-----------|---------|
| `_choose_device` | Line 37 | `(device: Optional[torch.device]) -> torch.device` | Resolves target device: returns explicit device if given, else `cuda:{LOCAL_RANK}` if CUDA available, else `cpu`. |
| `_iter_pipeline_modules` | Line 46 | `(pipe: DiffusionPipeline) -> Iterable[Tuple[str, nn.Module]]` | Yields `(name, module)` pairs. Prefers the `pipe.components` registry dict (standard in modern Diffusers); falls back to scanning public attributes for `nn.Module` instances. |
| `_move_module_to_device` | Line 66 | `(module: nn.Module, device: torch.device, torch_dtype: Any) -> None` | Moves a single module to device+dtype. Handles `"auto"` (skip dtype cast), `torch.dtype` objects, and string dtypes via `dtype_from_str`. |

---

## 4. State Flow

### Pipeline Loading and Parallelization Flow

The entire flow is contained in `NeMoAutoDiffusionPipeline.from_pretrained` (lines 92-136). Here is the step-by-step sequence:

```
1. GUARD: Check DIFFUSERS_AVAILABLE (line 103)
   - If False: raise RuntimeError
   - If True: continue

2. LOAD: DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, ...) (line 108)
   - Delegates entirely to HuggingFace Diffusers
   - Returns a DiffusionPipeline instance with all components loaded
   - torch_dtype and extra **kwargs are forwarded

3. RESOLVE DEVICE: _choose_device(device) (line 116)
   - Explicit device arg? -> use it
   - CUDA available? -> torch.device("cuda", LOCAL_RANK)
   - Else -> torch.device("cpu")

4. MOVE TO DEVICE (conditional on move_to_device=True, default) (lines 119-121)
   - For each (name, module) in _iter_pipeline_modules(pipe):
     - _move_module_to_device(module, dev, torch_dtype)
       - "auto" -> module.to(device=dev) [no dtype cast]
       - string -> dtype_from_str(torch_dtype) -> module.to(device=dev, dtype=resolved)
       - torch.dtype -> module.to(device=dev, dtype=torch_dtype)

5. PARALLELIZE (conditional on parallel_scheme is not None) (lines 124-135)
   - Assert torch.distributed.is_initialized()
   - For each (comp_name, comp_module) in _iter_pipeline_modules(pipe):
     - Look up manager = parallel_scheme.get(comp_name)
     - If no manager for this component: skip (only listed components are parallelized)
     - Try: new_m = manager.parallelize(comp_module)
       - If new_m is a different object: setattr(pipe, comp_name, new_m)
       - If same object: no-op (parallelization was in-place)
     - Except: log warning, continue to next component

6. RETURN: pipe (line 136)
   - Returns the (now parallelized, device-placed) DiffusionPipeline instance
```

### Component Discovery Flow (`_iter_pipeline_modules`)

```
1. Check if pipe has a `components` attribute that is a dict (line 48)
   - YES (modern Diffusers): iterate pipe.components.items(), yield only nn.Module values
   - NO (fallback): iterate dir(pipe), skip private attrs (_*), yield nn.Module attributes
```

This two-path discovery ensures compatibility with both modern Diffusers (which has a `components` registry) and older or custom pipeline classes that expose modules as plain attributes.

### Real-World Usage Flow (Wan2.2 Example)

From `examples/diffusion/wan2.2/wan_generate.py`:

```
1. initialize_distributed(backend="nccl") (line 80)
2. Load VAE separately: AutoencoderKLWan.from_pretrained(...) (line 97)
3. Create FSDP2Manager with TP/CP/PP/DP sizes (lines 101-109)
4. Build parallel_scheme = {"transformer": fsdp2_manager, "transformer_2": fsdp2_manager} (lines 113-115)
5. NeMoAutoDiffusionPipeline.from_pretrained(
     "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
     vae=vae,             # pre-loaded VAE passed via kwargs (forwarded to Diffusers)
     torch_dtype=bf16,
     device=device,
     parallel_scheme=parallel_scheme,
   ) (lines 118-124)
6. Run inference: pipe(prompt=..., ...) (lines 133-141)
7. Export video on rank 0 (lines 143-144)
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding Support for a New Parallelization Backend (e.g., MegatronFSDP)

**Goal**: Allow `parallel_scheme` to accept `MegatronFSDPManager` in addition to `FSDP2Manager`.

**What to change**:
- The `parallel_scheme` type hint at line 97 of `auto_diffusion_pipeline.py` currently specifies `Dict[str, FSDP2Manager]`. Change this to accept any manager that implements a `parallelize(module: nn.Module) -> nn.Module` interface. You could define a `Protocol` or use `Union[FSDP2Manager, MegatronFSDPManager]`.
- The actual runtime code (lines 124-135) only calls `manager.parallelize(comp_module)`, so any object with that method already works at runtime -- the change is primarily for type safety and documentation.
- Update the import at line 22 if the type annotation references additional classes.
- Add test cases in `tests/unit_tests/_diffusers/test_auto_diffusion_pipeline.py` following the existing pattern in `test_from_pretrained_parallel_scheme_applies_managers_and_sets_attrs` (line 175).

### Scenario 2: Adding Post-Load Hooks (e.g., Activation Checkpointing, torch.compile)

**Goal**: Apply additional transformations to pipeline components after loading and parallelization.

**What to change**:
- Add a new optional parameter to `from_pretrained`, e.g., `post_hooks: Optional[Dict[str, Callable[[nn.Module], nn.Module]]]`, at line 100 of `auto_diffusion_pipeline.py`.
- After the parallelization loop (after line 135), add a new loop that iterates over `_iter_pipeline_modules(pipe)` and applies any matching hook:
  ```python
  if post_hooks is not None:
      for comp_name, comp_module in _iter_pipeline_modules(pipe):
          hook = post_hooks.get(comp_name)
          if hook is not None:
              new_m = hook(comp_module)
              if new_m is not comp_module:
                  setattr(pipe, comp_name, new_m)
  ```
- Follow the same error-handling pattern (try/except with warning log) used in the parallelization loop.
- Add corresponding tests in `tests/unit_tests/_diffusers/test_auto_diffusion_pipeline.py`.

### Scenario 3: Supporting Pipeline-Parallel (PP) Stage Splitting for Diffusion Models

**Goal**: Split a large diffusion transformer across multiple pipeline-parallel stages, analogous to how LLM recipes use PP.

**What to change**:
- Extend `parallel_scheme` to optionally carry PP configuration per component. This could be a new data structure, e.g., `ParallelConfig(fsdp2_manager: FSDP2Manager, pp_split_fn: Optional[Callable])`, or a separate `pp_scheme: Dict[str, PipelineParallelConfig]` parameter on `from_pretrained`.
- In the parallelization loop (lines 124-135 of `auto_diffusion_pipeline.py`), after calling `manager.parallelize`, apply the PP split if configured. This would need to coordinate with `FSDP2Manager` which already has a `pp_size` attribute (as seen in the Wan2.2 example, line 105).
- The `_iter_pipeline_modules` helper may need extension to handle split sub-modules if PP produces multiple stage objects for a single original component.
- Add functional tests under `tests/functional_tests/` since PP requires multiple GPUs.

### Scenario 4: Adding Component-Specific Dtype Override

**Goal**: Allow different dtype per component (e.g., transformer in bf16, VAE in fp32).

**What to change**:
- Add a new parameter `component_dtypes: Optional[Dict[str, Any]] = None` to `from_pretrained` at line 100 of `auto_diffusion_pipeline.py`.
- In the device-move loop (lines 119-121), look up per-component dtype override:
  ```python
  for name, module in _iter_pipeline_modules(pipe):
      comp_dtype = (component_dtypes or {}).get(name, torch_dtype)
      _move_module_to_device(module, dev, comp_dtype)
  ```
- The existing `_move_module_to_device` helper already handles all dtype formats, so no changes are needed there.
- Add tests covering mixed-dtype scenarios.

### Scenario 5: Registering Custom Diffusion Pipeline Types

**Goal**: Support custom pipeline classes that are not registered in HuggingFace's `DiffusionPipeline` auto-detection.

**What to change**:
- Add a `pipeline_class: Optional[type] = None` parameter to `from_pretrained` at line 97 of `auto_diffusion_pipeline.py`.
- At line 108, instead of always calling `DiffusionPipeline.from_pretrained`, use `(pipeline_class or DiffusionPipeline).from_pretrained(...)`.
- This follows the same pattern HuggingFace uses internally in `DiffusionPipeline.from_pretrained` with its `custom_pipeline` kwarg, but gives AutoModel explicit control.
- The rest of the flow (device placement, parallelization) works unchanged since it operates on generic `nn.Module` discovery via `_iter_pipeline_modules`.
