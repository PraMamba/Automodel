---
name: automodel-dr-autonvtx
description: Use when working with the autonvtx module of automodel — autonomous NVTX annotation for profiling
---

# autonvtx Module Deep-Read

## 1. Module Purpose & Capabilities

The `autonvtx` module (`/home/scbjtfy/Automodel/nemo_automodel/autonvtx/__init__.py`) provides automatic NVIDIA Tools Extension (NVTX) range annotations for PyTorch `nn.Module` hierarchies. It recursively patches every module in a model tree with NVTX push/pop hooks on forward and (optionally) backward passes, enabling fine-grained GPU profiling in tools like Nsight Systems without requiring any manual annotation in model code.

Inspired by the open-source project at `https://github.com/zasdfgbnm/autonvtx`, this implementation adds thread-safety and recursion guards to handle activation checkpointing correctly, where forward passes are re-executed during the backward pass.

### Public API

There is a single public entry point:

```python
def patch(model, name=None, add_backward_hooks=True) -> nn.Module
```

- **`model`** (`nn.Module`): The root module to annotate. All children are patched recursively.
- **`name`** (`str | None`): Optional human-readable label. When `None`, the class name is used (e.g., `"LlamaForCausalLM"`). For child modules, the name becomes `"{child_name}: {ClassName}"`.
- **`add_backward_hooks`** (`bool`): When `True` (default), backward passes are also annotated with NVTX ranges.
- **Returns**: The same `model` object, now instrumented with hooks.

The module exports only `patch` via `__all__ = ["patch"]` (line 97).

## 2. Core Design Logic

### Why This Design Exists

NVTX annotations are essential for GPU profiling, but manually inserting `torch.cuda.nvtx.range_push()` / `range_pop()` calls throughout a model's forward/backward code is error-prone and creates maintenance burden. This module solves the problem by using PyTorch's hook mechanism to automatically inject NVTX ranges at module boundaries.

### Key Architectural Decisions

**Decision 1: Hook-based instrumentation instead of code modification.**
Rather than monkey-patching `forward()` methods, the module uses PyTorch's first-class hook API (`register_forward_pre_hook`, `register_forward_hook`, `register_full_backward_pre_hook`, `register_full_backward_hook`). This is non-invasive -- hooks compose cleanly with other features like FSDP2, pipeline parallelism, and torch.compile.

**Decision 2: Thread-local recursion guard for activation checkpointing safety.**
Activation checkpointing (gradient checkpointing) re-runs forward passes during the backward pass. Without protection, this would create nested NVTX ranges with the same name, leading to corrupted profiling timelines. The module uses a thread-local `set` (`_thread_local.active_ranges` at line 23) to track which NVTX range names are currently open. If `push_fwd` is called while its range name is already active (indicating a re-entrant forward from checkpointing), the push is skipped and the module is flagged with `_nvtx_skipped = True` so the corresponding pop is also skipped (lines 39-43, 47-50).

**Decision 3: Idempotent patching via `_nvtx_patched` sentinel.**
Both `_add_nvtx_hooks` (line 35-36) and `patch` (line 83-84) check for a `_nvtx_patched` attribute on the module before proceeding. This prevents double-hooking if `patch()` is called multiple times on the same model or overlapping subgraphs.

**Decision 4: Lazy import in recipes.**
The module is imported lazily (`import nemo_automodel.autonvtx as autonvtx`) only when `self.enable_nvtx` is `True` in the recipe (lines 951 and 958 of `train_ft.py`). This avoids any overhead when profiling is disabled, which is the common case.

**Decision 5: Hierarchical naming convention.**
Child modules receive names like `"layers: ModuleList"`, `"self_attn: LlamaAttention"` -- the `child_name` from `named_children()` is prepended, separated by `": "`. The root module uses the class name alone (or a user-supplied label). This produces a readable call-tree in Nsight Systems.

## 3. Core Data Structures

### Thread-Local State

| Symbol | Type | Location | Purpose |
|--------|------|----------|---------|
| `_thread_local` | `threading.local` | Line 23 | Thread-isolated storage for active range tracking |
| `_thread_local.active_ranges` | `set[str]` | Lazily created in `_get_active_ranges()` (lines 26-30) | Set of NVTX range name strings currently pushed but not yet popped on this thread |

### Module-Level Sentinels (set as attributes on `nn.Module` instances)

| Attribute | Type | Set In | Purpose |
|-----------|------|--------|---------|
| `_nvtx_patched` | `bool` (`True`) | `_add_nvtx_hooks` line 74 | Prevents double-hooking a module |
| `_nvtx_skipped` | `bool` | `push_fwd` lines 40-42, `push_bwd` lines 59-61 | Signals the corresponding pop hook to skip when the push was suppressed due to recursion |

### Functions

| Function | Location | Visibility | Purpose |
|----------|----------|------------|---------|
| `_get_active_ranges()` | Lines 26-30 | Private | Returns (lazily initializing) the thread-local active-ranges set |
| `_add_nvtx_hooks(model, name, add_backward_hooks)` | Lines 33-74 | Private | Registers four hooks (fwd pre, fwd post, bwd pre, bwd post) on a single module |
| `patch(model, name, add_backward_hooks)` | Lines 77-93 | Public | Recursively instruments an entire module tree |

### Hook Functions (closures inside `_add_nvtx_hooks`)

| Closure | Registered Via | Triggers |
|---------|----------------|----------|
| `push_fwd(module, *args, **kwargs)` | `register_forward_pre_hook` (line 52) | Before each forward call |
| `pop_fwd(module, *args, **kwargs)` | `register_forward_hook` (line 53) | After each forward call |
| `push_bwd(module, grad_input)` | `register_full_backward_pre_hook` (line 71) | Before backward through this module |
| `pop_bwd(module, grad_input, grad_output)` | `register_full_backward_hook` (line 72) | After backward through this module |

## 4. State Flow

### Entry Points

The sole entry point is `patch()`. It is called from `TrainFinetuneRecipeForNextTokenPrediction.setup()` in `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_ft.py`:

- **Non-pipeline path** (line 957-961): `autonvtx.patch(model, name=model.__class__.__name__)` -- patches the entire model as a single unit.
- **Pipeline-parallel path** (line 950-955): Iterates over `model.parts` (pipeline stages) and calls `autonvtx.patch(part, name=f"PipelineStage_{i}")` for each stage independently.

The `enable_nvtx` flag is set from the YAML config at line 817: `self.enable_nvtx = bool(self.cfg.get("nvtx", False))`. A YAML example exists at `/home/scbjtfy/Automodel/examples/llm_finetune/gemma/functiongemma_xlam.yaml` (line 110: `nvtx: false`).

### Data Flow During a Patched Forward Pass

1. **`patch(root_model, name)`** is called once during setup.
2. `patch` calls `_add_nvtx_hooks(root_model, name)`, which registers hooks on the root module and sets `root_model._nvtx_patched = True`.
3. `patch` then iterates `root_model.named_children()` and recursively calls `patch(child, child_name)` for each child.
4. During training, when `root_model.forward()` is called:
   - `push_fwd` fires: checks `_get_active_ranges()`. If `name` is not in the set, adds it and calls `torch.cuda.nvtx.range_push(name)`.
   - The module's actual `forward()` runs (which triggers child hooks in depth-first order).
   - `pop_fwd` fires: calls `torch.cuda.nvtx.range_pop()` and removes `name` from the active set.
5. During backward (if `add_backward_hooks=True`):
   - `push_bwd` fires with the same recursion guard logic.
   - Gradients flow through the module.
   - `pop_bwd` fires, popping the NVTX range.

### Recursion Guard Flow (Activation Checkpointing)

When activation checkpointing replays a forward during backward:
1. `push_fwd` is called for a module whose name is already in `_get_active_ranges()` (because backward hooks pushed it).
2. The guard at line 39 detects this, sets `module._nvtx_skipped = True`, and returns without pushing an NVTX range.
3. When `pop_fwd` fires, line 47 checks `_nvtx_skipped` and returns without popping.
4. Result: no spurious nested NVTX ranges, no unbalanced push/pop.

### Error Handling

The module has no explicit error handling (no try/except). If `torch.cuda.nvtx.range_push` or `range_pop` fails (e.g., CUDA not available), the exception propagates to the caller. The module assumes it will only be activated on CUDA-capable systems, which is enforced by the recipe-level `nvtx: true` config gate.

### Side Effects

- Mutates module instances by setting `_nvtx_patched` and `_nvtx_skipped` attributes.
- Registers hooks that persist for the lifetime of the module (no removal mechanism).
- Modifies thread-local state (`_thread_local.active_ranges`) during forward/backward execution.
- Calls `torch.cuda.nvtx.range_push()` / `range_pop()`, which emit NVTX markers visible to external profilers.

## 5. Common Modification Scenarios

### Scenario 1: Adding NVTX support to a new recipe (e.g., VLM fine-tuning)

To add profiling support to a new recipe at `/home/scbjtfy/Automodel/nemo_automodel/recipes/vlm/finetune.py`:

1. In the recipe's `__init__` or setup method, read the config flag: `self.enable_nvtx = bool(self.cfg.get("nvtx", False))`.
2. After model construction, conditionally import and patch:
   ```python
   if self.enable_nvtx:
       import nemo_automodel.autonvtx as autonvtx
       autonvtx.patch(model, name=model.__class__.__name__)
   ```
3. If the recipe supports pipeline parallelism, iterate over pipeline stages and patch each one with a `PipelineStage_{i}` name, following the pattern at lines 950-955 of `train_ft.py`.
4. Add `nvtx: false` to the recipe's default YAML config.

### Scenario 2: Adding selective module filtering (patch only certain layer types)

Currently `patch()` instruments every `nn.Module` in the tree. To add filtering:

1. Add a `filter_fn` parameter to `patch()` with a default of `None`:
   ```python
   def patch(model, name=None, add_backward_hooks=True, filter_fn=None):
   ```
2. Before calling `_add_nvtx_hooks`, check the filter:
   ```python
   if filter_fn is None or filter_fn(model, name):
       _add_nvtx_hooks(model, name, add_backward_hooks=add_backward_hooks)
   ```
3. Pass `filter_fn` through the recursive call at line 91.
4. Update `__all__` -- no change needed since `patch` is already exported.

Key files to modify:
- `/home/scbjtfy/Automodel/nemo_automodel/autonvtx/__init__.py` (add filter logic)
- `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_ft.py` (optionally pass filter_fn)

### Scenario 3: Adding hook removal / unpatch capability

The module currently provides no way to remove NVTX hooks. To add this:

1. In `_add_nvtx_hooks`, capture the hook handles returned by `register_forward_pre_hook`, etc.:
   ```python
   h1 = model.register_forward_pre_hook(push_fwd)
   h2 = model.register_forward_hook(pop_fwd)
   ```
2. Store handles on the module: `model._nvtx_hooks = [h1, h2, ...]`.
3. Add an `unpatch(model)` function that iterates `model.modules()`, calls `handle.remove()` for each stored hook, and deletes the `_nvtx_patched` sentinel.
4. Export `unpatch` via `__all__`.

Key file: `/home/scbjtfy/Automodel/nemo_automodel/autonvtx/__init__.py`.

### Scenario 4: Adding custom metadata to NVTX ranges (e.g., tensor shapes, dtypes)

NVTX supports payload/message annotations beyond simple string names. To enrich the profiling data:

1. Modify `push_fwd` to inspect `args`/`kwargs` and include shape info in the range name:
   ```python
   def push_fwd(module, input):
       shape_str = ""
       if isinstance(input, torch.Tensor):
           shape_str = f" [{list(input.shape)}]"
       elif isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
           shape_str = f" [{list(input[0].shape)}]"
       range_name = f"{name}{shape_str}"
       ...
   ```
2. This would change the range name dynamically, so the recursion guard logic in `_get_active_ranges()` would need adjustment -- the set key would need to remain the static `name` while the NVTX label uses the enriched string.

Key file: `/home/scbjtfy/Automodel/nemo_automodel/autonvtx/__init__.py`.

### Scenario 5: Integrating with the benchmark recipe

The benchmark recipe at `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/benchmark.py` already uses manual `torch.cuda.nvtx.range_push`/`range_pop` calls (lines 234, 249) and `torch.autograd.profiler.emit_nvtx` (line 211) for coarse-grained profiling. To add module-level granularity:

1. After model construction in the benchmark recipe, call `autonvtx.patch(model)`.
2. The existing manual `range_push("iteration_{i}_ga_step_{ga_step_idx}")` calls will nest correctly around the automatic per-module ranges, producing a two-level profiling hierarchy: iteration-level (manual) and module-level (automatic).
3. No changes to `autonvtx` itself are needed -- the ranges compose naturally because NVTX supports arbitrary nesting.
