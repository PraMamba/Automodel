---
name: automodel-dr-shared
description: Use when working with the shared module of automodel — provides utility functions for dtype conversion, safe imports, torch patches, and import helpers used across all other modules
---

# Module: `nemo_automodel/shared/`

## 1. Module Purpose & Capabilities

The `shared` module is the cross-cutting utility layer of NeMo AutoModel. It provides three categories of functionality used pervasively by components, recipes, and model implementations: (a) safe/deferred importing of optional dependencies so that missing GPU-only or optional packages do not crash at import time, (b) a `dtype_from_str` converter that translates YAML/CLI string representations of torch dtypes into actual `torch.dtype` objects, and (c) an idempotent set of monkey patches for PyTorch compatibility regressions. Unlike `components/`, this module has no independence constraints -- it is intentionally imported by everything.

### Public API Surface

**Exported from `__init__.py`:**

- `dtype_from_str(val, default=torch.bfloat16) -> torch.dtype` -- The only symbol in `__all__`.

**From `import_utils.py` (imported directly by consumers):**

| Symbol | Type | Purpose |
|---|---|---|
| `safe_import(module, *, msg=None, alt=None)` | function | Import a module; on failure return `(False, placeholder)` instead of raising |
| `safe_import_from(module, symbol, *, msg=None, alt=None, fallback_module=None)` | function | Import a symbol from a module with same deferred-error semantics |
| `gpu_only_import(module, *, alt=None)` | function | Wrapper around `safe_import` with GPU-specific error message |
| `gpu_only_import_from(module, symbol, *, alt=None)` | function | Wrapper around `safe_import_from` with GPU-specific error message |
| `is_unavailable(obj)` | function | Check if an object is an `UnavailableMeta` placeholder |
| `UnavailableError` | exception | Raised when a placeholder object is actually used |
| `UnavailableMeta` | metaclass | Creates placeholder classes that raise `UnavailableError` on any operation |
| `null_decorator` | context-manager/decorator | A no-op decorator/context-manager used as fallback when a real decorator (e.g., `triton.jit`) is unavailable |
| `get_torch_version()` | function | Returns `packaging.version.Version` for the installed PyTorch |
| `is_torch_min_version(version, check_equality=True)` | function | Boolean check: is installed torch >= (or >) a version string |
| `get_te_version()` | function | Returns `packaging.version.Version` for TransformerEngine |
| `is_te_min_version(version, check_equality=True)` | function | Boolean check for TransformerEngine version |
| `get_transformers_version()` | function | Returns `packaging.version.Version` for HuggingFace transformers |
| `is_transformers_min_version(version, check_equality=True)` | function | Boolean check for transformers version |
| `get_check_model_inputs_decorator()` | function | Returns the correct `check_model_inputs` decorator for the installed transformers version, handling the API change at 4.57.3 |
| `MISSING_TRITON_MSG` | str constant | Human-readable install instructions for triton |
| `MISSING_QWEN_VL_UTILS_MSG` | str constant | Human-readable install instructions for qwen-vl-utils |
| `MISSING_CUT_CROSS_ENTROPY_MSG` | str constant | Human-readable install instructions for cut-cross-entropy |
| `MISSING_TORCHAO_MSG` | str constant | Human-readable install instructions for torchao |
| `GPU_INSTALL_STRING` | str constant | Generic GPU package install instructions |

**From `torch_patches.py` (imported directly by consumers):**

| Symbol | Type | Purpose |
|---|---|---|
| `apply_torch_patches()` | function | Idempotent function that applies monkey patches for PyTorch compatibility issues |

**From `utils.py`:**

| Symbol | Type | Purpose |
|---|---|---|
| `dtype_from_str(val, default=torch.bfloat16)` | function | Convert a string like `"torch.bfloat16"`, `"bf16"`, or `"float32"` to `torch.dtype` |

---

## 2. Core Design Logic

### Why deferred import errors?

NeMo AutoModel has many optional dependencies (triton, transformer-engine, torchao, cut-cross-entropy, qwen-vl-utils) that are only needed for specific code paths (e.g., triton kernels, FP8 training, specific VLM datasets). The alternative -- guarding every usage with `try/except` -- is noisy and error-prone. The `UnavailableMeta` pattern centralizes this: you import symbols eagerly at module scope, and if the dependency is missing, you get a placeholder that looks like a real class/module but raises `UnavailableError` the moment it is actually *used*. This means:

1. Modules can be imported on CPU-only machines without crashing.
2. Errors appear at *usage* time with clear install instructions, not at import time with cryptic tracebacks.
3. `isinstance` checks against placeholders correctly return `False` rather than crashing.

### Why the metaclass approach?

`UnavailableMeta` is a *metaclass*, not a regular class. This matters because the placeholder is itself a *class* (not an instance), so overriding dunder methods at the metaclass level ensures that operations like `placeholder + 1`, `len(placeholder)`, `hash(placeholder)`, and `placeholder()` all raise `UnavailableError`. A regular class with dunder overrides would only catch operations on instances, not on the class object itself.

The metaclass overrides ~30 dunder methods covering: comparison (`__eq__`, `__lt__`, etc.), arithmetic (`__add__`, `__mul__`, etc.), unary (`__neg__`, `__abs__`, `__invert__`), container protocol (`__len__`, `__iter__`, `__setitem__`, `__delitem__`), descriptor protocol (`__get__`, `__delete__`), context manager (`__enter__`), and `__hash__`/`__index__`. The naming convention `MISSING{name}` (set in `__new__`) makes placeholders identifiable in tracebacks.

### Why `null_decorator` is a context manager

The `null_decorator` is decorated with `@contextmanager` but also works as a plain decorator. This dual nature exists because it substitutes for triton decorators (`triton.jit`, `triton.autotune`, `triton.heuristics`) which may be used either as `@triton.jit` (plain decorator) or `@triton.autotune(configs=[...])` (decorator factory / context manager). The implementation checks: if called with a single callable argument, return it directly; otherwise return an identity wrapper. This lets it replace both usage patterns seamlessly.

### Why torch patches are lazy

The docstring in `torch_patches.py` is explicit: patches are *not* applied at `import nemo_automodel` time. This keeps tokenizer-only imports (which do not need torch at all) lightweight. Instead, `apply_torch_patches()` is called from exactly two entry points: `nemo_automodel/_transformers/auto_model.py` (module scope, runs when any auto-model is loaded) and `nemo_automodel/recipes/base_recipe.py` (module scope, runs when any recipe is loaded). A module-level boolean `_TORCH_PATCHES_APPLIED` ensures idempotency.

### Why `dtype_from_str` uses a lookup table

YAML configs specify dtypes as strings (e.g., `"torch.bfloat16"` or `"bf16"`). A lookup table (rather than `getattr(torch, ...)`) provides: (a) case-insensitive matching, (b) support for aliases like `"bf16"`, `"half"`, `"double"`, `"long"`, (c) controlled scope -- only known dtypes are accepted, preventing injection of arbitrary torch attributes. If `val` is already a `torch.dtype`, it passes through unchanged. If `val` is `None`, the `default` parameter is returned.

### Trade-offs

- **Broad dunder coverage in UnavailableMeta**: Not every possible Python dunder is covered (e.g., `__or__`, `__and__`, `__contains__`). Uncovered operations will raise `TypeError` instead of `UnavailableError`, which is less informative but still prevents silent misuse.
- **Module-level logger in import_utils**: The logger is configured with `StreamHandler` and `INFO` level at import time. This is a side effect that can affect the root logger's handler list if the shared module is imported.
- **Version-pinned DeviceMesh patch**: Patch #2 in `torch_patches.py` is conditionally applied only for PyTorch `2.10.0` with `nv25.11`. This is fragile but intentional -- it targets a specific regression and includes a TODO to remove it.

---

## 3. Core Data Structures

### `UnavailableMeta` (metaclass)
**File:** `/home/scbjtfy/Automodel/nemo_automodel/shared/import_utils.py`, line 65

A Python metaclass. Classes created with this metaclass have:
- `_msg` (str): The error message shown when the placeholder is used. Set either explicitly or defaults to `"{name} could not be imported"`.
- `__name__` (str): Always `"MISSING{original_name}"`.

Every dunder method on the metaclass raises `UnavailableError(cls._msg)`.

### `UnavailableError` (exception)
**File:** `/home/scbjtfy/Automodel/nemo_automodel/shared/import_utils.py`, line 44

A simple exception subclass of `Exception`. No additional fields.

### `dtype_from_str` lookup table
**File:** `/home/scbjtfy/Automodel/nemo_automodel/shared/utils.py`, line 32

An inline dict mapping 21 string keys to `torch.dtype` values. Keys include both fully-qualified names (`"torch.float32"`) and shorthand (`"bf16"`). The lookup is case-insensitive (keys are compared after `.lower()`).

### Module-level constants (error messages)
**File:** `/home/scbjtfy/Automodel/nemo_automodel/shared/import_utils.py`, lines 31-41

Five string constants used as `msg` arguments when creating placeholders for commonly-missing optional dependencies:
- `GPU_INSTALL_STRING` -- generic CUDA pip install instructions
- `MISSING_TRITON_MSG` -- triton
- `MISSING_QWEN_VL_UTILS_MSG` -- qwen-vl-utils
- `MISSING_CUT_CROSS_ENTROPY_MSG` -- cut-cross-entropy
- `MISSING_TORCHAO_MSG` -- torchao

### `_TORCH_PATCHES_APPLIED` (module-level boolean)
**File:** `/home/scbjtfy/Automodel/nemo_automodel/shared/torch_patches.py`, line 28

A global flag ensuring `apply_torch_patches()` runs at most once.

---

## 4. State Flow

### Safe Import Flow

```
Caller calls safe_import("some_module")
  |
  +-> importlib.import_module("some_module")
  |     |
  |     +-> Success: return (True, module_object)
  |     +-> ImportError: log debug traceback
  |     +-> Other Exception: re-raise immediately
  |
  +-> ImportError path:
        |
        +-> alt provided? return (False, alt)
        +-> alt not provided? return (False, UnavailableMeta(name, (), {"_msg": msg}))
```

`safe_import_from` follows the same pattern but adds:
- An `AttributeError` path: if the module imports but the symbol is missing, it optionally tries `fallback_module` via recursive call (with `fallback_module=None` to prevent infinite recursion).

`gpu_only_import` and `gpu_only_import_from` are thin wrappers that inject `GPU_INSTALL_STRING` into the error message.

### Placeholder Usage Flow

```
Any operation on placeholder (call, attribute access, arithmetic, etc.)
  |
  +-> UnavailableMeta.__<dunder>__ is invoked (because placeholder is a class, not instance)
  |
  +-> raise UnavailableError(cls._msg)
```

Special case: `isinstance(some_obj, placeholder)` works via Python's built-in `isinstance` mechanism which calls `__instancecheck__` on the metaclass. Since `UnavailableMeta` does not override `__instancecheck__`, Python's default behavior applies and returns `False` without raising.

### dtype_from_str Flow

```
dtype_from_str(val, default=torch.bfloat16)
  |
  +-> val is None? return default (after asserting it is a torch.dtype)
  +-> val is already torch.dtype? return val as-is
  +-> val is str:
        |
        +-> lowercase val, check in lookup table -> return if found
        +-> prepend "torch." to lowercase val, check in lookup table -> return if found
        +-> raise KeyError("Unknown dtype string: {val}")
```

### Torch Patches Flow

```
apply_torch_patches()
  |
  +-> _TORCH_PATCHES_APPLIED is True? return immediately (idempotent)
  +-> import torch; if fails, return (no-op in torch-less environments)
  |
  +-> Patch #1: pin_memory compatibility
  |     Check if torch.utils.data._utils.pin_memory.pin_memory signature
  |     lacks "device" parameter. If so, wrap both pin_memory and
  |     _pin_memory_loop to accept and ignore the device argument.
  |
  +-> Patch #2: DeviceMesh _get_slice_mesh_layout regression fix
  |     Only for PyTorch 2.10.0+nv25.11. Replaces _MeshEnv._get_slice_mesh_layout
  |     with a version that bypasses the "stride < pre_stride" check and
  |     correctly handles _dim_group_names for sliced meshes.
  |
  +-> Set _TORCH_PATCHES_APPLIED = True
```

### Entry Points (where patches are applied)

1. `nemo_automodel/_transformers/auto_model.py` -- called at module scope (line 30): `apply_torch_patches()`
2. `nemo_automodel/recipes/base_recipe.py` -- called at module scope (line 28): `apply_torch_patches()`

Both are import-time side effects, meaning patches are applied as soon as any recipe or auto-model class is imported.

### Error Handling

- `safe_import` / `safe_import_from`: `ImportError` is caught and logged at DEBUG level. All other exceptions propagate. This is important -- a syntax error in an optional module will still crash.
- `get_torch_version()` / `get_te_version()` / `get_transformers_version()`: Return `PkgVersion("0.0.0")` on any failure, ensuring callers never crash from missing packages.
- `apply_torch_patches()`: Each patch is wrapped in its own `try/except`. If a patch fails, it logs at DEBUG and continues to the next patch. The function never raises.

### Side Effects

- `import_utils.py` at import time: creates a logger (`__name__`), sets it to INFO, adds a `StreamHandler`.
- `apply_torch_patches()`: mutates `torch.utils.data._utils.pin_memory.pin_memory`, `torch.utils.data._utils.pin_memory._pin_memory_loop`, and potentially `torch.distributed.device_mesh._MeshEnv._get_slice_mesh_layout`.

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new optional dependency

If you need to import from a new optional package (e.g., `flash_attn`):

1. **Add a `MISSING_*_MSG` constant** in `/home/scbjtfy/Automodel/nemo_automodel/shared/import_utils.py` (around line 41), following the pattern of existing messages:
   ```python
   MISSING_FLASH_ATTN_MSG = "flash_attn is not installed. Please install it with `pip install flash-attn`."
   ```

2. **Use `safe_import` or `safe_import_from` in the consuming module** (NOT in `shared/`). Example:
   ```python
   from nemo_automodel.shared.import_utils import MISSING_FLASH_ATTN_MSG, safe_import
   _ok, flash_attn = safe_import("flash_attn", msg=MISSING_FLASH_ATTN_MSG)
   ```
   The consuming code can then use `flash_attn` freely; it will only error if actually called on a machine without flash-attn.

3. **No changes needed in `__init__.py`** -- message constants are imported directly from `import_utils`.

### Scenario 2: Adding support for a new torch dtype string

If a new dtype alias is needed (e.g., `"fp8"` for `torch.float8_e4m3fn`):

1. **Edit the lookup table** in `/home/scbjtfy/Automodel/nemo_automodel/shared/utils.py` at line 32, inside the `lut` dict:
   ```python
   "fp8": torch.float8_e4m3fn,
   "torch.float8_e4m3fn": torch.float8_e4m3fn,
   ```

2. **Add test cases** in `/home/scbjtfy/Automodel/tests/unit_tests/shared/test_shared_utils.py` in the `test_cases` list of `test_dtype_from_str_valid_inputs`.

No other files need modification. The lookup is case-insensitive, so `"FP8"` will also work automatically.

### Scenario 3: Adding a new torch compatibility patch

If a new PyTorch version introduces a regression that needs patching:

1. **Add a new numbered patch section** in `/home/scbjtfy/Automodel/nemo_automodel/shared/torch_patches.py`, after the existing patches and before the `_TORCH_PATCHES_APPLIED = True` line (~line 147). Follow the existing pattern:
   ```python
   # -------------------------------------------------------------------------
   # Patch #3: <description>
   # -------------------------------------------------------------------------
   try:
       # Version check if applicable
       if "<version>" in _torch.__version__:
           # Apply patch
           pass
   except (ImportError, AttributeError) as e:
       _logger.debug(f"Could not apply <name> patch: {e}")
   ```

2. **Key constraints**: Each patch must be wrapped in its own `try/except` so that failure of one patch does not block others. Version-gating is strongly recommended. Add a TODO comment for removal when the upstream fix ships.

3. **No registration needed** -- `apply_torch_patches()` is a single function that runs all patches sequentially. The existing call sites in `auto_model.py` and `base_recipe.py` will automatically pick up the new patch.

### Scenario 4: Handling a breaking API change in a dependency

The `get_check_model_inputs_decorator()` function (line 447 of `import_utils.py`) demonstrates the pattern for handling API changes across dependency versions. If another dependency changes its API:

1. **Add a version check function** (like `get_transformers_version` / `is_transformers_min_version`) if one does not already exist for that dependency.
2. **Write a compatibility wrapper** that inspects the version and returns the correct API. Place it in `import_utils.py` alongside `get_check_model_inputs_decorator`.
3. **Consumers import the wrapper** instead of the dependency's symbol directly.

### Scenario 5: Changing what is exported from `nemo_automodel.shared`

Currently only `dtype_from_str` is in `__all__` (in `/home/scbjtfy/Automodel/nemo_automodel/shared/__init__.py`). To add another symbol:

1. Add the import in `__init__.py`.
2. Add the name to the `__all__` list.

However, most consumers import directly from the submodules (`from nemo_automodel.shared.import_utils import safe_import`) rather than from `nemo_automodel.shared`. The current convention in this codebase is to import from the specific submodule, so adding to `__all__` is only needed if you want a cleaner top-level API.

---

## File Summary

| File | Lines | Role |
|---|---|---|
| `nemo_automodel/shared/__init__.py` | 17 | Package init; re-exports `dtype_from_str` |
| `nemo_automodel/shared/import_utils.py` | 468 | Safe import system, UnavailableMeta, version checkers, error message constants |
| `nemo_automodel/shared/torch_patches.py` | 147 | Idempotent monkey patches for PyTorch compatibility regressions |
| `nemo_automodel/shared/utils.py` | 60 | `dtype_from_str` string-to-torch.dtype converter |
