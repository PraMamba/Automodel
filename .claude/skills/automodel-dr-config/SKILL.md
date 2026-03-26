---
name: automodel-dr-config
description: Use when working with the config module of automodel — YAML configuration system with CLI overrides, Hydra-style _target_ instantiation, and ConfigNode tree
---

# Config Module

**Source**: `nemo_automodel/components/config/`
**Files** (908 lines total):

| File | Lines | Role |
|------|-------|------|
| `loader.py` | 804 | Core engine: `ConfigNode` class, YAML loading, target resolution, env var interpolation, security policy |
| `_arg_parser.py` | 91 | CLI argument parsing: `--config` path extraction + `--dotted.path=value` overrides |
| `__init__.py` | 13 | License header only; public API is imported directly from submodules |

---

## 1. Module Purpose & Capabilities

The config module is the single entry point for all configuration in automodel. It provides:

- **YAML-to-object tree loading** -- `load_yaml_config(path)` in `loader.py` (line 792) reads a YAML file via `yaml.safe_load` and wraps the resulting dict in a recursive `ConfigNode` tree.
- **Hydra-style `_target_` instantiation** -- Any YAML node containing a `_target_` key is resolvable to a Python class/function. Calling `ConfigNode.instantiate()` (line 433) imports the target, passes sibling keys as kwargs, and returns the constructed object. Nested `_target_` nodes are instantiated recursively.
- **Automatic `*_fn` key resolution** -- Keys ending in `_fn` (e.g., `collate_fn`, `loss_fn`) are resolved to callables at parse time via `_resolve_target()` (line 265), not at instantiation time. This is handled in `ConfigNode._wrap()` (line 358).
- **CLI override system** -- `parse_cli_argv()` in `_arg_parser.py` (line 20) extracts `--config <path>` and collects any `--dotted.path=value` overrides. `parse_args_and_load_config()` (line 77) loads the YAML then applies each override via `ConfigNode.set_by_dotted()`.
- **Environment variable interpolation** -- `resolve_yaml_env_vars()` in `loader.py` (line 176) supports `${VAR}`, `${VAR,default}`, `$VAR`, and back-compat `${oc.env:VAR}` / `${oc.env:VAR,default}` forms inside string values.
- **Secret-safe printing** -- Resolved env vars are wrapped in `_OrigValueStr` (line 164) so that `str(cfg)` and `repr(cfg)` display the original placeholder (e.g., `${oc.env:HF_TOKEN}`) rather than the secret value. Sensitive keys (password, token, api_key, etc.) are redacted by `_redact()` (line 108) in error messages.
- **Automatic type coercion** -- `translate_value()` (line 120) converts YAML string values to Python types: `"123"` becomes `123`, `"True"` becomes `True`, `"[1,2,3]"` becomes `[1,2,3]`, etc., using `ast.literal_eval` with special-case handling.
- **Security policy for imports** -- `_resolve_target()` enforces an allowlist (`ALLOWED_IMPORT_PREFIXES` at line 32) and blocks private/dunder attribute traversal (`_is_safe_attr()` at line 103). User-defined modules can be opted-in via `NEMO_ENABLE_USER_MODULES=1` env var or `set_enable_user_modules(True)` (line 51).

---

## 2. Core Design Logic

### Why a custom config system instead of Hydra/OmegaConf?

The module is a deliberate lightweight replacement for Hydra/OmegaConf. It keeps the same `_target_` convention that the ML community is familiar with but avoids Hydra's complexity (config groups, defaults lists, multirun). The entire system is three files, zero external config DSL dependencies, and the `ConfigNode` is a plain Python object (attribute access, no schema compilation).

### Config resolution order

1. **YAML parse** -- `yaml.safe_load` produces a raw Python dict.
2. **Recursive wrapping** -- `ConfigNode.__init__()` calls `_wrap(k, v)` for every key-value pair:
   - `dict` values become nested `ConfigNode` instances.
   - `list` values are recursively wrapped element-by-element.
   - `_target_` and `*_fn` keys are resolved immediately to Python callables via `_resolve_target()`.
   - All other string values pass through env var resolution (`resolve_yaml_env_vars`) then type coercion (`translate_value`).
3. **CLI overrides** -- Applied after YAML load via `set_by_dotted()`, which creates intermediate `ConfigNode` objects as needed and wraps the leaf value through `_wrap()`.
4. **Lazy instantiation** -- `_target_` callables are resolved at parse time, but the actual object construction happens only when `instantiate()` is explicitly called by recipe code.

### Why `_target_` is resolved at parse time but objects are constructed lazily

Resolving `_target_` to a callable during `_wrap()` means import errors surface immediately at config load, before any GPU setup or distributed init. But construction is deferred to `instantiate()` so that recipes can inspect, modify, or conditionally skip config subtrees before committing resources.

### Raw config preservation

`ConfigNode.__init__()` stores `self._raw_config = deepcopy(d)` (line 342). This immutable snapshot is used by the checkpoint system (`save_config` in the checkpoint module) to persist the original YAML structure. Finetune scripts may modify the live config in place, but `raw_config` preserves the starting state.

### Env var interpolation with secret safety

When a string value contains `$`, `_wrap()` resolves it via `resolve_yaml_env_vars()` and wraps the result in `_OrigValueStr(resolved, original)` (line 387). This subclass of `str` carries the original token. Both `__repr__()` (line 709) and `to_yaml_dict(use_orig_values=True)` (line 567) check for `_orig_value` and display the placeholder instead of the secret. This means `print(cfg)` never leaks tokens or passwords.

---

## 3. Core Data Structures

### `ConfigNode` (loader.py, line 325)

The central data structure. A recursive attribute-access wrapper around a YAML dict.

**Key instance attributes:**
- `_raw_config: dict` -- Deep copy of the original dict, immutable after init.
- `_original_strings: dict` -- Maps `_target_` and `*_fn` key names to their original string values before `_resolve_target()` converted them to callables. Used by `get_as_string()`.
- `raise_on_missing_attr: bool` -- Controls whether `__getattr__` raises `AttributeError` (default `True`) or returns `None`.
- All other entries from the YAML dict are set as instance attributes via `__dict__.update()`.

**Key methods:**
- `instantiate(*args, **kwargs)` (line 433) -- Calls `_resolve_target(self._target_)`, collects all non-internal attributes as kwargs (recursively instantiating nested `_target_` nodes via `_instantiate_value`), merges explicit `kwargs` overrides, resolves env vars, then calls `func(*args, **config_kwargs)`.
- `instantiate_path(dotted_path, default=None, *args, **kwargs)` (line 401) -- Convenience: `self.get(dotted_path, default).instantiate(...)`. Used in recipes like `self.cfg.instantiate_path("peft")`.
- `get(key, default=None)` (line 659) -- Dotted-path traversal. Supports list indexing (e.g., `"arr.values.1"`). Returns `default` on missing path.
- `set_by_dotted(dotted_key, value)` (line 693) -- Creates intermediate `ConfigNode` objects for missing path segments, then wraps and sets the leaf.
- `__contains__(key)` (line 771) -- Dotted-path membership test.
- `to_dict()` (line 512) -- Recursively unwraps to plain Python dicts/lists.
- `to_yaml_dict(resolve_env, redact_sensitive, use_orig_values)` (line 567) -- YAML-serialization-ready dict. Converts callables back to dotted path strings via `_to_dotted_path()`. Optionally resolves env vars and redacts sensitive keys.
- `get_as_string(key, default=None)` (line 637) -- Returns the original import path string for `_target_` and `*_fn` keys.
- `__repr__` / `__str__` (lines 709, 760) -- Indented tree display. Defaults to showing original env var placeholders (safe logging).

### `_OrigValueStr` (loader.py, line 164)

A `str` subclass that stores the original unresolved value. Created when `_wrap()` resolves an env var in a string. Has two extra attributes:
- `_orig_value: str` -- The original token (e.g., `${oc.env:HF_TOKEN}`).
- `_no_env_resolve: bool` -- Set to `True` to prevent double-resolution.

### `translate_value(v)` (loader.py, line 120)

Not a data structure but a critical transform. Converts string tokens to Python objects:
- Special symbols: `"none"/"None"` -> `None`, `"true"/"True"` -> `True`, `"false"/"False"` -> `False`.
- `ast.literal_eval` for numbers, dicts, lists, tuples.
- Falls back to the raw string for anything unparseable.
- Strings longer than 1000 chars are returned as-is (safety guard against pathological inputs).

### Security constants (loader.py, lines 32-48)

- `ALLOWED_IMPORT_PREFIXES` (line 32): `("nemo_automodel", "torch", "transformers", "torchdata", "torchao", "liger_kernel")` -- Modules that `_resolve_target()` will import without opt-in.
- `SAFE_BASE_DIR` (line 35): Resolved to `nemo_automodel/` parent (repo root). Used for file-path `_target_` safety checks.
- `ENABLE_USER_MODULES` (line 38): Controlled by `NEMO_ENABLE_USER_MODULES` env var. When True, all import restrictions are bypassed.
- `SENSITIVE_KEY_SUBSTRINGS` (line 40): Tuple of substrings (`password`, `secret`, `token`, `apikey`, `api_key`, `authorization`, `auth`) used by `_redact()` to mask values in error output.

---

## 4. State Flow

### Full lifecycle: YAML file to running object

```
YAML file on disk
       |
       v
load_yaml_config(path)            [loader.py:792]
  yaml.safe_load(f)  -->  raw dict
  ConfigNode(raw)    -->  recursive _wrap()
       |
       |  For each key-value pair:
       |    dict        --> nested ConfigNode (recurse)
       |    list        --> wrapped element-by-element
       |    _target_    --> _resolve_target() --> callable stored as attribute
       |    *_fn key    --> _resolve_target() --> callable stored as attribute
       |    "$..." str  --> resolve_yaml_env_vars() --> translate_value() --> _OrigValueStr or typed value
       |    other str   --> translate_value() --> int/float/bool/list/dict/str
       |
       v
ConfigNode tree (in memory)
       |
       v  (optional)
parse_args_and_load_config()       [_arg_parser.py:77]
  parse_cli_argv()  -->  (cfg_path, overrides)
  load_yaml_config(cfg_path)
  for each override:
    cfg.set_by_dotted(key, translate_value(val))
       |
       v
Recipe code (e.g., train_ft.py, base_recipe.py)
  cfg.model.instantiate()   -->  _resolve_target(self._target_) then func(**config_kwargs)
  cfg.dataset.instantiate()
  cfg.optimizer.instantiate()
  cfg.instantiate_path("peft", default=None)
       |
       v
Running Python objects (model, dataset, optimizer, etc.)
```

### `_resolve_target()` resolution paths (loader.py, line 265)

Two forms are supported:

1. **File path with colon** -- `"path/to/file.py:ClassName"`:
   - Validates `.py` suffix and file existence.
   - Calls `load_module_from_file()` to dynamically import.
   - Checks `_is_safe_attr()` on the attribute name.
   - Returns `getattr(module, attr)`.

2. **Dotted module path** -- `"torch.optim.Adam"`:
   - Tries longest-prefix module import: `torch.optim.Adam`, then `torch.optim`, then `torch`.
   - Each prefix is checked against `_is_allowed_module()`.
   - Remaining segments are resolved via `getattr()` with `_is_safe_attr()` checks.

### `instantiate()` internals (loader.py, line 433)

1. Calls `_resolve_target(self._target_)` (which may be a no-op if already a callable).
2. Builds `config_kwargs` from all non-internal attributes. For each value:
   - If it's a `ConfigNode` with `_target_` -> recursively `instantiate()` it.
   - If it's a plain `ConfigNode` -> `resolve_yaml_env_vars(v.to_dict())`.
   - If it's a list -> recursively process elements.
   - Otherwise -> `translate_value(resolve_yaml_env_vars(v))`.
3. Merges explicit `**kwargs` overrides (these take precedence).
4. Calls `resolve_yaml_env_vars(config_kwargs)` on the full dict (last-moment env resolution).
5. Calls `func(*args, **config_kwargs)`.
6. On failure: prints a diagnostic with the function signature, args, and redacted kwargs, then re-raises.

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new allowed import prefix

If you need `_target_` to resolve a new third-party package (e.g., `deepspeed`):

**Where**: `loader.py`, line 32, the `ALLOWED_IMPORT_PREFIXES` tuple.
**What**: Add the top-level package name to the tuple.
```python
ALLOWED_IMPORT_PREFIXES = ("nemo_automodel", "torch", "transformers", "torchdata", "torchao", "liger_kernel", "deepspeed")
```
**Why**: `_is_allowed_module()` (line 73) checks the top-level name against this tuple when `find_spec` returns `None` and the module is not in `sys.modules`. Without adding it, `_resolve_target("deepspeed.SomeClass")` would raise `ImportError: Cannot resolve target (blocked or not found)`.

### Scenario 2: Adding a new CLI flag beyond `--config`

If you need a new top-level CLI flag (e.g., `--dry-run`) that is not a config override:

**Where**: `_arg_parser.py`, `parse_cli_argv()` function (line 20).
**What**: Add a new `if tok in ("--dry-run",):` block before the generic `--` handler at line 50. Extract the value and return it alongside `cfg_path` and `overrides`.
**Impact**: The return signature of `parse_cli_argv()` and `parse_args_and_load_config()` would change. All callers in recipe entry points (e.g., `train_ft.py`, `finetune.py`) import `parse_args_and_load_config` and would need updating.

### Scenario 3: Adding a new special key convention (like `_target_` or `*_fn`)

If you need a new key suffix (e.g., `*_cls`) to auto-resolve to a class without instantiation:

**Where**: `ConfigNode._wrap()` in `loader.py` (line 358).
**What**: Add a new `elif` branch before the final `else`:
```python
elif k.endswith("_cls"):
    if isinstance(v, str):
        self._original_strings[k] = v
    return _resolve_target(v)
```
**Impact**: Also update `ConfigNode.instantiate()` (line 456) to skip `*_cls` keys the same way it skips `*_fn` keys (line 459), so they are passed as-is (callables) rather than through `_instantiate_value()`. Update `to_yaml_dict._convert()` (line 596) to recognize the new suffix as target-like.

### Scenario 4: Adding a new env var interpolation syntax

If you need to support a new form like `#{VAR}`:

**Where**: `resolve_yaml_env_vars()` in `loader.py` (line 176), specifically the inner `_resolve_in_str()` function.
**What**: Add a new regex pattern and replacement function after the existing `braced_pattern` and `dollar_pattern` blocks.
**Impact**: Must also update the early-exit check `if "$" not in value` (line 191) to also check for `"#"`.

### Scenario 5: Making a ConfigNode key optional with a default

When recipe code needs to safely access a config key that may not exist:

**Where**: Recipe code (not the config module itself).
**What**: Use `cfg.get("dotted.path", default_value)` (line 659) or construct with `ConfigNode(d, raise_on_missing_attr=False)` so `__getattr__` returns `None` on missing keys instead of raising `AttributeError`.
**Example**: In `train_biencoder.py`: `self.dist_env = build_distributed(self.cfg.get("dist_env", {}))`.

### Scenario 6: Persisting modified config to YAML

To serialize the current (possibly modified) config tree back to YAML:

**Where**: Use `ConfigNode.to_yaml_dict()` (line 567).
**What**: Call `cfg.to_yaml_dict(use_orig_values=True, redact_sensitive=True)` to get a dict safe for logging/saving. Pass to `yaml.dump()`. The method converts callables back to dotted path strings, preserves env var placeholders, and redacts sensitive keys.
**Contrast with `to_dict()`**: `to_dict()` (line 512) does not convert callables to strings and does not handle env var placeholders -- it is meant for internal dict conversion, not serialization.
