---
name: automodel-dr-optim
description: Use when working with the optim module of automodel — optimizer creation and learning rate scheduler configuration
---

# Optim Module Deep Read

**Module path:** `nemo_automodel/components/optim/`
**Files:** 3 files, ~594 lines total
**Public API** (from `__init__.py`): `OptimizerParamScheduler`, `build_dion_optimizer`, `is_dion_optimizer`

---

## 1. Module Purpose & Capabilities

The optim module provides two capabilities:

1. **Learning rate and weight decay scheduling** via `OptimizerParamScheduler` (in `scheduler.py`). This is a standalone scheduler that wraps any `torch.optim.Optimizer` and manages per-step LR warmup, decay, and weight decay annealing. It is not a subclass of any PyTorch LR scheduler; it directly modifies `param_group["lr"]` and `param_group["weight_decay"]` values on every call to `step()`.

2. **Dion-family optimizer construction** via `build_dion_optimizer` and `is_dion_optimizer` (in `utils.py`). These functions detect whether a config targets a Dion/Muon/NorMuon/Dion2 optimizer and build it with intelligent parameter grouping (matrix params vs. scalar/vector params vs. embeddings vs. lm_head).

The module is a self-contained component under the project's architecture: it has zero imports from other `nemo_automodel/components/` packages and is consumed exclusively by recipe-layer code.

---

## 2. Core Design Logic

### Why a custom scheduler instead of `torch.optim.lr_scheduler`?

`OptimizerParamScheduler` (in `scheduler.py`) exists because it co-schedules **both** learning rate and weight decay in a single object with checkpoint save/restore semantics. PyTorch's built-in LR schedulers do not manage weight decay annealing, and they rely on a different step-counting convention. This scheduler uses a simple `step(increment)` interface where the caller passes how many steps have elapsed (always `1` during training; a restored `num_steps` value on checkpoint load). The scheduler computes the new LR and WD from scratch each time based on `self.num_steps`, making it fully deterministic from the step count alone.

The scheduler supports per-param-group overrides via `max_lr`, `min_lr`, `lr_mult`, and `wd_mult` keys in param groups (see `get_lr()` at line 188-189 and `step()` at line 255-256 of `scheduler.py`). This allows different parameter groups to have scaled learning rates and weight decays while sharing a single scheduler instance.

### Why a dedicated Dion builder instead of generic `cfg.instantiate()`?

Dion-family optimizers (Dion, Dion2, Muon, NorMuon from the `dion` package) require parameter grouping by tensor dimensionality: 2D (matrix) parameters use the Muon/Dion algorithm, while 1D (vector/bias), embedding, and lm_head parameters use a scalar optimizer (AdamW or Lion). The function `_separate_param_groups()` (in `utils.py`, line 41) performs this classification automatically. This grouping logic is too complex and optimizer-specific for the generic `cfg.instantiate(params=...)` path used by standard optimizers.

The builder also handles:
- Extracting the correct 1D `dp_shard_cp` submesh from a multi-dimensional `DeviceMesh` via `_get_dion_mesh()` (line 134).
- Introspecting the target constructor's signature to filter out unsupported kwargs (`inspect.signature` at line 189-191).
- Consuming config-only keys (`scalar_opt`, `scalar_betas`, `scalar_eps`, `scalar_lr`, `embed_lr`, `lm_head_lr`, `no_compile`) before passing remaining kwargs to the optimizer constructor.

### Checkpoint restore backward compatibility

`load_state_dict()` (line 298) supports old checkpoint key names: `start_lr` -> `max_lr`, `warmup_iter`/`warmup_steps` -> `lr_warmup_steps`, `end_iter`/`decay_steps` -> `lr_decay_steps`, `decay_style` -> `lr_decay_style`, `num_iters` -> `num_steps`. It also handles the case where weight decay fields are absent (partial restore).

---

## 3. Core Data Structures

### `OptimizerParamScheduler` (class, `scheduler.py` line 14)

The central scheduler class. Key attributes:

| Attribute | Type | Set by | Purpose |
|---|---|---|---|
| `optimizer` | `torch.optim.Optimizer` | constructor | The wrapped optimizer whose param_groups are modified |
| `num_steps` | `int` | `step()` / `load_state_dict()` | Current global step count (accumulates via increment) |
| `init_lr` | `float` | constructor | LR at step 0 (start of warmup) |
| `max_lr` | `float` | constructor | Peak LR (reached at end of warmup) |
| `min_lr` | `float` | constructor | Floor LR (reached at end of decay) |
| `lr_warmup_steps` | `int` | constructor | Steps for linear warmup from `init_lr` to `max_lr` |
| `lr_decay_steps` | `int` | constructor | Total steps over which decay schedule is defined |
| `lr_decay_style` | `str` | constructor | One of: `"constant"`, `"linear"`, `"cosine"`, `"inverse-square-root"`, `"WSD"` |
| `wsd_decay_steps` | `int` or `None` | constructor | Steps for final WSD annealing phase (required when `lr_decay_style="WSD"`) |
| `lr_wsd_decay_style` | `str` or `None` | constructor | Sub-decay style for WSD: `"linear"`, `"cosine"`, `"exponential"`, `"minus_sqrt"` |
| `start_wd` | `float` | constructor | Weight decay at step 0 |
| `end_wd` | `float` | constructor | Weight decay at end of `wd_incr_steps` |
| `wd_incr_steps` | `int` | constructor | Steps over which WD ramps from `start_wd` to `end_wd` |
| `wd_incr_style` | `str` | constructor | One of: `"constant"`, `"linear"`, `"cosine"` |
| `override_opt_param_scheduler` | `bool` | constructor | If True, `load_state_dict` uses class values, ignoring checkpoint |
| `use_checkpoint_opt_param_scheduler` | `bool` | constructor | If True, `load_state_dict` adopts checkpoint values silently |

Key methods:
- **`get_lr(param_group) -> float`** (line 181): Computes LR for a param group at current `num_steps`. Supports per-group `max_lr`/`min_lr` overrides.
- **`get_wd() -> float`** (line 156): Computes weight decay at current `num_steps`.
- **`step(increment: int)`** (line 244): Advances `num_steps` by `increment`, then sets `lr` and `weight_decay` on every param group. Applies `lr_mult` and `wd_mult` multipliers if present.
- **`state_dict() -> dict`** (line 258): Returns serializable state for checkpointing.
- **`load_state_dict(state_dict: dict)`** (line 298): Restores from checkpoint with backward-compatible key handling.

### `_separate_param_groups()` (function, `utils.py` line 41)

Classifies model parameters into four groups for Dion-family optimizers:

| Group | Classification Rule | Optimizer Algorithm | Weight Decay |
|---|---|---|---|
| `matrix_params` | `param.ndim == 2` and not in Embedding or lm_head | Default (Muon/Dion2) | From base config |
| `vector_params` | `param.ndim != 2` and not in Embedding or lm_head | `scalar_opt` (e.g., `"adamw"`) | Explicit `weight_decay` |
| `embed_params` | Parent module is `nn.Embedding` | `scalar_opt` | Always `0.0` |
| `lm_head_params` | `"lm_head"` in parameter name | `scalar_opt` | Always `0.0` |

The lm_head LR defaults to `base_lr / sqrt(d_in)` where `d_in` is the last dimension of the weight tensor (line 123-124), following Dion documentation recommendations.

### `is_dion_optimizer(cfg_opt) -> bool` (function, `utils.py` line 34)

Detection heuristic: returns `True` if `cfg_opt._target_.__module__` starts with `"dion"` OR `cfg_opt._target_.__name__` is in `{"Dion", "Dion2", "Muon", "NorMuon"}`.

### `build_dion_optimizer(cfg_opt, model, distributed_mesh) -> Any` (function, `utils.py` line 151)

Factory that:
1. Raises `RuntimeError` if `dion` package import failed (checked via module-level `_import_error`).
2. Pops config-only keys (`no_compile`, `scalar_opt`, `scalar_betas`, `scalar_eps`, `scalar_lr`, `embed_lr`, `lm_head_lr`).
3. If `no_compile=True`, disables `torch._dynamo` globally.
4. Calls `_separate_param_groups()` to build param groups.
5. Extracts 1D submesh via `_get_dion_mesh()`.
6. Filters kwargs by target constructor signature via `inspect.signature`.
7. Calls `target(param_groups, **cleaned_kwargs)` and returns the optimizer instance.

### `_get_dion_mesh(distributed_mesh) -> Any` (function, `utils.py` line 134)

Mesh extraction logic:
- Returns `None` if input is `None`.
- Returns input as-is if `ndim == 1` or `ndim` is missing.
- For multi-dimensional meshes, tries `mesh[("dp_replicate", "dp_shard_cp")]["dp_shard_cp"]` to extract the 1D data-parallel shard submesh.
- Falls back to returning the original mesh on `KeyError`/`RuntimeError`/`TypeError`.

---

## 4. State Flow

### Standard optimizer creation and scheduling flow

```
YAML config
    |
    v
build_model_and_optimizer()           [recipes/llm/train_ft.py, line 236-269]
    |
    |-- is_dion_optimizer(cfg_opt)?
    |       |
    |       |-- YES --> build_dion_optimizer(cfg_opt, model, mesh)
    |       |              |-- _separate_param_groups(model, ...)
    |       |              |-- _get_dion_mesh(mesh)
    |       |              |-- target(param_groups, **kwargs)
    |       |              `--> returns optimizer
    |       |
    |       `-- NO  --> cfg_opt.instantiate(params=trainable_params)
    |                      `--> returns standard optimizer (e.g., AdamW)
    |
    v
optimizer = [opt] (always wrapped in a list)
    |
    v
build_lr_scheduler(cfg, optimizer, step_scheduler)
    [recipes/llm/train_ft.py, line 596]
    |
    |-- For each opt in optimizer:
    |       |-- Extract base_lr from opt.param_groups[0]["lr"]
    |       |-- Compute defaults: init_lr = base_lr * 0.1, min_lr = base_lr * 0.01
    |       |-- lr_warmup_steps = min(1000, total_steps // 10)
    |       |-- Merge with user config overrides
    |       `-- OptimizerParamScheduler(optimizer=opt, ...)
    |
    v
lr_scheduler = [OptimizerParamScheduler, ...]
    |
    v
Training loop [recipes/llm/train_ft.py, line 1274-1276]:
    for scheduler in self.lr_scheduler:
        scheduler.step(1)    # Called after each optimizer step
    |
    v
Checkpoint save:
    scheduler.state_dict()   # Serializes num_steps, LR/WD config
    |
    v
Checkpoint restore:
    scheduler.load_state_dict(state_dict)  # Restores num_steps, resumes schedule
```

### Key integration points in recipes

- **`BaseRecipe`** (`recipes/base_recipe.py`, line 92-106): `is_lr_scheduler()` checks if an object is an `OptimizerParamScheduler` (or list of them) for automatic checkpoint tracking.
- **`BaseRecipe.save_checkpoint`** (line 260): Detects lr_scheduler among tracked state, saves via `state_dict()`.
- **`BaseRecipe.load_checkpoint`** (line 384): Detects lr_scheduler among tracked state, loads via `load_state_dict()`.
- **`BaseRecipe._log_model_and_optimizer_details`** (line 479): Logs scheduler `__repr__()` output.

### Checkpoint state dict format

Saved keys: `max_lr`, `lr_warmup_steps`, `num_steps`, `lr_decay_style`, `lr_decay_steps`, `min_lr`, `start_wd`, `end_wd`, `wd_incr_style`, `wd_incr_steps`.

Backward-compatible aliases on load: `start_lr`, `warmup_iter`, `warmup_steps`, `end_iter`, `decay_steps`, `decay_style`, `num_iters`.

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new LR decay style

1. In `scheduler.py`, add a new `elif` branch inside `get_lr()` (after line 237, before the `else` at line 238). The new branch must compute a `coeff` value between 0.0 and 1.0.
2. The formula is: `final_lr = min_lr + coeff * (max_lr - min_lr)`.
3. Add corresponding test cases in `tests/unit_tests/optim/test_scheduler.py` following the pattern of `test_get_lr_cosine_decay` (parametrized with `num_steps, expected_lr` pairs).
4. No changes needed in recipes or config loading -- the `lr_decay_style` string value comes from YAML config and is passed through directly.

**Files to modify:**
- `/home/scbjtfy/Automodel/nemo_automodel/components/optim/scheduler.py` -- add `elif` in `get_lr()`
- `/home/scbjtfy/Automodel/tests/unit_tests/optim/test_scheduler.py` -- add parametrized test

### Scenario 2: Supporting a new Dion-family optimizer variant

1. In `utils.py`, add the new optimizer class name to the detection set in `is_dion_optimizer()` (line 38): update `name in {"Dion", "Dion2", "Muon", "NorMuon", "NewOptimizer"}`.
2. If the new optimizer requires different parameter grouping logic (e.g., different treatment of 3D tensors), modify `_separate_param_groups()` (line 41).
3. If the new optimizer accepts non-standard constructor arguments, they will be passed through automatically if they match the constructor signature (handled by `inspect.signature` filtering at line 189-191). If they need special preprocessing, add `pop` logic in `build_dion_optimizer()` similar to the `scalar_opt`/`scalar_betas`/etc. handling (lines 179-184).
4. Add the optimizer to the optional import block at the top of `utils.py` (line 26): `from dion import Dion, Dion2, Muon, NorMuon, NewOptimizer`.
5. Create a YAML config example in `examples/` following `examples/llm_finetune/qwen/qwen2_5_7b_squad_muon.yaml`.
6. Add tests in `tests/unit_tests/optim/test_dion_optimizer_utils.py` following `TestIsDionOptimizer` and `TestBuildDionOptimizer` patterns.

**Files to modify:**
- `/home/scbjtfy/Automodel/nemo_automodel/components/optim/utils.py` -- update detection and optional imports
- `/home/scbjtfy/Automodel/tests/unit_tests/optim/test_dion_optimizer_utils.py` -- add tests

### Scenario 3: Adding a new weight decay annealing style

1. In `scheduler.py`, add a new `elif` branch inside `get_wd()` (after line 175, before the `else` at line 177). Compute a `coeff` between 0.0 and 1.0.
2. The formula is: `wd = start_wd + coeff * (end_wd - start_wd)`.
3. Add parametrized test cases in `tests/unit_tests/optim/test_scheduler.py` following `test_get_wd_linear` / `test_get_wd_cosine` patterns.

**Files to modify:**
- `/home/scbjtfy/Automodel/nemo_automodel/components/optim/scheduler.py` -- add `elif` in `get_wd()`
- `/home/scbjtfy/Automodel/tests/unit_tests/optim/test_scheduler.py` -- add parametrized test

### Scenario 4: Adding per-group LR categories to Dion builder

To add a new parameter category (e.g., attention heads with a separate LR):

1. In `_separate_param_groups()` in `utils.py`, add a new list (e.g., `attn_params = []`) and a classification condition in the `for name, param in model.named_parameters()` loop (starting at line 71).
2. Add the new group to the `param_groups` list (line 105-115) with appropriate `algorithm`, `lr`, and `weight_decay` settings.
3. Add a new config key (e.g., `attn_lr`) to the `pop` block in `build_dion_optimizer()` (lines 179-184) and pass it to `_separate_param_groups()`.
4. Update the YAML example and tests.

**Files to modify:**
- `/home/scbjtfy/Automodel/nemo_automodel/components/optim/utils.py` -- modify `_separate_param_groups()` and `build_dion_optimizer()`
- `/home/scbjtfy/Automodel/tests/unit_tests/optim/test_dion_optimizer_utils.py` -- add tests for new group

### Scenario 5: Changing default LR scheduler behavior in recipes

The default scheduler parameters are computed in `build_lr_scheduler()` at `recipes/llm/train_ft.py` line 623-629. Defaults are:
- `lr_warmup_steps = min(1000, total_steps // 10)` (10% warmup capped at 1000)
- `init_lr = base_lr * 0.1` (warmup starts at 10% of base LR)
- `min_lr = base_lr * 0.01` (decays to 1% of base LR)
- `lr_decay_style = "cosine"`
- `wd_incr_style = "constant"` (weight decay stays constant)

All defaults can be overridden from YAML via the `lr_scheduler:` section. The user config is merged via `default_kwargs.update(user_kwargs)` at line 646, so any YAML key wins over the computed default.

**File to modify:**
- `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_ft.py` -- modify `build_lr_scheduler()` defaults

---

## File Index

| File | Lines | Purpose |
|---|---|---|
| `nemo_automodel/components/optim/__init__.py` | 23 | Public API exports |
| `nemo_automodel/components/optim/scheduler.py` | 352 | `OptimizerParamScheduler` class (LR + WD scheduling) |
| `nemo_automodel/components/optim/utils.py` | 222 | Dion optimizer detection, param grouping, builder |
| `tests/unit_tests/optim/test_scheduler.py` | 922 | Unit tests for `OptimizerParamScheduler` |
| `tests/unit_tests/optim/test_dion_optimizer_utils.py` | 702 | Unit tests for Dion utilities |
| `examples/llm_finetune/qwen/qwen2_5_7b_squad_muon.yaml` | ~104 | Example YAML with Muon/Dion optimizer config |
