---
name: automodel-dr-training
description: Use when working with the training module of automodel — training loop, step scheduler, gradient accumulation, and training state management
---

# Training Module — Deep Code Read

**Module path**: `nemo_automodel/components/training/`
**Files (6, 1464 lines total)**:

| File | Lines | Purpose |
|---|---|---|
| `__init__.py` | 0 | Empty package marker |
| `step_scheduler.py` | 258 | Gradient accumulation scheduler, epoch/step iteration, checkpoint/validation triggers |
| `utils.py` | 339 | Gradient clipping (sharding-aware), gradient accumulation FSDP helpers, model offloading |
| `timers.py` | 558 | Distributed-aware timing infrastructure with min/max/all reporting across ranks |
| `signal_handler.py` | 170 | Distributed SIGTERM handling via all-gather across ranks |
| `rng.py` | 139 | Reproducible RNG state management for Python, NumPy, and PyTorch |

---

## 1. Module Purpose and Capabilities

This module provides the **training orchestration layer** that sits between high-level recipe scripts and low-level model/optimizer code. It does NOT contain the training loop itself (that lives in recipe scripts like `nemo_automodel/recipes/llm/train_ft.py`), but it provides the building blocks that every training loop uses:

- **Step scheduling**: `StepScheduler` in `step_scheduler.py` manages the iteration over dataloader batches, automatically grouping micro-batches for gradient accumulation, tracking global step/epoch counters, and deciding when to checkpoint, validate, or stop.
- **Gradient utilities**: `utils.py` provides sharding-aware gradient clipping (`clip_grad_norm`, `_clip_grad_norm_impl`) that correctly handles DTensor parameters with different placements across device meshes, plus FSDP gradient accumulation preparation (`prepare_for_grad_accumulation`, `prepare_for_final_backward`).
- **Distributed timing**: `timers.py` provides a `Timers` registry that tracks wall-clock time per named region across all distributed ranks, with configurable aggregation (max, minmax, all) and output to stdout, TensorBoard, or WandB.
- **Graceful shutdown**: `signal_handler.py` provides `DistributedSignalHandler` which catches SIGTERM on any rank and broadcasts the signal to all ranks via `all_gather`, enabling coordinated checkpoint-then-exit.
- **Reproducible RNG**: `rng.py` provides `StatefulRNG` (persistent RNG state for checkpointing) and `ScopedRNG` (temporary RNG context for deterministic operations like initialization).

---

## 2. Core Design Logic

### Why StepScheduler wraps the dataloader

The central design decision is that `StepScheduler` is both an **iterator** (yielding grouped micro-batches) and a **state machine** (tracking step/epoch and deciding checkpoint/validation triggers). This design exists because:

1. **Gradient accumulation is transparent to recipe code.** The recipe loop does `for batch_buffer in step_scheduler:` and gets a list of `grad_acc_steps` micro-batches. The recipe never manually counts micro-batches. The formula is `grad_acc_steps = global_batch_size // (local_batch_size * dp_size)` (line 86 of `step_scheduler.py`).

2. **Checkpoint/validation scheduling is coupled to step counting.** Because `StepScheduler.__iter__` increments `self.step` after yielding each accumulated group (line 153), the properties `is_ckpt_step`, `is_val_step`, and `is_last_step` always reflect the step that was just completed. This avoids off-by-one errors in recipe code.

3. **SIGTERM handling is integrated into iteration.** The scheduler checks `self.sigterm_flag` on every iteration (line 155) and on `is_ckpt_step` (line 196), so a SIGTERM triggers a checkpoint and clean exit at the next micro-batch boundary rather than mid-computation.

### Why gradient clipping is sharding-aware

The `_clip_grad_norm_impl` function in `utils.py` groups parameters by their `(device_mesh_id, placements)` key (lines 80-95) and computes norms per group before combining them. This is necessary because `torch.nn.utils.clip_grads_with_norm_` cannot mix DTensors from different device meshes in a single call. Pipeline parallel norms are reduced separately via `all_reduce` on the PP mesh (lines 135-141).

### Why RNG management is split into two classes

`StatefulRNG` is for persistent RNG state that survives across the entire training run and can be checkpointed via `state_dict()`/`load_state_dict()`. `ScopedRNG` is a context manager for operations that need a deterministic seed temporarily (e.g., model initialization with `ScopedRNG` in KD recipes, see `nemo_automodel/recipes/llm/kd.py` line 53) without disturbing the training RNG sequence.

### Why Timers use DummyTimer for disabled levels

The `Timers` class (line 257 in `timers.py`) uses a log-level gating system: timers with a log level above `self._log_level` return a `DummyTimer` that no-ops on `start()`/`stop()` but raises on `elapsed()`. This avoids conditional checks in recipe code while preventing silent misuse of timers that were never actually started.

---

## 3. Core Data Structures

### StepScheduler (`step_scheduler.py`, line 48)

Implements `torch.distributed.checkpoint.stateful.Stateful` for DCP integration.

**Key fields:**
- `self.grad_acc_steps: int` -- computed as `global_batch_size // (local_batch_size * dp_size)`
- `self.step: int` -- current global optimizer step (starts at `start_step`)
- `self.epoch: int` -- current epoch (starts at `start_epoch`)
- `self.max_steps: int` -- total steps to run; computed from `num_epochs * epoch_len` if not provided
- `self.epoch_len: int | None` -- steps per epoch; `None` for `IterableDataset`
- `self.ckpt_every_steps: int` -- checkpoint frequency; defaults to `epoch_len` or `max_steps // 2`
- `self.val_every_steps: int | None` -- validation frequency
- `self.log_remote_every_steps: int` -- remote logging frequency (default 1)
- `self.sig_handler: DistributedSignalHandler` -- entered at init (line 133)
- `self.sigterm_flag: bool` -- latched True once any rank receives SIGTERM

**Key properties (all `@property`):**
- `is_ckpt_step` -- True at checkpoint interval, last batch, last step, or SIGTERM
- `is_val_step` -- True at validation interval or checkpoint step (but not SIGTERM)
- `is_last_step` -- True when `step + 1 >= max_steps`
- `is_last_batch` -- True when step is at epoch boundary
- `is_remote_logging_step` -- True every `log_remote_every_steps` steps
- `sigterm_received` -- checks all ranks via `sig_handler.signals_received()`, latches

**Serialization:** `state_dict()` returns `{"step": min(max_steps, step + 1), "epoch": epoch}`. The `step + 1` accounts for the fact that checkpointing happens after yield but before the iterator increments (line 247-249).

### RNGState (`rng.py`, line 49)

Dataclass with four fields:
- `random_rng_state: tuple` -- Python `random` module state
- `np_rng_state: tuple` -- NumPy random state
- `torch_rng_state: torch.Tensor` -- PyTorch CPU RNG state
- `cuda_rng_state: torch.Tensor` -- all CUDA device RNG states

### StatefulRNG (`rng.py`, line 83)

Wraps `init_all_rng()` with `state_dict()`/`load_state_dict()` for checkpoint integration. Used in `nemo_automodel/recipes/base_recipe.py` line 47.

### ScopedRNG (`rng.py`, line 115)

Context manager that saves current RNG state on `__enter__`, seeds with a fixed value (default 95050), and restores original state on `__exit__`. Used in KD and VLM recipes for deterministic model init.

### DistributedSignalHandler (`signal_handler.py`, line 91)

Context manager fields:
- `self.sig: int` -- signal number (default `signal.SIGTERM`)
- `self._signal_received: bool` -- set True by signal handler on this rank
- `self.released: bool` -- prevents double-release
- `self.original_handler` -- saved for restoration

`signals_received()` (line 117) calls `all_gather_item(self._signal_received, dtype=torch.int32)` to collect signal status from all ranks.

### Timer hierarchy (`timers.py`)

- `TimerBase(ABC)` (line 19) -- abstract base with `start()`, `stop()`, `reset()`, `elapsed()`, and context manager support via `with_barrier()`.
- `DummyTimer(TimerBase)` (line 104) -- no-op implementation; raises on `elapsed()` and `active_time()` to catch misuse.
- `Timer(TimerBase)` (line 152) -- real timer using `time.time()` with `torch.cuda.synchronize()` for accurate GPU timing. Tracks both `_elapsed` (resettable) and `_active_time` (cumulative, never reset).
- `Timers` (line 257) -- registry that creates `Timer` or `DummyTimer` based on log level. Callable interface: `timers("name", log_level=1)` returns a timer usable as context manager.

### Utility functions in `utils.py`

- `clip_grad_norm()` (line 152) -- public API; collects parameters from `model_parts`, delegates to `_clip_grad_norm_impl()`.
- `_clip_grad_norm_impl()` (line 56) -- groups parameters by DTensor sharding pattern, computes per-group norms via `torch.nn.utils.get_total_norm()`, combines across groups, reduces across PP mesh, clips per group.
- `scale_grads_and_clip_grad_norm()` (line 257) -- applies PP scaling (`grad / (num_label_tokens / dp_group_size)`) and EP scaling before clipping. Used by pipeline-parallel and MoE recipes.
- `prepare_for_grad_accumulation()` (line 218) -- calls `set_is_optim_step(False)` and `mp.prepare_for_grad_accumulation()` on each model part to set FSDP into accumulation mode (no gradient sync).
- `prepare_for_final_backward()` (line 237) -- calls `set_is_optim_step(True)` and `mp.prepare_for_final_backward()` to enable gradient sync for the last micro-batch.
- `count_tail_padding()` (line 27) -- counts trailing `ignore_label` tokens via `cumprod` trick on flipped labels. Used in KD and VLM recipes for PP gradient scaling.
- `move_to_device()` (line 317) -- moves model and buffers to device, clears CUDA cache.
- `ScopedModuleOffloading` (line 326) -- context manager that moves model to CUDA on enter and back to CPU on exit. Used in KD recipes for teacher model offloading.

---

## 4. State Flow

### Training loop lifecycle (as orchestrated by recipes using this module)

```
Recipe init
  |
  v
StatefulRNG(seed) -----> init_all_rng() seeds Python/NumPy/PyTorch/CUDA
  |
  v
StepScheduler(global_batch_size, local_batch_size, dp_size, dataloader, ...)
  |  - Computes grad_acc_steps = global_batch_size // (local_batch_size * dp_size)
  |  - Computes epoch_len = ceil(len(dataloader) / grad_acc_steps)
  |  - Computes max_steps from num_epochs * epoch_len (or uses provided max_steps)
  |  - Installs DistributedSignalHandler for SIGTERM
  |
  v
for epoch in step_scheduler.epochs:      <-- yields epoch indices, stops on max_steps or SIGTERM
  |
  step_scheduler.set_epoch(epoch)         <-- sets sampler epoch for shuffling
  |
  for batch_buffer in step_scheduler:     <-- yields list of grad_acc_steps micro-batches
    |
    prepare_for_grad_accumulation(model_parts)   <-- set_is_optim_step(False), disable grad sync
    |
    for i, micro_batch in enumerate(batch_buffer):
      |
      if i == len(batch_buffer) - 1:
        prepare_for_final_backward(model_parts)  <-- set_is_optim_step(True), enable grad sync
      |
      forward(micro_batch) --> loss
      loss.backward()
    |
    clip_grad_norm(max_grad_norm, model_parts)   <-- sharding-aware norm computation + clipping
    |  (or scale_grads_and_clip_grad_norm for PP/EP)
    |
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    |
    if step_scheduler.is_ckpt_step:
      save_checkpoint()                   <-- step_scheduler.state_dict() saves step+1
    |
    if step_scheduler.is_val_step:
      run_validation()
    |
    step_scheduler.step is incremented by __iter__
  |
  epoch ends, step_scheduler.epoch incremented by __iter__
```

### Checkpoint resume flow

```
StepScheduler.load_state_dict({"step": N, "epoch": E})
  |  - Sets self.step = N, self.epoch = E
  |  - __iter__ skips if step >= max_steps
  |
StatefulRNG.load_state_dict(rng_state)
  |  - Restores all four RNG states
  |
Training resumes from step N, epoch E
```

### SIGTERM flow

```
SIGTERM received on rank K
  |
  signal_handler sets _signal_received = True on rank K
  |
  StepScheduler.sigterm_received property called (at iteration boundary)
    |
    all_gather_item(self._signal_received) broadcasts across all ranks
    |
    self.sigterm_flag latched True on ALL ranks
    |
    is_ckpt_step returns True --> checkpoint saved
    |
    __iter__ returns (line 156) --> training stops cleanly
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new periodic action (e.g., evaluation on a different dataset)

Add a new property to `StepScheduler` following the pattern of `is_val_step`:

**File**: `nemo_automodel/components/training/step_scheduler.py`

1. Add a constructor parameter (e.g., `custom_eval_every_steps: Optional[int] = None`) and store it.
2. Add a property:
```python
@property
def is_custom_eval_step(self):
    if self.custom_eval_every_steps and self.custom_eval_every_steps > 0:
        return self.step % self.custom_eval_every_steps == self.custom_eval_every_steps - 1
    return False
```
3. In the recipe training loop, check `step_scheduler.is_custom_eval_step` after each optimizer step.

Note: Follow the `val_every_steps - 1` convention (uses 0-indexed modular check) to trigger at the END of a period, not the beginning.

### Scenario 2: Adding a new gradient scaling strategy (e.g., for a new parallelism scheme)

Modify `scale_grads_and_clip_grad_norm()` in `nemo_automodel/components/training/utils.py`:

1. Add new parameters for the scaling factor or mesh.
2. Add a new scaling branch in the single-pass loop (lines 293-303) following the EP scaling pattern: check if the parameter is a DTensor, check if it belongs to the relevant mesh dimension, and apply `p.grad.div_(factor)`.
3. The final `clip_grad_norm()` call at line 306 handles the rest -- no changes needed there because `_clip_grad_norm_impl` already groups by `(device_mesh_id, placements)` automatically.

### Scenario 3: Adding a new timer output sink (e.g., MLflow)

Add a new method to the `Timers` class in `nemo_automodel/components/training/timers.py` following the `write_to_wandb()` pattern (line 538):

1. Create a method like `write_to_mlflow(self, names, writer, iteration, normalizer, reset, barrier)`.
2. Call `self._get_global_min_max_time(names, reset, barrier, normalizer)` to get per-timer `(min, max)` tuples.
3. Write `max_time` values using the MLflow client API.

The existing `_get_global_min_max_time()` handles all the distributed all-gather and aggregation.

### Scenario 4: Changing the checkpoint trigger logic

Modify `StepScheduler.is_ckpt_step` property in `step_scheduler.py` (line 188). Currently it triggers on: periodic interval OR last batch OR last step OR SIGTERM. To add a new condition (e.g., checkpoint on validation loss improvement), add an additional `or` clause to the return statement at line 196. The `state_dict()` method at line 238 already handles the step+1 accounting correctly regardless of why the checkpoint was triggered.

### Scenario 5: Supporting a new distributed backend in signal handling

Modify `signal_handler.py`:

1. `get_device()` (line 24) currently supports only `nccl` and `gloo`. Add a new `elif` branch for the backend.
2. `all_gather_item()` (line 51) uses `torch.distributed.all_gather` which is backend-agnostic, so it likely works without changes.
3. `DistributedSignalHandler.signals_received()` delegates to `all_gather_item()`, so no changes needed there.
