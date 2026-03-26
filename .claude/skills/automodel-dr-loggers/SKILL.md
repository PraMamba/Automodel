---
name: automodel-dr-loggers
description: Use when working with the loggers module of automodel — training metric logging with W&B, TensorBoard, and custom logger backends
---

# Loggers Module -- Deep Read

Module path: `/home/scbjtfy/Automodel/nemo_automodel/components/loggers/`

## 1. Module Purpose & Capabilities

The loggers module provides three independent logging subsystems consumed by NeMo AutoModel recipes during training and validation:

1. **JSONL Metric Logging** (`metric_logger.py`) -- A local, file-based structured logger that writes one JSON object per line. It handles GPU tensor-to-CPU conversion in batches, supports buffered writes with optional fsync, is thread-safe, and has a distributed variant that silences all non-rank-0 processes. This is the "always-on" local log; every recipe uses it.

2. **MLflow Experiment Tracking** (`mlflow_utils.py`) -- A full-featured MLflow client that creates experiments, logs parameters/metrics/artifacts/models, and manages run lifecycle. Only active on rank 0. Lazily imports the `mlflow` package so it is an optional dependency.

3. **W&B Noise Suppression** (`wandb_utils.py`) -- A utility that silences W&B's verbose footer and upload messages by raising the wandb logger level to CRITICAL and monkey-patching internal `_footer*` functions. W&B itself is initialized in the recipes (via `wandb.init()`), not in this module; this module only suppresses noise.

4. **Application Log Setup** (`log_utils.py`) -- Configures Python's `logging` for the entire process: rank filtering (only rank 0 logs), colorized output with ANSI codes, per-module filtering, and warning suppression. This is the first thing recipes call at startup.

There is no TensorBoard logger in this module. The three external logging backends available to recipes are: JSONL (local files via this module), W&B (initialized in recipes, noise-suppressed here), and MLflow (fully managed here).

## 2. Core Design Logic

### Why four separate files with no shared base class

Each file addresses a different concern and has different dependency requirements. The module avoids a common `BaseLogger` ABC because the three logging targets have fundamentally different APIs:

- `MetricLogger` writes structured JSONL to disk -- it deals with file I/O, buffering, and tensor-to-CPU conversion.
- `MLflowLogger` is a thin wrapper around the mlflow SDK, managing experiment/run lifecycle.
- W&B integration is handled by `wandb.init()` in recipe code; this module only patches away noisy output.
- `setup_logging()` configures Python's built-in logging framework, which is orthogonal to metric logging.

Forcing these into a polymorphic hierarchy would add coupling without benefit. Instead, recipes compose them freely: every recipe uses `MetricLogger` + `setup_logging()`, and optionally adds W&B and/or MLflow based on YAML config presence.

### Rank-0-only logging as a cross-cutting pattern

Distributed training means multiple processes execute the same code. Every logger in this module guards against non-rank-0 writes, but each uses a different mechanism appropriate to its design:

- `MetricLoggerDist` replaces `log` and `close` with lambda no-ops on non-rank-0 at `__init__` time (line 152-156 of `metric_logger.py`). This avoids per-call rank checks.
- `MLflowLogger` only calls `mlflow.start_run()` when `dist.get_rank() == 0` (line 61 of `mlflow_utils.py`), and every logging method checks `dist.get_rank() == 0 and self.run is not None` before proceeding.
- `RankFilter` in `log_utils.py` reads the `RANK` environment variable and calls `logging.disable(logging.CRITICAL)` permanently on non-zero ranks (line 46-47 of `log_utils.py`).

### Batched tensor-to-CPU transfer

The function `stack_and_move_tensor_metrics_to_cpu()` in `metric_logger.py` (lines 44-80) groups buffered `MetricsSample` objects by their tensor metric names, stacks tensors per metric name across samples, moves the entire stack to CPU in a single transfer, then writes scalar `.item()` values back. This avoids N individual GPU-to-CPU synchronization points when the buffer flushes N samples.

### Lazy/optional dependency imports

`MLflowLogger.__init__()` imports `mlflow` inside the constructor (lines 49-53 of `mlflow_utils.py`) and raises a clear `ImportError` if not installed. `wandb_utils.py` wraps its monkey-patching in `try/except ImportError` blocks (lines 35-52). This keeps the module functional even when optional backends are not installed.

## 3. Core Data Structures

### MetricsSample (metric_logger.py, line 27)

```python
@dataclass
class MetricsSample:
    step: int
    epoch: int
    metrics: Dict[str, float]  # default_factory=dict
    timestamp: str  # auto-set in __post_init__ to UTC ISO 8601
```

The universal metric record. Created by recipes with training/validation metrics (loss, grad_norm, lr, tps, mem, etc.) and passed to all loggers. The `to_dict()` method merges `step`, `epoch`, `timestamp` with the flat `metrics` dict for serialization.

### MetricLogger (metric_logger.py, line 83)

Thread-safe buffered JSONL writer. Key attributes:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `filepath` | `str` | Absolute path to the `.jsonl` output file |
| `flush` | `bool` | If True, `os.fsync()` after each buffer flush |
| `buffer_size` | `int` | Number of `MetricsSample` to accumulate before writing (default 100) |
| `buffer` | `List[MetricsSample]` | In-memory accumulator |
| `_lock` | `threading.Lock` | Guards file writes |
| `_fp` | file object | Open file handle (append or write mode) |

### MetricLoggerDist (metric_logger.py, line 145)

Subclass of `MetricLogger`. Asserts `torch.distributed` is initialized. On non-rank-0 processes, replaces `log`, `close`, `__enter__`, and `__exit__` with no-op lambdas at construction time.

### MLflowLogger (mlflow_utils.py, line 24)

Full experiment tracking client. Key attributes:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `mlflow` | module | The imported `mlflow` package |
| `experiment_name` | `str` | MLflow experiment name |
| `run_name` | `Optional[str]` | Name for the current run |
| `tags` | `Dict[str, str]` | Tags attached to the run |
| `run` | `mlflow.Run` or `None` | Active run object (None on non-rank-0) |

Methods: `log_params()`, `log_metrics()`, `log_artifacts()`, `log_artifact()`, `log_model()`. All are rank-0-guarded and None-run-guarded.

### RankFilter (log_utils.py, line 25)

A `logging.Filter` subclass that reads the `RANK` env var. On rank > 0, permanently disables logging via `logging.disable(logging.CRITICAL)`.

### ColorFormatter (log_utils.py, line 62)

A `logging.Formatter` subclass that applies ANSI color codes to log level names. Respects `NO_COLOR` and `FORCE_COLOR` environment variables. Default format: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.

## 4. State Flow

### Recipe startup sequence (as seen in `train_ft.py`)

```
Recipe.__init__()
  |
  +---> setup_logging()                          [log_utils.py]
  |       Sets root logger level (INFO or env LOGGING_LEVEL)
  |       Installs ColorFormatter on root handler
  |       Adds RankFilter to root + all existing loggers
  |       Optionally adds warning_filter, module_filter
  |
  +---> suppress_wandb_log_messages()            [wandb_utils.py]  (if cfg.wandb exists)
  |       Sets wandb logger to CRITICAL
  |       Monkey-patches _footer* functions to no-ops
  |
  +---> wandb.init(...)                          [in recipe, not in this module]
  |       Creates W&B run; config passed from YAML
  |
  +---> build_mlflow(cfg)                        [mlflow_utils.py] (if cfg.mlflow exists)
  |       Creates MLflowLogger
  |         -> mlflow.set_tracking_uri() if provided
  |         -> mlflow.get_experiment_by_name() / create_experiment()
  |         -> mlflow.start_run()
  |       Calls mlflow_logger.log_params(cfg.to_dict())
  |
  +---> build_metric_logger(filepath)            [metric_logger.py]
          If dist initialized: returns MetricLoggerDist (rank-0-only)
          Else: returns MetricLogger
          Creates training.jsonl and validation_*.jsonl files
```

### Training loop metric logging

```
Each training step:
  |
  +---> Recipe constructs MetricsSample(step, epoch, metrics={loss, grad_norm, lr, mem, tps, ...})
  |
  +---> log_train_metrics(log_data)
          |
          +---> (if is_remote_logging_step):
          |       wandb.log(log_data.to_dict(), step=step)          [W&B]
          |       mlflow_logger.log_metrics(log_data.to_dict(), step=step)  [MLflow]
          |
          +---> metric_logger_train.log(log_data)                   [JSONL, every step]
                  Appends to buffer
                  If buffer full (>=100 samples):
                    stack_and_move_tensor_metrics_to_cpu(buffer)
                    Serialize each MetricsSample to JSON
                    Write all lines under lock
                    Optionally fsync
```

### Validation metric logging

```
Each validation step:
  |
  +---> Recipe constructs MetricsSample(step, epoch, metrics={val_loss, lr, num_label_tokens, mem})
  |
  +---> log_val_metrics(val_name, log_data, metric_logger)
          |
          +---> wandb.log(log_data.to_dict() | {"val_name": val_name}, step=step)
          +---> mlflow_logger.log_metrics(log_data.to_dict(), step=step)
          +---> metric_logger.log(log_data)        [writes to validation_<name>.jsonl]
```

### Shutdown

```
Training complete:
  |
  +---> metric_logger_train.close()
  |       Flushes remaining buffer to disk
  |       Flush + close file handle
  |
  +---> metric_logger_valid[name].close()  (for each validation set)
  |
  +---> mlflow_logger.__exit__()  (if used as context manager)
  |       Calls mlflow.end_run()
  |
  +---> wandb.finish()  (in recipe code, not in this module)
```

## 5. Common Modification Scenarios

### Scenario 1: Adding a new logging backend (e.g., Neptune, ClearML)

Create a new file at `nemo_automodel/components/loggers/<backend>_utils.py`. Follow the `MLflowLogger` pattern:

- Lazy-import the SDK in `__init__()` with a clear `ImportError` message.
- Guard all operations with `dist.get_rank() == 0` and a null-run check.
- Provide a `build_<backend>(cfg)` factory function that reads from `cfg.<backend>`.
- Support the context manager protocol (`__enter__`/`__exit__`).
- Accept `Dict[str, float]` metrics and an optional `step` parameter in `log_metrics()`.

Then integrate in the recipe (e.g., `train_ft.py`):
- Add conditional initialization in the recipe's `__init__` alongside the existing W&B and MLflow blocks (around line 827-836 of `train_ft.py`).
- Add logging calls in `log_train_metrics()` and `log_val_metrics()`.

No changes to other component files are needed -- the component independence constraint is preserved.

### Scenario 2: Changing the JSONL buffer size or adding compression

The `MetricLogger` buffer size is hardcoded to 100 in the default parameter of `__init__()` at line 93 of `metric_logger.py`. To make it configurable:

1. Add a `buffer_size` parameter to `build_metric_logger()` at line 159.
2. Thread it through from recipe config (e.g., `cfg.step_scheduler.log_buffer_size`).
3. For gzip compression, modify `_save()` (line 118) to write to a `gzip.open()` file handle instead of a plain file, and change the file extension to `.jsonl.gz`. The `_move_to_cpu()` method and buffer logic remain unchanged.

### Scenario 3: Adding system metrics (GPU utilization, memory) to MLflow

The `log_metrics()` method in `MLflowLogger` (line 106 of `mlflow_utils.py`) has an explicit `# TODO: add system metrics to mlflow` comment at line 127. To implement:

1. Before calling `self.mlflow.log_metrics()`, query `torch.cuda.memory_allocated()`, `torch.cuda.utilization()`, etc.
2. Merge these into `float_metrics` with keys like `system/gpu_memory_gb`, `system/gpu_utilization`.
3. These will be logged alongside training metrics at the same step.

### Scenario 4: Making RankFilter pipeline-parallelism aware

The `RankFilter` class at line 25 of `log_utils.py` has a TODO comment: `# TODO(@akoumparouli): make this PP aware.` Currently it checks the `RANK` env var globally. For pipeline parallelism, you may want the last pipeline stage (which computes loss) to also log. To implement:

1. Accept a `log_ranks: set[int]` parameter in `RankFilter.__init__()`.
2. In `filter()`, check if the current rank is in `log_ranks` instead of only allowing rank 0.
3. Update `setup_logging()` to accept and forward this parameter.
4. Recipes with pipeline parallelism would pass the set of ranks that should log.

### Scenario 5: Switching from per-call rank checks to a NullLogger pattern in MLflow

Currently every method in `MLflowLogger` (e.g., `log_params`, `log_metrics`, `log_artifacts`, `log_artifact`, `log_model`) repeats the guard `if not dist.get_rank() == 0 or self.run is None: return`. An alternative is to follow the `MetricLoggerDist` pattern: replace all methods with no-ops at `__init__` time when rank != 0. This eliminates per-call overhead and centralizes the guard logic. The change is confined to `MLflowLogger.__init__()` in `mlflow_utils.py`.

## File Index

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 14 | License header only; no public API exports |
| `metric_logger.py` | 164 | `MetricsSample`, `MetricLogger`, `MetricLoggerDist`, `build_metric_logger()`, `stack_and_move_tensor_metrics_to_cpu()` |
| `mlflow_utils.py` | 229 | `MLflowLogger`, `build_mlflow()` |
| `wandb_utils.py` | 53 | `suppress_wandb_log_messages()` |
| `log_utils.py` | 218 | `RankFilter`, `ColorFormatter`, `warning_filter()`, `module_filter()`, `add_filter_to_all_loggers()`, `setup_logging()` |

## Test Coverage

Unit tests are at `/home/scbjtfy/Automodel/tests/unit_tests/loggers/`:

| Test File | What It Covers |
|-----------|----------------|
| `test_metric_logger.py` | JSONL output correctness, append vs. write modes, thread safety, fsync behavior, `MetricLoggerDist` rank-0 vs. non-rank-0 behavior |
| `test_mlflow_utils.py` | `build_mlflow` tag enrichment, `log_params` flattening, `log_metrics` type conversion (int/float/tensor), rank guard + null-run guard, context manager `end_run` |
| `test_wandb_utils.py` | W&B suppression patching |
| `test_log_utils.py` | `warning_filter`, `module_filter` |
