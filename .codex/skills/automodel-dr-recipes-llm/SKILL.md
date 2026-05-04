---
name: automodel-dr-recipes-llm
description: Use when working with LLM recipes of automodel — pretraining, supervised fine-tuning (SFT), knowledge distillation (KD), and PEFT training workflows
---

# LLM Recipes Module

## 1. Module Purpose and Capabilities

The `nemo_automodel/recipes/llm/` module contains four recipe classes that orchestrate end-to-end LLM training workflows. Each recipe composes independent components (model, optimizer, dataset, distributed, checkpoint, loss) into a complete training pipeline. The module sits at the second layer of NeMo AutoModel's three-layer architecture: components -> **recipes** -> CLI.

### Recipe inventory

| File | Class | Purpose |
|---|---|---|
| `train_ft.py` | `TrainFinetuneRecipeForNextTokenPrediction` | Pretraining and supervised fine-tuning (SFT) for causal LMs; supports FSDP2, TP, CP, PP, PEFT, FP8, QAT, sequence packing |
| `kd.py` | `KnowledgeDistillationRecipeForNextTokenPrediction` | Knowledge distillation from a frozen teacher to a student model; extends `train_ft.py` |
| `benchmark.py` | `BenchmarkingRecipeForNextTokenPrediction` | Performance benchmarking with MFU calculation, nsys profiling, and timer instrumentation; extends `train_ft.py` |
| `train_seq_cls.py` | `TrainFinetuneRecipeForSequenceClassification` | Sequence classification fine-tuning (e.g., BERT on Yelp); standalone recipe extending `BaseRecipe` |
| `base_recipe.py` (parent) | `BaseRecipe` | Automatic state tracking, checkpoint save/load, distributed helper methods |

### Key capabilities

- **SPMD execution**: Same recipe script runs on 1 GPU or 1000+ GPUs. Parallelism dimensions (DP, TP, CP, PP) are determined by the YAML config `distributed` section.
- **HuggingFace day-0 support**: Any HF `AutoModelForCausalLM` or `AutoModelForSequenceClassification` works via `NeMoAutoModelForCausalLM.from_pretrained` / `from_config`.
- **PEFT (LoRA, QLoRA)**: Configured via `peft` YAML section; the recipe freezes base weights and creates adapters automatically.
- **FP8 training**: Enabled via `fp8` config section with torchao integration.
- **QAT (Quantization-Aware Training)**: Delayed fake-quant toggling via `qat.fake_quant_after_n_steps`.
- **Pipeline parallelism**: Composed with FSDP2 (3D parallelism) via `AutoPipeline`.
- **Sequence packing**: Dataset-level packing for improved throughput.
- **Multiple validation datasets**: The NTP recipe supports any number of `validation_dataset*` sections in the YAML.
- **Async checkpointing**: Deferred symlink updates for non-blocking checkpoint writes.

---

## 2. Core Design Logic

### 2.1 Why recipes are structured this way

Recipes exist to separate **orchestration** from **implementation**. Components (datasets, distributed strategies, optimizers, checkpointers) are independently developed and tested. Recipes import and compose them, enforced by import-linter which forbids cross-component imports.

The inheritance hierarchy:

```
BaseRecipe                           (base_recipe.py)
  |-- TrainFinetuneRecipeForNTP      (train_ft.py)       -- the central recipe
  |     |-- KnowledgeDistillationRecipeForNTP (kd.py)    -- extends NTP with teacher model
  |     |-- BenchmarkingRecipeForNTP (benchmark.py)      -- extends NTP with timers/profiling
  |-- TrainFinetuneRecipeForSeqCls   (train_seq_cls.py)  -- standalone classification recipe
```

`TrainFinetuneRecipeForNextTokenPrediction` (train_ft.py) is the canonical recipe. It defines all the stateless builder functions (`build_model_and_optimizer`, `build_dataloader`, `build_distributed`, `build_step_scheduler`, `build_lr_scheduler`, `build_checkpoint_config`, `build_wandb`, `calculate_loss`) as module-level functions, making them reusable by other recipes (kd.py, benchmark.py, train_seq_cls.py all import from train_ft.py).

### 2.2 Stateless builders + stateful recipe

The design separates **construction** (stateless module-level functions prefixed with `build_*`) from **execution** (instance methods on the recipe class). This makes builders independently testable and composable.

### 2.3 BaseRecipe's automatic state tracking

`BaseRecipe.__setattr__` intercepts every attribute assignment to the recipe instance. It classifies objects into categories (model, optimizer, lr_scheduler, dataloader, tokenizer, config, stateful RNG) and registers them in `__state_tracked`. This set drives automatic checkpoint save/load: every tracked attribute is serialized without the recipe needing to maintain an explicit list. Validation/eval/test/loss-related attributes are excluded from tracking to avoid saving ephemeral state.

### 2.4 Configuration-driven instantiation

All recipes use `ConfigNode` objects loaded from YAML. The pattern `cfg.model.instantiate(**kwargs)` resolves `_target_` fields to Python callables and calls them with merged keyword arguments. This means changing the model, optimizer, dataset, or loss function requires only a YAML edit, not code changes.

### 2.5 Gradient accumulation via StepScheduler

The `StepScheduler` (from `components/training/step_scheduler.py`) is an iterator that yields lists of micro-batches. Each yielded list has `grad_acc_steps` entries, where `grad_acc_steps = global_batch_size // (local_batch_size * dp_size)`. The recipe calls `_run_train_optim_step(batches)` with the full list, which loops over micro-batches doing forward+backward, then takes a single optimizer step.

---

## 3. Core Data Structures

### 3.1 BaseRecipe (base_recipe.py, line 127)

```
BaseRecipe
  __state_tracked: set[str]    -- names of tracked attributes for checkpoint save/load
  _best_val_loss: float        -- tracks lowest validation loss for LOWEST_VAL symlink
  checkpointer: Checkpointer   -- handles DCP save/load with SafeTensors
  device_mesh: DeviceMesh      -- PyTorch DeviceMesh for parallelism dimensions

  save_checkpoint(epoch, step, train_loss, val_loss, best_metric_key)
  load_checkpoint(restore_from)
  _get_dp_group() / _get_dp_rank() / _get_tp_rank() / _get_pp_rank()
  _dp_allreduce(tensor, op, include_cp)
```

### 3.2 TrainFinetuneRecipeForNextTokenPrediction (train_ft.py, line 785)

Key instance attributes set during `setup()`:

| Attribute | Type | Source |
|---|---|---|
| `cfg` | `ConfigNode` | YAML config |
| `dist_env` | `DistInfo` | `build_distributed()` |
| `device_mesh` | `DeviceMesh` or `None` | from `model_wrapper` |
| `moe_mesh` | `DeviceMesh` or `None` | from `model_wrapper` (MoE-specific) |
| `model_wrapper` | FSDP2Manager / MegatronFSDPManager / None | `cfg.distributed.instantiate()` |
| `model_parts` | `list[nn.Module]` | single model or PP stage list |
| `pp` | `AutoPipeline` or `None` | pipeline parallelism orchestrator |
| `pp_enabled` | `bool` | whether PP is active |
| `optimizer` | `list[Optimizer]` | always a list (one per PP stage) |
| `lr_scheduler` | `list[OptimizerParamScheduler]` or `None` | LR scheduler per optimizer |
| `loss_fn` | `nn.Module` | `FusedLinearCrossEntropy`, `MaskedCrossEntropy`, or custom |
| `dataloader` | `StatefulDataLoader` | training dataloader |
| `val_dataloaders` | `dict[str, DataLoader]` | named validation dataloaders |
| `tokenizer` | `PreTrainedTokenizerBase` | HF tokenizer |
| `step_scheduler` | `StepScheduler` | gradient accumulation and epoch management |
| `checkpointer` | `Checkpointer` | checkpoint manager |
| `peft_config` | PEFT config or `None` | LoRA/adapter config |
| `rng` | `StatefulRNG` | checkpointable RNG |
| `max_grad_norm` | `float` | gradient clipping threshold (default 1.0) |

### 3.3 KnowledgeDistillationRecipeForNextTokenPrediction (kd.py, line 148)

Extends the NTP recipe with:

| Attribute | Type | Source |
|---|---|---|
| `teacher_model` | `nn.Module` | frozen teacher from `_build_teacher_model()` |
| `kd_loss_fn` | `nn.Module` | KL-divergence loss (default `KLDivLoss(reduction="batchmean")`) |
| `kd_ratio` | `float` | mixing coefficient: `loss = (1-kd_ratio)*CE + kd_ratio*KD` |
| `_kd_loss_buffer` | `list[Tensor]` | per-step KD loss accumulator |
| `_ce_loss_buffer` | `list[Tensor]` | per-step CE loss accumulator |

### 3.4 BenchmarkingRecipeForNextTokenPrediction (benchmark.py, line 33)

Extends the NTP recipe with:

| Attribute | Type | Source |
|---|---|---|
| `timers` | `Timers` | min/max timer instrumentation |
| `tflops` | `float` | theoretical TFLOPs per GPU |
| `_bench_warmup_steps` | `int` | warmup iterations excluded from timing |
| `_bench_peak_tflops` | `float` | hardware peak for MFU calculation |
| `_bench_nsys_start/end/ranks` | `int`/`list` | nsys profiling window |

### 3.5 MetricsSample (components/loggers/metric_logger.py, line 27)

```python
@dataclass
class MetricsSample:
    step: int
    epoch: int
    metrics: Dict[str, float]
    timestamp: str  # auto-set to UTC ISO format
```

Used as the universal return type from `_run_train_optim_step()` and `_run_validation_epoch()`.

### 3.6 Module-level builder functions (train_ft.py)

| Function | Line | Returns | Used by |
|---|---|---|---|
| `build_model_and_optimizer()` | 130 | `(model, list[Optimizer], loss_fn)` | all recipes |
| `build_dataloader()` | 366 | `(DataLoader, tokenizer)` | all recipes |
| `build_distributed()` | 556 | `DistInfo` | all recipes |
| `build_step_scheduler()` | 570 | `StepScheduler` | all recipes |
| `build_lr_scheduler()` | 596 | `list[OptimizerParamScheduler]` or `None` | all recipes |
| `build_checkpoint_config()` | 274 | `CheckpointingConfig` | all recipes |
| `build_wandb()` | 658 | `wandb.Run` | NTP, seq_cls |
| `build_validation_dataloader()` | 726 | `dict[str, DataLoader]` | NTP, KD |
| `calculate_loss()` | 679 | `Tensor` | NTP, KD |

---

## 4. State Flow

### 4.1 Lifecycle: Config -> Setup -> Train -> Checkpoint

```
YAML config
    |
    v
parse_args_and_load_config()  -->  ConfigNode
    |
    v
Recipe.__init__(cfg)           -- stores cfg only
    |
    v
Recipe.setup()                 -- builds ALL components:
    |  1. build_distributed()         -> self.dist_env
    |  2. setup_logging()
    |  3. cfg.distributed.instantiate() -> self.model_wrapper, self.device_mesh
    |  4. build_wandb() / build_mlflow()
    |  5. AutoPipeline construction (if PP enabled)
    |  6. build_model_and_optimizer() -> self.model_parts, self.optimizer, self.loss_fn
    |  7. build_dataloader()          -> self.dataloader, self.tokenizer
    |  8. build_validation_dataloader() -> self.val_dataloaders
    |  9. build_step_scheduler()      -> self.step_scheduler
    |  10. build_lr_scheduler()       -> self.lr_scheduler
    |  11. Checkpointer construction  -> self.checkpointer
    |  12. load_checkpoint()          -- resume if checkpoint exists
    v
Recipe.run_train_validation_loop()
    |
    for epoch in step_scheduler.epochs:
        step_scheduler.set_epoch(epoch)
        for batches in step_scheduler:        -- yields grad_acc_steps micro-batches
            train_log = _run_train_optim_step(batches)
            log_train_metrics(train_log)
            if is_val_step:
                for val_name, val_dl in val_dataloaders:
                    val_log = _run_validation_epoch(val_dl)
                    log_val_metrics(val_name, val_log)
            if is_ckpt_step:
                save_checkpoint(epoch, step, loss, val_losses)
    |
    v
Close metric loggers, checkpointer
```

### 4.2 Single training step flow (_run_train_optim_step)

```
batches (list of grad_acc_steps micro-batches)
    |
    v
count num_label_tokens, num_tokens_in_batch
dp_allreduce both across DP group
    |
    v
prepare_for_grad_accumulation(model_parts)   -- disable grad sync for micro-batches
    |
    for i, batch in enumerate(batches):
        if last micro-batch:
            prepare_for_final_backward()      -- re-enable grad sync
        _forward_backward_step(i, batch, ...)
            |-- move batch to device
            |-- make_cp_batch_and_ctx()       -- context parallelism setup
            |-- pop labels from batch
            |-- if PP: schedule.step(input_ids, target=labels, ...)
            |-- else:
            |     model(**batch)              -- forward
            |     calculate_loss(loss_fn, logits, labels, ...)
            |     (loss * dp_group_size).backward()   -- backward
            |-- append loss to loss_buffer
    |
    v
scale_grads_and_clip_grad_norm()
optimizer.step() + zero_grad()
lr_scheduler.step(1)
    |
    v
compute throughput (tps), reporting_loss
return MetricsSample(step, epoch, {loss, grad_norm, lr, mem, tps, ...})
```

### 4.3 Checkpoint save flow (BaseRecipe.save_checkpoint)

```
save_checkpoint(epoch, step, train_loss, val_loss)
    |
    v
checkpointer.async_wait()     -- wait for previous async write
update LATEST symlink for previous checkpoint (if async)
update LOWEST_VAL symlink for previous checkpoint (if async + better)
    |
    v
create directory: checkpoint_dir/epoch_{E}_step_{S}/
write losses.json
    |
    v
iterate __state_tracked:
    - model  -> model.save_pretrained() or checkpointer.save_model() (PP)
    - optimizer -> checkpointer.save_optimizer()
    - dataloader -> checkpointer.save_on_dp_ranks()
    - StatefulRNG -> checkpointer.save_on_dp_ranks()
    - config -> save_config()
    - other stateful -> torch.save(state_dict)
    |
    v
update LATEST / LOWEST_VAL symlinks (sync) or defer (async)
```

### 4.4 KD-specific flow (kd.py _forward_backward_step override)

```
_forward_backward_step(idx, batch, ...)
    |
    v
move batch to device, pop labels
make_cp_batch_and_ctx()
    |
    v
with torch.inference_mode():
    teacher_logits = teacher_model(**batch).logits.detach().clone()
    |
    v
student_out = model(**batch)
ce_loss = calculate_loss(loss_fn, student_logits, labels)
kd_loss = kd_loss_fn(student_logits, teacher_logits, labels, num_batch_labels)
local_loss = (1 - kd_ratio) * ce_loss + kd_ratio * kd_loss
local_loss.backward()
return (local_loss, kd_loss, ce_loss)
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new training recipe

To add a new recipe (e.g., reinforcement learning from human feedback):

1. Create `nemo_automodel/recipes/llm/rlhf.py`.
2. Subclass `TrainFinetuneRecipeForNextTokenPrediction` (or `BaseRecipe` for a fully custom loop).
3. Import builder functions from `train_ft.py` for any shared setup:
   ```python
   from nemo_automodel.recipes.llm.train_ft import (
       build_model_and_optimizer, build_dataloader, build_distributed,
       build_step_scheduler, build_lr_scheduler, build_checkpoint_config,
   )
   ```
4. Override `setup()` to add reward-model-specific components (call `super().setup()` first for shared setup).
5. Override `_forward_backward_step()` or `_run_train_optim_step()` to inject the RLHF loss computation.
6. Add a `main()` entry point and a YAML config under `examples/`.
7. Key constraint: the recipe must assign stateful attributes (model, optimizer, dataloader, etc.) as instance attributes so `BaseRecipe.__setattr__` auto-tracks them for checkpointing. Avoid naming attributes with "val", "eval", "test", or "loss" substrings unless they should be excluded from checkpoint tracking (see `base_recipe.py` line 165).

### Scenario 2: Customizing the training loop (e.g., adding a regularization term)

To add weight regularization to the NTP training step:

1. Subclass `TrainFinetuneRecipeForNextTokenPrediction`.
2. Override `_run_train_optim_step()`:
   - Call the parent implementation or replicate its logic.
   - After `calculate_loss()` in `_forward_backward_step()`, add the regularization term to `local_loss` before `.backward()`.
   - Alternatively, override `_forward_backward_step()` directly (as `kd.py` does at line 188).
3. The return value must be a `MetricsSample` (from `nemo_automodel.components.loggers.metric_logger`). Add custom metric keys to `metrics` dict; they will be logged to WandB and JSONL automatically if you also override `log_train_metrics()`.
4. If the regularization requires additional model state (e.g., an EMA model), assign it as an instance attribute so it is checkpoint-tracked.

### Scenario 3: Adding a new validation metric

To add perplexity or BLEU alongside validation loss:

1. Override `_run_validation_epoch()` in a subclass.
2. After computing `val_loss`, compute the additional metric (e.g., `perplexity = torch.exp(val_loss)`).
3. Include it in the returned `MetricsSample.metrics` dict:
   ```python
   return MetricsSample(step=..., epoch=..., metrics={"val_loss": val_loss, "perplexity": perplexity, ...})
   ```
4. Override `log_val_metrics()` to format the new metric in the log line.
5. If the metric should determine "best" checkpoint, set `best_metric_key` in config (`checkpoint.best_metric_key`) and ensure the key appears in the `val_loss` dict passed to `save_checkpoint()`.

### Scenario 4: Supporting a new distributed strategy

To add a new parallelism strategy:

1. Implement the strategy in `nemo_automodel/components/distributed/` (must not import other components).
2. The strategy manager must expose `device_mesh` (a `torch.distributed.device_mesh.DeviceMesh`) and optionally `moe_mesh`, `pp_size`, and `defer_fsdp_grad_sync`.
3. Register it as a `_target_` in YAML under the `distributed:` section.
4. The recipe's `setup()` at `train_ft.py` line 822-825 instantiates it via `cfg.distributed.instantiate(world_size=...)` and extracts `device_mesh` and `moe_mesh`.
5. Key integration points in the recipe:
   - `build_model_and_optimizer()` receives `model_wrapper` and passes it to `NeMoAutoModelForCausalLM` which calls `model_wrapper.parallelize(model)`.
   - `_forward_backward_step()` uses `get_sync_ctx(model, ...)` for gradient synchronization.
   - `_get_dp_group()`, `_get_dp_rank()`, `_get_tp_rank()`, `_get_pp_rank()` in `BaseRecipe` query `self.device_mesh`.

### Scenario 5: Adding a new loss function

1. Implement the loss in `nemo_automodel/components/loss/`.
2. Register it as a `_target_` in YAML under the `loss_fn:` section.
3. The recipe calls `build_loss_fn(cfg.loss_fn)` at `train_ft.py` line 901, which calls `cfg_loss.instantiate()`.
4. If the loss needs special handling (like `FusedLinearCrossEntropy` which operates on hidden states instead of logits), update `calculate_loss()` at `train_ft.py` line 679 to detect the new loss type and pass the correct arguments.
5. If the model does not support `logits_to_keep`, the recipe automatically falls back to `MaskedCrossEntropy` at `train_ft.py` line 221-223.

---

## File Paths

| File | Absolute Path |
|---|---|
| Base recipe | `/home/scbjtfy/Automodel/nemo_automodel/recipes/base_recipe.py` |
| NTP training/fine-tuning | `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_ft.py` |
| Knowledge distillation | `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/kd.py` |
| Benchmarking | `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/benchmark.py` |
| Sequence classification | `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_seq_cls.py` |
| StepScheduler | `/home/scbjtfy/Automodel/nemo_automodel/components/training/step_scheduler.py` |
| MetricsSample | `/home/scbjtfy/Automodel/nemo_automodel/components/loggers/metric_logger.py` |
| AutoPipeline | `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/pipelining.py` |
| Checkpointer | `/home/scbjtfy/Automodel/nemo_automodel/components/checkpoint/checkpointing.py` |
