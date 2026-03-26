# Reference: LLM Recipes Module API

## BaseRecipe (base_recipe.py)

### Class: `BaseRecipe`

**Location**: `/home/scbjtfy/Automodel/nemo_automodel/recipes/base_recipe.py`, line 127

Provides automatic state tracking and checkpoint save/load for all recipes.

#### `__setattr__(self, key, value)` (line 132)
Intercepts attribute assignment. Classifies `value` using type-checking helpers (`is_model`, `is_optimizer`, `is_lr_scheduler`, `is_dataloader`, `is_tokenizer`, `has_load_restore_state`, `isinstance(ConfigNode)`). Adds `key` to `__state_tracked` set unless the key name contains "val", "eval", "test", or "loss" (case-insensitive substring match, line 165).

#### `save_checkpoint(self, epoch, step, train_loss, val_loss=None, best_metric_key="default")` (line 170)
- Waits for any async checkpoint via `self.checkpointer.async_wait()`.
- Updates LATEST/LOWEST_VAL symlinks for previously-pending async checkpoints.
- Creates `checkpoint_dir/epoch_{epoch}_step_{step}/`.
- Writes `losses.json` with train_loss and val_loss on rank 0.
- Iterates `__state_tracked`: saves model via `save_pretrained()` (single model) or `checkpointer.save_model()` (PP multi-stage), optimizer via `checkpointer.save_optimizer()`, dataloaders/RNG via `checkpointer.save_on_dp_ranks()`, config via `save_config()`.
- For async: defers LATEST/LOWEST_VAL symlink updates to next call. For sync: updates immediately.
- Skips `teacher_model` during save (line 253).

#### `load_checkpoint(self, restore_from=None)` (line 355)
- Resolves latest checkpoint via `_find_latest_checkpoint()` (LATEST symlink or highest step_* directory).
- Loads model via `checkpointer.load_model()`, optimizer via `checkpointer.load_optimizer()`, dataloaders/RNG via `checkpointer.load_on_dp_ranks()`.
- Skips tokenizer and ConfigNode (not loaded from checkpoint).

#### Distributed helpers (lines 533-573)
- `_get_dp_group(include_cp=False)`: Returns DP process group from `device_mesh["dp"]` or `device_mesh["dp_cp"]`.
- `_get_dp_group_size(include_cp=False)`: Returns DP group size.
- `_get_dp_rank(include_cp=False)`: Returns local DP rank.
- `_get_tp_rank()`: Returns local TP rank.
- `_get_pp_rank()`: Returns local PP rank.
- `_dp_allreduce(tensor, op=SUM, include_cp=False)`: All-reduce tensor across DP group.

---

## TrainFinetuneRecipeForNextTokenPrediction (train_ft.py)

### Class: `TrainFinetuneRecipeForNextTokenPrediction(BaseRecipe)`

**Location**: `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_ft.py`, line 785

#### `__init__(self, cfg)` (line 791)
Stores `cfg` (a `ConfigNode`). No other initialization.

#### `setup(self)` (line 800)
Builds all components in sequence:
1. `build_distributed(cfg.dist_env)` -> `self.dist_env` (DistInfo with rank, world_size, device, is_main)
2. `setup_logging()` and `apply_cache_compatibility_patches()`
3. `StatefulRNG(seed)` -> `self.rng`
4. `cfg.distributed.instantiate(world_size=...)` -> `self.model_wrapper`, `self.device_mesh`, `self.moe_mesh`
5. WandB/MLflow initialization (rank 0 only)
6. Pipeline parallelism: `AutoPipeline` construction if `model_wrapper.pp_size > 1`
7. `cfg.peft.instantiate()` -> `self.peft_config`
8. `build_loss_fn(cfg.loss_fn)` -> `self.loss_fn`
9. `build_checkpoint_config(...)` -> `CheckpointingConfig` -> `Checkpointer` -> `self.checkpointer`
10. `build_model_and_optimizer(...)` -> `model`, `self.optimizer`, `self.loss_fn`
11. Model wrapping: if `AutoPipeline`, `self.model_parts = model.parts`; else `self.model_parts = [model]`
12. `build_dataloader(...)` -> `self.dataloader`, `self.tokenizer`
13. `build_validation_dataloader(...)` -> `self.val_dataloaders` (dict)
14. `build_step_scheduler(...)` -> `self.step_scheduler`
15. `build_lr_scheduler(...)` -> `self.lr_scheduler`
16. QAT setup via `_setup_qat()`
17. JSONL metric loggers
18. `self.load_checkpoint(restore_from)` to resume

#### `run_train_validation_loop(self)` (line 1070)
Main training loop. Sets all model parts to train mode. Iterates epochs from `step_scheduler.epochs`, and within each epoch iterates micro-batch groups from `step_scheduler`. For each group:
- Calls `_enable_qat_if_delayed()` if QAT configured.
- Calls `_run_train_optim_step(batches, max_grad_norm)`.
- Calls `log_train_metrics(train_log_data)`.
- If `step_scheduler.is_val_step`: iterates `val_dataloaders`, calls `_run_validation_epoch()`, `log_val_metrics()`.
- If `step_scheduler.is_ckpt_step`: calls `save_checkpoint()`.
- On completion: closes metric loggers and checkpointer.

#### `_forward_backward_step(self, idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train=True)` (line 1119)
Core forward/backward logic. Handles both PP and non-PP paths:
- **Non-PP**: Calls `model(**batch)`, `calculate_loss()`, `loss.backward()`. Uses `get_sync_ctx()` to defer FSDP grad sync until the last micro-batch.
- **PP**: Uses `self.pp.info.schedule.step()` (training) or `.eval()` (validation) to execute the pipeline schedule.
- Context parallelism: `make_cp_batch_and_ctx()` splits batch and creates CP context.
- For `FusedLinearCrossEntropy`: passes `logits_to_keep=1` to model forward to avoid materializing full logits.

#### `_run_train_optim_step(self, batches, max_grad_norm=None)` (line 1216)
Complete optimization step:
1. Counts `num_label_tokens` and `num_tokens_in_batch`, all-reduces across DP.
2. `prepare_for_grad_accumulation()` to disable FSDP grad sync.
3. Loops over micro-batches, calling `_forward_backward_step()`.
4. `scale_grads_and_clip_grad_norm()` after all micro-batches.
5. `optimizer.step()` + `zero_grad()`.
6. Updates MoE gate bias if applicable.
7. `lr_scheduler.step(1)`.
8. Precomputes FP8 scales if configured.
9. Computes throughput (tokens/second) and loss.
10. Returns `MetricsSample`.

#### `_run_validation_epoch(self, val_dataloader)` (line 1329)
Runs validation with `torch.no_grad()` and `ScopedRNG(seed=1)`. Iterates all batches, accumulates loss and label token counts. All-reduces across DP. For PP, sends loss from last stage to main rank. Returns `MetricsSample` with `val_loss`.

#### `log_train_metrics(self, log_data)` (line 1422)
Logs to WandB, MLflow (if `is_remote_logging_step`), JSONL (always), and console. Resets peak memory stats.

#### `log_val_metrics(self, val_name, log_data, metric_logger=None)` (line 1385)
Logs validation metrics to WandB, MLflow, JSONL, and console. Includes `val_name` for multi-dataset support.

---

## Module-level builder functions (train_ft.py)

### `build_model_and_optimizer(cfg_model, cfg_opt, cfg_peft, model_wrapper, seed, ...)` (line 130)
Returns `(model, list[Optimizer], loss_fn)`.
- Detects NeMoAutoModel vs custom model via `cfg_model._target_`.
- For NeMoAutoModel: `cfg_model.instantiate(**kwargs)` with tp_size, cp_size, has_packed_sequence, autopipeline, parallelize_fn, peft_config, model_wrapper, loss_fn, fp8_config, compile_config, quantization_config, qat_quantizer.
- For custom: `cfg_model.instantiate()` then `apply_model_infrastructure()`.
- Falls back to `MaskedCrossEntropy` if model doesn't support `logits_to_keep`.
- Unfreezes specified modules after PEFT (line 226-229).
- Creates optimizer per PP stage (if `model.parts` exists) or single optimizer.
- For MegatronFSDP: calls `fully_shard_optimizer()` (line 268).
- For Dion optimizer: calls `build_dion_optimizer()` (line 241-255).

### `build_dataloader(cfg_ds, cfg_dl, cfg_model, cfg_ps, seed, ...)` (line 366)
Returns `(DataLoader, tokenizer)`.
- Builds tokenizer via `_build_tokenizer()`.
- Handles MegatronPretraining, map-style, and iterable datasets.
- For IterableDataset: shards via `dataset.shard()` or `split_dataset_by_node()`.
- Applies sequence packing if `packed_sequence_size > 0` and model supports `seq_lens`.
- For MegatronPretraining: creates Megatron sampler.
- For map-style: creates `StatefulDistributedSampler`.
- Chains collate functions with PP mask precomputation if PP enabled.

### `build_distributed(cfg_dist)` (line 556)
Returns `DistInfo`. Calls `initialize_distributed(backend, timeout_minutes)`.

### `build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size)` (line 570)
Returns `StepScheduler`. Merges config with defaults (num_epochs=10, global_batch_size=32, ckpt_every_steps=100).

### `build_lr_scheduler(cfg, optimizer, step_scheduler)` (line 596)
Returns `list[OptimizerParamScheduler]` or `None`. Calculates total_steps, creates one scheduler per optimizer. Defaults: 10% warmup (max 1000 steps), cosine decay, init_lr=base*0.1, min_lr=base*0.01.

### `build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft)` (line 274)
Returns `CheckpointingConfig`. Defaults: enabled=True, safetensors format, save_consolidated=True.

### `build_wandb(cfg)` (line 658)
Returns `wandb.Run`. Uses model name as default run name.

### `build_validation_dataloader(cfg, dp_world_size, dp_rank, pp_enabled, model=None)` (line 726)
Returns `dict[str, DataLoader]`. Scans config for all keys starting with `validation_dataset`, builds a dataloader for each.

### `calculate_loss(loss_fn, **kwargs)` (line 679)
Returns `Tensor`. Dispatches based on loss function type:
- `FusedLinearCrossEntropy`: finds `lm_head` weight, passes hidden_states + labels + lm_weight.
- Other: passes logits + labels.
- Always passes `num_label_tokens` for token-level loss normalization.

---

## KnowledgeDistillationRecipeForNextTokenPrediction (kd.py)

### Class: `KnowledgeDistillationRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction)`

**Location**: `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/kd.py`, line 148

#### `setup(self)` (line 151)
- Verifies tokenizer compatibility between student and teacher (vocab size and pad token must match).
- Calls `super().setup()` for full student setup.
- Raises `ValueError` if PP is enabled (not supported for KD).
- Builds teacher model via `_build_teacher_model()` (frozen, eval mode, no PEFT/FP8/QAT).
- Builds KD loss via `_build_kd_loss_fn()` (default: `KLDivLoss(reduction="batchmean")`).
- Sets `kd_ratio` from config (default: 0.5).

#### `_forward_backward_step(self, idx, batch, *, num_label_tokens, num_batches, is_train=True)` (line 188)
Overrides parent. Returns `(local_loss, kd_loss, ce_loss)` instead of appending to loss_buffer.
- Teacher forward with `torch.inference_mode()` and optional CPU offloading (`ScopedModuleOffloading`).
- Student forward with optional `logits_to_keep=1` for `FusedLinearCrossEntropy`.
- Computes `ce_loss` via `calculate_loss()`, `kd_loss` via `self.kd_loss_fn()`.
- Combined: `local_loss = (1 - kd_ratio) * ce_loss + kd_ratio * kd_loss`.

#### `_run_train_optim_step(self, batches, max_grad_norm=None)` (line 256)
Overrides parent. Returns `MetricsSample` with additional metrics: `ce_loss`, `kd_loss`, `kd_ratio`, `temperature`.

#### `_run_validation_epoch(self, val_dataloader)` (line 359)
Overrides parent. Returns `MetricsSample` with `val_loss`, `ce_loss`, `kd_loss`.

### Helper: `_build_teacher_model(cfg_teacher, seed, ...)` (line 70)
Builds teacher using same infrastructure as student but without PEFT/FP8/QAT. Sets `eval()` mode and `requires_grad_(False)`.

### Helper: `_verify_tokenizer_compatibility(student_cfg, teacher_cfg, ...)` (line 130)
Validates student and teacher tokenizers have same vocab_size and pad_token.

---

## BenchmarkingRecipeForNextTokenPrediction (benchmark.py)

### Class: `BenchmarkingRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction)`

**Location**: `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/benchmark.py`, line 33

#### `__init__(self, cfg)` (line 41)
Extracts benchmarking params from `cfg.benchmark`: `warmup_steps`, `peak_tflops`, `nsys_start/end/ranks`, `json_output_path`. Infers `max_steps` from `step_scheduler`, `seq_len` from dataset. Injects `vocab_size` and `batch_size` into dataset config.

#### `setup(self)` (line 86)
Calls `super().setup()` with timer wrapping. Clears validation dataloader. Calculates theoretical TFLOPs via `get_flops_formula_for_hf_config()`. For PEFT: adjusts TFLOPs with `(2 + lora_params/frozen_params) / 3` multiplier.

#### `run_benchmark(self)` (line 166)
Custom benchmarking loop (does not use `run_train_validation_loop`):
- Computes gradient accumulation steps.
- For each step: manual gradient accumulation with `prepare_for_grad_accumulation/final_backward`, `_forward_backward_step()`, optimizer step.
- nsys profiling via `cudaProfilerStart/Stop` in configurable step range.
- Timer instrumentation for iteration, forward_backward, optimizer phases.
- Logs MFU per iteration via `calculate_mfu()`.
- Final summary: avg iteration time, avg MFU, saved to JSON and WandB table.

---

## TrainFinetuneRecipeForSequenceClassification (train_seq_cls.py)

### Class: `TrainFinetuneRecipeForSequenceClassification(BaseRecipe)`

**Location**: `/home/scbjtfy/Automodel/nemo_automodel/recipes/llm/train_seq_cls.py`, line 44

Standalone classification recipe (not subclassing NTP recipe). Uses standard `CrossEntropyLoss` on classification logits. Computes and logs accuracy in addition to loss. Imports `build_*` functions from `train_ft.py` but does not use PP, FP8, sequence packing, or QAT.

#### `setup(self)` (line 50)
Similar to NTP setup but simpler: no PP, no FP8, no QAT. Sets `use_hf_fa2 = False`. Uses `build_model_and_optimizer()` with `unfreeze_modules=["classifier"]` when PEFT is active. Single validation dataloader (not dict).

#### `run_train_validation_loop(self)` (line 181)
Simplified training loop without PP support.

#### `_run_train_optim_step(self, batches)` (line 213)
Forward pass computes logits, CE loss, and accuracy. No gradient accumulation abstraction via prepare_for_grad_accumulation (simpler than NTP). Uses `clip_grad_norm` directly instead of `scale_grads_and_clip_grad_norm`.

#### `_validate_one_epoch(self, dataloader)` (line 298)
Returns `MetricsSample` with `val_loss` and `val_accuracy`.

---

## Entry Points

Each recipe file exposes a `main()` function:

| File | Default config |
|---|---|
| `train_ft.py` | `llama_3_2_1b_hellaswag.yaml` (relative to file) |
| `kd.py` | `examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml` |
| `benchmark.py` | `examples/benchmark/configs/moonlight_16b_torch.yaml` |
| `train_seq_cls.py` | `examples/llm_sequence_classification/yelp/yelp_bert.yaml` |

All use `parse_args_and_load_config(config_path)` to load YAML, construct the recipe, call `setup()`, and run the training loop.
