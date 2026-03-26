---
name: automodel-dr-recipes-vlm
description: Use when working with VLM recipes of automodel — vision-language model fine-tuning with SFT and PEFT support
---

# VLM Recipes Module

## 1. Module Purpose and Capabilities

The `nemo_automodel/recipes/vlm/` module contains a single recipe class, `FinetuneRecipeForVLM`, that orchestrates end-to-end fine-tuning of vision-language models (VLMs). It follows the same three-layer architecture as the LLM recipes (components -> **recipes** -> CLI) but is specialized for multimodal inputs: images, audio, and text are jointly processed through HuggingFace `AutoModelForImageTextToText`-compatible models.

### Recipe inventory

| File | Class | Purpose |
|---|---|---|
| `finetune.py` | `FinetuneRecipeForVLM` | SFT fine-tuning for VLMs; supports FSDP2, TP, CP, PP, PEFT, FP8, MoE, and model-specific collate functions |
| `base_recipe.py` (parent) | `BaseRecipe` | Automatic state tracking, checkpoint save/load, distributed helper methods |

### Key capabilities

- **Multimodal input handling**: The recipe processes batches containing `pixel_values`, `image_grid_hws`, `image_grid_thw`, `input_ids`, `attention_mask`, and `labels`. Batch elements can be tensors or nested dicts of tensors (e.g., for Phi-4 audio inputs), handled by recursive device transfer in `_forward_backward_step()` (finetune.py line 738-745).
- **Model-specific collate functions**: A registry `COLLATE_FNS` in `components/datasets/vlm/collate_fns.py` (line 649-656) maps processor type names to specialized collate functions. Supported processors: `Qwen2_5_VLProcessor`, `Qwen3OmniMoeProcessor`, `KimiVLProcessor`, `KimiK25Processor`, `NemotronParseProcessor`, and a `default` fallback. Custom collate functions can also be specified via the YAML `dataloader.collate_fn._target_` field.
- **Vision-specific pipeline parallelism**: For PP, the recipe implements custom chunking logic for `pixel_values` and `image_grid_hws`/`image_grid_thw` tensors (finetune.py lines 765-823). These tensors have non-standard structures that cannot be naively split along the batch dimension, so they are pre-chunked per microbatch and stored on the first PP stage model (`_vlm_pixel_values_chunks`, `_vlm_image_grid_hws_chunks`, `_vlm_chunk_idx`).
- **Processor-based tokenization**: Unlike the LLM recipe which uses a tokenizer, VLM uses a HuggingFace `Processor` (e.g., `AutoProcessor.from_pretrained`) that combines tokenization with image/audio preprocessing. The processor is loaded in `build_dataloader()` (finetune.py line 239-253) and passed to collate functions.
- **Freeze configuration**: Supports freezing specific model components via `freeze_config` in YAML (e.g., `freeze_embeddings`, `freeze_vision_tower`, `freeze_language_model`), passed to `NeMoAutoModelForImageTextToText` at model construction time (finetune.py line 116-117).
- **PEFT (LoRA)**: Configured via `peft` YAML section with pattern-based module exclusion (e.g., `"*vision_tower*"`, `"*lm_head*"`) to keep vision encoder frozen while adapting language layers.
- **Conversation-based label masking**: All collate functions use `build_labels()` (collate_fns.py line 84-143) which masks non-assistant tokens with -100 (ignored by loss), only training on assistant response tokens. This is the SFT label construction strategy.
- **FusedLinearCrossEntropy support**: The recipe can use `FusedLinearCrossEntropy` which operates on hidden states instead of logits, avoiding full logits materialization (finetune.py lines 843-861). Falls back to `MaskedCrossEntropy` if the model does not support `logits_to_keep` (finetune.py line 140-144).
- **Validation loop**: Optional validation with a separate dataloader. Validation is not supported with pipeline parallelism (finetune.py line 704).

### Supported model families (via example configs)

| Model | Example Config |
|---|---|
| Gemma 3 VL 4B | `examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml` |
| Gemma 3N VL 4B | `examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix.yaml` |
| Qwen 2.5 VL 3B | `examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml` |
| Qwen 3 VL 4B/8B | `examples/vlm_finetune/qwen3/qwen3_vl_4b_instruct_rdr.yaml` |
| Qwen 3 VL MoE 30B/235B | `examples/vlm_finetune/qwen3/qwen3_vl_moe_30b_te_deepep.yaml` |
| Qwen 3 Omni MoE 30B | `examples/vlm_finetune/qwen3/qwen3_omni_moe_30b_te_deepep.yaml` |
| Phi-4 MM (audio) | `examples/vlm_finetune/phi4/phi4_mm_cv17.yaml` |
| Ministral 3 (3B/8B/14B) | `examples/vlm_finetune/mistral/ministral3_3b_medpix.yaml` |
| InternVL 3.5 4B | `examples/vlm_finetune/internvl/internvl_3_5_4b.yaml` |
| Nemotron-Parse | `examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml` |
| Kimi VL / Kimi K2.5 VL | `examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml` |

---

## 2. Core Design Logic

### 2.1 Why the VLM recipe is a separate module from the LLM recipe

The VLM recipe (`FinetuneRecipeForVLM`) is **not** a subclass of the LLM recipe (`TrainFinetuneRecipeForNextTokenPrediction`). It is a standalone class that extends `BaseRecipe` directly. This separation exists because:

1. **Different model class**: VLM uses `NeMoAutoModelForImageTextToText` instead of `NeMoAutoModelForCausalLM`. The `build_model_and_optimizer()` function (finetune.py line 86-164) enforces this with a strict check at line 130-137: it raises `ValueError` if the model target is not `NeMoAutoModelForImageTextToText.from_config` or `.from_pretrained`.

2. **Processor instead of tokenizer**: VLMs use a `ProcessorMixin` (which combines tokenizer + image processor) rather than a standalone tokenizer. The `build_dataloader()` function (finetune.py line 212-287) loads an `AutoProcessor` and passes it to collate functions for joint text+image preprocessing.

3. **Non-standard batch structure**: VLM batches contain heterogeneous tensors (`pixel_values`, `image_grid_hws`, `image_grid_thw`) with shapes that do not correspond to the batch dimension of `input_ids`. The `_forward_backward_step()` method (finetune.py line 728-864) has recursive device transfer logic (line 738-745) to handle nested dicts and optional None values.

4. **Vision-specific PP chunking**: Pipeline parallelism for VLMs requires custom microbatch splitting of vision inputs (finetune.py lines 765-823), which has no analogue in the LLM recipe.

### 2.2 Self-contained builder functions (no cross-import with LLM recipe)

Unlike the LLM recipe where `kd.py` and `benchmark.py` import builders from `train_ft.py`, the VLM `finetune.py` defines its own complete set of builder functions. This is intentional: the VLM builders have different signatures and VLM-specific logic (e.g., `build_dataloader` returns `(DataLoader, ProcessorMixin)` rather than `(DataLoader, tokenizer)`). The builders are module-level stateless functions following the same pattern as the LLM recipe.

### 2.3 Collate function dispatch

The collate function is resolved in `build_dataloader()` (finetune.py line 262-270) via a two-tier strategy:
1. If `cfg_dl.collate_fn` is specified in YAML, it is instantiated via `_target_` (line 264).
2. Otherwise, the processor's class name (e.g., `"Qwen2_5_VLProcessor"`) is looked up in the `COLLATE_FNS` registry (collate_fns.py line 649-656). If the processor type is unknown, `"default"` is used with a warning.

This dispatch allows supporting new VLM architectures by either adding a collate function to the registry or specifying a custom one in the YAML config.

### 2.4 Label construction strategy

All VLM collate functions share the same label construction logic via `build_labels()` (collate_fns.py line 84-143). This function:
1. Initializes labels as all -100 (ignore index for cross-entropy loss).
2. For each assistant turn in the conversation, tokenizes the assistant text.
3. Pattern-matches the tokenized assistant text within the full tokenized sequence using `_find_pattern_indices()` (collate_fns.py line 44-51).
4. Copies the original token IDs into the labels only at matched assistant positions.
5. Extends labels by one token if the next token after the assistant response is a stop token (e.g., `<end_of_turn>`, `<|im_end|>`).

This ensures the model is trained only on generating assistant responses, not on reproducing user prompts or image tokens.

### 2.5 Configuration-driven instantiation

Same pattern as LLM: `ConfigNode` objects loaded from YAML resolve `_target_` fields to Python callables. The VLM recipe requires these YAML sections:

| Section | Required | Purpose |
|---|---|---|
| `model` | Yes | `_target_` must be `NeMoAutoModelForImageTextToText.from_pretrained` or `.from_config` |
| `loss_fn` | Yes | Usually `MaskedCrossEntropy` or `FusedLinearCrossEntropy` |
| `dataset` | Yes | Dataset factory (e.g., `make_cord_v2_dataset`) |
| `dataloader` | Yes | DataLoader config (typically `StatefulDataLoader`) |
| `optimizer` | Yes | Optimizer (e.g., `torch.optim.AdamW`) |
| `step_scheduler` | Yes | Batch size, checkpointing frequency, max steps |
| `distributed` | No | FSDP2Manager, MegatronFSDPManager, or omit for single-GPU |
| `peft` | No | LoRA/adapter config |
| `freeze_config` | No | Component freezing (`freeze_vision_tower`, etc.) |
| `checkpoint` | No | Checkpoint directory and format |
| `processor` | No | Custom processor config; if absent, `AutoProcessor.from_pretrained` is used |
| `validation_dataset` / `validation_dataloader` | No | Validation data |
| `lr_scheduler` | No | Learning rate schedule |
| `wandb` | No | Weights & Biases logging |
| `fp8` | No | FP8 training |
| `compile` | No | torch.compile config |
| `autopipeline` | No | Pipeline parallelism config |
| `parallelizer` | No | Custom parallelization function |
| `clip_grad_norm` | No | Gradient clipping (default `max_norm=1.0`) |

---

## 3. Core Data Structures

### 3.1 FinetuneRecipeForVLM (finetune.py line 483)

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
| `loss_fn` | `nn.Module` | `MaskedCrossEntropy`, `FusedLinearCrossEntropy`, or custom |
| `dataloader` | `StatefulDataLoader` | training dataloader |
| `val_dataloader` | `DataLoader` or `None` | single validation dataloader |
| `processor` | `ProcessorMixin` or `None` | HF processor (tokenizer + image processor) |
| `step_scheduler` | `StepScheduler` | gradient accumulation and epoch management |
| `checkpointer` | `Checkpointer` | checkpoint manager |
| `peft_config` | PEFT config or `None` | LoRA/adapter config |
| `rng` | `StatefulRNG` | checkpointable RNG |
| `max_grad_norm` | `float` | gradient clipping threshold (default 1.0) |
| `best_metric_key` | `str` | validation metric key for best checkpoint selection |
| `metric_logger_train` | `MetricLogger` | JSONL logger for training metrics |
| `metric_logger_valid` | `MetricLogger` | JSONL logger for validation metrics |

### 3.2 Key differences from LLM recipe attributes

| Aspect | LLM Recipe | VLM Recipe |
|---|---|---|
| Model class | `NeMoAutoModelForCausalLM` | `NeMoAutoModelForImageTextToText` |
| Tokenizer/Processor | `self.tokenizer` (PreTrainedTokenizerBase) | `self.processor` (ProcessorMixin) |
| Validation dataloaders | `self.val_dataloaders` (dict of named loaders) | `self.val_dataloader` (single loader or None) |
| Batch contents | `input_ids`, `attention_mask`, `labels`, optional `position_ids` | `input_ids`, `attention_mask`, `labels`, `pixel_values`, `image_grid_hws`/`image_grid_thw`, plus model-specific keys |
| PP vision chunking | N/A | `_vlm_pixel_values_chunks`, `_vlm_image_grid_hws_chunks`, `_vlm_chunk_idx` on stage-0 model |

### 3.3 Module-level builder functions (finetune.py)

| Function | Line | Returns | Purpose |
|---|---|---|---|
| `build_model_and_optimizer()` | 86 | `(model, list[Optimizer], loss_fn)` | Enforces VLM model class, builds model+optimizer |
| `build_dataloader()` | 212 | `(DataLoader, ProcessorMixin)` | Loads processor, dataset, sampler, and dispatches collate fn |
| `build_distributed()` | 290 | `DistInfo` | Initializes distributed backend |
| `build_step_scheduler()` | 304 | `StepScheduler` | Configures gradient accumulation and epoch scheduling |
| `build_lr_scheduler()` | 330 | `list[OptimizerParamScheduler]` or `None` | Creates cosine decay with warmup |
| `build_checkpoint_config()` | 167 | `CheckpointingConfig` | Checkpoint directory, format, PEFT compatibility |
| `build_loss_fn()` | 200 | `nn.Module` | Instantiates loss from config |
| `build_wandb()` | 391 | `wandb.Run` | Initializes W&B logging |
| `calculate_loss()` | 412 | `Tensor` | Routes to FusedLinearCE (hidden states) or standard CE (logits) |
| `parallelize_for_pp()` | 459 | `nn.Module` | Default PP parallelization via model_wrapper |

### 3.4 VLM Dataset factories (components/datasets/vlm/datasets.py)

| Function | Default dataset | Modality | Output format |
|---|---|---|---|
| `make_rdr_dataset()` | `quintend/rdr-items` | Image + text | `{"conversation": [...]}` |
| `make_cord_v2_dataset()` | `naver-clova-ix/cord-v2` | Image + structured text | `{"conversation": [...]}` |
| `make_medpix_dataset()` | `medpix-dataset/medpix-dataset` | Medical image + QA | `{"conversation": [...]}` |
| `make_cv17_dataset()` | `ysdede/commonvoice_17_tr_fixed` | Audio + text | `{"conversation": [...], "audio": (...)}` |
| `make_unimm_chat_dataset()` | `Yirany/UniMM-Chat` | Image + multi-turn chat | `{"conversation": [...]}` |

All dataset factories return lists of dicts with a `"conversation"` key following the OpenAI chat format: `[{"role": "user", "content": [...]}, {"role": "assistant", "content": [...]}]`. Image content items use `{"type": "image", "image": <PIL.Image or path>}`.

### 3.5 COLLATE_FNS registry (components/datasets/vlm/collate_fns.py line 649-656)

```python
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3OmniMoeProcessor": qwen3_omni_collate_fn,
    "KimiVLProcessor": kimi_vl_collate_fn,
    "KimiK25Processor": kimi_k25_vl_collate_fn,
    "NemotronParseProcessor": nemotron_parse_collate_fn,
    "default": default_collate_fn,
}
```

Each collate function follows the same contract: `(examples: list[dict], processor) -> dict[str, Tensor]`. The returned dict always contains `input_ids`, `attention_mask`, and `labels`, plus model-specific keys like `pixel_values`, `image_grid_hws`, or `decoder_input_ids` (Nemotron-Parse).

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
FinetuneRecipeForVLM.__init__(cfg)    -- stores cfg only
    |
    v
FinetuneRecipeForVLM.setup()          -- builds ALL components:
    |  1. build_distributed()                -> self.dist_env
    |  2. setup_logging()
    |  3. apply_cache_compatibility_patches()
    |  4. StatefulRNG(seed)                  -> self.rng
    |  5. cfg.distributed.instantiate()      -> self.model_wrapper, self.device_mesh, self.moe_mesh
    |  6. build_wandb() (if wandb in config) -> W&B run
    |  7. _log_experiment_details(), _log_library_versions()
    |  8. AutoPipeline construction (if PP)
    |  9. build_loss_fn()                    -> self.loss_fn
    |  10. build_checkpoint_config()         -> checkpoint_config
    |  11. Checkpointer construction         -> self.checkpointer
    |  12. build_model_and_optimizer()       -> self.model_parts, self.optimizer, self.loss_fn
    |  13. build_dataloader()                -> self.dataloader, self.processor
    |  14. build_dataloader() for validation -> self.val_dataloader (if configured)
    |  15. build_step_scheduler()            -> self.step_scheduler
    |  16. build_lr_scheduler()              -> self.lr_scheduler
    |  17. build_metric_logger()             -> self.metric_logger_train, self.metric_logger_valid
    |  18. load_checkpoint()                 -- resume if checkpoint exists
    v
FinetuneRecipeForVLM.run_train_validation_loop()
    |
    for epoch in step_scheduler.epochs:
        step_scheduler.set_epoch(epoch)
        for batches in step_scheduler:           -- yields grad_acc_steps micro-batches
            train_log = _run_train_optim_step(batches)
            log_train_metrics(train_log)
            if is_val_step and val_dataloader:
                if pp_enabled: warn & skip
                else:
                    val_log = _run_validation_epoch(val_dataloader)
                    log_val_metrics(val_log)
                    set model_parts back to train()
            if is_ckpt_step:
                save_checkpoint(epoch, step, loss, val_loss)
    |
    v
Close metric loggers and checkpointer
```

### 4.2 Data pipeline flow (VLM-specific)

```
Dataset factory (e.g., make_cord_v2_dataset)
    |
    v
Returns list of {"conversation": [...]} dicts
    |
    v
DistributedSampler (dp-aware sharding)
    |
    v
Collate function (dispatched by processor type)
    |  1. Extract conversations from examples
    |  2. Apply processor.apply_chat_template() for text
    |  3. Extract images via process_vision_info() (Qwen) or from content items
    |  4. Call processor(text=texts, images=images, ...) -> tokenized batch
    |  5. build_labels(): mask non-assistant tokens with -100
    |  6. Shift inputs: remove last token for autoregressive alignment
    v
Batch dict: {input_ids, attention_mask, labels, pixel_values, image_grid_*}
    |
    v
_forward_backward_step(): recursive device transfer for all values
```

### 4.3 Single training step flow (_run_train_optim_step, finetune.py line 866)

```
batches (list of grad_acc_steps micro-batches)
    |
    v
count num_label_tokens (tokens where labels != -100)
dp_allreduce num_label_tokens across DP group
    |
    v
count num_tokens_in_batch (excluding tail padding)
dp_allreduce num_tokens_in_batch
    |
    v
prepare_for_grad_accumulation(model_parts)     -- disable grad sync
    |
    for i, batch in enumerate(batches):
        if last micro-batch:
            prepare_for_final_backward()        -- re-enable grad sync
        _forward_backward_step(i, batch, ...)
            |-- recursive device transfer (handles nested dicts, None values)
            |-- make_cp_batch_and_ctx()
            |-- pop labels from batch
            |-- if PP:
            |     pop pixel_values, image_grid_hws/image_grid_thw
            |     chunk vision inputs per microbatch
            |     store chunks on stage-0 model
            |     schedule.step(input_ids, target=labels, ...)
            |     clear stored chunks
            |-- else:
            |     model(**batch)                 -- forward (with or without logits_to_keep)
            |     calculate_loss()
            |     (loss * dp_group_size).backward()
            |-- append loss to loss_buffer
    |
    v
scale_grads_and_clip_grad_norm(max_grad_norm)
optimizer.step() + zero_grad(set_to_none=True)
update_moe_gate_bias() if applicable
lr_scheduler.step(1)
precompute_float8_dynamic_scale (if FP8 enabled)
    |
    v
compute tps (tokens per second), aggregate loss across DP ranks
PP: send loss from last PP stage to rank 0
return MetricsSample(step, epoch, {loss, grad_norm, lr, mem, tps, tps_per_gpu, ...})
```

### 4.4 PP vision chunking detail (finetune.py lines 765-823)

When pipeline parallelism is enabled and the first stage has pixel_values:

```
pixel_values: (N_patches, C, H, W)      -- all patches concatenated
image_grid: (N_images, 2 or 3)          -- per-image grid dimensions
    |
    v
Compute patch_counts = image_grid.prod(dim=1)  -- patches per image
cumsum = cumulative sum of patch_counts
    |
    v
If n_images == batch_size (1 image per sample):
    Split into n_microbatches groups of (batch_size // n_microbatches) images
    For each microbatch: slice pixel_values by cumulative patch range
Else:
    Give all images to first microbatch, empty tensors to rest
    (with a warning about non-standard multi-image layouts)
    |
    v
Store on stage-0 model:
    model._vlm_pixel_values_chunks = [chunk_0, chunk_1, ...]
    model._vlm_image_grid_hws_chunks = [grid_0, grid_1, ...]
    model._vlm_chunk_idx = 0
    |
    v
PP schedule step retrieves chunks via the stored attributes
    |
    v
After PP step: clear stored chunks (set to None)
```

### 4.5 Validation flow (_run_validation_epoch, finetune.py line 990)

```
@torch.no_grad() with ScopedRNG(seed=1)
    |
    v
model_parts -> eval mode
    |
    for batch in val_dataloader:
        move batch to device
        pop labels, count num_label_tokens
        add position_ids if CP>1 or TP>1 and not already present
        make_cp_batch_and_ctx()
        model(**batch) -> out
        calculate_loss() -> local_loss
        accumulate total_loss weighted by num_label_tokens
    |
    v
dp_allreduce total_loss, total_tokens
val_loss = total_loss / total_tokens
return MetricsSample({val_loss, lr, num_label_tokens, mem})
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding support for a new VLM architecture

To add a new VLM (e.g., LLaVA-Next):

1. **Create a collate function** in `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/collate_fns.py`. Follow the contract: `def llava_next_collate_fn(examples: list[dict], processor) -> dict[str, Tensor]`. Extract conversations, process with the model's specific processor, call `build_labels()` for SFT label masking, and shift inputs for autoregressive alignment.
2. **Register in `COLLATE_FNS`** (collate_fns.py line 649): add `"LlavaNextProcessor": llava_next_collate_fn`.
3. **Create a dataset factory** in `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/datasets.py` if needed, following the conversation-format output pattern.
4. **Create a YAML config** under `examples/vlm_finetune/llava_next/`. Set `model._target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained` and point to the HF model name. The collate function will be auto-dispatched by processor type.
5. **If the model has non-standard vision tensor layouts** for pipeline parallelism, update `_forward_backward_step()` (finetune.py lines 765-823) to handle the new tensor keys.

### Scenario 2: Adding a new loss function for VLM training

To add a custom VLM loss (e.g., a vision-language contrastive loss):

1. Implement the loss in `/home/scbjtfy/Automodel/nemo_automodel/components/loss/`.
2. Register it as a `_target_` in the YAML `loss_fn` section.
3. If the loss requires different forward arguments than logits+labels (like `FusedLinearCrossEntropy` requires hidden states), update `calculate_loss()` at finetune.py line 412-456 to detect the new loss type and pass the correct arguments.
4. If the loss needs additional model outputs, ensure the model's forward pass is called with the right flags (e.g., `output_hidden_states=True` for hidden-state-based losses).

### Scenario 3: Enabling PEFT (LoRA) for VLM fine-tuning

To switch from full fine-tuning to LoRA:

1. Add a `peft` section to the YAML config:
   ```yaml
   peft:
     _target_: nemo_automodel.components._peft.lora.PeftConfig
     match_all_linear: false
     exclude_modules:
       - "*vision_tower*"
       - "*vision*"
       - "*visual*"
       - "*image_encoder*"
       - "*lm_head*"
     dim: 8
     alpha: 32
   ```
2. The recipe detects `peft` in config at finetune.py line 576 and instantiates the PEFT config.
3. The PEFT config is passed to `build_model_and_optimizer()` -> `NeMoAutoModelForImageTextToText.from_pretrained()` which applies LoRA adapters and freezes base weights.
4. Change `checkpoint.model_save_format` to `safetensors` (PEFT does not support `torch_save`, enforced at finetune.py line 192-195).
5. Reference config: `/home/scbjtfy/Automodel/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml`.

### Scenario 4: Adding a new dataset for VLM fine-tuning

1. Create a dataset factory function in `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/datasets.py`:
   ```python
   def make_my_dataset(path_or_dataset="org/my-dataset", split="train", **kwargs):
       dataset = load_dataset(path_or_dataset, split=split)
       def format(example):
           return {
               "conversation": [
                   {"role": "user", "content": [
                       {"type": "image", "image": example["image"]},
                       {"type": "text", "text": example["question"]},
                   ]},
                   {"role": "assistant", "content": [
                       {"type": "text", "text": example["answer"]},
                   ]},
               ],
           }
       return [format(example) for example in dataset]
   ```
2. The function must return a list of dicts with a `"conversation"` key in the OpenAI chat format. Image content uses `{"type": "image", "image": <PIL Image or URL>}`.
3. Reference the function in YAML:
   ```yaml
   dataset:
     _target_: nemo_automodel.components.datasets.vlm.datasets.make_my_dataset
     path_or_dataset: org/my-dataset
     split: train
   ```
4. If the model's processor requires special handling not covered by existing collate functions, also add a collate function (see Scenario 1).

### Scenario 5: Extending the VLM recipe with knowledge distillation

To create a VLM KD recipe (similar to how `kd.py` extends the LLM recipe):

1. Create `nemo_automodel/recipes/vlm/kd.py`.
2. Subclass `FinetuneRecipeForVLM`.
3. Override `setup()` to additionally build a frozen teacher VLM (using `NeMoAutoModelForImageTextToText.from_pretrained` with `torch.no_grad()`).
4. Override `_forward_backward_step()` to:
   - Run the teacher forward with the full batch (including `pixel_values`) under `torch.inference_mode()`.
   - Compute KD loss between student and teacher logits.
   - Combine CE loss and KD loss: `loss = (1-ratio)*ce + ratio*kd`.
5. Key consideration: the teacher model must process the same multimodal inputs, so it needs the same processor. Store the teacher as `self.teacher_model` -- `BaseRecipe.__setattr__` at `base_recipe.py` line 253 skips checkpoint-tracking for `"teacher_model"` by name.
6. Assign KD-specific attributes (e.g., `kd_loss_fn`) as instance attributes, but name them to avoid substring matches with "val"/"eval"/"test"/"loss" if they should be checkpoint-tracked (see `base_recipe.py` line 165).

---

## File Paths

| File | Absolute Path |
|---|---|
| VLM fine-tuning recipe | `/home/scbjtfy/Automodel/nemo_automodel/recipes/vlm/finetune.py` |
| Base recipe | `/home/scbjtfy/Automodel/nemo_automodel/recipes/base_recipe.py` |
| VLM collate functions | `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/collate_fns.py` |
| VLM dataset factories | `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/datasets.py` |
| VLM utilities | `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/utils.py` |
| VLM datasets __init__ | `/home/scbjtfy/Automodel/nemo_automodel/components/datasets/vlm/__init__.py` |
| NeMoAutoModelForImageTextToText | `/home/scbjtfy/Automodel/nemo_automodel/_transformers/auto_model.py` |
| Example config (Gemma3 SFT) | `/home/scbjtfy/Automodel/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml` |
| Example config (Gemma3 PEFT) | `/home/scbjtfy/Automodel/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml` |
| StepScheduler | `/home/scbjtfy/Automodel/nemo_automodel/components/training/step_scheduler.py` |
| MetricsSample | `/home/scbjtfy/Automodel/nemo_automodel/components/loggers/metric_logger.py` |
| AutoPipeline | `/home/scbjtfy/Automodel/nemo_automodel/components/distributed/pipelining.py` |
| Checkpointer | `/home/scbjtfy/Automodel/nemo_automodel/components/checkpoint/checkpointing.py` |
