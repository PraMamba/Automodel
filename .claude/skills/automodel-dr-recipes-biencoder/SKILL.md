---
name: automodel-dr-recipes-biencoder
description: Use when working with biencoder recipes of automodel — contrastive biencoder training for embedding models
---

# Biencoder Recipes Module

**Module path:** `nemo_automodel/recipes/biencoder/`
**Files (3):**
- `__init__.py` (19 lines) -- exports `TrainBiencoderRecipe`, `MineHardNegativesRecipe`
- `train_biencoder.py` (819 lines) -- contrastive training recipe
- `mine_hard_negatives.py` (1321 lines) -- hard negative mining recipe

**Base class:** `nemo_automodel/recipes/base_recipe.py` -- `BaseRecipe` (checkpoint save/load, state tracking, distributed helpers)

---

## 1. Module Purpose & Capabilities

This module provides two standalone recipes that form the biencoder embedding-model workflow:

1. **`TrainBiencoderRecipe`** (`train_biencoder.py`) -- End-to-end contrastive training of a dual-encoder (biencoder) model. Given query-document pairs with positives and hard negatives, it fine-tunes a shared or separate query/passage encoder using cross-entropy loss over in-batch contrastive similarity scores. Supports FSDP2 data parallelism, LoRA/PEFT, gradient accumulation, WandB logging, and checkpoint resume.

2. **`MineHardNegativesRecipe`** (`mine_hard_negatives.py`) -- Offline hard negative mining. Loads a trained biencoder checkpoint, encodes the full query set and document corpus into dense embeddings, then selects high-similarity non-relevant documents as hard negatives for each query. Outputs an enriched JSON file that can be fed directly into the training recipe. Supports multi-GPU distributed embedding generation via rank-sharded caching.

Together they implement the standard iterative embedding-model training loop: **train model -> mine hard negatives -> retrain with harder data**.

---

## 2. Core Design Logic

### Why two separate recipes instead of one

Training and mining have fundamentally different resource profiles. Training requires gradient computation, optimizer state, and an online data pipeline. Mining requires only inference-mode forward passes followed by a large matrix-multiply scoring phase. Separating them allows:
- Mining to unload the model after embedding generation (`_unload_model()` at `mine_hard_negatives.py:927`) to reclaim GPU memory for the scoring phase.
- Mining to run on a different GPU count or node configuration than training.
- Iterative refinement: mine, inspect results, retrain, without coupling the two.

### Why the recipe inherits from BaseRecipe (training only)

`TrainBiencoderRecipe` extends `BaseRecipe` (`base_recipe.py:127`) to inherit automatic state tracking via `__setattr__` (`base_recipe.py:132`). Any `nn.Module`, optimizer, LR scheduler, dataloader, or `ConfigNode` assigned as a recipe attribute is automatically registered for checkpoint save/load. This means `save_checkpoint()` and `load_checkpoint()` work without explicit registration.

`MineHardNegativesRecipe` does NOT inherit `BaseRecipe` because mining is a stateless inference pipeline -- there is no optimizer state, no training loop, and no need for checkpoint persistence of the recipe itself.

### Contrastive learning approach

The loss is standard in-batch cross-entropy over a similarity matrix. The `BiencoderModel.forward()` method (`nemo_automodel/components/models/biencoder/llama_bidirectional_model.py:419`) calls `_compute_scores()` which:
1. Encodes queries through `lm_q` and passages through `lm_p` (shared or separate encoders).
2. Pools hidden states via the configured strategy (avg, cls, last) using `pool()`.
3. Optionally L2-normalizes embeddings and applies temperature scaling (`scores / self.t`).
4. Computes `contrastive_scores_and_labels()` to build the similarity matrix and diagonal labels.
5. Applies `nn.CrossEntropyLoss` over the score matrix.

The recipe's `_forward_backward_step()` (`train_biencoder.py:567`) wraps this with `torch.amp.autocast("cuda", dtype=torch.bfloat16)` and gradient accumulation sync contexts.

### Distributed embedding generation in mining

Mining shards work across ranks using a file-based coordination pattern (not collective communication for tensors):
- Each rank computes its partition via `_compute_rank_partition()` (`mine_hard_negatives.py:102`).
- Each rank writes its shard to `{cache_dir}/corpus_chunks/chunk_XXXX_rankYYYY.npz`.
- After a barrier, rank 0 assembles all shards into the final chunk file.
- Only rank 0 performs the actual mining (similarity scoring and top-k selection) since it holds the full embeddings.

### Hard negative margin filtering

The mining algorithm applies a margin-based filter to avoid selecting "false negatives" (documents so similar to positives they may actually be relevant). The margin can be percentage-based (`perc`) or absolute (`abs`), controlled by `hard_neg_margin` and `hard_neg_margin_type` (defaults: `0.95` / `perc`). Documents scoring above `min_positive_score * margin` are set to `-inf` before top-k selection (`mine_hard_negatives.py:1044-1057`).

---

## 3. Core Data Structures

### `TrainBiencoderRecipe` (`train_biencoder.py:299`)

Extends `BaseRecipe`. Key attributes set during `setup()`:

| Attribute | Type | Source |
|---|---|---|
| `cfg` | `ConfigNode` | Constructor arg; parsed from YAML |
| `dist_env` | `DistInfo` | `build_distributed()` at line 318 |
| `rng` | `StatefulRNG` | `train_biencoder.py:324` |
| `device_mesh` | `DeviceMesh \| None` | From `model_wrapper` if distributed config present |
| `model_parts` | `list[nn.Module]` | `[model]` (single element; PP not supported yet, line 417) |
| `pp` | `AutoPipeline \| None` | Always `None` for biencoder (PP raises `NotImplementedError`) |
| `optimizer` | `list[Optimizer]` | Single-element list, built from `cfg.optimizer` |
| `tokenizer` | `PreTrainedTokenizerBase` | From `cfg.tokenizer.instantiate()` |
| `dataloader` | `StatefulDataLoader` | Built via `build_dataloader()` |
| `val_dataloader` | `StatefulDataLoader \| None` | Optional validation loader |
| `step_scheduler` | `StepScheduler` | Controls epochs, grad accum, checkpoint frequency |
| `lr_scheduler` | `list[OptimizerParamScheduler] \| None` | LR warmup/decay |
| `checkpointer` | `Checkpointer` | From `build_checkpoint_config()` |
| `peft_config` | `PeftConfig \| None` | LoRA config if `cfg.peft` present |
| `metric_logger_train` | `MetricLoggerDist` | JSONL logger for training metrics |
| `metric_logger_valid` | `MetricLoggerDist` | JSONL logger for validation metrics |

### `MineHardNegativesRecipe` (`mine_hard_negatives.py:138`)

Plain class (no BaseRecipe inheritance). Key attributes:

| Attribute | Type | Purpose |
|---|---|---|
| `cfg` | `ConfigNode` | Constructor arg |
| `dist_env` | `DistInfo` | Distributed environment |
| `mining_cfg` | `ConfigNode` | Sub-config under `cfg.mining` |
| `model` | `NeMoAutoModelBiencoder \| None` | Loaded via `from_pretrained`, set to `None` after `_unload_model()` |
| `tokenizer` | `NeMoAutoTokenizer` | Configured with optional BOS/EOS overrides |
| `questions_dataset` | loaded dataset | From `load_datasets()` |
| `documents_dataset` | corpus object | Single corpus from `load_datasets()` |
| `doc_to_idx` / `idx_to_doc` | `dict` | Bidirectional document-ID-to-index mapping |
| `questions` | `list[str]` | Query texts (with `EMPTY_QUESTION` placeholder for blanks) |
| `question_ids` / `corpus_ids` | `list` | IDs for alignment |
| `pos_doc_indices` | `list[list[int]]` | Positive document indices per query |
| `supplied_neg_doc_indices` | `list[list[int]]` | Existing negatives from input file |
| `query_embeddings` | `np.ndarray` | Shape `[num_queries, dim]` |
| `document_embeddings` | `np.ndarray` | Shape `[num_docs, dim]` |
| `mined_neg_indices` | `list[list[int]]` | Mined hard negative indices per query |
| `mined_neg_scores` | `list[list[float]]` | Similarity scores for negatives |
| `pos_scores` | `list[list[float]]` | Similarity scores for positives |

### `MINING_DEFAULTS` (`mine_hard_negatives.py:47-72`)

Dictionary of default mining parameters. Key entries:
- `hard_negatives_to_mine`: 20
- `hard_neg_margin`: 0.95 (percentage-based by default)
- `corpus_chunk_size`: 50000 (documents encoded per chunk for memory management)
- `query_prefix` / `passage_prefix`: empty strings (override for prefix-trained models)
- `add_bos_token` / `add_eos_token`: `None` (use Automodel tokenizer defaults)

### Batch unpacking: `_unpack_qp()` (`train_biencoder.py:55`)

Splits a flat batch dictionary with `q_*` and `d_*` prefixed keys into separate query and passage dictionaries. Also extracts `kd_labels` (knowledge distillation labels) if present, attaching them to the query dict.

### Key external types referenced

- `BiencoderModel` (`nemo_automodel/components/models/biencoder/llama_bidirectional_model.py:375`) -- the actual `nn.Module` with `lm_q`, `lm_p` encoders, `cross_entropy` loss, and `_compute_scores()`.
- `NeMoAutoModelBiencoder` (`nemo_automodel/components/models/biencoder/biencoder_model.py:33`) -- factory class wrapping `BiencoderModel.build()` with kernel patching (Liger, SDPA).
- `RetrievalBiencoderCollator` (`nemo_automodel/components/datasets/llm/retrieval_collator.py:40`) -- tokenizes queries/passages at batch time with prefixes and dynamic padding.
- `make_retrieval_dataset` (`nemo_automodel/components/datasets/llm/retrieval_dataset.py:329`) -- loads QA JSON files into a dataset with positive/negative document pairs.

### Example config: `examples/biencoder/llama3_2_1b_biencoder.yaml`

Key sections: `model` (target: `NeMoAutoModelBiencoder.from_pretrained`), `tokenizer`, `dataloader` (with `make_retrieval_dataset` + `RetrievalBiencoderCollator`), `optimizer` (FusedAdam), `lr_scheduler`, `checkpoint`, `distributed` (FSDP2).

### Example config: `examples/biencoder/mining_config.yaml`

Minimal config with only `dist_env` and `mining` sections. Required params (`model_name_or_path`, `train_qa_file_path`, `train_file_output_path`) must be supplied via CLI overrides.

---

## 4. State Flow

### Training flow (`TrainBiencoderRecipe`)

```
main() [train_biencoder.py:803]
  |-> parse_args_and_load_config(default_config_path)
  |-> TrainBiencoderRecipe(cfg)
  |-> recipe.setup()
  |     |-> build_distributed() -> self.dist_env
  |     |-> setup_logging()
  |     |-> StatefulRNG(seed) -> self.rng
  |     |-> cfg.distributed.instantiate() -> self.model_wrapper (FSDP2Manager)
  |     |-> build_wandb(cfg) [if wandb configured, rank 0 only]
  |     |-> _log_experiment_details(), _log_library_versions()
  |     |-> cfg.peft.instantiate() -> self.peft_config [optional]
  |     |-> build_checkpoint_config() -> checkpoint_config
  |     |-> Checkpointer(config=checkpoint_config) -> self.checkpointer
  |     |-> cfg.model.instantiate() -> model [NeMoAutoModelBiencoder]
  |     |-> apply_lora_to_linear_modules(model) [if peft_config]
  |     |-> model_wrapper.parallelize(model) [if distributed]
  |     |-> model.to(device) -> self.model_parts = [model]
  |     |-> cfg.optimizer.instantiate(params=trainable_params) -> self.optimizer
  |     |-> cfg.tokenizer.instantiate() -> self.tokenizer
  |     |-> build_dataloader() -> self.dataloader
  |     |-> build_dataloader() -> self.val_dataloader [optional]
  |     |-> build_step_scheduler() -> self.step_scheduler
  |     |-> build_lr_scheduler() -> self.lr_scheduler
  |     |-> MetricLoggerDist() -> self.metric_logger_train, self.metric_logger_valid
  |     |-> self.load_checkpoint(restore_from) [resume if exists]
  |
  |-> recipe.run_train_validation_loop()
        |-> for epoch in step_scheduler.epochs:
        |     |-> step_scheduler.set_epoch(epoch)
        |     |-> for batches in step_scheduler:   # yields grad-accum batch groups
        |           |-> _run_train_optim_step(batches)
        |           |     |-> for idx, batch in batches:
        |           |     |     |-> _forward_backward_step(idx, batch)
        |           |     |           |-> batch to device
        |           |     |           |-> _unpack_qp(batch) -> query, passage
        |           |     |           |-> model(query=query, passage=passage)
        |           |     |           |-> loss.backward() [scaled by 1/num_batches]
        |           |     |-> scale_grads_and_clip_grad_norm()
        |           |     |-> optimizer.step() + zero_grad()
        |           |     |-> lr_scheduler.step(1)
        |           |     |-> dp_allreduce(loss) for reporting
        |           |     |-> return MetricsSample
        |           |
        |           |-> log_train_metrics()
        |           |-> if is_val_step: _run_validation_epoch() + log_val_metrics()
        |           |-> if is_ckpt_step: save_checkpoint()
        |
        |-> close loggers and checkpointer
```

### Validation flow (`_run_validation_epoch`, `train_biencoder.py:675`)

Switches model to eval mode. Iterates over `val_dataloader` with `torch.no_grad()`. Collects `outputs.scores` and `outputs.labels` across all batches. Computes:
- **val_loss**: average cross-entropy loss (allreduced across DP ranks)
- **val_acc1**: accuracy@1 (top prediction matches label)
- **val_mrr**: mean reciprocal rank

### Mining flow (`MineHardNegativesRecipe`)

```
main() [called from examples/biencoder/mine_hard_negatives.py]
  |-> parse_args_and_load_config()
  |-> MineHardNegativesRecipe(cfg)
  |-> recipe.setup()
  |     |-> build_distributed() -> self.dist_env
  |     |-> _extract_mining_params() [from cfg.mining with MINING_DEFAULTS fallback]
  |     |-> _validate_mining_params()
  |     |-> NeMoAutoModelBiencoder.from_pretrained() -> self.model [inference mode]
  |     |-> _configure_tokenizer() -> self.tokenizer [with optional BOS/EOS overrides]
  |     |-> _load_data() -> self.questions_dataset, self.documents_dataset
  |     |-> _build_document_mappings() -> self.doc_to_idx, self.idx_to_doc
  |     |-> _prepare_data() -> self.questions, self.question_ids, self.pos_doc_indices, etc.
  |
  |-> recipe.run()
        |-> _generate_embeddings()
        |     |-> [try cache first if load_embeddings_from_cache]
        |     |-> _encode_queries_sharded() or _encode_queries()
        |     |     |-> _encode_texts(encoder_type="query") per shard
        |     |-> _encode_all_documents()
        |     |     |-> for each chunk of corpus_chunk_size:
        |     |           |-> _encode_documents_chunk()
        |     |                 |-> _encode_chunk_distributed() or _encode_chunk_local()
        |     |                       |-> _encode_texts(encoder_type="passage")
        |     |-> _save_embeddings_to_cache() [rank 0]
        |
        |-> _unload_model() [free GPU memory]
        |-> _synchronize_ranks()
        |
        |-> [rank 0 only]:
              |-> _mine_hard_negatives()
              |     |-> for query batch:
              |           |-> scores = query_embs @ doc_embs.T
              |           |-> extract pos_scores, mask positives to -inf
              |           |-> apply margin filter (perc or abs threshold)
              |           |-> topk selection with TOPK_BUFFER_MULTIPLIER=2
              |           |-> post-filter remaining positives, limit to num_negs
              |
              |-> _write_output()
                    |-> load original JSON, add mining metadata
                    |-> replace neg_doc with mined negatives + scores
                    |-> add similarity scores to pos_doc entries
                    |-> write enriched JSON to train_file_output_path
```

### Checkpoint state management

`BaseRecipe.__setattr__()` (`base_recipe.py:132`) intercepts all attribute assignments. It checks each value against type predicates (`is_model`, `is_optimizer`, `is_dataloader`, `is_tokenizer`, `is_lr_scheduler`, `has_load_restore_state`, `isinstance(ConfigNode)`). Matching attributes are added to `__state_tracked` (a `set`), UNLESS the attribute name contains "val", "eval", "test", or "loss" (to exclude validation components from checkpointing).

During `save_checkpoint()` (`base_recipe.py:170`), tracked attributes are iterated in sorted order, dispatched by type:
- Models: `save_pretrained()` for HF compatibility (single model) or `checkpointer.save_model()` (multi-stage PP)
- Optimizers: `checkpointer.save_optimizer()`
- Dataloaders/RNG: `checkpointer.save_on_dp_ranks()`
- Config: `save_config(config.raw_config)`
- Others: `torch.save(obj.state_dict())`

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new loss function (e.g., triplet loss or knowledge distillation loss)

The loss is computed inside `BiencoderModel.forward()` (`llama_bidirectional_model.py:434`) as `self.cross_entropy(scores, labels)`. To add a new loss:

1. Add the loss module as an attribute in `BiencoderModel.__init__()` (line 382-405).
2. Modify `forward()` to compute and combine the new loss with the existing cross-entropy.
3. The `kd_labels` key is already extracted by `_unpack_qp()` (`train_biencoder.py:64-68`) and passed into the query dict -- this provides an existing hook for knowledge distillation signals.
4. No changes needed in the recipe itself since it calls `model(query=query, passage=passage)` and uses `outputs.loss` opaquely.

### Scenario 2: Adding a new validation metric (e.g., NDCG, Recall@K)

Modify `_run_validation_epoch()` (`train_biencoder.py:675`). The method already collects `all_scores` and `all_labels` tensors. To add Recall@K:

1. After line 719, compute: `topk_preds = torch.topk(scores, k=K, dim=1).indices` and `recall = (topk_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()`.
2. Add the new metric to the `metrics` dict at line 731.
3. Update `log_val_metrics()` (`train_biencoder.py:774`) format string to include the new metric.
4. If using WandB, no additional changes needed -- the dict is logged as-is at line 785.

### Scenario 3: Supporting a new encoder architecture (beyond Llama)

Currently `BiencoderModel.build()` (`llama_bidirectional_model.py:518`) only supports Llama-based models (line 569-575). To add support for another architecture (e.g., Mistral):

1. Implement a new bidirectional model class analogous to `LlamaBidirectionalModel` (line 1-374 of `llama_bidirectional_model.py`), ensuring it supports `output_hidden_states=True` and `return_dict=True`.
2. Add an `elif` branch in `BiencoderModel.build()` at line 569 to detect the new model type from `config.json` and select the new class.
3. Register the new class in `NeMoAutoModelBiencoder.from_pretrained()` (`biencoder_model.py:44`) if kernel patching differences apply.
4. No recipe changes needed -- the recipe only interacts with the model through `model(query=query, passage=passage)`.

### Scenario 4: Customizing the mining margin strategy

The margin filter in `_mine_hard_negatives()` (`mine_hard_negatives.py:1044-1057`) currently supports `perc` (percentage) and `abs` (absolute) margin types. To add a new strategy (e.g., adaptive per-query margin):

1. Add the new type string to `_validate_mining_params()` valid_types list (line 321).
2. Add an `elif` branch in the margin filtering block at line 1048-1053.
3. The new margin computation receives `min_pos_tensor` (shape `[batch_size]`) and must produce a `threshold` tensor of the same shape.
4. Add the new default to `MINING_DEFAULTS` at line 47 if a default value is appropriate.

### Scenario 5: Enabling pipeline parallelism for biencoder training

Currently PP is explicitly blocked with `NotImplementedError` (`train_biencoder.py:417-420`). The `setup()` method already has the PP infrastructure scaffolded (lines 343-390), including `AutoPipeline` configuration. To enable PP:

1. Remove the `NotImplementedError` raise at line 417.
2. Implement the PP-specific forward pass in `_forward_backward_step()` -- the current implementation at line 587 assumes `self.model_parts[0]` is the full model, but PP requires scheduling microbatches through pipeline stages.
3. Handle the biencoder-specific complication that query and passage encoding are independent forward passes, so the pipeline schedule may need to interleave them or run them sequentially per microbatch.
4. Update `self.checkpointer.config.model_state_dict_keys` (line 454) to handle multi-stage model parts.

### Scenario 6: Changing the data format for mining input/output

The mining recipe expects JSON files loaded by `load_datasets()` (`nemo_automodel/components/datasets/llm/retrieval_dataset.py`). The output format is defined in `_write_output()` (`mine_hard_negatives.py:1159-1214`). Key conventions:
- Each row has `question`, `question_id`, `corpus_id`, `pos_doc` (list of `{"id": ...}`), `neg_doc` (list of `{"id": ..., "score": ...}`).
- Mining adds a top-level `"mining"` key with `"args"` metadata for reproducibility.
- Document text extraction uses `_get_document_text()` (line 456) which concatenates `title` and `text` fields.
- To support a different schema, modify `_prepare_data()` (line 406), `_get_document_text()` (line 456), and `_write_output()` (line 1159).
