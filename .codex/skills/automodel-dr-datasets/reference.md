# Datasets Module Reference

Complete file-by-file reference for `nemo_automodel/components/datasets/`.

## Directory Structure

```
nemo_automodel/components/datasets/
    __init__.py                         # Package init (license only)
    utils.py                            # Shared collaters, mask creation, SFTSingleTurnPreprocessor
    reservoir_sampler.py                # ReservoirSampler for bounded-memory streaming shuffle
    llm/
        __init__.py                     # Exports: ChatDataset, ColumnMapped*, NanogptDataset, make_*_dataset, etc.
        formatting_utils.py             # Core tokenization: format_prompt_completion, format_chat_template, _package_tokenized_example
        chat_dataset.py                 # ChatDataset: OpenAI-format chat with tool-calling
        column_mapped_text_instruction_dataset.py         # ColumnMappedTextInstructionDataset: map-style SFT
        column_mapped_text_instruction_iterable_dataset.py # ColumnMappedTextInstructionIterableDataset: streaming SFT
        delta_lake_dataset.py           # DeltaLakeDataset, DeltaLakeIterator, is_delta_lake_path
        packed_sequence.py              # pack_dataset, create_block_causal_mask, _pad_pack
        squad.py                        # make_squad_dataset
        xlam.py                         # make_xlam_dataset (tool-calling SFT)
        hellaswag.py                    # HellaSwag benchmark dataset
        seq_cls.py                      # GLUE_MRPC sequence classification
        nanogpt_dataset.py              # NanogptDataset: binary shard IterableDataset
        mock.py                         # build_unpacked_dataset (test utility)
        mock_packed.py                  # build_packed_dataset (test utility)
        mock_iterable_dataset.py        # MockIterableDataset (benchmarking)
        retrieval_dataset.py            # make_retrieval_dataset (corpus-id based)
        retrieval_dataset_inline.py     # make_retrieval_dataset (inline text)
        retrieval_collator.py           # RetrievalBiencoderCollator
        megatron_dataset.py             # MegatronPretraining orchestrator
        megatron/
            __init__.py
            builder.py                  # BlendedDataset, BlendedMegatronDatasetBuilder
            gpt_dataset.py              # GPTDataset, GPTDatasetConfig, BlendedMegatronDatasetConfig, Split
            indexed_dataset.py          # IndexedDataset, IndexedDatasetBuilder, _IndexReader, _BinReader
            sampler.py                  # BaseMegatronSampler, MegatronPretrainingSampler, MegatronPretrainingRandomSampler, create_megatron_sampler
            helpers.py                  # build_sample_idx (Python wrapper for C++ helpers)
            megatron_utils.py           # get_blend_from_list, compile_helper
            helpers.cpp                 # C++ implementation of index building
            Makefile                    # Builds helpers_cpp shared object
    vlm/
        __init__.py                     # Exports: make_rdr_dataset, make_cord_v2_dataset, make_unimm_chat_dataset
        datasets.py                     # VLM dataset factories: make_rdr/cord_v2/medpix/cv17/unimm_chat_dataset
        collate_fns.py                  # All VLM collate functions + build_labels + COLLATE_FNS registry
        utils.py                        # default_stop_tokens, json2token, process_text_batch
```

## File Details

### `utils.py`

**Functions:**
- `batchify(tensor, default_tensor_cls)` (line 22): Ensures tensor has batch dim. Adds unsqueeze(0) if 1D.
- `extract_key_from_dicts(batch, key)` (line 40): Extracts values for a key across a list of dicts.
- `pad_within_micro(batch, pad_token_id, pad_seq_len_divisible)` (line 55): Pads list-of-lists to max length.
- `find_last_non_pad_token(lst, value)` (line 79): Finds index of last token that is not `value`.
- `get_pad_token_from_key(val, pad_token_ids)` (line 96): Returns pad token id for a given key name. Defaults: labels=-100, attention_mask=0, loss_mask=0, input_ids=0.
- `make_attention_mask_from_labels(ids, ignore_token)` (line 109): Creates binary attention mask from labels.
- `create_causal_mask_mapping(model_config, batch_size, seq_len, ...)` (line 127): Creates 4D causal masks using HF `create_causal_mask` and optionally `create_sliding_window_causal_mask`.
- `add_causal_masks_to_batch(batch_dict, model_config)` (line 183): Wraps `create_causal_mask_mapping` for batched dicts.
- `default_collater(batch, pad_seq_len_divisible)` (line 221): Standard collator. Pops `___PAD_TOKEN_IDS___`, pads per key, converts to LongTensor.
- `packed_sequence_thd_collater(batch)` (line 249): THD collator. Stacks input_ids/labels/position_ids, pads seq_lens/seq_lens_padded with -1000, sets `qkv_format="thd"`.

**Classes:**
- `SFTSingleTurnPreprocessor` (line 337): Tokenizes context+target, builds labels (-100 for context), optionally pads to max length. Used by `HellaSwag`.

### `reservoir_sampler.py`

**Class: `ReservoirSampler`** (line 21)
- `__init__(iterator, buffer_size, seed)`: Validates inputs. Buffer size must be > 0.
- `__iter__()`: Fills buffer, shuffles, then reservoir-samples: for each new item, randomly evict one from buffer. Yields evicted items. On exhaustion, yields remaining buffer (filtered for None).
- `__len__()`, `__getitem__()`: Raise RuntimeError (not supported).

### `llm/formatting_utils.py`

**Constants:**
- `GENERATION_REGEX` (line 24): `re.compile(r"\{%-?\s+generation\s+-?%\}")` - detects `{% generation %}` in chat templates.

**Functions:**
- `_pad_to_seq_length(sample, pad_token_id, seq_length)` (line 27): Simple list padding.
- `_add_pad_token(tokenizer)` (line 35): Sets pad_token_id to eos_token_id if missing. Returns pad_token_id.
- `_has_chat_template(tokenizer)` (line 47): Returns True if tokenizer has `chat_template` attr and callable `apply_chat_template`.
- `_package_tokenized_example(tokenizer, input_ids, assistant_masks, eos_token_id, pad_token_id, seq_length, truncation, padding)` (line 62): Core packaging function.
  - Shifts: `input_ids = tokens[:-1]`, `labels = tokens[1:]` (labels shifted by removing BOS).
  - Masks prompt tokens in labels with -100 using assistant_masks.
  - Returns dict with `input_ids`, `labels`, `attention_mask`, `___PAD_TOKEN_IDS___`.
- `format_prompt_completion(tokenizer, prompt, answer, eos_token_id, pad_token_id, ...)` (line 118): Tokenizes prompt+answer, builds assistant_masks from prompt length, calls `_package_tokenized_example`.
- `format_chat_template(tokenizer, formatted_text, eos_token_id, pad_token_id, tools, answer_only_loss_mask, ...)` (line 177): Applies chat template. Three masking strategies:
  1. Template has `{% generation %}`: uses `return_assistant_tokens_mask=True`.
  2. No generation keyword + answer_only_loss_mask: manually splits by tokenizing prompt-only and computing length difference.
  3. Neither: all tokens are answer (mask = all 1s).

### `llm/chat_dataset.py`

**Class: `ChatDataset(Dataset)`** (line 119)
- Constructor: Loads OpenAI-format messages from HF or local JSON/JSONL via `_load_openai_messages()`. Validates tokenizer has chat template.
- `__getitem__(idx)`: Normalizes messages, calls `format_chat_template()` with optional tools list.
- Helper `_normalize_messages(messages)` (line 98): Validates roles, converts list content to text string.

### `llm/column_mapped_text_instruction_dataset.py`

**Class: `ColumnMappedTextInstructionDataset(Dataset)`** (line 148)
- Constructor: Validates column_mapping has answer + (context and/or question). Loads dataset via `_load_dataset()`.
- `__getitem__(idx)` (line 251): Maps columns, calls `_apply_tokenizer()`. Skips samples with no valid labels (all -100) by advancing index.
- `_apply_tokenizer(sample)` (line 269): Dispatches to `format_chat_template()` or `format_prompt_completion()` based on `use_hf_chat_template` flag and tokenizer capability.

**Enum: `ColumnTypes`** (line 42): `Context="context"`, `Question="question"`, `Answer="answer"`.

### `llm/column_mapped_text_instruction_iterable_dataset.py`

**Class: `ColumnMappedTextInstructionIterableDataset(IterableDataset, ColumnMappedTextInstructionDataset)`** (line 108)
- Inherits `_apply_tokenizer()` from parent.
- Constructor: Always loads in streaming mode via `_load_streaming_dataset()`. Supports Delta Lake detection.
- `__iter__()` (line 201): Infinite loop. For each row: map columns, tokenize, skip invalid, yield. On exhaustion: increment epoch, call `set_epoch()`, repeat.
- `shard(num_shards, index)` (line 235): Delegates to underlying HF dataset.
- `shuffle(buffer_size, seed)` (line 240): Delegates to underlying HF dataset.

### `llm/delta_lake_dataset.py`

**Function: `is_delta_lake_path(path)`** (line 193): Checks for `delta://`, `dbfs:/`, `abfss://`, `s3://`, `s3a://`, `gs://` prefixes or local `_delta_log` directory.

**Class: `DeltaLakeIterator`** (line 342)
- Three iteration backends: `_iter_with_deltalake()`, `_iter_with_spark()`, `_iter_with_databricks_sql()`.
- Auto-detects Unity Catalog paths (catalog.schema.table format).
- Handles deletion vectors by falling back from deltalake to Spark.
- Supports sharding via `_shard_info` tuple.

**Class: `DeltaLakeDataset`** (line 696)
- HF-compatible wrapper. `shard()` delegates to iterator. `shuffle()` wraps iterator with `ReservoirSampler`. `take()` creates `_LimitedDeltaLakeDataset` wrapper.

### `llm/packed_sequence.py`

**Constants:**
- `CROSS_ENTROPY_IGNORE_IDX = -100` (line 23)
- `PACK_TYPE = dict[str, torch.Tensor | list[int]]` (line 24)

**Functions:**
- `pack_dataset(dataset, split, packed_sequence_size, max_packs, padding_idx, drop_long_samples, cp_size)` (line 202): Main packing function. Greedy first-fit. Returns HF Dataset.
- `_pad_pack(pack, padding_idx, packed_sequence_size, cross_entropy_ignore_idx, cp_size)` (line 37): Pads a single pack. `seq_lens` stores originals, `seq_lens_padded` has CP padding + pack padding on last sequence.
- `_split_and_add_pack(current_pack, packs, previous_sample_boundary, ...)` (line 156): Splits at boundary, processes completed pack, returns overflow.
- `create_block_causal_mask(seq_lens)` (line 321): Creates batch of block-diagonal causal masks from seq_lens. Returns `[batch_size, 1, packed_size, packed_size]` tensor.
- `packed_block_causal_mask(seq_lens)` (line 365): Alias for `create_block_causal_mask`.

### `llm/nanogpt_dataset.py`

**Constants:**
- `MAGIC = 278895051`, `LEGACY_MAGIC = 20240520` (lines 71-72)
- `HEADER_BYTES = 256 * 4` (line 74)

**Functions:**
- `load_bin_shard(path)` (line 152): Memory-maps a .bin shard. Supports both legacy (uint16) and new (uint16/uint32) formats.
- `_load_bos_index(path)` (line 91): Loads .bos.idx companion file for efficient BOS alignment.

**Class: `NanogptDataset(IterableDataset)`** (line 261)
- `__init__(file_pattern, seq_len, bos_token, shuffle_files, align_to_bos)`: Globs files.
- `__iter__()`: Sets up worker context (DDP rank + DL worker), yields from all files in infinite loop.
- `_process_file_tokens()` (line 354): Slides a window of `seq_len+1` over mmap'd tokens, yielding `{input_ids, labels}`.

### `llm/squad.py`

**Function: `make_squad_dataset(tokenizer, seq_length, limit_dataset_samples, ...)` (line 67)**: Loads SQuAD, applies prompt-completion or chat-template formatting, returns mapped HF Dataset.

### `llm/xlam.py`

**Function: `make_xlam_dataset(tokenizer, ...)` (line 154)**: Loads xLAM function-calling dataset. Converts tool definitions to OpenAI schema via `_convert_tools()`. Formats as chat with `tool_calls` in assistant messages.

### `llm/hellaswag.py`

**Class: `HellaSwag`** (line 20): Wraps HellaSwag with `SFTSingleTurnPreprocessor`. `get_context()` returns `examples["ctx"]`, `get_target()` returns correct ending by label index.

### `llm/seq_cls.py`

**Class: `GLUE_MRPC`** (line 22): Sentence pair classification. Tokenizes sentence1+sentence2, labels are classification ints wrapped in list.

### `llm/retrieval_dataset.py`

**Classes:**
- `AbstractDataset` (line 28): ABC with `get_document_by_id`, `get_all_ids`.
- `TextQADataset(AbstractDataset)` (line 38): Loads text-only corpus from HF.
- `CorpusInfo` (line 61): Dataclass linking corpus metadata to dataset object.

**Functions:**
- `load_datasets(data_dir_list, concatenate)` (line 155): Loads JSON files with `corpus` and `data` sections. Normalizes to `question_id`, `question`, `corpus_id`, `pos_doc`, `neg_doc`.
- `make_retrieval_dataset(data_dir_list, data_type, train_n_passages, ...)` (line 329): Uses `set_transform()` for lazy doc resolution.

### `llm/retrieval_dataset_inline.py`

Inline variant that expects pos_doc/neg_doc texts directly in records (no corpus files). `_resolve_doc_to_example(doc)` normalizes inline docs.

### `llm/retrieval_collator.py`

**Class: `RetrievalBiencoderCollator`** (line 40): Tokenizes queries and documents separately at batch time. Produces `q_input_ids`, `q_attention_mask`, `d_input_ids`, `d_attention_mask`, `labels` (dummy zeros).

### `llm/megatron_dataset.py`

**Class: `MegatronPretraining`** (line 33)
- `__init__()`: Compiles C++ helpers, validates paths, parses blend config.
- `build()`: Calculates sample counts from trainer params, invokes `BlendedMegatronDatasetBuilder`.
- `get_dataset(split)`: Returns built dataset for "train"/"validation"/"test".
- `gpt_dataset_config` property: Creates `GPTDatasetConfig` with all parameters.

**Function: `try_load_blend_from_json(path)` (line 359)**: Loads JSON blend config, normalizes split aliases (valid/val/dev -> validation).

### `llm/megatron/gpt_dataset.py`

**Dataclasses:**
- `BlendedMegatronDatasetConfig` (line 46): `random_seed`, `sequence_length`, `blend`, `blend_per_split`, `split`, `path_to_cache`, `mmap_bin_files`, `tokenizer`, `mid_level_dataset_surplus`.
- `GPTDatasetConfig(BlendedMegatronDatasetConfig)` (line 136): Adds `reset_position_ids`, `reset_attention_mask`, `eod_mask_loss`, `create_attention_mask`, `drop_last_partial_validation_sequence`, `add_extra_token_to_sequence`.

**Class: `GPTDataset(torch.utils.data.Dataset)`** (line 238)
- `__init__()`: Builds document_index, sample_index, shuffle_index (cached as .npy files).
- `__getitem__(idx)`: Queries indices, extracts tokens from IndexedDataset, builds masks and position_ids.
- `_build_document_sample_shuffle_indices()` (line 490): Core index building with rank-0 priority and caching.

### `llm/megatron/builder.py`

**Class: `BlendedDataset(Dataset)`** (line 35): Conjugates multiple GPTDatasets with weighted sampling. Uses C++ `build_blending_indices` or `build_exhaustive_blending_indices`.

**Class: `BlendedMegatronDatasetBuilder`** (line 207)
- `build()` (line 243): Orchestrates building blended or per-split datasets.
- `_build_megatron_datasets_parallel()` (line 530): Parallel building with ThreadPoolExecutor. Rank-0 builds first with scaled thread count, barrier, then other ranks (guaranteed cache hit).

### `llm/megatron/indexed_dataset.py`

**Class: `IndexedDataset(Dataset)`** (line 425): Memory-mapped .bin/.idx reader.
- `get(idx, offset, length)`: Read a subsequence of a document.
- `sequence_lengths` property: numpy array of document lengths.

**Class: `IndexedDatasetBuilder`** (line 509): Writer for .bin/.idx pairs. `add_item()`, `add_document()`, `end_document()`, `finalize()`.

### `llm/megatron/sampler.py`

**Class: `MegatronPretrainingSampler(BaseMegatronSampler)`** (line 114): Sequential deterministic sampler. Slices global batch by data_parallel_rank. Optional padding for last batch.

**Class: `MegatronPretrainingRandomSampler(BaseMegatronSampler)`** (line 189): Per-epoch randomized sampler. Seeds with `seed + epoch`. Shuffles within per-rank buckets.

**Function: `create_megatron_sampler(dataset_len, micro_batch_size, global_batch_size, dataloader_type, ...)` (line 292)**: Factory returning "single" or "cyclic" sampler.

### `vlm/collate_fns.py`

**COLLATE_FNS dict** (line 649): Maps processor class names to collate functions:
- `"Qwen2_5_VLProcessor"` -> `qwen2_5_collate_fn`
- `"Qwen3OmniMoeProcessor"` -> `qwen3_omni_collate_fn`
- `"KimiVLProcessor"` -> `kimi_vl_collate_fn`
- `"KimiK25Processor"` -> `kimi_k25_vl_collate_fn`
- `"NemotronParseProcessor"` -> `nemotron_parse_collate_fn`
- `"default"` -> `default_collate_fn`

**Function: `build_labels(input_ids_batch, conversations, processor)`** (line 84): Creates label tensors. For each assistant turn: tokenizes text, pattern-matches via `_find_pattern_indices()`, extends span by 1 if next token is a stop token.

**Function: `_expand_image_tokens(input_ids, attention_mask, grid_thws, media_token_id, merge_kernel_size)`** (line 334): Pre-expands single image placeholder to N tokens based on grid dimensions. Used by `kimi_k25_vl_collate_fn` for PP compatibility.

### `vlm/datasets.py`

All factory functions return `list[dict]` where each dict has `conversation` key containing OpenAI-style message list with multimodal content.

- `make_rdr_dataset()` (line 24): Image + "Describe this image."
- `make_cord_v2_dataset()` (line 58): Document image + parsed JSON as text via `json2token()`.
- `make_medpix_dataset()` (line 99): Medical image + question/answer.
- `make_cv17_dataset()` (line 120): Audio transcription. Returns `(array, sampling_rate)` tuple in `audio` key.
- `make_unimm_chat_dataset()` (line 140): Multi-turn image chat. Handles `<image>` placeholders in human messages.

### `vlm/utils.py`

- `default_stop_tokens(processor)` (line 20): Returns tuple of common stop tokens: `"<end_of_turn>"`, `"<|im_end|>"`, `"<|eot_id|>"`, plus tokenizer's eos_token.
- `json2token(obj, sort_json_key)` (line 33): Converts JSON dict to token sequence using `<s_key>...</s_key>` markup. Used by CORD-V2 dataset.
- `process_text_batch(processor, texts, images)` (line 52): Batch-processes texts+images through a VLM processor.
