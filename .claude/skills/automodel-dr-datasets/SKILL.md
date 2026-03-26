---
name: automodel-dr-datasets
description: Use when working with the datasets module of automodel — dataset loading, sequence packing, tokenization, chat templates, LLM and VLM data pipelines
---

# NeMo AutoModel Datasets Module

## 1. Module Purpose & Capabilities

The datasets module (`nemo_automodel/components/datasets/`) provides the complete data pipeline for LLM and VLM training in NeMo AutoModel. It is a self-contained component (no cross-component imports, enforced by import-linter) that handles:

- **Dataset loading** from HuggingFace Hub, local JSON/JSONL, NanoGPT binary shards, Megatron indexed datasets, and Delta Lake tables (including Databricks Unity Catalog).
- **Tokenization and formatting** via HuggingFace tokenizers, with both prompt-completion and chat-template modes (including tool-calling support).
- **Sequence packing** for throughput optimization: greedy bin-packing of variable-length samples into fixed-size sequences with correct position IDs and block-causal masks.
- **Collation** for both standard padded batching (BSHD) and THD packed-sequence format with seq_lens metadata.
- **VLM data processing** for vision-language models (Qwen2.5-VL, Qwen3 Omni, KimiVL, Phi-4 MM, Nemotron-Parse) with image/audio/video multimodal inputs.
- **Streaming datasets** via `IterableDataset` with reservoir-sampled shuffling for bounded-memory operation.
- **Megatron-style pretraining datasets** with indexed binary files, document/sample/shuffle indices, blended multi-dataset support, and distributed-aware building.
- **Retrieval/biencoder datasets** for dense retrieval training with query-document pairs and configurable negative mining.

The module is organized into three subdirectories:
- `datasets/` (top-level): Shared utilities, collaters, `ReservoirSampler`
- `datasets/llm/`: All LLM-specific dataset classes
- `datasets/vlm/`: VLM dataset factories and collate functions

## 2. Core Design Logic

### Why this architecture

The datasets module follows NeMo AutoModel's component independence principle: it must not import from other components (e.g., `distributed`, `models`, `optim`). Recipes compose datasets with other components. This means dataset classes produce plain dicts of `input_ids`, `labels`, `attention_mask` (and optionally `position_ids`, `seq_lens`, `loss_mask`) -- a universal interface consumed by any model.

### Two formatting strategies

The module supports two tokenization/formatting strategies, chosen at dataset construction time:

1. **Prompt-completion format** (`format_prompt_completion` in `formatting_utils.py`): Concatenates a prompt string and answer string, tokenizes them, and constructs labels where prompt tokens are masked with -100 (via `assistant_masks`). Used when no chat template is available.

2. **Chat-template format** (`format_chat_template` in `formatting_utils.py`): Uses `tokenizer.apply_chat_template()` to format multi-turn conversations. If the template contains a `{% generation %}` block, it leverages `return_assistant_tokens_mask=True` to automatically identify answer tokens. Otherwise, it manually splits at the last assistant turn. Supports tool-calling via the `tools` parameter.

Both strategies share `_package_tokenized_example()` which shifts input_ids vs labels by one position (input_ids = tokens[:-1], labels = tokens[1:]), applies assistant masking, and handles padding.

### Map vs. Streaming datasets

- **Map-style** (`torch.utils.data.Dataset`): `ColumnMappedTextInstructionDataset`, `ChatDataset`, `HellaSwag`, `GLUE_MRPC`, `GPTDataset`. Support `__getitem__`/`__len__`, suitable for fixed-size datasets loaded into memory.
- **Iterable/streaming** (`torch.utils.data.IterableDataset`): `ColumnMappedTextInstructionIterableDataset`, `NanogptDataset`, `MockIterableDataset`, `DeltaLakeDataset`. Support `__iter__` only, suitable for large-scale or remote data. The iterable variant inherits tokenization logic from the map-style parent via `ColumnMappedTextInstructionDataset._apply_tokenizer()`.

### Packing strategy

Sequence packing (`packed_sequence.py`) uses a greedy first-fit approach:
1. Iterate through tokenized samples.
2. Accumulate into a buffer (`current_pack`) until the buffer exceeds `packed_sequence_size`.
3. Split at the previous sample boundary -- the completed pack goes to `packs`, the overflow starts the next pack.
4. Each pack gets position_ids that reset per-sequence, `seq_lens` recording original lengths, and `seq_lens_padded` including CP padding.
5. Final packs are padded to `packed_sequence_size` with padding tokens and -100 labels.

Context parallel (CP) support: when `cp_size > 1`, each individual sequence within a pack is padded to be divisible by `2 * cp_size` before being placed into the pack.

### Collation strategy

Two collation paths exist:
- `default_collater()` (in `utils.py`): Standard padding-based collation. Pads all sequences in a batch to the max length, using per-key pad tokens from `___PAD_TOKEN_IDS___` metadata embedded in each sample.
- `packed_sequence_thd_collater()` (in `utils.py`): THD-format collation for packed sequences. Stacks fixed-length token tensors and pads the variable-length `seq_lens`/`seq_lens_padded` arrays with sentinel value -1000. Sets `qkv_format: "thd"` in output.

### VLM collation

VLM collate functions (`vlm/collate_fns.py`) are processor-specific because each VLM processor has unique image token expansion, audio handling, and chat template application. They share `build_labels()` which finds assistant response tokens via pattern matching in the tokenized sequence and masks everything else with -100.

### Delta Lake / Databricks integration

`delta_lake_dataset.py` provides a three-backend strategy: (1) `deltalake` Python library for file-based tables, (2) PySpark for Databricks runtime tables with deletion vectors, (3) `databricks-sql-connector` for Unity Catalog access. It auto-detects the best backend and falls back gracefully. Streaming is enforced to prevent accidental materialization of large tables.

### Megatron pretraining data pipeline

The `megatron/` subdirectory provides a full port of Megatron-LM's indexed dataset system:
- `IndexedDataset` (`indexed_dataset.py`): Memory-mapped .bin/.idx file pairs for fast on-disk access.
- `GPTDataset` (`gpt_dataset.py`): Builds document/sample/shuffle indices with caching, supporting multi-epoch training with proper shuffling. Indices are built on rank 0 first, then loaded from cache on other ranks.
- `BlendedDataset` / `BlendedMegatronDatasetBuilder` (`builder.py`): Multi-dataset blending with configurable weights, parallel building with ThreadPoolExecutor, and support for per-split blend configurations.
- `MegatronPretrainingSampler` / `MegatronPretrainingRandomSampler` (`sampler.py`): Data-parallel-aware batch samplers that slice global batches across ranks.

## 3. Core Data Structures

### LLM Dataset Classes

| Class | File | Type | Purpose |
|-------|------|------|---------|
| `ColumnMappedTextInstructionDataset` | `llm/column_mapped_text_instruction_dataset.py` | `Dataset` | Generic instruction-tuning dataset with configurable column mapping (context/question/answer). Supports both prompt-completion and chat-template formatting. |
| `ColumnMappedTextInstructionIterableDataset` | `llm/column_mapped_text_instruction_iterable_dataset.py` | `IterableDataset` | Streaming variant of the above. Inherits tokenization from the map-style parent. Supports Delta Lake, HF streaming, infinite repetition with epoch-based reshuffle. |
| `ChatDataset` | `llm/chat_dataset.py` | `Dataset` | OpenAI-format chat dataset with tool-calling support. Requires tokenizer with chat template. Loads from HF or local JSON/JSONL. |
| `NanogptDataset` | `llm/nanogpt_dataset.py` | `IterableDataset` | Streams from NanoGPT binary shards (.bin files). Supports BOS-aligned slicing, multi-file sharding across DataLoader workers and DDP ranks. Infinite iteration. |
| `HellaSwag` | `llm/hellaswag.py` | wrapper | HellaSwag benchmark dataset. Uses `SFTSingleTurnPreprocessor` for tokenization+padding. |
| `GLUE_MRPC` | `llm/seq_cls.py` | wrapper | GLUE MRPC sentence pair classification dataset. |
| `DeltaLakeDataset` | `llm/delta_lake_dataset.py` | streaming wrapper | HF-compatible wrapper around `DeltaLakeIterator`. Supports shard/shuffle/take/set_epoch. |
| `MockIterableDataset` | `llm/mock_iterable_dataset.py` | `IterableDataset` | Synthetic random-token dataset for benchmarking. Yields pre-batched data. |
| `GPTDataset` | `llm/megatron/gpt_dataset.py` | `Dataset` | Megatron-style pretraining dataset backed by `IndexedDataset`. Builds document/sample/shuffle indices. Returns `input_ids`, `labels`, `attention_mask`, `loss_mask`. |
| `BlendedDataset` | `llm/megatron/builder.py` | `Dataset` | Blends multiple `GPTDataset` instances with weighted sampling. Uses C++ helpers for index building. |
| `MegatronPretraining` | `llm/megatron_dataset.py` | orchestrator | High-level class that configures and builds Megatron pretraining datasets (train/val/test splits). Handles JSON blend config loading, path validation, C++ helper compilation. |

### Retrieval Dataset Classes

| Class/Function | File | Purpose |
|----------------|------|---------|
| `make_retrieval_dataset()` | `llm/retrieval_dataset.py` | Loads corpus-id based retrieval data with separate corpus files. Uses `set_transform()` for lazy doc resolution. |
| `make_retrieval_dataset()` | `llm/retrieval_dataset_inline.py` | Loads inline retrieval data (pos_doc/neg_doc text embedded in records). No external corpus dependency. |
| `RetrievalBiencoderCollator` | `llm/retrieval_collator.py` | Batch-time tokenization collator for biencoder training. Separately tokenizes/pads queries and documents, producing `q_input_ids`, `d_input_ids`, etc. |

### VLM Dataset Functions and Collators

| Function | File | Purpose |
|----------|------|---------|
| `make_rdr_dataset()` | `vlm/datasets.py` | RDR image-to-text dataset in conversation format. |
| `make_cord_v2_dataset()` | `vlm/datasets.py` | CORD-V2 document understanding dataset. Uses `json2token()` for structured output. |
| `make_unimm_chat_dataset()` | `vlm/datasets.py` | UniMM-Chat multi-turn image conversation dataset. |
| `make_medpix_dataset()` | `vlm/datasets.py` | MedPix medical image QA dataset. |
| `make_cv17_dataset()` | `vlm/datasets.py` | CommonVoice 17 audio transcription dataset. |
| `qwen2_5_collate_fn()` | `vlm/collate_fns.py` | Collator for Qwen2.5-VL. Uses `qwen_vl_utils.process_vision_info`. |
| `qwen3_omni_collate_fn()` | `vlm/collate_fns.py` | Collator for Qwen3 Omni (audio+image+video). Uses `qwen_omni_utils.process_mm_info`. |
| `kimi_vl_collate_fn()` | `vlm/collate_fns.py` | Collator for KimiVL. |
| `kimi_k25_vl_collate_fn()` | `vlm/collate_fns.py` | Collator for Kimi K2.5 VL with pre-expanded image tokens for pipeline parallelism. |
| `nemotron_parse_collate_fn()` | `vlm/collate_fns.py` | Collator for Nemotron-Parse document parsing models. |
| `default_collate_fn()` | `vlm/collate_fns.py` | Default VLM collator using `processor.apply_chat_template`. |
| `build_labels()` | `vlm/collate_fns.py` | Shared label builder for VLM collators. Pattern-matches assistant tokens in the encoded sequence. |

### Shared Utilities

| Item | File | Purpose |
|------|------|---------|
| `SFTSingleTurnPreprocessor` | `utils.py` | Generic tokenize-then-pad preprocessor for single-turn SFT. Labels mask prompt with -100. |
| `default_collater()` | `utils.py` | Standard batch collator with per-key padding. |
| `packed_sequence_thd_collater()` | `utils.py` | THD-format collator for packed sequences. Pads `seq_lens` with -1000 sentinel. |
| `create_causal_mask_mapping()` | `utils.py` | Creates 4D causal masks (full + sliding window) for pipeline parallelism. |
| `add_causal_masks_to_batch()` | `utils.py` | Adds precomputed causal masks to an already-batched dict. |
| `ReservoirSampler` | `reservoir_sampler.py` | Bounded-memory streaming shuffle. Fills a buffer, then randomly evicts/replaces. Used by `DeltaLakeDataset.shuffle()`. |
| `pack_dataset()` | `llm/packed_sequence.py` | Greedy bin-packing function. Returns an HF `Dataset` of packed sequences with `input_ids`, `labels`, `position_ids`, `seq_lens`, `seq_lens_padded`. |
| `create_block_causal_mask()` | `llm/packed_sequence.py` | Builds 4D block-diagonal causal masks from `seq_lens` for packed sequences. |
| `IndexedDataset` | `llm/megatron/indexed_dataset.py` | Memory-mapped .bin/.idx reader for Megatron-format pretraining data. |
| `BaseMegatronSampler` | `llm/megatron/sampler.py` | Abstract base for Megatron data-parallel-aware batch samplers. |
| `create_megatron_sampler()` | `llm/megatron/sampler.py` | Factory function for Megatron samplers ("single" or "cyclic"). |

### Key Enums and Config Dataclasses

| Item | File | Purpose |
|------|------|---------|
| `ColumnTypes` | `llm/column_mapped_text_instruction_dataset.py` | Enum: `Context`, `Question`, `Answer` for column mapping. |
| `Split` | `llm/megatron/gpt_dataset.py` | Enum: `train=0`, `valid=1`, `test=2` for dataset splits. |
| `GPTDatasetConfig` | `llm/megatron/gpt_dataset.py` | Dataclass configuring Megatron GPT datasets: `random_seed`, `sequence_length`, `tokenizer`, `blend`/`blend_per_split`, `split`, `path_to_cache`, etc. |
| `BlendedMegatronDatasetConfig` | `llm/megatron/gpt_dataset.py` | Parent dataclass for blended dataset configuration. Auto-computes `split_matrix` from `split` string. |

## 4. State Flow

### LLM SFT Data Pipeline (ColumnMappedTextInstructionDataset)

```
Raw Data (HF/JSON/JSONL)
    |
    v
_load_dataset() -> datasets.Dataset
    |
    v
ColumnMappedTextInstructionDataset.__getitem__(idx)
    |
    v
Column mapping: {dest: row[src]} -- maps raw columns to context/question/answer
    |
    v
_apply_tokenizer(mapped_sample)
    |
    +--[has chat template]--> format_chat_template()
    |                            |
    |                            v
    |                         tokenizer.apply_chat_template(messages, tools=...)
    |                            |
    |                            v
    |                         Build assistant_masks (0 for prompt, 1 for answer)
    |                            |
    |                            v
    |                         _package_tokenized_example()
    |
    +--[no chat template]---> format_prompt_completion()
                                |
                                v
                             tokenizer(prompt + answer)
                                |
                                v
                             Build assistant_masks from prompt length
                                |
                                v
                             _package_tokenized_example()
                                    |
                                    v
                                 Shift: input_ids = tokens[:-1], labels = tokens[1:]
                                 Mask: labels[~assistant] = -100
                                 Optional padding to seq_length
                                    |
                                    v
                                 {input_ids, labels, attention_mask, ___PAD_TOKEN_IDS___}
```

### Sequence Packing Pipeline

```
Tokenized Dataset (input_ids, labels per sample)
    |
    v
pack_dataset(dataset, split, packed_sequence_size, cp_size)
    |
    v
For each sample:
    - If cp_size > 1: pad seq to divisible by 2*cp_size
    - Append to current_pack buffer
    - Track position_ids (reset per sequence), seq_lens
    |
    v
When len(current_pack) > packed_sequence_size:
    - _split_and_add_pack(): split at previous boundary
    - _tensorize_and_pad_pack(): convert to tensors, pad to packed_sequence_size
    - _pad_pack(): creates seq_lens_padded (with CP padding applied)
    |
    v
Dataset.from_dict({input_ids, labels, position_ids, seq_lens, seq_lens_padded})
    |
    v
DataLoader with packed_sequence_thd_collater()
    - Stacks fixed-size input_ids/labels/position_ids
    - Pads variable-length seq_lens/seq_lens_padded with -1000
    - Sets qkv_format="thd"
```

### VLM Data Pipeline

```
make_*_dataset() returns list of conversation dicts
    |
    v
Each example = {conversation: [{role, content: [{type, text/image}]}]}
    |
    v
DataLoader calls collate_fn(batch, processor)
    |
    v
Processor-specific collate_fn (e.g., qwen2_5_collate_fn):
    1. Extract conversations from batch
    2. processor.apply_chat_template() -> text prompts
    3. process_vision_info() -> image inputs (processor-specific)
    4. processor(text=texts, images=images, padding=True) -> batch dict
    5. build_labels(input_ids, conversations, processor):
       - For each assistant turn: tokenize assistant text, pattern-match in encoded sequence
       - Labels = -100 everywhere except matched assistant spans
    6. Shift: labels = labels[:, 1:], input_ids = input_ids[:, :-1]
    |
    v
{input_ids, attention_mask, labels, pixel_values, image_grid_thw, ...}
```

### Megatron Pretraining Data Pipeline

```
Binary shard files (.bin + .idx)
    |
    v
MegatronPretraining.__init__(paths, seq_length, tokenizer, ...)
    - Validate paths, compile C++ helpers if needed
    - Parse blend config (weights + prefixes or per-split blends)
    |
    v
MegatronPretraining.build()
    - Calculate num_train/val/test_samples from max_steps * global_batch_size
    |
    v
BlendedMegatronDatasetBuilder.build()
    |
    v
For each prefix:
    GPTDataset.build_low_level_dataset() -> IndexedDataset (mmap .bin/.idx)
    GPTDataset.__init__():
        - _build_document_sample_shuffle_indices():
          1. document_index: shuffled array of doc indices (num_epochs * num_docs)
          2. sample_index: 2D [doc_idx, offset] built by C++ helpers.build_sample_idx()
          3. shuffle_index: random permutation
        - Cache all indices as .npy files
    |
    v
BlendedDataset (if multiple prefixes):
    - C++ helpers build blending indices from weights
    - Maps (dataset_id, sample_id) pairs
    |
    v
MegatronPretrainingSampler / MegatronPretrainingRandomSampler
    - Slices global batch across data-parallel ranks
    - Yields per-rank micro-batch index lists
```

### NanoGPT Binary Dataset Pipeline

```
.bin shard files (header + uint16/uint32 tokens)
    |
    v
NanogptDataset.__init__(file_pattern, seq_len, bos_token, ...)
    - glob files, sort
    |
    v
NanogptDataset.__iter__():
    - _setup_worker_context(): assign files to DDP rank * DL worker
    - For single-file mode: split token range across workers
    |
    v
For each file:
    load_bin_shard() -> mmap as torch tensor
    _load_bos_index() -> optional .bos.idx for efficient BOS search
    |
    v
    Slide window of seq_len+1 tokens:
        inputs = buf[:-1], labels = buf[1:]
        If align_to_bos: advance to next BOS token
        yield {input_ids, labels}
    |
    v
    Infinite loop: reshuffle files, repeat
```

## 5. Common Modification Scenarios

### Scenario 1: Adding a New LLM Dataset Type

To add a new dataset (e.g., a custom QA format):

1. **Create a new file** at `nemo_automodel/components/datasets/llm/my_dataset.py`.
2. **Choose your base**: If map-style, extend `torch.utils.data.Dataset`. If streaming, extend `torch.utils.data.IterableDataset`.
3. **Implement tokenization**: Use the shared formatting utilities from `formatting_utils.py`:
   - Import `_add_pad_token`, `format_prompt_completion` or `format_chat_template` from `nemo_automodel.components.datasets.llm.formatting_utils`.
   - In `__getitem__` (or `__iter__`), return a dict with `input_ids`, `labels`, `attention_mask`, and `___PAD_TOKEN_IDS___` metadata for the collator.
4. **Register in `__init__.py`**: Add the import and export to `nemo_automodel/components/datasets/llm/__init__.py`.
5. **Use in recipe YAML**: Reference via `_target_: nemo_automodel.components.datasets.llm.my_dataset.MyDataset`.

Pattern to follow: `ColumnMappedTextInstructionDataset` at `nemo_automodel/components/datasets/llm/column_mapped_text_instruction_dataset.py` for a map-style dataset, or `ColumnMappedTextInstructionIterableDataset` at `nemo_automodel/components/datasets/llm/column_mapped_text_instruction_iterable_dataset.py` for a streaming dataset.

Key contract: Every sample dict must include `input_ids` (list of ints) and `labels` (list of ints with -100 for masked positions). Include `___PAD_TOKEN_IDS___` dict if you want `default_collater()` to use correct per-key padding (see `_package_tokenized_example()` in `formatting_utils.py`, line 110-115).

### Scenario 2: Changing the Packing Strategy

The current packing in `pack_dataset()` (`nemo_automodel/components/datasets/llm/packed_sequence.py`, line 202) uses greedy first-fit: samples are appended sequentially and split when the pack overflows.

To change the packing strategy (e.g., to best-fit decreasing for better utilization):

1. **Modify `pack_dataset()`** in `packed_sequence.py`. The main loop starts at line 247. Replace the sequential iteration with your binning algorithm.
2. **Preserve the output contract**: Each pack must be a dict with keys `input_ids`, `labels`, `position_ids` (as tensors), and `seq_lens` (as tensor of original sequence lengths). The `_pad_pack()` function (line 37) adds `seq_lens_padded` and pads to `packed_sequence_size`.
3. **Position IDs**: Must reset to 0 at each sequence boundary within a pack. See line 280: `current_pack["position_ids"] += [x % packed_sequence_size for x in range(len(input_ids))]`.
4. **CP compatibility**: If `cp_size > 1`, individual sequences must be padded to divisible by `2 * cp_size` before packing (lines 265-273).
5. **Block causal mask**: `create_block_causal_mask()` (line 321) and `packed_block_causal_mask()` (line 365) build masks from `seq_lens`. These are compatible as long as `seq_lens` accurately records original lengths.

### Scenario 3: Adding a New VLM Collate Function

To support a new VLM processor (e.g., for a new vision-language model):

1. **Add the function** to `nemo_automodel/components/datasets/vlm/collate_fns.py`.
2. **Follow the collate pattern**: Accept `(examples, processor)`, where examples are dicts with `conversation` key. See `qwen2_5_collate_fn()` (line 180) as a template.
3. **Build labels using `build_labels()`** (line 84): This shared function pattern-matches assistant response tokens in the encoded sequence and creates a -100-masked label tensor.
4. **Apply the input/label shift**: After building labels, do `labels = labels[:, 1:]` and trim all input-shaped tensors by removing the last position: `value[:, :-1]`. This is the standard autoregressive shift.
5. **Register in `COLLATE_FNS`** dict (line 649): Add `"MyProcessor": my_collate_fn`.
6. **Handle stop tokens**: `build_labels()` extends answer spans by one token if the next token matches `default_stop_tokens()` from `vlm/utils.py` (line 20). Add model-specific stop tokens there if needed.

### Scenario 4: Adding a New Data Source Backend

To add a new data source (e.g., a custom database or cloud storage):

1. **For streaming datasets**: The `_load_streaming_dataset()` function in `column_mapped_text_instruction_iterable_dataset.py` (line 29) is the dispatch point. Add a new detection check (like `is_delta_lake_path()`) and return an iterable that yields row dicts.
2. **Follow the `DeltaLakeDataset` pattern** at `delta_lake_dataset.py`: Implement `__iter__` yielding dicts, plus `shard()`, `shuffle()`, `take()`, `set_epoch()` for compatibility with the iterable dataset wrapper.
3. **For shuffling**: Use `ReservoirSampler` from `reservoir_sampler.py` to wrap your iterator with bounded-memory shuffling.

### Scenario 5: Modifying the Megatron Data Blend Configuration

The `MegatronPretraining` class (`llm/megatron_dataset.py`, line 33) supports three path formats:

1. **Single path or glob**: `paths="path/to/prefix"` or `paths="path/to/*.bin"`. Split using the `split` parameter (e.g., `"900,50,50"`).
2. **Weighted blend list**: `paths=["30", "path/to/ds1", "70", "path/to/ds2"]`. Weights are floats, paths alternate.
3. **Per-split dict**: `paths={"train": [...], "validation": [...], "test": [...]}`. Each value follows format 1 or 2. Can also be loaded from a JSON file via `try_load_blend_from_json()` (line 359).

To modify: Edit `MegatronPretraining.__init__()` for path parsing, or `BlendedMegatronDatasetBuilder._build_blended_dataset_splits()` in `builder.py` (line 302) for the blend construction logic. The `mid_level_dataset_surplus` config parameter (default 0.005) controls over-provisioning of mid-level datasets to prevent oversampling errors.
