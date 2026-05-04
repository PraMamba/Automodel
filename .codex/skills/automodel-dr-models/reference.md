# Models Module — Reference

Complete class, function, and file reference for `nemo_automodel/components/models/`.

## Directory Structure

```
nemo_automodel/components/models/
  __init__.py                          # Exports build_gpt2_model
  gpt2.py                             # Standalone GPT-2 (pure PyTorch, no HF dep)
  common/
    __init__.py                        # Re-exports all common utilities
    utils.py                           # BackendConfig, initialize_linear_module, initialize_rms_norm_module, _patch_te_modules
    hf_checkpointing_mixin.py          # HFCheckpointingMixin (save_pretrained via Checkpointer)
    combined_projection/
      __init__.py                      # Re-exports CombinedQKVAttentionMixin, CombinedGateUpMLP
      combined_qkv.py                  # CombinedQKVAttentionMixin (setup_qkv_projection, compute_qkv)
      combined_mlp.py                  # CombinedGateUpMLP (gate_up_proj -> act(gate) * up -> down_proj)
      state_dict_adapter.py            # CombinedProjectionStateDictAdapter (from_hf, to_hf)
  llama/
    __init__.py
    model.py                           # LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
    rope_utils.py                      # LlamaRotaryEmbedding, apply_rotary_pos_emb (also aliased as Qwen2RotaryEmbedding)
    state_dict_adapter.py              # LlamaStateDictAdapter (extends CombinedProjectionStateDictAdapter)
  qwen2/
    __init__.py                        # Exports Qwen2ForCausalLM
    model.py                           # Qwen2Attention, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM
    state_dict_adapter.py              # Qwen2StateDictAdapter (extends CombinedProjectionStateDictAdapter)
  deepseek_v3/
    __init__.py
    model.py                           # Block, DeepseekV3Model, DeepseekV3ForCausalLM
    layers.py                          # MLA (Multi-head Latent Attention with LoRA-compressed Q/KV)
    rope_utils.py                      # precompute_freqs_cis, apply_rotary_emb (complex-exponential), freqs_cis_from_position_ids, yarn_get_mscale
    state_dict_adapter.py              # DeepSeekV3StateDictAdapter (FP8 dequant, MoE expert merging, Triton kernel)
  deepseek_v32/
    __init__.py
    config.py                          # DeepseekV32Config (extends PretrainedConfig, adds index_n_heads/index_head_dim/index_topk)
    model.py                           # DeepseekV32Block, DeepseekV32Model, DeepseekV32ForCausalLM (extends V3)
    layers.py                          # DeepseekV32Indexer (top-k sparse attn), DeepseekV32MLA (MLA + Indexer)
    state_dict_adapter.py              # DeepSeekV32StateDictAdapter (extends V3 adapter, handles indexer keys)
  gpt_oss/
    __init__.py
    model.py                           # Block, GptOssModel, GptOssForCausalLM
    layers.py                          # GptOssAttention (sliding window, sink tokens, FlexAttention)
    rope_utils.py                      # RotaryEmbedding (YaRN/NTK-by-parts), apply_rotary_emb, apply_rotary_emb_qk, position_ids_to_freqs_cis
    state_dict_adapter.py              # GPTOSSStateDictAdapter (mxfp4 dequant, key remapping, FP4_VALUES lookup)
  qwen3_moe/
    __init__.py
    model.py                           # Block, Qwen3MoeModel, Qwen3MoeForCausalLM
    layers.py                          # Qwen3MoeAttention (per-head q_norm/k_norm RMSNorm)
    state_dict_adapter.py              # Qwen3MoeStateDictAdapter (MoESplitExpertsStateDictMixin)
  qwen3_next/
    __init__.py
    model.py                           # Block (hybrid linear_attention + full_attention), Qwen3NextModel, Qwen3NextForCausalLM
    layers.py                          # Qwen3NextAttention, Qwen3NextRMSNorm
    state_dict_adapter.py              # Qwen3NextStateDictAdapter
  qwen3_omni_moe/
    __init__.py
    model.py                           # Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeThinkerForConditionalGeneration
    state_dict_adapter.py              # Qwen3OmniMoeStateDictAdapter (handles thinker. prefix)
  qwen3_vl_moe/
    __init__.py
    model.py                           # Qwen3VLMoeBlock, Qwen3VLMoeModel, Qwen3VLMoeTextModelBackend, Qwen3VLMoeForConditionalGeneration
    state_dict_adapter.py              # Qwen3VLMoeStateDictAdapter (aggregated expert format, no per-expert split)
  glm4_moe/
    __init__.py
    model.py                           # Block, Glm4MoeModel, Glm4MoeForCausalLM
    layers.py                          # Glm4MoeAttention (optional QK norm, partial rotary)
    state_dict_adapter.py              # Glm4MoeStateDictAdapter (shared_expert handling)
  step3p5/
    __init__.py                        # Exports ModelClass = Step3p5ForCausalLM
    model.py                           # Block, Step3p5Model, Step3p5ForCausalLM (explicit moe_layers_enum)
    layers.py                          # Step3p5Attention, Step3p5MLP, Step3p5RMSNorm
    state_dict_adapter.py              # Step3p5StateDictAdapter
  nemotron_v3/
    __init__.py
    model.py                           # NemotronV3Model, NemotronV3ForCausalLM (Mamba2 + Attention + MoE hybrid)
    layers.py                          # NemotronV3Block (dispatches to Mamba2, Attention, or MLP/MoE per layer type)
    state_dict_adapter.py              # NemotronV3StateDictAdapter
  mistral3/
    __init__.py
    model.py                           # Ministral3Config, Ministral3Attention, Ministral3MLP, Ministral3Model, Ministral3ForCausalLM, _register_ministral3_with_transformers()
  biencoder/
    __init__.py                        # Exports BiencoderModel, NeMoAutoModelBiencoder, LlamaBidirectional*
    biencoder_model.py                 # NeMoAutoModelBiencoder (extends _BaseNeMoAutoModelClass)
    llama_bidirectional_model.py       # LlamaBidirectionalConfig, LlamaBidirectionalModel, LlamaBidirectionalForSequenceClassification, BiencoderModel, BiencoderOutput
    state_dict_adapter.py              # BiencoderStateDictAdapter (lm_q. <-> model. prefix)
  nemotron_parse/
    __init__.py
    model.py                           # NemotronParseConfig, NemotronParseDecoder, RadioWithNeck, NemotronParseForConditionalGeneration
    nemotron_parse_loss.py             # NemotronParseLoss (cross-entropy with coordinate token weighting)
  kimivl/
    __init__.py
    model.py                           # KimiVL VLM: MoonViT encoder, PatchMergerMLP projector, DeepseekV3 backend
  kimi_k25_vl/
    __init__.py
    model.py                           # KimiK25VL VLM: MoonViT3d encoder (temporal), PatchMergerMLP, DeepseekV3 backend
    state_dict_adapter.py              # KimiK25VL state dict adapter
```

## Key Classes Reference

### common/utils.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `BackendConfig` | dataclass | 46 | Backend selection: attn, linear, rms_norm, rope_fusion, enable_deepep, etc. |
| `HAVE_TE` | bool | 24 | Whether TransformerEngine is available |
| `HAVE_DEEP_EP` | bool | 25 | Whether DeepEP is available |
| `initialize_rms_norm_module()` | function | 63 | Creates TE or torch RMSNorm based on backend string |
| `initialize_linear_module()` | function | 95 | Creates TE or torch Linear based on backend string |
| `is_tensor_unallocated()` | function | 28 | Detects meta/fake tensors for PP shape inference |
| `_patch_te_modules()` | function | 132 | Patches TE Linear/RMSNorm to handle unallocated tensors |

### common/combined_projection/combined_qkv.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `CombinedQKVAttentionMixin` | class | 25 | Mixin providing `setup_qkv_projection()` and `compute_qkv()` |
| `setup_qkv_projection()` | method | 49 | Creates single `qkv_proj` Linear with (Q+2*KV)*head_dim output |
| `compute_qkv()` | method | 79 | Projects and splits; split sizes computed from actual tensor dim for TP |

### common/combined_projection/combined_mlp.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `CombinedGateUpMLP` | class | 26 | SwiGLU MLP: `gate_up_proj` (2*intermediate) -> split -> act(gate)*up -> `down_proj` |

### common/combined_projection/state_dict_adapter.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `CombinedProjectionStateDictAdapter` | class | 34 | Generic HF <-> combined-projection converter |
| `from_hf()` | method | 77 | Concatenates Q/K/V -> qkv_proj, gate/up -> gate_up_proj, handles tied weights |
| `to_hf()` | method | 170 | Splits qkv_proj -> Q/K/V, gate_up_proj -> gate/up; handles TP-sharded DTensors |

### common/hf_checkpointing_mixin.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `HFCheckpointingMixin` | class | 39 | Provides `save_pretrained()` delegating to Checkpointer |

### llama/model.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `LlamaAttention` | class | 61 | Combined QKV attention (uses CombinedQKVAttentionMixin) |
| `LlamaMLP` | class | 140 | Inline combined gate_up MLP |
| `LlamaDecoderLayer` | class | 167 | GradientCheckpointingLayer subclass |
| `LlamaPreTrainedModel` | class | 231 | Supports flash_attn, sdpa, flex_attn |
| `LlamaModel` | class | 254 | Transformer body (embed + layers + norm + rotary_emb) |
| `LlamaForCausalLM` | class | 385 | Top-level; `_tp_plan = {"lm_head": "colwise_rep"}` |
| `ModelClass` | variable | 526 | `= LlamaForCausalLM` (for registry) |

### deepseek_v3/layers.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `MLA` | class | 37 | Multi-head Latent Attention with q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim |

### deepseek_v3/state_dict_adapter.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `DeepSeekV3StateDictAdapter` | class | 121 | FP8 dequant + MoE expert merging |
| `dequantize_from_fp8()` | function | 375 | Block-scale FP8 dequantization (Triton or torch fallback) |
| `_weight_dequant_kernel` | Triton kernel | 96 | GPU-accelerated FP8 dequantization |

### deepseek_v32/layers.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `DeepseekV32Indexer` | class | 95 | Top-k sparse attention selection using Q/K scores + per-head weights + ReLU |
| `DeepseekV32MLA` | class | 272 | MLA + Indexer integration; builds sparse mask for TE or SDPA backend |

### gpt_oss/layers.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `GptOssAttention` | class | 39 | Separate Q/K/V/O projections, sliding window via `window_size`, sink tokens via learnable `sinks` parameter |

### gpt_oss/rope_utils.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `RotaryEmbedding` | class | 57 | YaRN/NTK-by-parts with `partial_rotary_factor` support |
| `apply_rotary_emb_qk()` | function | 137 | Dispatches to TE fused rope or non-fused (cos/sin) |
| `position_ids_to_freqs_cis()` | function | 185 | Converts position_ids to freqs_cis for fused or non-fused rope |

### gpt_oss/state_dict_adapter.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `GPTOSSStateDictAdapter` | class | 51 | Key remapping (router -> gate), mxfp4 dequantization |
| `FP4_VALUES` | list | 31 | 16-element FP4 lookup table for mxfp4 dequantization |

### qwen3_moe/layers.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `Qwen3MoeAttention` | class | 34 | Separate Q/K/V/O with per-head `q_norm`/`k_norm` RMSNorm |

### mistral3/model.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `Ministral3Config` | class | 45 | Vendored config with YaRN rope_parameters |
| `Ministral3RotaryEmbedding` | class | 188 | Uses `ROPE_INIT_FUNCTIONS` from transformers |
| `_register_ministral3_with_transformers()` | function | 562 | Registers config+model with AutoConfig/AutoModel/AutoModelForCausalLM |
| `ModelClass` | variable | 617 | `= Ministral3ForCausalLM` |

### biencoder/llama_bidirectional_model.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `LlamaBidirectionalConfig` | class | 129 | Extends LlamaConfig with pooling and temperature |
| `LlamaBidirectionalModel` | class | 158 | LlamaModel with `is_causal = False` on all layers |
| `BiencoderModel` | class | 375 | Dual-encoder (lm_q, lm_p) with contrastive loss |
| `BiencoderOutput` | dataclass | 364 | q_reps, p_reps, loss, labels, scores |

### nemotron_parse/model.py

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `NemotronParseConfig` | class | 146 | Composite config: `encoder` (RADIO) + `decoder` (mBART) |
| `RadioWithNeck` | class | 366 | RADIO vision encoder with conv neck |
| `NemotronParseForConditionalGeneration` | class | 431 | Encoder-decoder VLM |

### Registry (nemo_automodel/_transformers/registry.py)

| Symbol | Type | Line | Purpose |
|---|---|---|---|
| `_ModelRegistry` | dataclass | 33 | Walks `MODELING_PATH`, imports modules, registers `ModelClass` by class name |
| `ModelRegistry` | singleton | 94 | `= get_registry()` — the global registry |
| `MODELING_PATH` | list | 29 | `["nemo_automodel.components.models"]` |

## State Dict Adapter Hierarchy

```
StateDictAdapter (components/checkpoint/state_dict_adapter.py)
  |
  +-- CombinedProjectionStateDictAdapter (common/combined_projection/state_dict_adapter.py)
  |     +-- LlamaStateDictAdapter (llama/state_dict_adapter.py)
  |     +-- Qwen2StateDictAdapter (qwen2/state_dict_adapter.py)
  |
  +-- DeepSeekV3StateDictAdapter + MoESplitExpertsStateDictMixin (deepseek_v3/state_dict_adapter.py)
  |     +-- DeepSeekV32StateDictAdapter (deepseek_v32/state_dict_adapter.py)
  |
  +-- GPTOSSStateDictAdapter (gpt_oss/state_dict_adapter.py)
  |
  +-- Qwen3MoeStateDictAdapter + MoESplitExpertsStateDictMixin (qwen3_moe/state_dict_adapter.py)
  +-- Qwen3OmniMoeStateDictAdapter + MoESplitExpertsStateDictMixin (qwen3_omni_moe/state_dict_adapter.py)
  +-- Qwen3VLMoeStateDictAdapter (qwen3_vl_moe/state_dict_adapter.py)
  +-- Glm4MoeStateDictAdapter + MoESplitExpertsStateDictMixin (glm4_moe/state_dict_adapter.py)
  +-- Step3p5StateDictAdapter (step3p5/state_dict_adapter.py)
  +-- NemotronV3StateDictAdapter (nemotron_v3/state_dict_adapter.py)
  +-- Qwen3NextStateDictAdapter (qwen3_next/state_dict_adapter.py)
  +-- BiencoderStateDictAdapter (biencoder/state_dict_adapter.py)
```

## RoPE Implementation Map

| RoPE module | Used by | Key features |
|---|---|---|
| `llama/rope_utils.py` | Llama, Qwen2 | Standard + Llama3 smooth interpolation, cos/sin cache |
| `gpt_oss/rope_utils.py` | GPT-OSS, Qwen3-MoE, GLM4-MoE, Step3p5, Qwen3-Next | YaRN/NTK-by-parts, partial_rotary_factor, TE fused rope, position_ids_to_freqs_cis |
| `deepseek_v3/rope_utils.py` | DeepSeek-V3, DeepSeek-V3.2 | Complex-exponential (torch.polar), interleaved, TE fused rope |
| `mistral3/model.py` (inline) | Ministral3 | `ROPE_INIT_FUNCTIONS` from transformers, dynamic_rope_update |

## ForCausalLM Class Hierarchy

```
PreTrainedModel (transformers)
  +-- LlamaPreTrainedModel
  |     +-- LlamaForCausalLM + HFCheckpointingMixin
  +-- Qwen2PreTrainedModel
  |     +-- Qwen2ForCausalLM + HFCheckpointingMixin
  +-- Ministral3PreTrainedModel
        +-- Ministral3ForCausalLM + HFCheckpointingMixin + GenerationMixin

nn.Module
  +-- DeepseekV3ForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin
  |     +-- DeepseekV32ForCausalLM
  +-- GptOssForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin
  +-- Qwen3MoeForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin
  +-- Qwen3NextForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin
  +-- Glm4MoeForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin
  +-- Step3p5ForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin
  +-- NemotronV3ForCausalLM + HFCheckpointingMixin + MoEFSDPSyncMixin

HF multimodal base + HFCheckpointingMixin + MoEFSDPSyncMixin
  +-- Qwen3OmniMoeThinkerForConditionalGeneration (extends HFQwen3OmniMoeThinkerForConditionalGeneration)
  +-- Qwen3VLMoeForConditionalGeneration (extends HFQwen3VLMoeForConditionalGeneration)

NemotronParsePreTrainedModel + HFCheckpointingMixin + GenerationMixin
  +-- NemotronParseForConditionalGeneration
```
