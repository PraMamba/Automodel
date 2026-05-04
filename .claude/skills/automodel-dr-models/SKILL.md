---
name: automodel-dr-models
description: Use when working with the models module of automodel — custom optimized model implementations (Llama, DeepSeek-V3, Qwen3-MoE, Mistral3, GPT-OSS) with tensor parallel plans and registration
---

# Models Module — Skill Document

## 1. Module Purpose & Capabilities

The `nemo_automodel/components/models/` module provides custom, optimized model implementations that replace or extend HuggingFace defaults. These custom implementations exist for three reasons:

1. **Combined projections** — Fusing Q/K/V into a single `qkv_proj` and gate/up into `gate_up_proj` reduces kernel launch overhead and improves memory access patterns. Dense models (Llama, Qwen2) use `CombinedQKVAttentionMixin` and `CombinedGateUpMLP` from `common/`.

2. **Backend flexibility** — The `BackendConfig` dataclass (`common/utils.py`) lets every layer pick between TransformerEngine (`te`) and PyTorch-native (`torch`/`sdpa`/`flex`) backends for linear, RMSNorm, and attention at construction time, without changing model code.

3. **MoE-native architectures** — DeepSeek-V3, GPT-OSS, Qwen3-MoE, GLM4-MoE, Step3p5, NemotronV3, and Qwen3-Next require Mixture-of-Experts layers that HuggingFace does not optimize for distributed training. These models compose `MoE`/`MLP` from `components/moe/layers.py` with custom attention layers.

### Model families covered

| Sub-directory | Top-level class | Architecture type | Key feature |
|---|---|---|---|
| `llama/` | `LlamaForCausalLM` | Dense, PreTrainedModel-based | Combined QKV/gate_up, TP/PP plans |
| `qwen2/` | `Qwen2ForCausalLM` | Dense, PreTrainedModel-based | Combined QKV/gate_up, sliding window attn |
| `deepseek_v3/` | `DeepseekV3ForCausalLM` | MoE, nn.Module-based | Multi-head Latent Attention (MLA) |
| `deepseek_v32/` | `DeepseekV32ForCausalLM` | MoE, extends V3 | Indexer for sparse attention (top-k) |
| `gpt_oss/` | `GptOssForCausalLM` | MoE, nn.Module-based | Sliding attention, FlexAttention |
| `qwen3_moe/` | `Qwen3MoeForCausalLM` | MoE, nn.Module-based | Per-head QK RMSNorm |
| `qwen3_next/` | `Qwen3NextForCausalLM` | Hybrid MoE, nn.Module-based | Gated DeltaNet linear attention |
| `qwen3_omni_moe/` | `Qwen3OmniMoeThinkerForConditionalGeneration` | VLM + MoE | MRoPE, DeepStack visual fusion |
| `qwen3_vl_moe/` | `Qwen3VLMoeForConditionalGeneration` | VLM + MoE | Vision encoder + MoE text decoder |
| `glm4_moe/` | `Glm4MoeForCausalLM` | MoE, nn.Module-based | Partial rotary embeddings, QK norm |
| `step3p5/` | `Step3p5ForCausalLM` | MoE, nn.Module-based | Explicit MoE layer enumeration |
| `nemotron_v3/` | `NemotronV3ForCausalLM` | Hybrid MoE, nn.Module-based | Mamba2 + Attention + MoE layers |
| `mistral3/` | `Ministral3ForCausalLM` | Dense, PreTrainedModel-based | Vendored Ministral3 config + YaRN RoPE |
| `biencoder/` | `BiencoderModel` / `NeMoAutoModelBiencoder` | Embedding/retrieval | Bidirectional Llama, contrastive loss |
| `nemotron_parse/` | `NemotronParseForConditionalGeneration` | Encoder-decoder VLM | RADIO vision encoder + mBART decoder |
| `kimivl/` | VLM wrapper | VLM | MoonViT + DeepseekV3 backend |
| `kimi_k25_vl/` | VLM wrapper | VLM | MoonViT3d + DeepseekV3 backend |
| `gpt2.py` | `GPT2LMHeadModel` (standalone) | Dense, nn.Module | Pure-PyTorch GPT-2, no HF dependency |

## 2. Core Design Logic

### 2.1 Why custom implementations exist

HuggingFace models use separate `q_proj`, `k_proj`, `v_proj` and separate `gate_proj`, `up_proj`. The custom implementations fuse these into single linear projections (`qkv_proj`, `gate_up_proj`) that:
- Issue one GEMM call instead of two or three
- Improve GPU utilization through larger matrix operations
- Are compatible with DTensor-based tensor parallelism (the split sizes are computed dynamically from the actual tensor shape, not from config values)

### 2.2 The two model families: PreTrainedModel vs nn.Module

**Dense models** (Llama, Qwen2, Mistral3) subclass `transformers.PreTrainedModel`. They follow HF conventions closely (same forward signature, `CausalLMOutputWithPast` return type, `post_init()` weight initialization, `_tp_plan`/`_pp_plan` dicts).

**MoE/hybrid models** (DeepSeek-V3, GPT-OSS, Qwen3-MoE, etc.) subclass `nn.Module` directly. They use a raw-tensor forward signature (`input_ids -> logits`) instead of the HF keyword-argument API. This is because:
- MoE routing requires custom `padding_mask` handling
- The `MoE` layer from `components/moe/layers.py` expects a specific calling convention
- These models mix `MoEFSDPSyncMixin` for gradient synchronization across expert-parallel groups

### 2.3 Registration pattern

Every model file that should be discoverable exports a module-level `ModelClass` variable. The `ModelRegistry` in `nemo_automodel/_transformers/registry.py` walks all subpackages of `nemo_automodel.components.models`, imports each non-package module, and if it has a `ModelClass` attribute, registers it by class name. Example from `llama/model.py` line 526: `ModelClass = LlamaForCausalLM`.

When `NeMoAutoModelForCausalLM.from_pretrained()` is called, it reads `config.json` to get the architecture name, looks it up in the registry, and instantiates the custom class instead of the HuggingFace default.

### 2.4 State dict adapter pattern

Every custom model attaches a `state_dict_adapter` attribute that implements bidirectional conversion between HF checkpoint format and the internal (combined-projection or grouped-expert) format. The adapter has two methods:
- `from_hf(hf_state_dict) -> native_state_dict` — Called during model loading
- `to_hf(native_state_dict) -> hf_state_dict` — Called during checkpoint saving
- `convert_single_tensor_to_hf(fqn, tensor)` — Called for per-tensor streaming conversion

**For dense models**, the adapter is `CombinedProjectionStateDictAdapter` (`common/combined_projection/state_dict_adapter.py`), which concatenates/splits Q/K/V and gate/up weight matrices. Llama and Qwen2 both subclass it.

**For MoE models**, the adapter extends `StateDictAdapter` (from `components/checkpoint/`) and `MoESplitExpertsStateDictMixin` (from `components/moe/`). It handles:
- Merging per-expert HF weights (`experts.{E}.gate_proj.weight`) into grouped tensors (`experts.gate_and_up_projs`)
- FP8 dequantization (DeepSeek-V3) or mxfp4 dequantization (GPT-OSS)
- Key remapping (GPT-OSS router -> gate)

### 2.5 BackendConfig

Defined in `common/utils.py` (line 46), `BackendConfig` is a `@dataclass(kw_only=True)` that controls:
- `attn`: `"te"` | `"sdpa"` | `"flex"` — attention backend
- `linear`: `"torch"` | `"te"` — linear layer backend
- `rms_norm`: `"torch"` | `"te"` — RMSNorm backend
- `rope_fusion`: `bool` — whether to use TE fused RoPE
- `enable_deepep`: `bool` — DeepEP expert parallelism
- `enable_hf_state_dict_adapter`: `bool` — whether to create the adapter
- `enable_fsdp_optimizations`: `bool`
- `gate_precision`: `str | torch.dtype | None`
- `fake_balanced_gate`: `bool`

Models pass this to every sub-layer constructor. Factory functions `initialize_linear_module()` and `initialize_rms_norm_module()` dispatch based on the backend string.

### 2.6 HFCheckpointingMixin

All custom models inherit `HFCheckpointingMixin` (`common/hf_checkpointing_mixin.py`). It provides `save_pretrained()` that delegates to the `Checkpointer` infrastructure. Critically, it does NOT override `state_dict()` or `load_state_dict()`, because PyTorch DCP depends on standard `nn.Module` behavior.

## 3. Core Data Structures

See `reference.md` for the complete class hierarchy and file paths.

### 3.1 Common building blocks

- `CombinedQKVAttentionMixin` (`common/combined_projection/combined_qkv.py`) — Mixin that adds `setup_qkv_projection()` and `compute_qkv()`. The `compute_qkv()` method dynamically computes split sizes from the actual tensor shape to handle TP sharding transparently.

- `CombinedGateUpMLP` (`common/combined_projection/combined_mlp.py`) — SwiGLU MLP with a single `gate_up_proj` linear. Same TP-aware dynamic split.

- `CombinedProjectionStateDictAdapter` (`common/combined_projection/state_dict_adapter.py`) — Generic adapter for converting between HF separate-projection and combined-projection state dicts. Handles QKV, gate_up, and tied lm_head/embed_tokens.

- `BackendConfig` (`common/utils.py`) — Backend selection dataclass.

- `HFCheckpointingMixin` (`common/hf_checkpointing_mixin.py`) — Provides `save_pretrained()`.

### 3.2 RoPE implementations

Three distinct RoPE implementations exist:
1. `llama/rope_utils.py` — Standard RoPE with Llama3-style smooth interpolation. Shared by Llama and Qwen2 (aliased as `Qwen2RotaryEmbedding`).
2. `gpt_oss/rope_utils.py` — YaRN/NTK-by-parts RoPE with partial rotary factor support. Shared by GPT-OSS, Qwen3-MoE, GLM4-MoE, Step3p5, Qwen3-Next.
3. `deepseek_v3/rope_utils.py` — Complex-exponential RoPE with interleaved format for MLA. Shared by DeepSeek-V3 and V3.2.

### 3.3 Attention implementations

- **Dense models** use HuggingFace's `ALL_ATTENTION_FUNCTIONS` dispatch (sdpa/flash_attention_2/flex_attn).
- **MoE models** use `components/attention/utils.py` which provides `initialize_attn_module_and_func()` to create TE `DotProductAttention` or SDPA/Flex attention with a unified `preprocess_args_and_kwargs_for_attn()` / `postprocess_output_for_attn()` API.
- **DeepSeek-V3** uses Multi-head Latent Attention (MLA) with LoRA-compressed Q and KV.
- **DeepSeek-V3.2** extends MLA with an `Indexer` module for top-k sparse attention selection.

## 4. State Flow

### 4.1 Model loading flow

```
NeMoAutoModelForCausalLM.from_pretrained(model_name)
  -> AutoConfig.from_pretrained(model_name)
  -> ModelRegistry.get_model_cls_from_model_arch(config.architectures[0])
  -> Custom ModelClass.__init__(config, backend=BackendConfig())
     -> Creates model with combined projections / MoE layers
     -> Attaches state_dict_adapter
  -> Checkpointer.load_base_model()
     -> Reads HF safetensors
     -> Calls model.state_dict_adapter.from_hf(hf_state_dict)
        -> Concatenates Q/K/V or merges per-expert weights
     -> Loads converted state dict into model
```

### 4.2 Forward pass flow (dense models, e.g. Llama)

```
LlamaForCausalLM.forward(input_ids, attention_mask, labels, ...)
  -> LlamaModel.forward(input_ids, ...)
     -> embed_tokens(input_ids) -> inputs_embeds
     -> rotary_emb(inputs_embeds, position_ids) -> (cos, sin)
     -> create_causal_mask(...)
     -> For each LlamaDecoderLayer:
        -> input_layernorm(hidden_states)
        -> LlamaAttention.forward(hidden_states, position_embeddings, mask)
           -> compute_qkv(hidden_states)  # CombinedQKVAttentionMixin
           -> apply_rotary_pos_emb(q, k, cos, sin)
           -> attention_interface(q, k, v, mask)  # sdpa/flash/flex
           -> o_proj(attn_output)
        -> residual + attn_output
        -> post_attention_layernorm(hidden_states)
        -> CombinedGateUpMLP.forward(hidden_states)
           -> gate_up_proj(x) -> split -> act(gate) * up -> down_proj
        -> residual + mlp_output
     -> norm(hidden_states)
  -> lm_head(hidden_states) -> logits
  -> loss_function(logits, labels) if labels provided
  -> CausalLMOutputWithPast(loss, logits, ...)
```

### 4.3 Forward pass flow (MoE models, e.g. DeepSeek-V3)

```
DeepseekV3ForCausalLM.forward(input_ids, position_ids, attention_mask, **attn_kwargs)
  -> squeeze_input_for_thd() if thd format
  -> DeepseekV3Model.forward(input_ids, ...)
     -> embed_tokens(input_ids)
     -> freqs_cis_from_position_ids(position_ids, self.freqs_cis, ...)
     -> For each Block:
        -> input_layernorm(x)
        -> MLA.forward(x, freqs_cis, attention_mask, **attn_kwargs)
           -> q_a_proj -> q_a_layernorm -> q_b_proj -> split q_nope, q_pe
           -> kv_a_proj_with_mqa -> split kv, k_pe -> kv_a_layernorm -> kv_b_proj
           -> apply_rotary_emb_qk(q_pe, k_pe, freqs_cis)
           -> cat [q_nope, q_pe], cat [k_nope, k_pe]
           -> preprocess_args_and_kwargs_for_attn(q, k, v, mask, backend)
           -> attn_func(q, k, v, **kwargs)
           -> o_proj(output)
        -> residual + attn_out
        -> post_attention_layernorm(x)
        -> MoE(x, padding_mask) or MLP(x)
        -> residual + mlp_out
     -> norm(h)
  -> lm_head(logits)
```

### 4.4 Checkpoint saving flow

```
Checkpointer.save_model(model, weights_path, ...)
  -> model.state_dict()  # Standard nn.Module state_dict
  -> model.state_dict_adapter.to_hf(state_dict)
     -> Splits qkv_proj back to q_proj/k_proj/v_proj
     -> Splits gate_up_proj back to gate_proj/up_proj
     -> Or splits grouped expert tensors back to per-expert format
  -> Saves as HF-compatible safetensors
```

## 5. Common Modification Scenarios

### Scenario 1: Adding a new dense model (e.g., "MyLlama")

1. Create `nemo_automodel/components/models/myllama/` directory with `__init__.py`, `model.py`, `state_dict_adapter.py`.
2. In `model.py`, subclass `PreTrainedModel`. Use `CombinedQKVAttentionMixin` in the attention class and `CombinedGateUpMLP` for the MLP. Inherit `HFCheckpointingMixin` in the top-level `ForCausalLM` class. Set `_tp_plan` and `_pp_plan` dicts.
3. In `state_dict_adapter.py`, subclass `CombinedProjectionStateDictAdapter` with the appropriate config class.
4. Attach `self.state_dict_adapter = MyLlamaStateDictAdapter(config)` in `__init__`.
5. Set `ModelClass = MyLlamaForCausalLM` at module level.
6. The registry will auto-discover the model by class name.

**Key files to reference:**
- `llama/model.py` (line 385: `LlamaForCausalLM`, line 526: `ModelClass`)
- `llama/state_dict_adapter.py` (line 27: `LlamaStateDictAdapter`)
- `common/combined_projection/combined_qkv.py` (line 25: `CombinedQKVAttentionMixin`)

### Scenario 2: Adding a new MoE model

1. Create a new subdirectory with `model.py`, `layers.py`, `state_dict_adapter.py`.
2. In `layers.py`, create an attention class using `initialize_attn_module_and_func()` and `initialize_linear_module()` from `common/`.
3. In `model.py`, create `Block(nn.Module)` composing attention + `MoE`/`MLP` from `components/moe/layers.py`. Create the `Model(nn.Module)` with embed_tokens, layers (ModuleDict), norm, rotary_emb. Create `ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin)` with `from_config()`, `from_pretrained()`, `forward()`, `initialize_weights()`.
4. In `state_dict_adapter.py`, subclass `MoESplitExpertsStateDictMixin, StateDictAdapter`. Define `from_hf()` to call `self._from_hf_w_merged_experts(hf_state_dict, device_mesh)` and `to_hf()` / `convert_single_tensor_to_hf()`.
5. Export `ModelClass` at module level.

**Key files to reference:**
- `deepseek_v3/model.py` (canonical MoE model pattern)
- `qwen3_moe/model.py` (softmax-routing MoE pattern)
- `qwen3_moe/state_dict_adapter.py` (line 29: `Qwen3MoeStateDictAdapter`)

### Scenario 3: Adding a TP plan for an existing model

For `PreTrainedModel`-based models, define `_tp_plan` and `_pp_plan` as class attributes on the `ForCausalLM` class:
```python
_tp_plan = {"lm_head": "colwise_rep"}
_pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
```

For `nn.Module`-based MoE models, TP plans are applied externally through the distributed infrastructure (`components/distributed/optimized_tp_plans.py`) rather than on the model class.

**Key files to reference:**
- `llama/model.py` (line 388-390: `_tied_weights_keys`, `_tp_plan`, `_pp_plan`)
- `qwen2/model.py` (line 367-369: same pattern)

### Scenario 4: Changing the attention or linear backend for a model

Pass a custom `BackendConfig` when constructing the model:
```python
backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
model = DeepseekV3ForCausalLM.from_config(config, backend=backend)
```

All layers use `initialize_linear_module(backend.linear, ...)` and `initialize_rms_norm_module(backend.rms_norm, ...)` so the choice propagates automatically.

**Key file:** `common/utils.py` (line 46-60: `BackendConfig`, line 63-129: factory functions)

### Scenario 5: Adding FP8 dequantization support

DeepSeek-V3 already handles FP8 dequantization in its state dict adapter (`deepseek_v3/state_dict_adapter.py`, function `dequantize_from_fp8()` at line 375). To add similar support for a new model:
1. Add `_dequantize()` method in the state dict adapter that detects `*_scale_inv` keys.
2. Use the Triton kernel `_weight_dequant_kernel` (line 96) for GPU dequantization or fall back to `_dequantize_with_torch()` (line 313).
3. Call `_dequantize()` in `from_hf()` before key remapping.

**Key file:** `deepseek_v3/state_dict_adapter.py` (line 121: `DeepSeekV3StateDictAdapter`)

### Scenario 6: Adding a VLM model

VLM models (Qwen3-VL-MoE, Qwen3-Omni-MoE, KimiVL) typically:
1. Inherit from the HF multimodal base class for vision/audio encoding.
2. Replace the text decoder with the custom backend text model (reusing `Block` from an existing MoE model).
3. Handle multimodal fusion (pixel_values, image_grid_thw) before the text forward pass.
4. Use MRoPE (multi-dimensional RoPE) for position encoding of visual tokens.

**Key files to reference:**
- `qwen3_vl_moe/model.py` (line 317: `Qwen3VLMoeForConditionalGeneration`)
- `qwen3_omni_moe/model.py` (line 177: `Qwen3OmniMoeThinkerForConditionalGeneration`)
