---
name: automodel-dr-attention
description: Use when working with the attention module of automodel — attention mechanisms, context parallel attention, and flash attention integration
---

# Attention Module Deep Read

## 1. Module Purpose & Capabilities

The attention component (`nemo_automodel/components/attention/`) provides a backend-agnostic abstraction layer for attention computation in NeMo AutoModel. It allows model layers to swap between three attention implementations without changing model code:

- **Transformer Engine (`te`)** -- NVIDIA's fused kernel attention via `transformer_engine.pytorch.attention.DotProductAttention`. Supports `bshd` and `thd` (packed/variable-length) QKV formats, padding-causal masks, sliding window, and cu_seqlens for sequence packing. This is the production-grade path for NVIDIA GPUs.

- **Scaled Dot-Product Attention (`sdpa`)** -- PyTorch-native `torch.nn.functional.scaled_dot_product_attention`. A lightweight fallback requiring no extra dependencies. Supports causal masking and GQA via `enable_gqa`.

- **FlexAttention (`flex`)** -- PyTorch's `torch.nn.attention.flex_attention` with `torch.compile(mode="max-autotune-no-cudagraphs")`. Provides programmable mask functions (causal, block-causal for packed sequences, sliding window, fixed-block) and attention sink rescaling. Adapted from the torchtitan project.

The module also handles tensor layout transformations (transposing between `[B,H,S,D]` and `[B,S,H,D]`) so that each backend receives tensors in its expected format.

### Key capabilities:
- Unified factory function to create any attention backend from a single string selector (`attn_impl`).
- Pre/post-processing utilities that adapt Q/K/V tensors and kwargs per backend, including mask inversion for TE and layout transposition for SDPA/Flex.
- Compiled FlexAttention with class-level `BlockMask` caching to amortize mask creation and compilation cost across forward passes and layers.
- Composable mask modifiers: causal, block-causal (EOS-delimited packed sequences), sliding window, and fixed-block size -- combinable via `_fixed_block_mask_mod`.
- Attention sink support: FlexAttention can return log-sum-exp (LSE) values and rescale outputs by `sigmoid(lse - sink_weight)` per head.


## 2. Core Design Logic

### Why a factory + pre/post-process pattern?

Model layers (DeepSeek-V3, Qwen3-MoE, GPT-OSS, GLM4-MoE, etc.) all need attention but each backend has incompatible input conventions: TE expects `[B,S,H,D]` with inverted padding masks and cu_seqlens kwargs; SDPA and Flex expect `[B,H,S,D]` and different kwarg sets. Rather than polluting every model with backend-specific branches, `utils.py` centralizes three concerns:

1. **`initialize_attn_module_and_func()`** -- returns a `(module, callable)` tuple. TE and Flex return an `nn.Module` (needed for parameter registration / hooks) plus its `__call__`; SDPA returns `(None, partial_func)` since it is stateless. This dual return lets callers optionally register the module as a submodule while always calling attention through a uniform callable.

2. **`preprocess_args_and_kwargs_for_attn()`** -- transforms Q/K/V shapes and builds the correct kwargs dict. For TE: inverts the boolean attention_mask, wraps cu_seqlens into the expected format, sets `qkv_format="thd"` for packed sequences. For SDPA/Flex: transposes dims 1 and 2 to go from `[B,H,S,D]` to `[B,S,H,D]`.

3. **`postprocess_output_for_attn()`** -- reverses the layout change for SDPA/Flex (transpose dims 1 and 2 back); TE output is already in the model's native layout so it passes through unchanged.

### Why class-level state on FlexAttention?

`FlexAttention` uses `ClassVar` for both the compiled `flex_attn` function and the `block_masks` dictionary. This is deliberate:

- **Compilation amortization**: `torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")` is expensive. Making it a class variable means all `FlexAttention` instances (one per layer) share the same compiled function, paying the compile cost only once.

- **Mask caching across layers**: The `block_masks` class-level dict is keyed by `FLEX_ATTN_MASK_T = tuple[str, int | None]` (or `tuple[int, int, int]` for the sliding-window/sink path). Since attention masks are identical across layers for a given batch, caching at the class level avoids redundant `create_block_mask` calls. Different layers can use different mask types (e.g., different `mask_key` attributes) and they will each get their own entry.

- **Per-instance `mask_key`**: Each `FlexAttention` instance stores a `mask_key` attribute (set externally by the model layer) that indexes into the shared `block_masks` dict. This lets a model with heterogeneous attention patterns (e.g., different layers using causal vs. block-causal) share the class infrastructure while still having per-layer mask selection.

### Why the sink_weights branch exists

The `forward()` method has two distinct paths:

1. **Without sink_weights**: Simple lookup of the pre-created `BlockMask` by `self.mask_key` and a single `flex_attn` call. This is the standard path.

2. **With sink_weights**: Creates/caches a mask on the fly keyed by `(sliding_window, S_q, S_kv)`, calls `flex_attn` with `return_lse=True`, then rescales output by `sigmoid(lse - w)` where `w` is a per-head learned weight. This implements "attention sinks" -- a technique for stabilizing KV cache eviction in long-context inference by giving certain positions persistent attention weight.

### Why `_compile=False` on `create_block_mask`

Line 91 of `flex_attention.py` sets `_compile=False` with an explicit note: compiling block mask creation causes hangs during sampling. This is a pragmatic workaround for a PyTorch limitation in the compile + CUDA graph interaction for mask creation.


## 3. Core Data Structures

### `FlexAttention` (class, `nn.Module`)
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/attention/flex_attention.py`, line 33

| Attribute | Type | Scope | Purpose |
|---|---|---|---|
| `flex_attn` | `ClassVar[Callable]` | Class | `torch.compile`d `flex_attention` function, shared across all instances |
| `block_masks` | `ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]]` | Class | Cache of created `BlockMask` objects keyed by mask type or `(window, S_q, S_kv)` |
| `mask_key` | `FLEX_ATTN_MASK_T` | Instance | Set externally; indexes into `block_masks` to select the mask for this layer |

### `FLEX_ATTN_MASK_T` (type alias)
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/attention/flex_attention.py`, line 28

```python
FLEX_ATTN_MASK_T = tuple[str, int | None]
```

Key type for `block_masks`. The string identifies mask type (e.g., `"causal"`, `"block_causal"`); the optional int carries auxiliary info like `fixed_block_size`. In the sink-weights path, a different key shape `tuple[int, int, int]` -- `(sliding_window, S_q, S_kv)` -- is used instead.

### `initialize_attn_module_and_func()` return type
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/attention/utils.py`, line 25

```python
def initialize_attn_module_and_func(...) -> tuple[nn.Module | None, Callable]
```

Returns `(attn_module, attn_func)`. For `te` and `flex`, `attn_module` is an `nn.Module`; for `sdpa`, it is `None` (SDPA is a pure function with no learnable parameters or state).

### `preprocess_args_and_kwargs_for_attn()` return type
**File**: `/home/scbjtfy/Automodel/nemo_automodel/components/attention/utils.py`, line 68

```python
def preprocess_args_and_kwargs_for_attn(...) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
```

Returns `(q, k, v, attn_kwargs)` where Q/K/V may have been transposed and `attn_kwargs` contains backend-specific keyword arguments ready to be unpacked into the attention callable.


## 4. State Flow

### Initialization flow (model construction time)

```
Model layer __init__()
  |
  v
initialize_attn_module_and_func(attn_impl, num_attention_heads, ...)
  |
  +--[attn_impl == "te"]--> DotProductAttention(...) --> (module, module.__call__)
  |
  +--[attn_impl == "sdpa"]--> functools.partial(F.scaled_dot_product_attention, ...) --> (None, partial_func)
  |
  +--[attn_impl == "flex"]--> FlexAttention() --> (module, module.__call__)
  |
  v
Model stores self.attn_module, self.attn_func
```

### Forward pass flow (standard, no sink weights)

```
Model layer forward(x, freqs_cis, attention_mask, **attn_kwargs)
  |
  v
Compute Q, K, V from projections, apply RoPE
  |
  v
preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask, attn_impl, **kwargs)
  |
  +--[te]: Invert attention_mask -> padding_mask, wrap as [B,1,1,S].
  |        OR detect cu_seqlens -> set qkv_format="thd", attn_mask_type="padding_causal".
  |        Pass through window_size. Return (q, k, v, attn_kwargs).
  |
  +--[sdpa/flex]: Transpose q,k,v from [B,H,S,D] -> [B,S,H,D].
  |               SDPA: set is_causal=True.
  |               Flex: pass through kwargs unchanged.
  |               Return (q_transposed, k_transposed, v_transposed, attn_kwargs).
  |
  v
self.attn_func(q, k, v, **attn_kwargs)
  |
  +--[te]: DotProductAttention.__call__(q, k, v, **attn_kwargs)
  |        Uses fused Flash/cuDNN kernels internally.
  |
  +--[sdpa]: F.scaled_dot_product_attention(q, k, v, **attn_kwargs)
  |          Dispatches to Flash/mem-efficient/math backend.
  |
  +--[flex]: FlexAttention.forward(q, k, v, **attn_kwargs)
  |          Looks up BlockMask from class cache via self.mask_key.
  |          Calls compiled flex_attention(q, k, v, block_mask=...).
  |
  v
postprocess_output_for_attn(output, attn_impl)
  |
  +--[te]: Pass through (no layout change).
  |
  +--[sdpa/flex]: Transpose [B,S,H,D] -> [B,H,S,D].
  |
  v
Model layer continues (output projection, residual, etc.)
```

### FlexAttention sink-weights flow (forward with `sink_weights != None`)

```
FlexAttention.forward(q, k, v, scale, sink_weights, sliding_window, enable_gqa)
  |
  v
Compute mask_key = (sliding_window, S_q, S_kv)
  |
  +--[not cached]--> Select mask_mod:
  |                    sliding_window > 0 ? _get_sliding_window_mask_mod(window)
  |                                      : _get_causal_mask_mod()
  |                  create_block_mask(mask_mod, B, H_q, S_q, S_kv, _compile=False)
  |                  Cache in FlexAttention.block_masks[mask_key]
  |
  v
flex_attn(q, k, v, block_mask, enable_gqa, return_lse=True) --> (out, lse)
  |
  v
scale = sigmoid(lse - sink_weights.view(1,-1,1)).unsqueeze(-1)  # [B,H,S,1]
out = out * scale
out = out.to(q.dtype)
  |
  v
Return out
```

### FlexAttention mask creation flow (for non-sink path, done externally before forward)

```
External caller (model/recipe):
  |
  v
Choose mask_mod via static methods:
  _get_causal_mask_mod()            --> q_idx >= kv_idx
  _get_block_causal_mask_mod(batch, eos_id)  --> same seq_idx AND q_idx >= kv_idx
  _get_sliding_window_mask_mod(window)       --> kv_idx <= q_idx AND q_idx - kv_idx < window
  |
  v
Optionally compose with _fixed_block_mask_mod(mask_mod, fixed_block_size):
  Restricts attention to fixed-size blocks within the sequence.
  q_block == kv_block AND inner_mask(local_q, local_kv)
  |
  v
create_block_mask(final_mask_mod, B, H, S_q, S_kv) --> BlockMask
  |
  v
FlexAttention.block_masks[(mask_type, fixed_block_size)] = block_mask
Set instance.mask_key = (mask_type, fixed_block_size)
```


## 5. Common Modification Scenarios

### Scenario 1: Adding a new attention backend

To add a fourth backend (e.g., a custom triton kernel):

1. **`utils.py` / `initialize_attn_module_and_func()`** (line 25): Add a new `elif attn_impl == "custom_triton":` branch that creates the module/function and returns `(module, func)`.

2. **`utils.py` / `preprocess_args_and_kwargs_for_attn()`** (line 68): Add an `elif attn_impl == "custom_triton":` branch that transforms Q/K/V into the layout your kernel expects and builds the appropriate kwargs dict.

3. **`utils.py` / `postprocess_output_for_attn()`** (line 149): Add the reverse layout transformation if your kernel outputs in a non-standard format. If it matches the model's native `[B,H,S,D]` like TE, just pass through.

4. Model layers do not need modification -- they already use the `attn_impl` string from config to select the backend via these utility functions.

### Scenario 2: Adding a new FlexAttention mask type

To add a new programmable mask (e.g., a "prefix-LM" mask where the first N tokens attend bidirectionally):

1. **`flex_attention.py`**: Add a new static method following the pattern of `_get_causal_mask_mod()` (line 126):
   ```python
   @staticmethod
   def _get_prefix_lm_mask_mod(prefix_length: int) -> _mask_mod_signature:
       def prefix_lm_mask(b, h, q_idx, kv_idx):
           # Bidirectional in prefix, causal after
           in_prefix = (q_idx < prefix_length) & (kv_idx < prefix_length)
           causal_after = (q_idx >= kv_idx)
           return in_prefix | causal_after
       return prefix_lm_mask
   ```

2. The caller then creates the `BlockMask` via `create_block_mask()` and stores it in `FlexAttention.block_masks` with a descriptive key like `("prefix_lm", prefix_length)`, then sets the instance's `mask_key` accordingly.

3. Optionally compose it with `_fixed_block_mask_mod()` (line 147) if you want to restrict attention to fixed-size blocks within the prefix-LM pattern.

### Scenario 3: Modifying the TE preprocessing for a new packed-sequence format

If a new dataset component emits packed sequences with a different metadata format (e.g., `offsets` instead of `cu_seqlens`):

1. **`utils.py` / `preprocess_args_and_kwargs_for_attn()`**, the `attn_impl == "te"` branch (line 73): Add an `elif "offsets" in kwargs:` block that converts `offsets` to the `cu_seqlens_q` / `cu_seqlens_kv` format expected by TE's `DotProductAttention`, and sets `qkv_format="thd"`.

2. The rest of the pipeline (postprocessing, model layer code) remains unchanged because the transformation is fully encapsulated in the preprocessing step.

### Scenario 4: Extending FlexAttention sink support to additional mask types

Currently the sink-weights path (line 71-108 in `flex_attention.py`) only supports causal and sliding-window masks. To support block-causal masks with sinks:

1. In `FlexAttention.forward()`, extend the `if sink_weights is not None` branch to accept the batch tensor and `eos_id` needed by `_get_block_causal_mask_mod()`.

2. Modify the mask_key to include the mask type (e.g., `("block_causal", sliding_window, S_q, S_kv)`) so block-causal sink masks are cached separately from causal ones.

3. After obtaining `(out, lse)` from `flex_attn`, apply the same `sigmoid(lse - w)` rescaling.

### Scenario 5: Changing tensor layout conventions

If the project moves to a different default QKV layout (e.g., `[B,S,H,D]` everywhere):

1. **`utils.py` / `preprocess_args_and_kwargs_for_attn()`**: Remove the `transpose(1,2)` calls in the SDPA and Flex branches (lines 135-137, 142-143) since the input would already be in the expected format.

2. **`utils.py` / `postprocess_output_for_attn()`**: Remove the corresponding reverse transpose for SDPA/Flex (line 152).

3. The TE branch may need a new transpose added since TE expects `[B,S,H,D]` natively but the model would now also be in that format -- so it might become a pass-through as well.
