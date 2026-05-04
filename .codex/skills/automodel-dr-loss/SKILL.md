---
name: automodel-dr-loss
description: Use when working with the loss module of automodel — loss functions including cross-entropy, knowledge distillation, chunked loss for memory efficiency
---

# Loss Module Deep Read

Source: `nemo_automodel/components/loss/` (7 files, ~1065 lines)

---

## 1. Module Purpose & Capabilities

The loss module provides a family of cross-entropy and knowledge-distillation loss functions designed for large-scale LLM/VLM training across distributed GPU topologies. Every loss class shares a common API contract -- accepting `logits` (or `hidden_states`), `labels`, an optional `mask`, and an optional `num_label_tokens` for per-token normalization -- so they can be swapped in YAML configs without changing recipe code.

**Five loss implementations are provided:**

| Class | File | Purpose |
|---|---|---|
| `MaskedCrossEntropy` | `masked_ce.py` | Standard cross-entropy with optional boolean mask, fp32 upcast, DTensor support |
| `ChunkedCrossEntropy` | `chunked_ce.py` | Memory-efficient CE that processes the sequence in fixed-size chunks via `torch.compile` |
| `FusedLinearCrossEntropy` | `linear_ce.py` | Fused lm_head linear + CE using `cut_cross_entropy` library; avoids materializing the full logits tensor |
| `TEParallelCrossEntropy` | `te_parallel_ce.py` | Tensor-parallel-aware CE using Triton kernels from TransformerEngine; handles DTensor/sharded logits |
| `KDLoss` | `kd_loss.py` | Forward KL divergence KL(P_teacher || P_student) for knowledge distillation |

**Supporting Triton kernels** live in `triton/te_cross_entropy.py`:
- `online_softmax_kernel` -- first-pass numerically-stable softmax (max + sum-of-exp) per TP rank
- `cross_entropy_kernel` -- fused second-pass: computes loss and in-place gradient simultaneously
- `element_mul_kernel` -- backward pass: elementwise multiply by upstream gradient

---

## 2. Core Design Logic

### 2a. Why chunked loss (`ChunkedCrossEntropy`)

For long-context training the logits tensor `[B, seq_len, vocab_size]` can be enormous (e.g., 128k tokens x 128k vocab = 64 GB in fp32). `ChunkedCrossEntropy` avoids holding the entire flattened logits in memory by splitting along the sequence dimension into chunks of `chunk_len` (default 32) tokens. Each chunk is processed independently through a `torch.compile`-d `compute_cross_entropy` kernel (see `chunked_ce.py:96-104`). The compiled kernel is cached in the module-level global `_compiled_compute_cross_entropy` and reused across calls.

### 2b. Why fused linear + CE (`FusedLinearCrossEntropy`)

`FusedLinearCrossEntropy` never materializes the full `[B*T, V]` logits matrix. Instead it delegates to Apple's `cut_cross_entropy.linear_cross_entropy`, which fuses the linear projection (`hidden_states @ lm_weight.T`) with the cross-entropy computation in a single Triton kernel pass. This dramatically reduces peak memory. The class requires a **different forward signature** than other losses: it takes `hidden_states` and `lm_weight` instead of `logits`. Recipe-level dispatch in `calculate_loss()` (`recipes/llm/train_ft.py:690-723`) handles this by extracting `model.get_output_embeddings().weight` and passing `hidden_states` from the model output.

A monkey-patch (`linear_ce.py:113-116`) replaces `cut_cross_entropy.tl_utils.is_triton_greater_or_equal` with a version that checks `pytorch-triton` (the NVIDIA fork) rather than upstream `triton`, ensuring compatibility in container environments.

### 2c. Why Triton-based parallel CE (`TEParallelCrossEntropy`)

When tensor parallelism shards the vocabulary dimension across ranks, standard `F.cross_entropy` would require an all-gather of the full logits. `TEParallelCrossEntropy` avoids this by using an online-softmax approach adapted from TransformerEngine:

1. Each TP rank computes local `(max, sum_exp, X_y)` via `online_softmax_kernel` (one pass over the local vocab shard).
2. An `all_gather_into_tensor` collects these 3-element summaries from all ranks (only 3 floats per token per rank, not the full logits).
3. `cross_entropy_kernel` merges the summaries and computes both the loss and the in-place gradient in a single fused pass.

This reduces the all-gather volume from `O(V)` to `O(3 * world_size)` per token.

The `TEParallelCrossEntropy.__call__` method (`te_parallel_ce.py:132-190`) also handles **DTensor interop**: if `logits` is a `DTensor` sharded on the vocab dimension, it auto-extracts the local tensor and infers the TP process group from the DTensor's `device_mesh`.

### 2d. Why KD loss is separate

`KDLoss` (`kd_loss.py:20-100`) implements forward KL divergence and has a fundamentally different interface: it takes both `student_logits` and `teacher_logits`, plus `labels` for masking. It is composed with a CE loss in the KD recipe (`recipes/llm/kd.py:249`): `local_loss = (1 - kd_ratio) * ce_loss + kd_ratio * kd_loss`.

### 2e. Shared design patterns across all losses

- **`ignore_index=-100`**: Every loss respects PyTorch's standard ignore index for padding tokens.
- **Mask-to-ignore conversion**: When a boolean `mask` is provided, `masked_fill_` sets masked positions to `ignore_index` in-place (`masked_ce.py:71`, `chunked_ce.py:92`, `te_parallel_ce.py:173`), converting the mask into label-space so the underlying CE kernel can use its native ignore_index path.
- **`num_label_tokens` normalization**: All CE losses support an optional `num_label_tokens` parameter. When provided with `reduction="sum"`, the sum-reduced loss is divided by this count. This enables correct per-token loss averaging across gradient accumulation steps and DP ranks, where the recipe pre-computes the total valid token count via `_dp_allreduce`.
- **`fp32_upcast`**: `MaskedCrossEntropy` and `KDLoss` upcast logits to fp32 before computing loss for numerical stability. The Triton kernel in `te_cross_entropy.py` also performs its math in fp32 (`tl.float32` casts on lines 87, 101, 204).
- **Device mismatch handling**: Multiple losses check `labels.device != logits.device` and move labels, handling the case where `CPUOffloadPolicy` places tensors on different devices.

---

## 3. Core Data Structures

### `MaskedCrossEntropy(nn.Module)` -- `nemo_automodel/components/loss/masked_ce.py:22`

```
__init__(fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = "sum")
forward(logits: Tensor[B, T, V], labels: Tensor[B, T], mask: Optional[Tensor], num_label_tokens: Optional[int]) -> Tensor
```

The baseline loss used by most recipes. Handles DTensor logits by calling `logits.full_tensor()` (`masked_ce.py:77`), making it safe to use when TP plans set `use_local_output=False`. This is the **fallback loss** -- if a model does not support `logits_to_keep`, the recipe automatically replaces the configured loss with `MaskedCrossEntropy` (`train_ft.py:221-223`).

### `ChunkedCrossEntropy(nn.Module)` -- `nemo_automodel/components/loss/chunked_ce.py:43`

```
__init__(chunk_len: int = 32, compile: bool = True, ignore_index: int = -100, reduction: str = "sum")
forward(logits: Tensor[B, T, V], labels: Tensor[B, T], mask: Optional[Tensor], num_label_tokens: Optional[int]) -> Tensor
```

Uses the module-level helper `compute_cross_entropy()` (`chunked_ce.py:23-40`), which is a thin wrapper around `F.cross_entropy` on flattened 2D tensors. The compiled version is stored in the global `_compiled_compute_cross_entropy` (`chunked_ce.py:20`).

### `FusedLinearCrossEntropy(nn.Module)` -- `nemo_automodel/components/loss/linear_ce.py:119`

```
__init__(ignore_index: int = -100, logit_softcapping: float = 0, reduction: str = "sum")
forward(hidden_states: Tensor, labels: Tensor, lm_weight: Tensor, num_label_tokens: Optional[int]) -> Tensor
```

Distinct forward signature: takes `hidden_states` (last hidden layer output) and `lm_weight` (the lm_head weight matrix) instead of pre-computed logits. Requires the optional `cut_cross_entropy` package. Supports `logit_softcapping` (Gemma-style tanh capping, 0 = disabled). The recipe-level `calculate_loss()` function (`train_ft.py:690-723`) extracts `lm_weight` by calling `model.get_output_embeddings().weight` and unshards it via `.full_tensor()`.

### `TEParallelCrossEntropy` -- `nemo_automodel/components/loss/te_parallel_ce.py:113`

```
__init__(ignore_index: int = -100, reduction: str = "sum", tp_group: Optional[ProcessGroup] = None)
__call__(logits: Tensor[B, T, V], labels: Tensor[B, T], mask: Optional[Tensor], num_label_tokens: Optional[int]) -> Tensor
```

Not an `nn.Module` -- it is a plain class with `__call__` (no learnable parameters). Wraps `CrossEntropyFunction` (`te_parallel_ce.py:46-107`), a custom `torch.autograd.Function` that calls the Triton kernels for forward and backward. The `tp_group` parameter can be explicitly provided or auto-inferred from DTensor placements (`te_parallel_ce.py:157-163`).

### `KDLoss(nn.Module)` -- `nemo_automodel/components/loss/kd_loss.py:20`

```
__init__(ignore_index: int = -100, temperature: float = 1.0, fp32_upcast: bool = True)
forward(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor, num_batch_labels: Optional[int]) -> Tensor
```

Computes `KL(P_teacher || P_student)`. Key details:
- Filters out padding tokens using `valid_mask = (labels != ignore_index)` before any computation (`kd_loss.py:60`).
- Applies temperature scaling via in-place `mul_(1/temperature)` on fp32-upcast logits (`kd_loss.py:80-82`).
- Masks out infinities from student logits to avoid NaN gradients (`kd_loss.py:91-94`): `torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)`.
- Returns zero tensor when the entire batch is padding (`kd_loss.py:63`).
- `num_batch_labels` controls normalization: when `None`, returns `mean(kl_per_token)`; when provided, returns `sum(kl_per_token) / num_batch_labels` for gradient-accumulation-correct averaging.

### `CrossEntropyFunction(torch.autograd.Function)` -- `nemo_automodel/components/loss/te_parallel_ce.py:46`

The autograd glue between `TEParallelCrossEntropy` and the Triton kernels. Forward calls `cross_entropy_forward()`, which launches `online_softmax_kernel` then `cross_entropy_kernel`. The kernel writes gradients **in-place into the input tensor** (`_input`), which is saved via `ctx.save_for_backward`. The backward pass calls `cross_entropy_backward()`, which multiplies by `grad_output` using `element_mul_kernel` (or skips if `grad_output == 1.0`).

### Triton Kernels -- `nemo_automodel/components/loss/triton/te_cross_entropy.py`

- `cross_entropy_forward(_input, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx)` -- Python orchestrator at line 291. Reshapes `[B, SQ, V]` input, allocates `loss_1d` and `m_d_X_y` buffers, launches kernels, handles `all_gather_into_tensor` for multi-rank. Returns `(loss, _input)` where `_input` now contains the gradient.
- `cross_entropy_backward(_input, grad_output)` -- Python orchestrator at line 373. Multiplies saved gradient by upstream `grad_output` via `element_mul_kernel`.
- `MAX_FUSED_SIZE = 65536 // 2` -- maximum Triton block size (`te_cross_entropy.py:253`).
- Modified from original TransformerEngine: uses `n_non_ignore = (target != ignore_idx).sum()` for correct mean-reduction when ignore_idx tokens are present (`te_cross_entropy.py:345`, `te_cross_entropy.py:368`).

---

## 4. State Flow

### Standard CE path (MaskedCrossEntropy / ChunkedCrossEntropy)

```
Model forward
  -> out.logits: [B, T, V]
  -> labels: [B, T]
       |
       v
calculate_loss() [train_ft.py:680]
  -> logits.view(-1, V), labels.view(-1)
  -> if mask: labels.masked_fill_(mask==0, -100)
  -> if fp32_upcast: logits = logits.float()
  -> F.cross_entropy(logits, labels, reduction="sum")
  -> if num_label_tokens: loss /= num_label_tokens
  -> return scalar loss
```

For `ChunkedCrossEntropy`, the flattened `[B*T, V]` tensor is split into chunks of `chunk_len` tokens, each independently processed through the compiled `compute_cross_entropy`, and losses are accumulated.

### Fused Linear CE path

```
Model forward(logits_to_keep=1)
  -> out.hidden_states[-1]: [B, T, H]   (last hidden layer)
  -> labels: [B, T]
       |
       v
calculate_loss() [train_ft.py:690]
  -> lm_weight = model.get_output_embeddings().weight.full_tensor()
  -> linear_cross_entropy(hidden_states, lm_weight, targets=labels, ...)
     [fused: hidden @ weight.T + CE in single Triton pass, never materializes [B*T, V]]
  -> if num_label_tokens: loss /= num_label_tokens
  -> return scalar loss
```

### TE Parallel CE path (tensor-parallel)

```
Model forward (TP-sharded)
  -> logits: DTensor[B, T, V/tp] or Tensor[B, T, V/tp]
  -> labels: [B, T]
       |
       v
TEParallelCrossEntropy.__call__()
  -> if DTensor: logits = logits.to_local(), infer tp_group from device_mesh
  -> if mask: labels.masked_fill_(mask==0, -100)
       |
       v
CrossEntropyFunction.forward()
  -> cross_entropy_forward():
       1. online_softmax_kernel: each rank computes (max, sum_exp, X_y) for local V/tp shard
       2. all_gather_into_tensor(m_d_X_y) across tp_group  [3 floats/token/rank]
       3. cross_entropy_kernel: merges summaries, computes loss + writes gradient in-place into _input
       -> returns (loss_1d, _input_with_grad)
  -> if reduce_loss: loss = sum(loss_1d) / n_non_ignore
  -> else: loss = loss_1d.reshape(B, SQ)
       |
       v
  -> if reduction=="sum": loss = loss.sum(); if num_label_tokens: loss /= num_label_tokens
  -> return scalar loss

CrossEntropyFunction.backward()
  -> cross_entropy_backward():
       if grad_output != 1.0: element_mul_kernel(saved_input, grad_output)
       -> returns gradient w.r.t. input logits
```

### KD Loss path (knowledge distillation recipe)

```
Student forward -> student_logits: [B, T, V]
Teacher forward -> teacher_logits: [B, T, V]
labels: [B, T]
       |
       v
KDLoss.forward()
  -> valid_mask = (labels != -100)
  -> filter to valid tokens only
  -> if fp32_upcast: cast to float32
  -> if temperature != 1: scale by 1/T in-place
  -> teacher_prob = softmax(t_logits)
  -> student_logprob = log_softmax(s_logits)
  -> inf_mask = isinf(s_logits)
  -> kl_per_token = masked_fill(teacher_prob * student_logprob, inf_mask, 0).sum(-1)
  -> if num_batch_labels: return -sum(kl) / num_batch_labels
  -> else: return -mean(kl)

Combined in KD recipe [kd.py:249]:
  local_loss = (1 - kd_ratio) * ce_loss + kd_ratio * kd_loss
```

### num_label_tokens computation (recipe level)

```
Recipe._run_train_optim_step() [train_ft.py:1224-1227]:
  num_label_tokens = sum((batch["labels"] != -100).sum() for batch in batches)
  num_label_tokens = dp_allreduce(num_label_tokens)  # sync across all DP ranks
  -> passed to loss_fn via calculate_loss()
```

This ensures consistent per-token averaging across data-parallel ranks and gradient accumulation micro-batches.

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new loss function (e.g., focal loss)

1. Create `nemo_automodel/components/loss/focal_ce.py` with a class inheriting `nn.Module`.
2. Implement `forward(self, logits, labels, mask=None, num_label_tokens=None)` following the same signature as `MaskedCrossEntropy.forward()` (`masked_ce.py:38-86`).
3. Include the `num_label_tokens` normalization guard: `if num_label_tokens is not None: assert self.reduction == "sum"; loss = loss / num_label_tokens`.
4. Reference it in YAML config: `loss_fn: { _target_: nemo_automodel.components.loss.focal_ce.FocalCrossEntropy }`.
5. No recipe changes needed -- `calculate_loss()` (`train_ft.py:715-721`) will pass `logits` and `labels` to any loss that is not `FusedLinearCrossEntropy`.

### Scenario 2: Adding logit softcapping to MaskedCrossEntropy

To add Gemma-style logit capping to the standard CE path:

1. Add a `logit_softcapping` parameter to `MaskedCrossEntropy.__init__()` (`masked_ce.py:23`), defaulting to `0.0`.
2. In `forward()`, after the fp32 upcast block (`masked_ce.py:73-74`), insert: `if self.logit_softcapping: logits = logits / self.logit_softcapping; logits = torch.tanh(logits); logits = logits * self.logit_softcapping`.
3. The rest of the flow (mask application, `F.cross_entropy`, `num_label_tokens` normalization) remains unchanged.
4. Update YAML configs to set the new parameter.

### Scenario 3: Switching a recipe from MaskedCrossEntropy to FusedLinearCrossEntropy

1. Change the YAML config from:
   ```yaml
   loss_fn:
     _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
   ```
   to:
   ```yaml
   loss_fn:
     _target_: nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
   ```
2. Ensure the model supports `logits_to_keep` in its `forward()` method. The recipe checks this via `_supports_logits_to_keep(model)` (`train_ft.py:221`); if missing, it auto-falls back to `MaskedCrossEntropy`.
3. Ensure the model config includes `output_hidden_states: True` so the recipe can extract `out.hidden_states[-1]` (`train_ft.py:1194-1199`).
4. Install the `cut_cross_entropy` package: `pip install cut-cross-entropy`.

### Scenario 4: Using TEParallelCrossEntropy with tensor parallelism

1. Set the loss in YAML:
   ```yaml
   loss_fn:
     _target_: nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy
   ```
2. If the TP plan sets `use_local_output=False` on the output projection, logits will arrive as a `DTensor` sharded on the vocab dimension. `TEParallelCrossEntropy.__call__()` (`te_parallel_ce.py:157-164`) automatically detects this, extracts the local shard via `.to_local()`, and infers the TP process group from the DTensor's `device_mesh`.
3. Alternatively, pass `tp_group` explicitly in YAML or at construction time.
4. Requires `triton` to be installed (checked via `HAVE_TRITON` flag, `te_cross_entropy.py:36-38`).

### Scenario 5: Adjusting KD temperature and loss mixing ratio

1. In the KD YAML config (`examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml`):
   ```yaml
   kd_loss_fn:
     _target_: nemo_automodel.components.loss.kd_loss.KDLoss
     temperature: 2.0    # default is 1.0
   kd_ratio: 0.7         # weight of KD loss vs CE loss
   ```
2. The KD recipe (`kd.py:249`) computes `local_loss = (1 - kd_ratio) * ce_loss + kd_ratio * kd_loss`.
3. Higher temperature softens the teacher distribution, making the KD signal less peaked. The temperature is applied as in-place `mul_(1/temperature)` before softmax/log_softmax (`kd_loss.py:80-82`).
4. `fp32_upcast` (default `True`) can be set to `False` if memory is critical and bf16 stability is acceptable.
