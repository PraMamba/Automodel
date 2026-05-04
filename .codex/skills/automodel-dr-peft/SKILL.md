---
name: automodel-dr-peft
description: Use when working with the PEFT module of automodel — LoRA adapter integration with FSDP2 and tensor parallel compatibility
---

# PEFT Module -- LoRA Adapter Integration

## 1. Module Purpose & Capabilities

The `_peft` module (`/home/scbjtfy/Automodel/nemo_automodel/components/_peft/`) provides LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of LLMs and VLMs. It supports:

- **Standard LoRA** on any `nn.Linear` layer, including TransformerEngine linear layers.
- **DoRA** (Weight-Decomposed Low-Rank Adaptation), which adds a learnable magnitude vector on top of LoRA for `nn.Linear` layers only.
- **QLoRA** via BitsAndBytes quantized base weights with full-precision LoRA adapters.
- **MoE LoRA** for Mixture-of-Experts architectures, with both standard grouped-GEMM (`GroupedExpertsLoRA`) and DeepEP-based (`GroupedExpertsDeepEPLoRA`) variants.
- **Triton-accelerated LoRA** kernels that fuse the `x @ A @ B` computation into a single kernel for forward and backward passes.
- **Flexible module targeting** via exact name matching, wildcard glob patterns, match-all-linear mode, or exclude-list mode.
- **HF PEFT-compatible checkpoint format**: saves `adapter_model.safetensors` + `adapter_config.json` so adapters load with HuggingFace's `peft` library.

The module is consumed by:
- `nemo_automodel/_transformers/auto_model.py` -- the `apply_model_infrastructure` function calls `apply_lora_to_linear_modules` after model instantiation but before FSDP2/TP sharding.
- `nemo_automodel/components/checkpoint/checkpointing.py` and `addons.py` -- PEFT-aware save/load logic that writes only adapter weights on rank 0 and broadcasts on load.
- `nemo_automodel/recipes/biencoder/train_biencoder.py` -- direct import for biencoder training.

### Files (5 files, 1844 lines total)

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 0 | Empty package marker |
| `module_matcher.py` | 115 | Module selection logic (which layers get LoRA) |
| `lora.py` | 605 | Core LoRA classes, patching, and orchestration entry point |
| `lora_kernel.py` | 588 | Triton kernels for fused LoRA forward/backward |
| `lora_moe.py` | 536 | MoE-specific LoRA wrappers (GroupedExperts, DeepEP) |

---

## 2. Core Design Logic

### Why monkey-patching instead of wrapper modules

The central design decision is that `patch_linear_module()` (in `lora.py`, line 331) mutates `nn.Linear` modules **in-place** by changing their `__class__` to a dynamically-created subclass (`PatchedLinearLoRA`), rather than replacing them with wrapper modules. This preserves:

1. **FQN (Fully Qualified Name) stability**: The `weight` and `bias` parameters remain at the same FQN in the model's state dict. This is critical because FSDP2 and TP sharding plans reference modules by FQN. If LoRA replaced a module, all downstream sharding and checkpoint logic would break.
2. **Meta-device compatibility**: When models are instantiated on meta device (for large models that do not fit in CPU memory), weight data cannot be copied. The static method `LinearLoRA._init_adapter()` (line 148) adds `lora_A` and `lora_B` sub-modules to the existing object without needing valid weight tensors. The `peft_ctx = init_empty_weights()` context in `auto_model.py` (line 622) ensures LoRA parameters are also on meta device.
3. **Quantized weight support**: When the original linear uses BitsAndBytes quantized weights (`quant_state` attribute), the monkey-patch stores the original `forward` as `super_fwd` (line 408) so the quantized forward path is preserved while LoRA is added on top.

### Why LoRA scale is applied before lora_B

In `LinearLoRA.forward()` (line 242), the scale factor is multiplied between `lora_A` and `lora_B`:
```python
lora_res = self.lora_B(self.lora_A(x) * self.scale)
```
This is intentional for DTensor/FSDP2 compatibility. The comment at line 239 explains: when TP is active, `lora_A(x)` produces a `Partial` tensor. Multiplying by a scalar keeps it `Partial`. If scale were applied after `lora_B`, it would trigger an implicit redistribution (Partial to Replicate), causing an extra reduce-scatter. By scaling in the middle, both `res` and `lora_res` remain `Partial`, requiring only one reduce-scatter after their addition.

### Why MoE LoRA duplicates parent forward logic

`GroupedExpertsLoRA.forward()` (lora_moe.py, line 225) and `GroupedExpertsDeepEPLoRA.forward()` (line 437) fully duplicate their parent class forward methods rather than calling `super().forward()`. This is because the base `GroupedExperts.forward()` does not expose hooks for injecting LoRA into the inner expert computation (between the gate/up projection and down projection). The LoRA injection must happen at the point where `x @ W` is computed, so the forward logic is replicated with LoRA additions interleaved.

### Triton kernel design: exploiting small LoRA rank

The Triton forward kernel `lora_forward_kernel` (lora_kernel.py, line 182) computes `D = X @ A @ B` in a single fused kernel by exploiting the fact that the LoRA rank `N` is very small (typically 8-64). The heuristic at line 180 sets `BLOCK_SIZE_N` to `next_power_of_2(N)` (minimum 16), so the entire LoRA-rank dimension fits in a single block. The `inner_kernel` computes `X @ A` into registers, then `block_vector_mul` multiplies by `B` and writes the result, avoiding materialization of the intermediate `M x N` matrix in global memory.

### Adapter init_lora_weights as a framework hook

Both `GroupedExpertsLoRA.init_lora_weights()` (lora_moe.py, line 204) and `GroupedExpertsDeepEPLoRA.init_lora_weights()` (line 416) carry a docstring warning: "This method is called by the PEFT framework's `_init_peft_adapters` after the model is materialized from meta device." The function `_init_peft_adapters()` in `checkpointing.py` (line 824) iterates all modules checking `hasattr(module, "init_lora_weights")` and calling it. This means any module with an `init_lora_weights` method will be re-initialized after meta-device materialization, which is the mechanism that ensures LoRA weights get proper random initialization even when the model was originally on meta device.

---

## 3. Core Data Structures

### PeftConfig (lora.py, line 42)

Dataclass that holds all LoRA hyperparameters. Created from YAML config via Hydra's `_target_` mechanism.

```
@dataclass
class PeftConfig:
    target_modules: list          # Module names or wildcard patterns (e.g., ["q_proj", "v_proj"])
    exclude_modules: list         # Module names to exclude
    match_all_linear: bool        # If True, targets all nn.Linear / TE Linear layers
    dim: int = 8                  # LoRA rank (r)
    alpha: int = 32               # LoRA scaling factor (alpha)
    use_dora: bool = False        # Enable DoRA (magnitude + direction decomposition)
    dropout: float = 0.0          # Dropout probability on LoRA path
    dropout_position: "pre"|"post" = "post"  # Apply dropout before or after LoRA
    lora_A_init: str = "xavier"   # Initialization: "xavier" or "uniform" (kaiming)
    lora_dtype: Optional[torch.dtype] = None  # Override dtype for LoRA weights (needed for QLoRA)
    use_triton: bool = False      # Use Triton-fused LoRA kernels
```

- `to_dict()` / `from_dict()` for serialization.
- Serialized to two JSON files at checkpoint time: `adapter_config.json` (HF-compatible subset: `r`, `lora_alpha`, `use_dora`, `target_modules`, `task_type`) and `automodel_peft_config.json` (remaining fields like `dropout`, `lora_A_init`, etc.). See `PeftAddon.pre_save()` in `/home/scbjtfy/Automodel/nemo_automodel/components/checkpoint/addons.py` line 126.

### ModuleMatcher (module_matcher.py, line 49)

Dataclass that encapsulates the matching logic for deciding which modules receive LoRA adapters.

```
@dataclass
class ModuleMatcher:
    target_modules: List[str]     # Patterns to match (exact name or wildcard)
    exclude_modules: List[str]    # Patterns to exclude
    match_all_linear: bool        # Match any nn.Linear or TE Linear
    is_causal_lm: bool            # (Unused currently, reserved for future task-aware matching)
```

- `match(m, name, prefix)` returns `True` if the module should receive LoRA.
- Three matching modes in priority order:
  1. `match_all_linear=True` -- matches any `nn.Linear` or `transformer_engine.pytorch.Linear` via `_is_linear_module()` (line 26).
  2. `target_modules` non-empty -- matches by exact `name == pattern` or `wildcard_match(pattern, full_name)` where wildcards use `*` (converted to regex `(.*)`) via `wildcard_match()` (line 30).
  3. `exclude_modules` only -- matches all linear modules except those in the exclude list (fallback mode).
- `target_modules` and `exclude_modules` are mutually exclusive (asserted at line 104).

### LinearLoRA (lora.py, line 76)

Subclass of `nn.Linear` that adds LoRA `lora_A` and `lora_B` sub-modules. Used both as a class for direct construction and as the base for monkey-patching.

Key attributes added by `_init_adapter()`:
- `lora_A`: `nn.Linear(in_features, dim, bias=False)` -- projects down to LoRA rank
- `lora_B`: `nn.Linear(dim, out_features, bias=False)` -- projects back up
- `scale`: `alpha / dim` (float)
- `dim`, `dropout_p`, `dropout_position`, `use_dora`
- `lora_magnitude`: `nn.Parameter` (only when `use_dora=True`) -- per-output-dimension magnitude vector initialized to `||W||` row-wise L2 norm (line 199)

### TritonLinearLoRA (lora.py, line 288)

Subclass of `LinearLoRA` that overrides `forward()` to use `LoRATritonFunction.apply()` (line 324) which calls the fused Triton kernels. Does not support DoRA.

### LoRATritonFunction (lora.py, line 550)

Custom `torch.autograd.Function` with `setup_context`, `forward`, and `backward` static methods. Forward calls `lora_forward_wrapper` (lora_kernel.py, line 257). Backward calls `lora_da_dx_update_wrapper` (line 419) for `d_lora_A` and `d_x`, and `lora_db_update_wrapper` (line 549) for `d_lora_B`.

### GroupedExpertsLoRA (lora_moe.py, line 116)

Subclass of `GroupedExperts` from `nemo_automodel/components/moe/layers.py`. Adds four 3D LoRA parameter tensors:
- `lora_gate_and_up_A`: `[n_routed_experts, dim, lora_dim]`
- `lora_gate_and_up_B`: `[n_routed_experts, lora_dim, moe_inter_dim * 2]`
- `lora_down_A`: `[n_routed_experts, moe_inter_dim, lora_dim]`
- `lora_down_B`: `[n_routed_experts, lora_dim, dim]`

The forward pass iterates over experts and dispatches tokens to `expert_activation_with_lora`, which is either `swiglu_with_lora` (line 30) or `quick_geglu_with_lora` (line 64), both implementing `out = (x @ W) + (x @ A @ B) * scale` inline.

### GroupedExpertsDeepEPLoRA (lora_moe.py, line 331)

Subclass of `GroupedExpertsDeepEP`. Same LoRA parameter structure as `GroupedExpertsLoRA` but uses `ops.gmm` (grouped GEMM from the `grouped_gemm` package) for batched matrix multiplications and `token_dispatcher.token_permutation2` / `token_unpermutation` for DeepEP-style all-to-all communication. The forward pass (line 437) interleaves LoRA additions between the base grouped-GEMM calls.

---

## 4. State Flow

### Config to Adapter Application Flow

```
YAML config (peft section)
    |
    v
Hydra _target_ instantiation --> PeftConfig dataclass
    |
    v
NeMoAutoModelForCausalLM.from_pretrained() / from_config()
    calls apply_model_infrastructure() [auto_model.py:531]
        |
        v
    _apply_peft_and_lower_precision() [auto_model.py:455]
        |-- disables Triton if TP > 1 or Pipeline Parallelism is active
        |-- calls apply_lora_to_linear_modules(model, peft_config) [lora.py:459]
            |
            v
        1. Freeze all base model parameters (requires_grad=False) [line 479]
        2. Detect is_causal_lm from model.config.architectures [line 483]
        3. Create ModuleMatcher from peft_config fields [line 495]
        4. Iterate model.named_modules():
           - If GroupedExperts or GroupedExpertsDeepEP and matcher matches:
             --> patch_moe_module() [line 416] creates GroupedExpertsLoRA
                 or GroupedExpertsDeepEPLoRA, replaces module in parent
           - If nn.Linear and matcher matches:
             --> patch_linear_module() [line 331] mutates __class__ in-place
                 to PatchedLinearLoRA (dynamic subclass of TritonLinearLoRA
                 or LinearLoRA + original class)
        5. Return count of matched modules
```

### FSDP2/TP Sharding After PEFT

PEFT application happens **before** FSDP2/TP sharding in `apply_model_infrastructure()`. The sequence is:
1. PEFT adapters added (on meta device if applicable, via `init_empty_weights()` context)
2. `pre_shard_hf_state_dict_keys` captured
3. EP/FSDP sharding or Pipeline Parallelism applied
4. If meta device: materialize to real device, load base checkpoint, call `_init_peft_adapters()` to re-initialize LoRA weights

This ordering means LoRA parameters participate in FSDP2 sharding automatically since they are regular `nn.Parameter`s on sub-modules when `fully_shard()` is called.

### Checkpoint Save Flow (PEFT-specific)

```
Checkpointer.save() [checkpointing.py]
    |
    v
PeftAddon.pre_save() [addons.py:126]
    |-- Rank 0 only:
    |   - Writes adapter_config.json (HF-compatible: task_type, peft_type="LORA",
    |     r=dim, lora_alpha=alpha, use_dora, target_modules, base_model_name_or_path)
    |   - Writes automodel_peft_config.json (all PeftConfig fields except dim/alpha)
    |   - Saves tokenizer
    |
    v
_do_save() [checkpointing.py:529]
    |-- PEFT + model save:
    |   - Rank 0 writes adapter_model.safetensors via safetensors.torch.save_file()
    |   - All ranks barrier
    |-- Non-PEFT or optimizer:
    |   - Standard DCP save
```

### Checkpoint Load Flow (PEFT-specific)

```
_do_load() [checkpointing.py:505]
    |-- PEFT + model + not init step:
    |   - Rank 0 reads adapter_model.safetensors via safetensors.torch.load_file()
    |   - (Broadcast implied by distributed setup)
    |-- Otherwise:
    |   - Standard DCP load
```

### Meta-Device Materialization Flow

When the model starts on meta device (common for large models):
1. `apply_model_infrastructure` wraps PEFT application in `init_empty_weights()` context
2. After sharding, `checkpointer.load_base_ckpt()` materializes weights
3. `_init_peft_adapters()` (checkpointing.py, line 824) walks all modules and calls `init_lora_weights(peft_init_method)` on any module that has this method, re-initializing LoRA A with xavier/kaiming and LoRA B with zeros

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new LoRA initialization method

To add a new initialization scheme (e.g., Gaussian with specific variance):

1. In `LinearLoRA.init_lora_weights()` (`lora.py`, line 134), add a new `elif` branch:
   ```python
   elif init_method == "gaussian":
       nn.init.normal_(self.lora_A.weight.data, std=0.02)
   ```
2. Mirror the same change in `GroupedExpertsLoRA.init_lora_weights()` (`lora_moe.py`, line 204) and `GroupedExpertsDeepEPLoRA.init_lora_weights()` (`lora_moe.py`, line 416) for MoE LoRA.
3. The `PeftConfig.lora_A_init` field (`lora.py`, line 52) already accepts a string, so no config change is needed -- just pass the new name via YAML.

### Scenario 2: Supporting LoRA on a new non-Linear module type

To apply LoRA to a custom module type (e.g., `nn.Embedding`):

1. Extend `_is_linear_module()` in `module_matcher.py` (line 26) or add a separate `_is_embedding_module()` check.
2. In `apply_lora_to_linear_modules()` (`lora.py`, line 459), add a new branch in the `for name, module` loop (currently line 499-545) that handles `nn.Embedding` instances, similar to the `GroupedExperts` branch.
3. Create a new patching function (analogous to `patch_linear_module`) that adds `lora_A` and `lora_B` to the embedding, adjusting shapes since embeddings project `vocab_size -> embed_dim` (the "A" matrix would be `embed_dim -> rank`, "B" would be `rank -> embed_dim`).
4. Ensure the new module has an `init_lora_weights()` method so `_init_peft_adapters()` in `checkpointing.py` (line 824) can re-initialize it after meta-device materialization.

### Scenario 3: Adding a new MoE activation function for LoRA

To support a new expert activation (beyond `swiglu` and `quick_geglu`) with LoRA:

1. Create a new function in `lora_moe.py` following the pattern of `swiglu_with_lora()` (line 30) or `quick_geglu_with_lora()` (line 64). The function must accept `lora_gate_and_up_A`, `lora_gate_and_up_B`, `lora_down_A`, `lora_down_B`, and `scale` kwargs and inject `(x @ A @ B) * scale` at the appropriate points in the activation computation.
2. Register it in `get_expert_activation_with_lora()` (`lora_moe.py`, line 104) with a new `elif` branch matching `config.expert_activation`.
3. The `GroupedExpertsLoRA._init_adapter()` (line 160) calls `get_expert_activation_with_lora(obj.config)` and stores the result as `obj.expert_activation_with_lora`, so no further changes are needed in the LoRA class itself.

### Scenario 4: Adjusting the target modules for a specific model architecture

To target only attention QKV projections and skip MLP layers for a HuggingFace model:

1. In YAML config, set specific `target_modules` in the `peft` section:
   ```yaml
   peft:
     _target_: nemo_automodel.components._peft.lora.PeftConfig
     target_modules: ["q_proj", "k_proj", "v_proj"]
     dim: 16
     alpha: 32
   ```
2. For layer-specific targeting, use wildcard patterns:
   ```yaml
   target_modules:
     - "*.layers.0.*.q_proj"
     - "*.layers.0.*.k_proj"
   ```
   The `wildcard_match()` function in `module_matcher.py` (line 30) converts `*` to `(.*)` regex and matches against the full dotted module name (e.g., `model.layers.0.self_attn.q_proj`).

### Scenario 5: Disabling Triton kernels for debugging

The Triton kernels are automatically disabled when TP > 1 or Pipeline Parallelism is active (in `_apply_peft_and_lower_precision()`, `auto_model.py` lines 459-465). They are also disabled for TransformerEngine linear layers (`patch_linear_module()`, `lora.py` line 373) and DoRA (`lora.py` line 383). To force-disable them in all cases, set `use_triton: False` in the YAML peft config. The selection between `TritonLinearLoRA` and `LinearLoRA` is made at line 385 in `patch_linear_module()`.
