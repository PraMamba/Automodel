# PR Review Task Templates

Review task templates organized by subsystem. Each template specifies model level and
review focus areas.

## Framework-Specific Templates (Opus)

### Distributed Parallelizer Review

**Model**: Opus
**Focus**: `nemo_automodel/components/distributed/parallelizer.py`
**Checklist**:
- DTensor placements correct for all parallel dimensions
- DeviceMesh construction matches expected topology
- Gradient sync across correct process groups
- Memory accounting for sharded vs replicated parameters
- Correct wrapping order for nested parallelism (TP inside FSDP)

### FSDP2 Core Review

**Model**: Opus
**Focus**: `nemo_automodel/components/distributed/fsdp2.py`
**Checklist**:
- FSDP2 wrap policy covers all model layers
- Mixed precision config consistent with model dtypes
- Activation checkpointing applied to correct layers
- State dict type (full vs sharded) matches save/load expectations
- Grad scaler interaction with mixed precision

### MegatronFSDP Review

**Model**: Opus
**Focus**: `nemo_automodel/components/distributed/megatron_fsdp.py`
**Checklist**:
- Pipeline stage boundaries correctly defined
- Communication between stages uses correct tensors
- Gradient accumulation across micro-batches
- Memory balance across pipeline stages

### Checkpoint Correctness Review

**Model**: Opus
**Focus**: `nemo_automodel/components/checkpoint/`
**Checklist**:
- State dict keys match between save and load
- Tensor shapes correct after resharding (different TP/DP)
- Optimizer state saved/loaded correctly
- Dataloader state (position) preserved
- Atomic write for crash safety

### MoE Layer Review

**Model**: Opus
**Focus**: `nemo_automodel/components/moe/`
**Checklist**:
- Router produces valid top-K selections
- Expert dispatch distributes tokens correctly across EP group
- Auxiliary load-balancing loss included
- State dict handles expert-parallel sharding
- FSDP wrapping policy correct for expert layers

### Tensor Parallel Plan Review

**Model**: Opus
**Focus**: `nemo_automodel/components/distributed/optimized_tp_plans.py`
**Checklist**:
- All linear layers in model have TP plan entries
- Column-wise vs row-wise parallelism matches layer semantics
- Embedding and output layers handle vocab parallel correctly
- Plan keys match actual model parameter names

## General Templates

### Model Architecture Review

**Model**: Opus
**Focus**: `nemo_automodel/components/models/*/model.py`
**Checklist**:
- Forward pass shape consistency (`[batch, seq, hidden]`)
- Attention mask handling correct
- RoPE / positional encoding applied correctly
- State dict adapter maps all keys (no orphaned parameters)
- Factory function signature matches YAML config expectations

### Recipe Training Loop Review

**Model**: Opus
**Focus**: `nemo_automodel/recipes/`
**Checklist**:
- Gradient accumulation steps computed correctly
- Loss scaling with distributed reduction
- Checkpoint save/load at correct points
- Evaluation loop uses `torch.no_grad()`
- Signal handler for graceful shutdown

### Configuration and Validation

**Model**: Sonnet
**Focus**: `nemo_automodel/components/config/`
**Checklist**:
- `_target_` paths resolve to valid callables
- CLI override parsing handles edge cases (lists, nested keys)
- Type translation covers all expected types
- Sensitive keys properly redacted
- Environment variable expansion secure

### Dataset Loader Review

**Model**: Sonnet
**Focus**: `nemo_automodel/components/datasets/`
**Checklist**:
- Data sharding across distributed ranks
- Sequence packing correctness (no cross-document attention)
- Tokenization consistent with model expectations
- Chat template applied correctly
- Streaming (IterableDataset) handles restarts

### Loss Function Review

**Model**: Sonnet
**Focus**: `nemo_automodel/components/loss/`
**Checklist**:
- Numerical stability (log-sum-exp, epsilon guards)
- Correct reduction across batch and sequence dimensions
- Label smoothing applied correctly if enabled
- Chunked CE memory savings vs correctness tradeoff
- Knowledge distillation temperature scaling

### PEFT/LoRA Review

**Model**: Sonnet
**Focus**: `nemo_automodel/components/_peft/`
**Checklist**:
- LoRA rank and alpha settings reasonable
- Target modules match model layer names
- FSDP2 compatibility (LoRA params wrapped correctly)
- Merge/unmerge for inference correctness
- QLoRA quantization config consistent

### Performance Regression Risk

**Model**: Sonnet
**Checklist**:
- No unnecessary GPU-CPU synchronization (`.item()`, `.tolist()`)
- No Python loops over tensor elements
- Communication operations overlap with computation where possible
- Memory usage not increased without justification

### Documentation Format

**Model**: Haiku
**Checklist**:
- Markdown filenames use hyphens (no underscores)
- Docstrings present on public functions
- YAML examples valid and up-to-date
- README references correct

### Test Coverage

**Model**: Haiku
**Checklist**:
- New code has corresponding tests
- GPU tests have skip markers
- Test naming follows `test_<what>_<condition>_<expected>`
- No hardcoded paths or secrets in tests

### Imports and Dependencies

**Model**: Haiku
**Checklist**:
- No cross-component imports (import-linter enforced)
- No wildcard imports
- Heavy dependencies conditionally imported
- New dependencies added to correct group in `pyproject.toml`
