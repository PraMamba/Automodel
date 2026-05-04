---
paths:
  - nemo_automodel/components/distributed/**
  - nemo_automodel/components/moe/**
---

# Distributed Code Rules

## Process Group Management

- **Never create global process group** in module-level code
- Always pass `process_group` explicitly, don't rely on default
- Use `dist.get_rank(group)` not `dist.get_rank()` when group matters
- Clean up process groups in teardown

## DeviceMesh & DTensor

- FSDP2 canonical 5D mesh: `("pp", "dp_replicate", "dp_shard", "cp", "tp")`
- MegatronFSDP mesh: `("dp", "cp", "tp")` (simpler, no PP/EP)
- MoE mesh (separate): `("pp", "ep_shard", "ep")`
- Flattened submeshes created automatically: `"dp"`, `"dp_shard_cp"`, `"dp_cp"`
- DTensor requires consistent mesh across all ranks
- Use `DTensor.from_local()` with correct placements
- Verify `pp_size * dp_replicate_size * dp_shard_size * cp_size * tp_size == world_size`

## FSDP2 Patterns

- Use `FSDP2Manager` for wrapping models with FSDP2
- FSDP2 is DTensor-native: uses `DTensor` internally for sharding
- Mixed precision via `MixedPrecision` policy, not manual casting
- Activation checkpointing configured via FSDP2 wrapper

## MegatronFSDP Patterns

- `MegatronFSDPManager` for hybrid Megatron + FSDP integration
- Pipeline parallelism requires stage definition in `pipelining/`
- Gradient accumulation handled by training loop, not FSDP

## Communication Patterns

- **All-reduce**: Must be called by all ranks in the group
- **Broadcast**: Specify `src` rank explicitly
- **Barrier**: Avoid unless necessary (debugging only)
- Check `NCCL_ASYNC_ERROR_HANDLING` for deadlock debugging

## Tensor Parallel Plans

- Defined in `distributed/optimized_tp_plans.py`
- Each model needs its own TP plan mapping layers to parallel styles
- Use `ColwiseParallel` / `RowwiseParallel` for linear layers
- Attention: parallelize Q/K/V projections + output projection

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| Hang | Mismatched collective calls | Ensure all ranks call same op |
| Wrong results | Incorrect reduction op | Check `ReduceOp` (SUM vs MEAN) |
| OOM | Unsharded tensor on wrong device | Verify DTensor placements |
| Slow | No communication overlap | Enable async all-reduce |
| Shape mismatch | Wrong TP/DP config | Verify sizes multiply to world_size |

## Debugging

- Set `TORCH_DISTRIBUTED_DEBUG=DETAIL` for verbose logging
- Use `NCCL_DEBUG=INFO` for NCCL-level issues
- Use `NCCL_DEBUG_SUBSYS=ALL` for detailed NCCL subsystem info
- Profile with NVTX annotations (see `autonvtx/` module)

## Import Constraints

- Components listed in the import-linter independence contract (checkpoint, config,
  datasets, distributed, launcher, loggers, loss, optim, training, utils) must NOT
  import each other
- Exception: `distributed.optimized_tp_plans` may import `models.llama.model`
- `components/models/`, `components/attention/`, `components/moe/`, `components/_peft/`
  are NOT in the independence contract and may import from other components
- Use factory pattern and `_target_` composition for cross-component integration
