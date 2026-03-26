---
paths:
  - nemo_automodel/components/config/**
  - examples/**/*.yaml
---

# Configuration & YAML Rules

## _target_ Instantiation Pattern

AutoModel uses Hydra-style `_target_` for object instantiation from YAML configs:

```yaml
model:
  _target_: nemo_automodel.components.models.llama.build_llama_model
  vocab_size: 32000
  hidden_size: 4096
```

**Rules:**

- `_target_` must point to a valid, importable callable (function or class)
- `_target_` paths must be within allowed import prefixes (security constraint)
- All other keys become keyword arguments to the callable
- Nested `_target_` objects are recursively instantiated

## ConfigNode System

The `ConfigNode` in `components/config/loader.py` provides:

- Hierarchical config with dotted-path access
- Environment variable expansion
- CLI override via `--key=value` or `--key value`
- Type translation: strings → int/float/bool/list/dict

**CLI Override Syntax:**

```bash
automodel --config base.yaml --model.hidden_size=8192 --step_scheduler.max_steps=10000
```

## YAML Config Structure

Standard top-level keys:

```yaml
step_scheduler:           # Training schedule (global_batch_size, max_steps, ckpt_every_steps)
dist_env:                 # Distributed environment (backend, timeout)
model:                    # Model factory (_target_ + model args)
dataset:                  # Dataset factory (_target_ + dataset args)
dataloader:               # DataLoader factory (_target_ + loader args)
loss_fn:                  # Loss function factory (_target_)
optimizer:                # Optimizer factory (_target_ + lr, weight_decay)
lr_scheduler:             # LR scheduler config (optional)
distributed:              # Parallelism manager (_target_ + dp/tp/pp/cp sizes)
loggers:                  # Logging backends (wandb, mlflow)
```

## Field Conventions

- **Required fields**: No default value, must be specified in YAML
- **Optional fields**: Have sensible defaults in the factory function
- **`null` values**: Typically mean "auto-compute" (e.g., `dp_size: null`)
- **Type safety**: Values are translated from strings at load time

## Validation

- Use `__post_init__` for validation in dataclass configs
- Raise `ValueError` with clear message:
  ```python
  if self.tp_size <= 0:
      raise ValueError(f"tp_size must be positive, got {self.tp_size}")
  ```

## Security

- `_target_` paths are validated against allowed import prefixes
- Sensitive keys (passwords, tokens) are redacted in repr/logging
- Untrusted YAML files should not be loaded without review

## Adding New Config Options

1. Add parameter to the factory function signature
2. Update example YAML configs in `examples/`
3. Document the parameter with type, default, and constraints
4. Add validation for invalid values
5. Update any affected tests

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `_target_` not found | Typo in import path | Verify the function/class exists |
| Type mismatch | YAML string vs expected int | Use proper YAML types (no quotes for numbers) |
| Override ignored | Wrong dotted path | Check config hierarchy matches override path |
| Circular reference | Nested `_target_` loop | Avoid self-referencing configs |
