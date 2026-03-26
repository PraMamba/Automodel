# Code Style Rules

Rules beyond pre-commit (Ruff format/lint).

## License Header (MANDATORY)

Every `.py` file must begin with the Apache 2.0 NVIDIA copyright header:

```python
# Copyright (c) <YEAR>, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

Use the current year for new files.

## Design Patterns

- **Prefer composition over inheritance**: Avoid deep class hierarchies
  - Good: Recipe holds a `Parallelizer` instance
  - Avoid: `ParallelRecipe(BaseRecipe)` → `FSDP2ParallelRecipe(ParallelRecipe)`
- Keep inheritance shallow (BaseRecipe → ConcreteRecipe, max 2 levels)
- Use mixins sparingly (e.g., `HFCheckpointingMixin`, `StateDictMixin`); prefer explicit
  delegation

## Component Independence

- **Components must not cross-import** (enforced by import-linter in `pyproject.toml`)
  - `nemo_automodel.components.models` CANNOT import from `nemo_automodel.components.distributed`
  - Each component is independently importable
  - Use `_target_` in YAML config to compose components at runtime
- **Shared utilities**: Only `nemo_automodel.shared.*` may be imported by all components

## Factory Pattern

- Model construction uses factory functions referenced by `_target_` in YAML:
  ```python
  # Good: standalone factory function
  def build_llama_model(vocab_size: int, hidden_size: int, ...) -> nn.Module:
      ...

  # Bad: class method
  class LlamaModel:
      @classmethod
      def from_config(cls, config): ...
  ```

## Logging

- Use `logging.getLogger(__name__)` from stdlib, NOT `print`
- Log levels:
  - DEBUG: Detailed tracing (avoid in hot paths)
  - INFO: Milestones (training start, checkpoint saved, epoch complete)
  - WARNING: Recoverable issues (fallback used, deprecated API)
  - ERROR: Failures requiring attention

## Performance Patterns

- **Avoid GPU-CPU sync**: `.item()`, `.tolist()`, `print(tensor)` cause sync
- **Prefer batch operations**: Avoid Python loops over tensor elements
- **In-place ops**: Use when safe, but careful with autograd (`.add_()` vs `+`)
- **Conditional imports**: Heavy dependencies inside functions using `shared/import_utils.py`

## Naming Conventions

| Type | Pattern | Example |
| ---- | ------- | ------- |
| Model factory | `build_xxx_model` | `build_llama_model`, `build_gpt2_model` |
| Dataset class | `XxxDataset` | `ChatDataset`, `NanogptDataset` |
| Config node | `ConfigNode` | Hierarchical YAML config node |
| Manager class | `XxxManager` | `FSDP2Manager`, `MegatronFSDPManager` |
| Loss function | `XxxLoss` or descriptive | `MaskedCrossEntropy`, `KDLoss` |
| State dict adapter | `state_dict_adapter.py` | Per-model state dict conversion |
| Parallelism style | `XxxStyle` | `ColwiseParallel`, `RowwiseParallel` |

## Tensor Conventions

- Shape convention: `[batch, seq_len, hidden]` or document clearly
- Use `torch.Size` assertions for shape validation in debug
- Prefer explicit dtype/device over implicit conversion

## Docstrings

- Use Google convention (`Args:`, `Returns:`, `Raises:` sections)
- Module-level docstrings with example YAML configs are encouraged for model modules
- Dataclass managers document `Attributes:` and `Methods:`
- `D101` and `D103` rules selected in Ruff (public class/function docstrings)

## Import Style

- Group: stdlib, third-party, nemo_automodel (Ruff handles order)
- Avoid `from x import *` (CLAUDE.md rule)
- Heavy deps inside functions or `TYPE_CHECKING` blocks:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from torch.optim import Optimizer
  ```
- Use `shared/import_utils.py` for `safe_import()` and `is_xxx_available()` checks
- `try/except ImportError` with fallback for optional dependencies
- `F401` is ignored in `__init__.py` files (re-exports are expected)
- Do NOT add `from __future__ import annotations` by default (only 5 files use it)

## Model Module Conventions

- Each model module must export `ModelClass = <ConcreteClass>` at module bottom
- Build functions must be standalone factory functions (not class methods)
- Custom models use shared building blocks from `models/common/`:
  `CombinedQKVAttentionMixin`, `CombinedGateUpMLP`, `HFCheckpointingMixin`

## Error Handling

- `ValueError` for invalid arguments, bad configs, shape mismatches
- `RuntimeError` for distributed/infrastructure failures
- `ImportError` with install instructions for missing optional deps
  (use `MISSING_*_MSG` constants from `shared/import_utils.py`)
- Do not create custom exception classes unless following the deferred-error pattern

## Formatting

- Line length: 120 characters
- Quote style: double quotes
- Ruff rules: F541, F841, F401, E741, F821, E266, I, D101, D103
- Markdown filenames: hyphens only, no underscores
- Tests excluded from Ruff lint rules (`tests/` directory)
