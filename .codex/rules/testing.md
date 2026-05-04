# Codex Guidance: testing

> Codex compatibility note: This guidance was migrated from `.claude/rules/testing.md`. Treat it as project guidance for matching files; this migration does not claim automatic path-scoped rule enforcement.

## Applies To

Original Claude rule selector metadata:

```yaml
paths:
  - '**/tests/**'
  - '*_test.py'
  - test_*.py
```

# Testing Rules

## Test Organization

```
tests/
├── functional_tests/      # Integration tests (often require GPU)
│   ├── checkpoint/       # Checkpoint save/load tests
│   ├── data/             # Dataset tests
│   ├── datasets/         # Dataset loading tests
│   ├── hf_dcp/           # HF DCP integration
│   ├── hf_peft/          # PEFT tests
│   ├── hf_transformer*/  # Transformer model tests
│   ├── context_parallel/ # Context parallelism tests
│   ├── training/         # Training loop tests
│   └── conftest.py       # Shared fixtures
├── unit_tests/            # Unit tests (mostly no GPU)
│   ├── components/       # Per-component tests
│   ├── models/           # Per-model tests
│   └── training/         # Training utility tests
└── utils/                 # Test utilities
```

## Pytest Markers

| Marker | When to Use |
| ------ | ----------- |
| `@pytest.mark.slow` | Takes > 10 seconds |
| `@pytest.mark.skipif(cond, reason=...)` | Conditional skip |
| `@pytest.mark.parametrize(...)` | Parameterized tests |

## Test Structure

```python
def test_<what>_<condition>_<expected>():
    """Test that <what> does <expected> when <condition>."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

## GPU Test Constraints

- **Always skip gracefully** when GPU unavailable:
  ```python
  import torch
  import pytest

  CUDA_AVAILABLE = torch.cuda.is_available()

  @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
  def test_gpu_feature():
      ...
  ```
- Clean up GPU memory: `torch.cuda.empty_cache()` in fixtures
- Use smallest possible model/batch for unit tests
- Check GPU before running: `python -c "import torch; print(torch.cuda.is_available())"`

## Distributed Test Patterns

- Use `torch.distributed.fake_pg` for unit tests when possible
- Mock `dist.get_rank()` and `dist.get_world_size()` explicitly
- Don't mock internals of FSDP2/DTensor - use functional tests instead
- Multi-GPU tests go in `functional_tests/` with appropriate skip markers

## Fixtures

- Prefer `tmp_path` over manual temp directories
- Use `monkeypatch` for environment variables
- Scope expensive fixtures appropriately (`session` > `module` > `function`)
- Share fixtures via `conftest.py` at appropriate directory level

## Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Avoid bare `assert tensor.equal()` - no useful error message
- For config tests, compare dictionaries with clear diff messages

## Running Tests

```bash
# Unit tests (no GPU needed)
uv run pytest tests/unit_tests/ -v

# Specific test file
uv run pytest tests/unit_tests/components/test_config.py -v

# Functional tests (GPU required)
uv run pytest tests/functional_tests/ -v

# With coverage
uv run pytest tests/ --cov=nemo_automodel --cov-report=html

# Quick subset
uv run pytest tests/unit_tests/ -v --timeout=60 -x
```

## Test Naming

- Test files: `test_<module>.py` or `<module>_test.py`
- Test functions: `test_<what>_<condition>_<expected>()`
- Test classes: `TestXxx` (group related tests)

## What to Test

- **Always test**: Public API functions, factory functions, state dict adapters
- **Test with care**: Config loading, YAML parsing, CLI overrides
- **Skip locally**: Multi-GPU distributed tests, large model tests
- **Document skips**: Always include reason in skip markers
