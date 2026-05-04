---
name: automodel-dr-cli
description: Use when working with the CLI module of automodel — job launcher for interactive and SLURM environments with dynamic recipe loading
---

# CLI Module (`nemo_automodel/_cli/`)

## 1. Module Purpose & Capabilities

The CLI module is a single-file job launcher (`nemo_automodel/_cli/app.py`) that serves as the unified entry point for running NeMo AutoModel training recipes. It is registered as a console script via `pyproject.toml` (`[project.scripts] automodel = "nemo_automodel._cli.app:main"`) so users invoke it as `automodel <command> <domain> -c <config.yaml>`.

Its three core capabilities:

1. **Interactive (local) launch** -- Detects available GPUs and either calls the recipe's `main()` directly (single GPU) or delegates to `torch.distributed.run` (multi-GPU). Handled by `run_interactive()` (line 266).

2. **SLURM cluster launch** -- When the YAML config contains a `slurm:` section, it generates an sbatch script with container mounts, env vars, and torchrun invocations, then submits via `sbatch`. Handled by `launch_with_slurm()` (line 125).

3. **Kubernetes launch (stub)** -- If the YAML contains `k8s:` or `kubernetes:`, raises `NotImplementedError`. Reserved for future work. Checked in `main()` (line 333).

The CLI is intentionally thin: it performs argument parsing and dispatch only, delegating all training logic to recipe scripts and all cluster mechanics to the `nemo_automodel.components.launcher.slurm` component.

## 2. Core Design Logic

### Why a single file?

The CLI deliberately avoids framework complexity (no Click, no Typer). It uses `argparse` with `parse_known_args` so that unknown flags (like `--model.pretrained_model_name_or_path`) pass through to the recipe's own config parser (`nemo_automodel/components/config/_arg_parser.py:parse_args_and_load_config`). This two-stage parsing is the key design decision: the CLI handles launch concerns, and the recipe handles config concerns.

### Dynamic recipe loading via file system convention

Rather than maintaining a registry of recipes, the CLI computes the recipe path from the `(command, domain)` pair using a fixed directory layout:

```
nemo_automodel/recipes/{domain}/{recipe_name}.py
```

The mapping from CLI command to recipe filename is done via `COMMAND_ALIASES` (line 45):

```python
COMMAND_ALIASES = {"finetune": "train_ft", "pretrain": "train_ft", "benchmark": "benchmark"}
```

Both `finetune` and `pretrain` map to `train_ft.py` because the same recipe handles both SFT and pretraining. The `kd` command has no alias so maps directly to `kd.py`. The function `get_recipe_script_path()` (line 48) performs this resolution.

For single-GPU interactive runs, the recipe is loaded dynamically via `load_function()` (line 64), which uses `importlib.util.spec_from_file_location` to import the recipe file and extract its `main` function. This avoids importing the entire module tree at CLI startup.

### SLURM path: config surgery

When launching via SLURM, `main()` pops the `slurm:` key from the loaded YAML config before writing the remainder as `job_config.yaml` into a timestamped job directory. This ensures the recipe never sees SLURM-specific keys, and the SLURM launcher gets only its own configuration. The timestamp directory (`str(int(time.time()))`) is validated in `launch_with_slurm()` with an assertion (line 130: 10-digit string check).

### Repo root detection

Two functions handle this:

- `get_automodel_repo_root()` (line 112) checks if `cwd` contains `nemo_automodel/components` and `examples/` -- used to detect editable/source installs.
- `get_repo_root()` (line 248) wraps the above and, when found, prepends the repo root to `PYTHONPATH` so that the source tree is importable by torchrun subprocesses. Falls back to `Path(__file__).parents[2]` (the installed package root).

### Multi-GPU decision logic

In `run_interactive()` (line 266), the number of available GPUs is detected via `torch.distributed.run.determine_local_world_size(nproc_per_node="gpu")`. The branching:

- If `args.nproc_per_node == 1` OR only 1 GPU detected: calls `load_function(script_path, "main")` and invokes the recipe directly in-process (no torchrun).
- Otherwise: constructs a `torchrun` argument namespace and delegates to `torch.distributed.run.run()`. It rewrites `training_script` and `training_script_args` so torchrun launches the correct recipe with the correct config path.

## 3. Core Data Structures

### `COMMAND_ALIASES` (dict, `app.py` line 45)

Maps user-facing CLI command names to recipe file stems. Keys: `"finetune"`, `"pretrain"`, `"benchmark"`. This is the canonical command-to-recipe mapping.

### `argparse.ArgumentParser` returned by `build_parser()` (`app.py` line 202)

Defines the CLI interface with:
- Positional: `command` (choices: `finetune`, `pretrain`, `kd`, `benchmark`)
- Positional: `domain` (choices: `llm`, `vlm`)
- Required flag: `-c`/`--config` (Path to YAML)
- Optional flag: `--nproc-per-node` (int, controls GPU count override)

All other flags are captured as extras by `parse_known_args()` and forwarded to the recipe or SLURM command.

### `SlurmConfig` (dataclass, `nemo_automodel/components/launcher/slurm/config.py` line 42)

Used by `launch_with_slurm()`. Key fields: `job_name`, `nodes`, `ntasks_per_node`, `time`, `container_image`, `hf_home`, `extra_mounts`, `master_port`, `wandb_key`, `hf_token`, `env_vars`, `command`, `chdir`, `nsys_enabled`. The CLI constructs a dict from the YAML `slurm:` section, augments it (adds `hf_home`, `extra_mounts`, computes `command`), then passes it as `SlurmConfig(**slurm_config, command=command, chdir=repo_root)`.

### `VolumeMapping` (frozen dataclass, `nemo_automodel/components/launcher/slurm/config.py` line 21)

Represents a host-to-container mount with `source: Path` and `dest: Path`. Used in `launch_with_slurm()` (line 198) to mount the repo root into the SLURM container.

### SLURM sbatch template (`nemo_automodel/components/launcher/slurm/template.py`)

A `TEMPLATE` string (line 42) with `{placeholders}` for SBATCH directives, env vars, and the user command. Rendered by `render_script(opts, job_dir)` (line 91). Uses `srun --container-image` for Pyxis/Enroot container execution.

## 4. State Flow

### CLI args to config to launch

```
User invokes: automodel finetune llm -c config.yaml [--extra-flags ...]
                |
                v
main() (line 302)
  |-- build_parser().parse_known_args()
  |     => args (Namespace with command, domain, config, nproc_per_node)
  |     => extra (list of unknown flags, forwarded downstream)
  |
  |-- load_yaml(config_path)
  |     => config (dict from YAML)
  |
  |-- Branch on config content:
  |
  |-- [A] config has "slurm" key:
  |     |-- Pop slurm section from config
  |     |-- Create timestamped job_dir: slurm.job_dir / unix_timestamp
  |     |-- Write remaining config as job_dir/job_config.yaml
  |     |-- launch_with_slurm(args, job_conf_path, job_dir, slurm_config, extra)
  |           |-- Resolve repo_root (from slurm.repo_root, cwd, or /opt/Automodel)
  |           |-- Set default hf_home if not provided
  |           |-- get_recipe_script_path(command, domain, repo_root) => script path
  |           |-- Build torchrun command string with script path + config path + extras
  |           |-- Optionally prepend nsys profile command
  |           |-- Append repo_root to extra_mounts
  |           |-- submit_slurm_job(SlurmConfig(...), job_dir)
  |                 |-- render_script(opts, job_dir) => sbatch script text
  |                 |-- Write .sbatch file to job_dir
  |                 |-- subprocess.Popen(["sbatch", script_path])
  |                 |-- Return exit code
  |
  |-- [B] config has "k8s" or "kubernetes" key:
  |     |-- raise NotImplementedError
  |
  |-- [C] Neither (interactive):
        |-- run_interactive(args)
              |-- get_repo_root() => repo_root (+ PYTHONPATH update)
              |-- get_recipe_script_path(command, domain, repo_root)
              |-- determine_local_world_size("gpu") => num_devices
              |
              |-- [C1] Single device (nproc_per_node==1 or num_devices==1):
              |     |-- load_function(script_path, "main") => recipe_main
              |     |-- recipe_main(config_path) -- direct in-process call
              |
              |-- [C2] Multi-device:
                    |-- get_args_parser() => torchrun_parser
                    |-- Parse torchrun args from sys.argv
                    |-- Rewrite training_script to recipe path
                    |-- Rewrite config path to absolute
                    |-- Set nproc_per_node = num_devices (if not overridden)
                    |-- torch.distributed.run.run(torchrun_args)
```

### Recipe interface contract

Every recipe script (`nemo_automodel/recipes/{domain}/{name}.py`) must expose a `def main(config_path=None)` function. When called by the CLI:
- In single-GPU mode: `main(config_path)` is called directly with an absolute `Path`.
- In multi-GPU mode: the script is launched as a subprocess by torchrun with `-c <config_path>` as a CLI argument. The recipe's `if __name__ == "__main__": main()` block runs, and `parse_args_and_load_config()` parses `-c` from `sys.argv`.

### Config override flow

Extra CLI flags (captured by `parse_known_args`) take two paths:
- **SLURM**: appended to the torchrun command string inside the sbatch script, so the recipe's `parse_args_and_load_config()` processes them at job runtime.
- **Interactive multi-GPU**: present in `sys.argv` when torchrun re-launches the script, so again handled by `parse_args_and_load_config()`.
- **Interactive single-GPU**: NOT forwarded in the current implementation (the direct `recipe_main(config_path)` call does not pass extras). This is a known gap.

## 5. Common Modification Scenarios

### Scenario 1: Adding a new CLI command (e.g., `evaluate`)

1. Add the command string to `build_parser()` choices at line 215: add `"evaluate"` to the `choices` list.
2. Add an alias entry in `COMMAND_ALIASES` (line 45) if the recipe filename differs from the command name. For example: `"evaluate": "eval"`.
3. Create the recipe at `nemo_automodel/recipes/{domain}/eval.py` with a `def main(config_path=None)` entry point.
4. Add a test in `tests/unit_tests/_cli/test_app.py` similar to `test_cli_accepts_pretrain` in `tests/unit_tests/_cli/test_pretrain_cli.py`.

### Scenario 2: Adding a new domain (e.g., `audio`)

1. Add `"audio"` to the `domain` choices in `build_parser()` at line 220.
2. Create the recipe directory `nemo_automodel/recipes/audio/` with the corresponding recipe files (e.g., `train_ft.py`).
3. The rest of the dispatch logic (path resolution, SLURM launch) works automatically because `get_recipe_script_path()` constructs the path from `{domain}`.

### Scenario 3: Adding Kubernetes support

1. In `main()` at line 333, replace the `NotImplementedError` raise with a call to a new launcher function (e.g., `launch_with_k8s(args, config_path, k8s_config, extra_args=extra)`).
2. Implement the launcher in `nemo_automodel/components/launcher/k8s/` following the pattern of the SLURM launcher: a config dataclass, a template renderer, and a submission utility.
3. Pop the `k8s`/`kubernetes` section from config (like SLURM does) so the recipe config stays clean.

### Scenario 4: Forwarding extra CLI overrides in single-GPU mode

Currently, `run_interactive()` does not forward `extra` args when calling `recipe_main(config_path)` directly (line 281). To fix:
1. Change the signature or approach so that `extra` args are injected into `sys.argv` before calling `recipe_main()`, or pass them as an additional parameter.
2. This would require updating the recipe's `main()` contract or using `monkeypatch`-style `sys.argv` manipulation before the call.

### Scenario 5: Customizing the SLURM sbatch template

The template lives in `nemo_automodel/components/launcher/slurm/template.py` as a `TEMPLATE` string constant (line 42). To add new SBATCH directives (e.g., `--mem-per-gpu`):
1. Add the field to `SlurmConfig` in `config.py`.
2. Add the `#SBATCH` directive in the `TEMPLATE` string with a corresponding `{placeholder}`.
3. Handle the new field in `render_script()` (e.g., conditionally add the directive like `gpus_per_node_directive`).
4. Users can then set the new field in their YAML `slurm:` section.

### Scenario 6: Supporting nsys profiling in interactive mode

`nsys_enabled` is currently only handled in the SLURM path (`launch_with_slurm()` line 160). To support it interactively:
1. Read the `nsys_enabled` flag from the config in `main()` before dispatching.
2. In `run_interactive()`, wrap the torchrun command or recipe call with nsys profiling if enabled.
