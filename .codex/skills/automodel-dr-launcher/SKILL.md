---
name: automodel-dr-launcher
description: Use when working with the launcher module of automodel — job launcher helpers for interactive and SLURM environments
---

# Launcher Module Deep Read

## 1. Module Purpose & Capabilities

The launcher module (`nemo_automodel/components/launcher/`) provides SLURM job submission infrastructure for NeMo AutoModel. It converts a declarative Python dataclass configuration into a fully rendered sbatch shell script, writes it to disk, and invokes `sbatch` to submit the job. The module is consumed exclusively by the CLI layer (`nemo_automodel/_cli/app.py`), never by recipes or other components.

**Capabilities:**

- Define SLURM job parameters (nodes, GPUs, wall time, partition, account) as a typed dataclass (`SlurmConfig` in `slurm/config.py`).
- Declare host-to-container volume mounts with path validation (`VolumeMapping` in `slurm/config.py`).
- Render a complete sbatch script from configuration, including multi-node environment variables (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`), NCCL tuning flags, W&B/HF credentials, and arbitrary user environment variables (`render_script()` in `slurm/template.py`).
- Assemble container mount strings from heterogeneous input types (dict, str, `VolumeMapping`) via `volume_map_to_str()` and `make_container_mounts()` in `slurm/utils.py`.
- Write the sbatch script to a job directory, invoke `sbatch` as a subprocess, capture stdout/stderr to files, and return the process exit code (`submit_slurm_job()` in `slurm/utils.py`).
- Support optional nsys profiling via `nsys_enabled` flag on `SlurmConfig` (wiring happens in `_cli/app.py:launch_with_slurm()`).

The module does **not** handle interactive (torchrun) launching -- that path lives entirely in `_cli/app.py:run_interactive()`. The launcher module is solely the SLURM path.

---

## 2. Core Design Logic

### Why a dataclass instead of a free-form dict?

`SlurmConfig` is a Python `dataclass` (not frozen) with `field(metadata=dict(help=...))` annotations on every attribute. This serves three purposes:
1. **Typed defaults** -- sensible defaults (1 node, 8 tasks-per-node, 5-minute wall time, default container image `nvcr.io/nvidia/nemo:dev`) mean a user only needs to supply `job_name` and `command` to get a valid script.
2. **Serialization** -- `dataclasses.asdict(config)` in `submit_slurm_job()` (line 68 of `slurm/utils.py`) converts the entire config to a plain dict that can be passed directly to `str.format()` inside the template. This avoids any manual dict construction.
3. **Validation at construction** -- `__post_init__` on both `VolumeMapping` and `SlurmConfig` enforces constraints (absolute paths, existing source directories, correct mount string format) before any script generation begins.

### Why frozen=True on VolumeMapping but not on SlurmConfig?

`VolumeMapping` (line 21-39 of `slurm/config.py`) is `frozen=True, slots=True` because mounts are value objects: once validated, they should never change. `SlurmConfig` is mutable because the CLI layer (`_cli/app.py:launch_with_slurm()`) mutates it after construction -- it pops `repo_root`, appends to `extra_mounts`, and conditionally sets `hf_home` and `job_name`.

### Why a string template instead of Jinja2 or programmatic construction?

The template in `slurm/template.py` uses Python `str.format()`. This keeps the module dependency-free (no Jinja2 requirement) and keeps the sbatch script shape readable as a single literal string. The `HEADER` is separated for metadata (user, host, timestamp) and prepended to the `TEMPLATE`.

### Why does make_container_mounts mutate the opts dict?

`make_container_mounts()` (line 51-62 of `slurm/utils.py`) pops `nemo_mount` and `extra_mounts` from `opts` after processing them. This is intentional: these keys are not valid `str.format()` placeholders in the template, so they must be removed before `render_script()` is called. The function converts them into the single `container_mounts` comma-separated string that the template expects.

### Why is hf_home selectively mounted?

`make_container_mounts()` only mounts `hf_home` if the path does NOT start with `~/` or `/home`. Home-directory paths are assumed to be available via the default container home mount or user environment, so they are excluded from explicit `--container-mounts` to avoid mount conflicts.

### Component independence

The launcher is listed in `pyproject.toml` under `[tool.importlinter.contracts]` as one of the independent components. It imports nothing from other `nemo_automodel.components.*` modules. The only intra-module imports are `slurm/utils.py` importing from `slurm/config.py` and `slurm/template.py`.

---

## 3. Core Data Structures

### VolumeMapping (frozen dataclass)

**File:** `nemo_automodel/components/launcher/slurm/config.py`, lines 21-39

```
@dataclass(frozen=True, slots=True)
class VolumeMapping:
    source: Path   # Absolute host path, must exist
    dest: Path     # Absolute container path
```

- `__post_init__` validates both paths are absolute and `source` exists on the filesystem.
- `to_str()` returns `"{source}:{dest}"` for sbatch mount syntax.

### SlurmConfig (mutable dataclass)

**File:** `nemo_automodel/components/launcher/slurm/config.py`, lines 42-87

Key fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `job_name` | `str` | (required) | SLURM `-J` flag |
| `nodes` | `int` | `1` | Number of nodes (`-N`) |
| `ntasks_per_node` | `int` | `8` | Tasks per node; also used for `torchrun --nproc_per_node` |
| `time` | `str` | `"00:05:00"` | Wall-clock limit |
| `account` | `str` | `None` | SLURM account (`-A`) |
| `partition` | `str` | `"batch"` | SLURM partition (`-p`) |
| `container_image` | `str` | `"nvcr.io/nvidia/nemo:dev"` | OCI/SquashFS image for `srun --container-image` |
| `nemo_mount` | `VolumeMapping` | `None` | Primary code mount |
| `hf_home` | `Path` | `"~/.cache/huggingface"` | HuggingFace cache directory |
| `extra_mounts` | `VolumeMapping` | `None` | Additional mounts (list, coerced in `__post_init__`) |
| `master_port` | `int` | `13742` | Rendezvous port for multi-node |
| `gpus_per_node` | `Optional[int]` | `None` | If set, adds `#SBATCH --gpus-per-node=N` |
| `wandb_key` | `str` | `os.environ.get("WANDB_API_KEY", "")` | W&B API key |
| `hf_token` | `str` | `os.environ.get("HF_TOKEN", "")` | HuggingFace token |
| `env_vars` | `dict` | `{}` | Arbitrary extra environment variables |
| `command` | `str` | `""` | Shell command to execute inside the container |
| `chdir` | `str` | `None` | Working directory for the job |
| `nsys_enabled` | `bool` | `False` | Enable nsys profiling prefix |

- `__post_init__` coerces `extra_mounts` elements from `str` (`"src:dst"`) to `VolumeMapping` instances.

### TEMPLATE (string constant)

**File:** `nemo_automodel/components/launcher/slurm/template.py`, lines 33-88

A `str.format()`-compatible sbatch script template. Key placeholder groups:
- SLURM directives: `{account}`, `{partition}`, `{nodes}`, `{time}`, `{job_name}`, `{gpus_per_node_directive}`, `{job_dir}`
- Multi-node env: `{master_port}`, `{num_gpus}`
- Credentials/env: `{wandb_key}`, `{hf_home}`, `{hf_token}`, `{custom_env_vars}`
- User payload: `{chdir}`, `{command}`, `{container_image}`, `{container_mounts}`

### render_script() function

**File:** `nemo_automodel/components/launcher/slurm/template.py`, lines 91-116

Signature: `render_script(opts: dict, job_dir) -> str`

Adds computed fields (`gpus_per_node_directive`, `num_gpus`, `custom_env_vars`, metadata from `getpass`/`socket`/`datetime`) then calls `TEMPLATE.format(...)`.

---

## 4. State Flow

The end-to-end flow from user invocation to job submission:

```
User runs: automodel finetune llm --config path/to/config.yaml
                    |
                    v
         _cli/app.py:main()
           - parse CLI args
           - load_yaml(config_path)
           - detect "slurm" key in config dict
                    |
                    v
         _cli/app.py:launch_with_slurm(args, job_conf_path, job_dir, slurm_config)
           - create job_dir = <slurm.job_dir or cwd/slurm_jobs>/<unix_timestamp>
           - write job_config.yaml (config minus slurm section) to job_dir
           - resolve repo_root (from yaml, from cwd, or default /opt/Automodel)
           - set hf_home to shared storage if not provided
           - set default job_name if empty
           - build torchrun command string with recipe script path + config path
           - optionally prepend nsys profile command
           - append repo_root as VolumeMapping to extra_mounts
           - construct: SlurmConfig(**slurm_config, command=command, chdir=repo_root)
                    |
                    v
         slurm/config.py:SlurmConfig.__post_init__()
           - coerce extra_mounts strings to VolumeMapping instances
                    |
                    v
         slurm/utils.py:submit_slurm_job(config, job_dir)
           - dataclasses.asdict(config) -> opts dict
           - make_container_mounts(opts):
               - collect hf_home (if non-home path), nemo_mount, extra_mounts
               - convert each via volume_map_to_str()
               - pop nemo_mount, extra_mounts from opts
               - return list of "src:dst" strings
           - join mounts into comma-separated string -> opts["container_mounts"]
           - render_script(opts, job_dir):
               - compute gpus_per_node_directive, num_gpus, custom_env_vars
               - inject user, host, timestamp metadata
               - TEMPLATE.format(**opts) -> sbatch script string
           - write script to <job_dir>/<job_name>.sbatch
           - subprocess.Popen(["sbatch", script_path])
           - capture stdout -> <job_dir>/subproc_sbatch.stdout
           - capture stderr -> <job_dir>/subproc_sbatch.stderr
           - return proc.returncode
```

**Artifacts written to job_dir:**

| File | Content |
|---|---|
| `job_config.yaml` | Training config (slurm section removed) |
| `<job_name>.sbatch` | Rendered sbatch shell script |
| `subproc_sbatch.stdout` | sbatch command stdout (contains SLURM job ID) |
| `subproc_sbatch.stderr` | sbatch command stderr |
| `slurm_<job_name>_<job_id>.out` | SLURM output log (written by SLURM at runtime) |

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a new SLURM directive (e.g., `--mem`, `--constraint`, or `--reservation`)

1. Add a new field to `SlurmConfig` in `nemo_automodel/components/launcher/slurm/config.py` with a sensible default:
   ```python
   mem: Optional[str] = field(default=None, metadata=dict(help="Memory per node (e.g. '500G')"))
   ```
2. Add a conditional directive line in `render_script()` in `nemo_automodel/components/launcher/slurm/template.py`, similar to how `gpus_per_node_directive` is handled (lines 92-98): check if the value is set, inject an `#SBATCH --mem={mem}` line.
3. Add a test case in `tests/unit_tests/launcher/test_template.py` following the pattern of `test_gpus_per_node_included()`.

### Scenario 2: Supporting a non-container (bare-metal) SLURM mode

Currently the template hardcodes `srun --container-image=... --container-mounts=...`. To support bare-metal:

1. Add a `container_mode: bool = True` field to `SlurmConfig` in `slurm/config.py`.
2. In `slurm/template.py`, create a second template (or conditional blocks) that replaces the `srun` stanza with a plain `srun bash -c "$CMD"` without container flags.
3. In `slurm/utils.py:submit_slurm_job()`, skip the `make_container_mounts()` call when `container_mode` is False.
4. In `_cli/app.py:launch_with_slurm()`, skip the VolumeMapping construction for `repo_root` when not using containers.

### Scenario 3: Adding a new credential or secret (e.g., Neptune API key)

1. Add the field to `SlurmConfig` in `slurm/config.py`:
   ```python
   neptune_api_token: str = field(
       default=os.environ.get("NEPTUNE_API_TOKEN", ""),
       metadata=dict(help="Neptune API token"),
   )
   ```
2. Add the `export NEPTUNE_API_TOKEN={neptune_api_token}` line in the `TEMPLATE` string in `slurm/template.py` (in the "Experiment env" block, around line 70).
3. No changes needed in `slurm/utils.py` because `dataclasses.asdict()` automatically picks up the new field and `str.format()` substitutes it.

Alternatively, for ad-hoc variables, users can already use the existing `env_vars` dict field on `SlurmConfig`, which renders via the `{custom_env_vars}` placeholder without any code changes.

### Scenario 4: Adding Kubernetes launch support

The CLI (`_cli/app.py:main()`, line 333-335) already detects `k8s` or `kubernetes` keys in the config but raises `NotImplementedError`. To implement:

1. Create `nemo_automodel/components/launcher/k8s/` mirroring the slurm structure: `config.py` (a `K8sConfig` dataclass), `template.py` (a Job/CronJob YAML template), `utils.py` (a `submit_k8s_job()` function using `kubectl apply`).
2. Wire the new `submit_k8s_job()` into `_cli/app.py:main()` in the `elif "k8s" in config` branch.
3. The launcher component independence constraint is preserved since k8s would live under the same `nemo_automodel.components.launcher` package.

### Scenario 5: Changing the job directory structure or adding metadata files

The job directory layout is established in two places:
- `_cli/app.py:main()` (lines 322-331) creates the directory and writes `job_config.yaml`.
- `slurm/utils.py:submit_slurm_job()` (lines 66-89) writes the `.sbatch`, `.stdout`, and `.stderr` files.

To add, for example, a `metadata.json` with git hash and CLI args:
1. In `_cli/app.py:launch_with_slurm()`, write the metadata file to `job_dir` before calling `submit_slurm_job()`.
2. No changes needed inside the launcher component itself, keeping it focused on script generation and submission.
