# Project Design Philosophy

> 范围：NVIDIA-NeMo/Automodel，工作树 `/root/Automodel/.worktrees/source_code_analysis`。  
> 方法：先读 README、CONTRIBUTING、docs、`.github`、CI、构建配置、核心源码、测试、examples，再用 `gh`/`git` 分析最近 PR/Issue/review 与高频 contributor 提交。  
> 结论分级：**明确事实**=源码/文档/维护者明确说明；**强推断**=多类证据一致支持；**弱推断**=少量证据支持、需维护者确认；**未知**=证据不足。

## 1. 摘要

- **PyTorch/DTensor-native SPMD 是底层哲学**：并行策略应由 `DeviceMesh`、placement、FSDP2/TP/PP/CP/EP 配置驱动，而不是在模型或训练循环中写死。
- **组件独立、Recipe 组合是最重要架构边界**：`components/*` 是 dependency-light building blocks；`recipes/*` 才是端到端编排层。
- **Hugging Face 生态兼容是产品边界**：`NeMoAutoModel*` 是 HF drop-in wrapper；checkpoint 产物要能被 Transformers/vLLM/SGLang 等消费。
- **YAML `_target_` 是主要扩展协议**：model、dataset、loss、optimizer、logger 等通过 config object/DI 实例化；但 `distributed:` 使用固定 strategy schema，不是任意 `_target_` 注入。
- **模型专属逻辑应靠近模型目录**：近期 RFC/Issue 反复指出 model-specific parallelizer、patch、collate、checkpoint adapter 不应继续膨胀通用 infra 文件。
- **显式配置优于全局魔法**：历史 PR 从 `match_all_linear` 转向 `target_modules`，从 implicit workaround 转向清晰 adapter/registry/opt-in flag。
- **数值/兼容/分布式证据是合入货币**：复杂模型或分布式改动需要 parity、loss curve、round-trip load、GPU/CI recipe 证据。
- **安全默认值不可妥协**：dataset/cache/checkpoint 反序列化必须避免 pickle/RCE 风险，secret config 打印必须 redaction。
- **PR 应小、单意图、可回滚**：高频 contributor 通常 1-3 files 小修复；巨大无关 diff 或无上下文 dependency bump 容易关闭。
- **测试与 examples 是设计的一部分**：新能力通常同时落在 `components`、`recipes`、`examples`、`tests/unit_tests` / `tests/functional_tests` / CI recipe。

## 2. 项目目标与非目标

### 目标

结论级别：明确事实

- 为 LLM/VLM/diffusion/retrieval 等训练与微调提供 PyTorch 原生分布式、GPU 加速、内存高效工作流。
- 提供 Day-0 Hugging Face model/tokenizer/checkpoint 兼容，并通过 recipes/examples 交付端到端可运行配置。
- 支持 FSDP2、MegatronFSDP、TP、PP、CP、EP、MoE、LoRA/QLoRA、FP8/QAT 等训练能力。

证据：

- 文档：`README.md`；`docs/repository-structure.md#What is NeMo AutoModel?`
- 文档：`docs/about/key-features.md`
- 源码：`nemo_automodel/_transformers/auto_model.py::NeMoAutoModelForCausalLM`
- gh CLI command：`gh repo view --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo`

### 非目标

结论级别：强推断

- 不是一个通用 CPU-first 训练框架；CPU 可用于部分测试/兼容路径，但核心优化面向 NVIDIA GPU/distributed。
- 不是 rigid trainer 框架；recipes 文档强调线性脚本暴露训练逻辑以便控制。
- 不是通过跨 component 继承树扩展；扩展主要通过 config/registry/adapter/strategy 与 recipe composition。
- 不鼓励为过期 runtime workaround 增加核心复杂度。

证据：

- 文档：`docs/guides/huggingface-api-compatibility.md#Differences`（CUDA expectation）
- 文档：`docs/repository-structure.md#Recipes Directory`（linear scripts / components）
- 配置：`pyproject.toml::tool.importlinter.contracts`（component independence）
- PR：#2070 关闭原因为最新 container 不再需要 PyTorch DCP plan caching workaround
- Issue：#2115 Gemma4 CP with TE 讨论明确受 FlashAttention/TE backend 限制约束

## 3. 核心设计哲学

### PyTorch/DTensor-native SPMD 优先

结论级别：明确事实

说明：同一 recipe 应尽量通过 mesh、parallelism size、placement/manager 选择改变并行策略；模型 forward/API 尽量维持 HF 语义。

证据：

- 文档：`README.md`（DeviceMesh、SPMD-first、parallelism is configuration）
- 源码：`nemo_automodel/components/distributed/mesh.py::MeshContext`
- 源码：`nemo_automodel/components/distributed/mesh_utils.py::init_device_mesh`
- 测试：`tests/unit_tests/test_validation.py`（并行 capability validation）
- PR：#2125 VLM context parallelism 要求验证 FSDP2 hook、CP/PP gate、padding/label 语义

对后续开发的要求：

- 新并行能力先明确 mesh 维度、strategy schema、manager wiring 与 train/val 一致性。
- 不要在模型层硬编码 rank/world-size 语义；模型层只保留必要的 tensor/attention/MoE 局部逻辑。
- 修改 TP/PP/CP/EP/FSDP2 时必须补 capability validation 和至少 unit wiring test；GPU/recipe 证据应在 PR 中说明。

### 组件独立，Recipe 组合

结论级别：明确事实

说明：组件是可替换 building blocks；recipe 是唯一合理的跨组件编排层。

证据：

- 文档：`docs/repository-structure.md#Components Directory`：components dependency-light、reusable、without cross-module imports
- 文档：`docs/repository-structure.md#Recipes Directory`：recipes 组合 model/dataset/training/checkpoint
- 配置：`pyproject.toml::tool.importlinter.contracts`，contract 名称 “Components must not import each other”
- 源码：`nemo_automodel/recipes/llm/train_ft.py::imports` 合法导入 `_transformers` 与多个 components
- 文档：`.codex/rules/code-style.md#Component Independence`

对后续开发的要求：

- component 需要共享逻辑时优先放 `nemo_automodel/shared` 或在 recipe 层组合，不要 component 互相 import。
- 新功能若需要多组件协作，应在 recipe、CLI 或 `_transformers` integration 层连接。
- 修改 import-linter contract 或新增 cross-component exception 要有非常明确的架构说明。

### Hugging Face 兼容是验收标准

结论级别：明确事实

说明：项目不是只要内部训练能跑；外部 artifact/API 兼容同等重要。

证据：

- 文档：`docs/guides/huggingface-api-compatibility.md#Drop-In Compatibility and Key Differences`
- 源码：`nemo_automodel/_transformers/auto_model.py::NeMoAutoModelForCausalLM`
- 源码：`nemo_automodel/__init__.py::__all__` 导出 `NeMoAutoModelForCausalLM`、VLM、sequence、retrieval、tokenizer、diffusion pipeline
- 文档：`docs/guides/checkpointing.md`（HF-compatible checkpoint layout）
- PR：#2117 Qwen3.5 checkpoint 隐藏 `_fp32_params` 包装以保证 HF load
- Issue：#2118 先用 loss/eval/adapter load 证据定位 QLoRA 问题，避免把 merge/load 问题误判为训练问题

对后续开发的要求：

- 修改 model/checkpoint/PEFT 时必须说明 HF load/save、adapter key、missing/unexpected key、safetensors layout 是否变化。
- 不要改 forward/generation/weight shape 等 public behavior，除非有迁移计划。
- 外部工具兼容（vLLM/SGLang/Transformers v4/v5）需要作为 PR risk 写清楚。

### 显式配置优于隐式魔法

结论级别：强推断

说明：项目使用 YAML `_target_` 提供灵活性，但危险或过宽的自动匹配会被收紧。

证据：

- 文档：`docs/guides/configuration.md#Use _target_ for Instantiation`
- 源码：`nemo_automodel/components/config/loader.py::ConfigNode.instantiate`
- 源码：`nemo_automodel/components/config/loader.py::_resolve_target` 限制私有/dunder 与 import prefix
- 测试：`tests/unit_tests/config/test_allowed_import_prefixes.py`
- PR：#2022 从 `match_all_linear` 改为 `target_modules`，避免 LoRA 误伤 `lm_head` 并破坏 vLLM
- PR：#2077 in-batch negatives 使用 opt-in flag，默认不改变旧行为

对后续开发的要求：

- 新配置项默认应 backward compatible；启用新行为用显式字段、target list、strategy name 或 opt-in flag。
- 不要用 “match all / auto detect everything” 作为默认路径，尤其是 PEFT、checkpoint、distributed、security-sensitive 场景。

### 安全与 secret redaction 是配置/IO 边界

结论级别：明确事实

说明：配置可以解析 env var，但日志输出不得泄漏 secret；dataset/cache/checkpoint 读取要避免危险反序列化。

证据：

- 文档：`docs/guides/configuration.md#Prevent Secret Leakage in Logs`
- 源码：`nemo_automodel/components/config/loader.py::config_to_yaml_str`
- 测试：`tests/unit_tests/config/test_loader.py::test_confignode_repr_uses_orig_value_for_oc_env`
- PR：#2045 将 unsafe `torch.load` dataset cache 改为更安全路径
- 配置：`.github/workflows/detect-secrets.yml`

对后续开发的要求：

- 不要在日志/exception/serialized config 中打印 resolved secret。
- 新增 cache/checkpoint/dataset IO 必须说明反序列化安全策略，避免 pickle/RCE 默认路径。

### 证据驱动合入，而不是“能跑就行”

结论级别：强推断

说明：复杂模型、分布式、checkpoint、PEFT 改动的 review 关注 parity、loss、round-trip、CI recipe。

证据：

- PR：#2039 DeepSeek V4 Flash support 带 per-tensor parity、cosine similarity、full fine-tune loss 等说明
- PR：#2125 VLM context parallelism 带大量 unit tests 与 8xH100 convergence evidence
- PR：#2117 checkpoint 改动验证 `to_hf/from_hf` 与 HF load missing/unexpected key
- Issue：#2143 DeepSeek-V4 dtype 讨论要求区分 FP32 casting 的真实贡献和 reference parity
- Issue：#2118 维护者要求 loss curve、merge script、eval logs 后再判断 LoRA 质量问题

对后续开发的要求：

- PR 设计说明必须写清楚验证矩阵：unit、functional、GPU、HF round-trip、numerical parity、benchmark 哪些已跑/未跑。
- 性能改动应绑定可复现实验或 benchmark recipe，而不是只给理论判断。

## 4. 架构总览

```text
User / CI / Examples
        |
        v
automodel CLI / launchers
        |
        v
recipes/*  ──────────────── orchestrate training/eval/checkpoint loops
  |     |     |     |
  |     |     |     +--> components/loggers, optim, training, loss
  |     |     +--------> components/checkpoint, _peft, quantization
  |     +--------------> components/datasets
  +--------------------> _transformers / _diffusers integration
                           |
                           v
components/models + HF registry / adapters

shared/* is the low-level utility layer allowed below all layers.
components/* should remain mutually independent; recipes and integration layers compose them.
```

核心依赖方向：

- `shared/*` → 可被各层使用。
- `components/*` → 自包含，原则上不互相 import。
- `_transformers/*` / `_diffusers/*` → 连接 HF/Diffusers 与 AutoModel components。
- `recipes/*` → 组合 components、integration、config、checkpoint、logger、optimizer。
- `cli/*` → 解析 config/launcher/recipe target，不承载训练领域逻辑。

证据：`docs/repository-structure.md`；`pyproject.toml::tool.importlinter.contracts`；`nemo_automodel/recipes/llm/train_ft.py`；`nemo_automodel/cli/app.py`。

## 5. 模块边界

| 边界 | 允许 | 禁止 | 证据 |
|---|---|---|---|
| `components/*` 之间 | 各自实现独立能力；通过 public config/recipe 组合 | 为方便直接 cross-import 另一个 component；把 recipe 编排下沉到 component | `docs/repository-structure.md#Components Directory`; `pyproject.toml::tool.importlinter.contracts` |
| `recipes/*` | 导入并组合多个 components；实现 train/val/checkpoint loop；解析高层 config | 把可复用底层 primitive 写成 recipe 私有逻辑后让其他 recipe copy-paste | `nemo_automodel/recipes/llm/train_ft.py`; `docs/guides/overview.md` |
| `distributed:` config | 使用 fixed strategy schema：`strategy: fsdp2/ddp/megatron_fsdp` + size/options | 把 `distributed:` 当任意 `_target_` 注入；绕过 strategy validation | `docs/guides/configuration.md#Distributed Section`; `nemo_automodel/recipes/_dist_setup.py` |
| `MeshContext` | 保存 typed runtime mesh/parallelism source of truth | 在 component 层解析 YAML/dict 或隐藏 mesh 事实源 | `nemo_automodel/components/distributed/mesh.py::MeshContext` |
| `_transformers/registry.py` | 维护 architecture → model mapping；提供 runtime `register()` | 新 model folder 不注册；在多个通用文件散落 model-specific if/else | `nemo_automodel/_transformers/registry.py::MODEL_ARCH_MAPPING`; `tests/unit_tests/_transformers/test_registry.py`; Issue #2163 |
| Public API `nemo_automodel.__all__` | 懒加载导出 `NeMoAutoModel*` / tokenizer / pipeline | eager import heavy torch/model deps；随意删除/改名 public wrapper | `nemo_automodel/__init__.py::__all__`; `tests/unit_tests/test_lazy_imports.py` |
| Checkpoint / PEFT artifacts | 保存 HF-compatible safetensors/config/tokenizer/adapter keys；测试 round-trip | 只保证内部 resume；破坏 HF/vLLM load；引入 unsafe pickle 默认 | `docs/guides/checkpointing.md`; PR #2117; PR #2045; Issue #2118 |
| CLI / launcher | 找 recipe target、分发 interactive/SLURM/SkyPilot/NeMo-Run | 在 CLI 写模型/训练细节；猜测集群配置 | `nemo_automodel/cli/app.py`; `docs/launcher/*`; `slurm.sub` |
| Tests | unit 覆盖组件/registry/config；functional/GPU 覆盖 recipe/distributed；CI recipe 固化场景 | 无测试改核心路径；GPU 测试不 skip/不说明硬件要求；污染全局状态 | `tests/unit_tests`; `tests/functional_tests`; PR #2167 |
| Docs / examples | 新能力配套 guide/example YAML/PR changelog | 只改代码不告诉用户如何配置或运行 | `.github/PULL_REQUEST_TEMPLATE.md`; `docs/guides/*`; `examples/*` |

### Public API / Internal API

- Public API：`nemo_automodel.NeMoAutoModelForCausalLM`、VLM/sequence/retrieval wrapper、`NeMoAutoTokenizer`、`NeMoAutoDiffusionPipeline`、documented YAML keys、CLI `automodel` invocation、HF-compatible checkpoints。
- Semi-public extension API：YAML `_target_` classes/functions、model registry mapping/runtime register、strategy dataclasses、dataset/loss/optimizer constructors、PEFT config。
- Internal API：`components/*` implementation details、private helpers、distributed manager internals、checkpoint adapter internals、test fixtures；可改但必须说明 blast radius 与兼容/round-trip evidence。

证据：`nemo_automodel/__init__.py::__all__`; `docs/guides/configuration.md`; `nemo_automodel/_transformers/registry.py`; `docs/guides/huggingface-api-compatibility.md`。

## 6. 已识别的设计模式

### Configuration Object + Dependency Injection

结论级别：明确事实

出现位置：

- `nemo_automodel/components/config/loader.py::ConfigNode`
- `nemo_automodel/components/config/loader.py::load_yaml_config`
- `nemo_automodel/recipes/llm/train_ft.py::build_dataset`
- examples YAML：`examples/llm_finetune/*/*.yaml`

解决的问题：用 YAML 描述可替换 component，保持 recipe 线性可读并避免硬编码。

为什么这是项目偏好的方式：docs 明确声明 components 通过 `_target_` 被 recipes 组合；functional tests 大量通过 CLI override 替换 `_target_`。

后续开发如何遵循：新增 dataset/loss/optimizer/logger/model loader 时提供可从 `_target_` 调用的 class/function，并让 YAML 参数与构造签名一致。

不应该怎么用：不要让 `distributed:` 成为任意 `_target_`；不要启用 out-of-tree user module 默认路径；不要让 secret 出现在 config repr。

证据：

- 源码：`nemo_automodel/components/config/loader.py::ConfigNode.instantiate`
- 测试：`tests/unit_tests/config/test_loader.py`
- 文档：`docs/guides/configuration.md#Use _target_ for Instantiation`
- PR / Issue / Review：#2022、#2077 体现显式配置和 opt-in
- 高频 contributor commits：`afea54d fix: switch from match_all_linear to target_modules (#2022)`

### Registry / Factory

结论级别：明确事实

出现位置：

- `nemo_automodel/_transformers/registry.py::MODEL_ARCH_MAPPING`
- `nemo_automodel/_transformers/registry.py::_ModelRegistry.register`
- `nemo_automodel/_transformers/auto_model.py::NeMoAutoModelForCausalLM.from_pretrained`
- `nemo_automodel/cli/utils.py::find_recipe_target`

解决的问题：把 architecture/recipe target 名称映射到实现，支持 HF-style auto loading 与 CLI recipe discovery。

为什么这是项目偏好的方式：项目要支持大量 Hugging Face model family；registry 能集中 external contract，避免用户手动 import model implementation。

后续开发如何遵循：新增 model 时不仅添加 `components/models/<name>`，还要注册 architecture mapping/runtime register，并补 registry coverage test。

不应该怎么用：不要让模型特例散落在 checkpoint/collate/parallelizer 通用文件；Issue #2163 已指出这是当前需要收敛的问题。

证据：

- 源码：`nemo_automodel/_transformers/registry.py::MODEL_ARCH_MAPPING`
- 测试：`tests/unit_tests/_transformers/test_registry.py::test_all_model_folders_registered`
- 文档：`docs/model-coverage/overview.md`
- PR / Issue / Review：Issue #2163 RFC model zoo vs infra
- 高频 contributor commits：DeepSeek/GPT-OSS/Qwen/Nemotron model commits 常同步改 model + registry + tests

### Adapter / Facade around Hugging Face

结论级别：明确事实

出现位置：

- `nemo_automodel/_transformers/auto_model.py::NeMoAutoModelForCausalLM`
- `nemo_automodel/_transformers/auto_tokenizer.py::NeMoAutoTokenizer`
- `nemo_automodel/_diffusers/auto_diffusion_pipeline.py`
- `nemo_automodel/components/checkpoint/*`

解决的问题：复用 HF mental model，同时插入 distributed、checkpoint、kernel、registration、compat shims。

为什么这是项目偏好的方式：docs 使用 “drop-in wrapper” 与 “HF-compatible checkpoints”；PR review 以 HF load/round-trip 作为验收。

后续开发如何遵循：对外保持 HF method 名、config、tokenizer、checkpoint layout；内部优化用 adapter/shim，不改变外部语义。

不应该怎么用：不要让 wrapper API 暴露 AutoModel 内部 manager 细节；不要使 HF `from_pretrained` 无法加载保存产物。

证据：

- 源码：`nemo_automodel/_transformers/auto_model.py::NeMoAutoModelForCausalLM`
- 文档：`docs/guides/huggingface-api-compatibility.md`
- 文档：`docs/guides/checkpointing.md`
- PR：#2117 checkpoint HF load compatibility
- Issue：#2118 PEFT/vLLM adapter load discussion

### Strategy

结论级别：明确事实

出现位置：

- `nemo_automodel/components/distributed/config.py` strategy dataclasses
- `nemo_automodel/components/distributed/mesh_utils.py::STRATEGY_MAP`
- `nemo_automodel/_transformers/infrastructure.py` manager instantiation
- `nemo_automodel/recipes/_dist_setup.py`

解决的问题：用固定 schema 表示 DDP/FSDP2/MegatronFSDP 及 TP/PP/CP/EP 组合，避免 arbitrary class injection。

为什么这是项目偏好的方式：分布式语义需要强 validation、typed mesh、capability checks，不能仅靠动态 `_target_`。

后续开发如何遵循：新增 strategy/backend 时定义 dataclass、map、validation、manager、tests，并说明与 TP/PP/CP/EP/FSDP2 的交互。

不应该怎么用：不要在 recipe 中用 string if/else 散落分布式行为；不要让 train/val gate 不一致。

证据：

- 源码：`nemo_automodel/components/distributed/config.py`
- 源码：`nemo_automodel/components/distributed/mesh_utils.py::create_mesh`
- 文档：`docs/guides/configuration.md#Distributed Section`
- 测试：`tests/unit_tests/test_validation.py`
- PR：#2125 context parallelism review comments

### Pipeline / Imperative Shell

结论级别：强推断

出现位置：

- `nemo_automodel/recipes/llm/train_ft.py`
- `nemo_automodel/recipes/base_recipe.py`
- `nemo_automodel/components/training/step_scheduler.py`
- `tests/functional_tests/*/L2_*.sh`

解决的问题：训练是 stateful、distributed、IO-heavy workflow；用线性 recipe 显式展示 load → prepare → train → validate → checkpoint。

为什么这是项目偏好的方式：docs 反对 rigid trainer，强调 linear scripts 和 YAML；functional tests 以 shell/YAML recipe 作为 E2E 合约。

后续开发如何遵循：新增 task/recipe 时保持训练步骤可读，复用 components；不要把 workflow 隐藏进深继承 trainer。

不应该怎么用：不要 copy-paste 大段 recipe 逻辑；可复用 piece 应抽成 component/shared helper。

证据：

- 文档：`docs/repository-structure.md#Recipes Directory`
- 源码：`nemo_automodel/recipes/llm/train_ft.py`
- 测试：`tests/functional_tests/*`
- 高频 contributor commits：热点目录 `examples/llm_finetune`、`tests/unit_tests`、`tests/functional_tests`、`recipes/llm`

### Lazy Import / Optional Dependency Boundary

结论级别：明确事实

出现位置：

- `nemo_automodel/__init__.py::__getattr__`
- `nemo_automodel/_transformers/__init__.py::__getattr__`
- `nemo_automodel/shared/import_utils.py`
- `tests/unit_tests/test_lazy_imports.py`

解决的问题：避免 import package 时拉起 heavy torch/model/optional dependency；提高 CPU/no-CUDA install 可用性。

为什么这是项目偏好的方式：CI 有 no-CUDA install tests，optional CUDA deps 很多，lazy import 行为有单元测试保护。

后续开发如何遵循：新增顶层 public export 应放入 lazy mapping 并补测试；optional dependency import 失败要给清晰提示。

不应该怎么用：不要在 `__init__.py` eager import CUDA/TE/DeepEP/bitsandbytes 等重依赖。

证据：

- 源码：`nemo_automodel/__init__.py::__getattr__`
- 源码：`nemo_automodel/shared/import_utils.py`
- 测试：`tests/unit_tests/test_lazy_imports.py`
- CI：`.github/workflows/install-test.yml` no-CUDA install matrix

### Compatibility Shim / Patch

结论级别：强推断

出现位置：

- `nemo_automodel/shared/patches.py`
- `nemo_automodel/shared/import_utils.py`
- `_transformers` wrappers and registry
- model-specific code under `components/models/*`

解决的问题：上游 Transformers/PyTorch/TE/NGC 演进很快，项目用小 shim 保持用户-facing API 与 artifact 稳定。

为什么这是项目偏好的方式：历史提交大量是 fix/compat regression；文档承认 Transformers v4/v5 forward-compat shims。

后续开发如何遵循：shim 应小、局部、测试覆盖，并说明何时可删除；不要把临时 workaround 永久埋进核心路径。

不应该怎么用：不要为过期环境加大范围分支；不要未定位根因就改 checkpoint/model 通用语义。

证据：

- 文档：`docs/guides/huggingface-api-compatibility.md#Transformers v5`
- PR：#2025 tokenizer+auto_map regression with transformers 5.5.0
- PR：#2070 关闭过期 workaround
- PR：#2071 root cause fixed elsewhere，避免错误 checkpoint patch
- 高频 contributor commits：`b74cdd5 fix: tokenizer+auto_map regression with transformers 5.5.0 (#2025)`

## 7. gh CLI 变更脉络分析

### 仓库基础信息

命令：

```bash
gh repo view --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo
```

结论级别：明确事实

- 默认分支：`main`
- 描述：PyTorch Distributed native training library for LLMs/VLMs with OOTB Hugging Face support
- topics 覆盖 `llm`、`vlm`、`finetuning`、`llama`、`qwen3`、`gpt-oss`、`deepseek-v4`、`agent` 等，说明项目主要围绕新模型快速支持与训练能力演进。
- licenseInfo 当前 gh 返回 `null`，但仓库存在 `LICENSE` 文件；准确 license 元数据需维护者确认。

### 最近 merged PR 偏好

命令：

```bash
gh pr list --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions
```

强信号：

- 多数近期 PR 是小修复/CI/docs：#2187 CI queue 1 file，#2173 DSV4 MoE fp32 clamp 1 file，#2159 DeepSeek V4 transpose 2 files，#2152 sparse attention mask 2 files。
- 大功能可接受但需要证据：#2125 VLM context parallelism 13 files / 1654 additions；#2039 DeepSeek V4 Flash support 大 PR，但有 parity/loss/PP stage 说明。
- Docs-only PR 有 label：#2157、#2138、#2126、#2124 等，表明文档路径与代码路径在 review/CI 中分流。

设计偏好：小 PR 默认更容易合入；复杂 PR 必须显式证明 correctness、compatibility、distributed behavior。

### 深入关键 PR / Review

命令示例：

```bash
gh pr view 2022 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels
gh pr view 2125 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels
gh api --paginate repos/NVIDIA-NeMo/Automodel/pulls/2125/comments
```

| PR | 设计信号 | 结论 |
|---|---|---|
| #2022 `match_all_linear -> target_modules` | LoRA 不应全局匹配 `lm_head`，否则破坏 vLLM 预期 | 显式 target 优于广义自动匹配 |
| #2045 unsafe `torch.load` | dataset cache 也被视为安全边界 | 安全默认值高优先级 |
| #2117 Qwen3.5 checkpoint | 隐藏 `_fp32_params` 包装，保证 HF load | checkpoint 外部兼容是验收标准 |
| #2125 VLM CP | review 关注 FSDP2 hook、CP/PP gate、validation `no_grad`、padding/label | 分布式路径 train/val 一致性必须证明 |
| #2039 DeepSeek V4 Flash | PR body 拆解 parity bug、loss curve、full fine-tune | 大 PR 可接受条件是证据充分、问题拆分清楚 |
| #2167 sys.modules pollution | test fixture 污染导致 order-dependent failures | 测试隔离是设计约束 |

### 最近 closed 未合并 PR 偏好

命令：

```bash
gh pr list --state closed --limit 100 --json number,title,author,closedAt,labels,comments,reviews
```

反复出现的问题：

- #2036 标题是 “fix 2 minor bugs”，但 diff 超大（子代理记录 1415 files / 291,776 additions），且 pre-check 未完成；属于不可审查 PR。
- #2070 为过期 runtime workaround，关闭而不是加核心复杂度。
- #2071 作者后续确认 root cause elsewhere，避免误改 checkpoint 通用语义。
- #2105 dependency bump 缺少动机/风险/测试上下文，关闭未合并。
- 多个 `beep boop` automation bump PR 未合并，说明 release automation 噪声会被过滤。

### Issue 中的设计讨论

命令：

```bash
gh issue list --state all --limit 100 --json number,title,author,createdAt,closedAt,labels,comments
gh issue view 2163 --json number,title,body,comments,labels,state
gh issue view 2143 --json number,title,body,comments,labels,state
gh issue view 2118 --json number,title,body,comments,labels,state
```

| Issue | 设计信号 | 后续要求 |
|---|---|---|
| #2163 RFC model zoo vs infra | model-specific logic 不应散落在 `parallelizer.py`、`optimized_tp_plans.py`、collate、checkpoint mapping | 新模型尽量 model-local，通用层只做 registry/dispatch |
| #2143 DeepSeek-V4 dtype | 维护者区分 FP32 对 reference parity 与训练指标影响 | dtype/precision 改动需要 ablation/parity evidence |
| #2118 LoRA regression | 维护者要求 loss curve、merge script、eval logs，最终指向 adapter/merge/load | 不要无证据归因训练质量；先做 input/artifact parity |
| #2115 Gemma4 CP with TE | backend 限制（head_dim、bidirectional attention、TE support）影响方案选择 | 新 backend 要陈述外部库能力边界与 fallback |
| #2164 MiniMax PEFT | 新模型 PEFT 构造兼容问题仍在 | model-specific constructor/adapter 兼容需要独立测试 |

## 8. 高频 Contributor 设计习惯

命令：

```bash
git shortlog -sn --all
git log --author="<AUTHOR>" --stat --oneline --date=short
git log --author="<AUTHOR>" --name-only --pretty=format:"%h %ad %s" --date=short
```

| Contributor | 高频修改模块 | 稳定设计习惯 | 代表 commits / PR | 对后续开发的启发 |
|---|---|---|---|---|
| Alexandros Koumparoulis | `examples/llm_finetune`, `tests/*`, `docs/guides`, `components/distributed`, `components/models` | 小粒度修复；兼容 HF/Transformers/vLLM；配置收紧 | `b74cdd5` Transformers 5.5.0 regression (#2025); `6853a56` PP seq len (#2024); `afea54d` target_modules (#2022) | PR 应围绕一个兼容/配置问题，补对应 test/example |
| Dong Hyuk Chang | CI、recipe list、functional tests、docker、examples | CI recipe 是产品合约；按 recipe triage | `d21d45d` vLLM deploy rc9 failures (#2047); `f4155ad` pipeline benchmark failures (#2040); `2e97d36` per-recipe env_vars (#1999) | 新能力要考虑 CI queue、recipe env、runtime matrix |
| Hemil Desai | `components/models`, `components/moe`, `components/distributed`, benchmark, LLM recipes | MoE/PP/TP 改动配合模型默认值、性能与 tests | `46e4ba7` MoE gate bias defaults (#1768); `7ab43ae` dynamic seq length for PP (#1689); `5106474` GPT-OSS THD fix (#1757) | 性能/分布式修复要限定作用域并给数值/recipe 依据 |
| Huiying / HuiyingLi | VLM recipes/examples, datasets, models, unit tests, docs | 多模态路径同步改 collate/dataset/model/tests/docs；重视 parity | `2bfb7de` NemotronOmni v3 dump + multimodal collate; `35c8da6` NemotronOmni tests; #2143/#2118 comments | VLM/新模型不要只改模型，必须覆盖 data path 和 eval evidence |
| Adil / adil-a | checkpoint robustness, PEFT recipes, functional tests, examples | 批量 checkpoint robustness 修复；HF dynamic module/race 兼容 | `b0a5aab` ckpt robustness (#1971); `375f24d` PEFT/ckpt robustness (#1984); `b403c9e` HellaSwag ckpt robustness | checkpoint/PEFT PR 要测试 resume + HF conversion + real recipe |
| Charlie Truong | CI、安全/docs、packaging、lockfile | 安全/packaging/CI 基础设施保守维护 | `439e95e` add SECURITY.md (#1996); `8564476` revert uv.lock for NGC CUDA (#1534) | 依赖/lockfile 变更必须说明 CUDA/NGC impact |
| Zhiyu Li | models, distributed, TP/FSDP performance, recipes/tests | 局部分布式优化，绑定 pre-shard/prefetch/TP plan | `dccd977` FSDP2 weight prefetch + async TP (#1711); `99cfccb` TP plan lookup (#1547); `3fe3679` FSDP pre-shard (#1357) | 性能优化要局部、可测、保护既有 TP/FSDP semantics |
| svcnvidia-nemo-ci | release/cherry-pick automation | release branch 同步、`cp:` commits | `cp: ... into r0.4.0` 系列 | 向后维护重要；breaking changes 需 release/branch 策略 |
| oliver könig | GitHub workflows/actions, codecov, external contributor CI | 改 CI runner/codecov 时粒度小、目标明确 | `e2b976a` AWS ephemeral runners (#1892); `e5b8a06` codecov base_sha (#2016) | CI 更改要最小化 blast radius，并保留外部 contributor 路径 |

稳定习惯（强推断）：

- 热点目录不是只有核心源码，而是 `examples/llm_finetune`、`tests/unit_tests`、`tests/functional_tests`、`components/models`、`components/distributed`、`components/checkpoint`。
- `fix` 远多于 `refactor`，说明项目偏向兼容/回归驱动的增量演进，而不是大重构。
- 命名携带模型和场景（Qwen、DeepSeek、GPT-OSS、NemotronOmni、HellaSwag、ckpt-robustness），避免过早抽象。

## 9. 推荐扩展方式

| 扩展目标 | 推荐位置 | 推荐模式 | 必须测试 | 禁止做法 |
|---|---|---|---|---|
| 新模型 architecture | `nemo_automodel/components/models/<name>/`; `_transformers/registry.py`; examples/model coverage docs | Registry + Adapter + model-local ownership | registry test、HF `from_config/from_pretrained`、forward parity、TP/FSDP/PEFT 如适用 | 只新增 model folder 不注册；在通用 infra 到处加 model-name if/else |
| 新 dataset/collator | `components/datasets/{llm,vlm,...}` + example YAML | `_target_` factory + composition | unit dataset/collate；functional recipe smoke；tokenizer injection path | 改 recipe 硬编码某 dataset；copy-paste collate 逻辑 |
| 新 loss | `components/loss/` | `nn.Module` + `_target_` | unit loss shape/mask/reduction；functional recipe smoke | 在 train loop 中写 task-specific loss 分支 |
| 新 optimizer/scheduler | `components/optim/` 或 YAML external target | Config object + factory | unit config instantiation；known parameter groups test | 绕过 config；改变 default LR/WD 语义不说明 |
| 新 PEFT/adapter | `components/_peft/` + checkpoint addon/tests | Adapter + explicit target_modules | adapter key round-trip、HF PEFT load、checkpoint save/load、vLLM path 如适用 | `match_all` 默认；保存非 HF-compatible adapter keys |
| 新 distributed strategy/backend | `components/distributed/config.py`, `mesh_utils.py`, manager/infrastructure | Strategy dataclass + capability validation | unit validation；multi-rank/GPU functional；train/val gate；PP/CP/FSDP hook evidence | 任意 `_target_` 注入；只测单卡；不说明 backend limitation |
| 新 checkpoint format/adapter | `components/checkpoint/` | Adapter + round-trip contract | DCP/SafeTensors round-trip；HF load missing/unexpected=0；resume | unsafe pickle 默认；只测内部 save 不测外部 load |
| 新 recipe | `recipes/<domain>/` + `examples/<domain>/` | Pipeline / imperative shell + component composition | config target test；functional L2 script；docs link | 复制完整老 recipe 后分叉；在 CLI 写训练逻辑 |
| 新 CLI/launcher | `nemo_automodel/cli/`, `components/launcher/`, `docs/launcher/` | Command facade + launcher adapter | CLI parse test；dry-run / local launcher; docs | 猜 cluster config；硬编码路径/secrets/endpoints |
| 新 docs | `docs/guides/`, `docs/model-coverage/`, examples README | MyST/Sphinx docs + examples | docs build when feasible; link from index/toctree | Markdown 文件名用 underscore；只写泛泛说明无可运行命令 |
| 新依赖 | `pyproject.toml`, `uv.lock`, install CI | Dependency justification + optional/lazy import | no-CUDA install；NGC CUDA install；import failure message | 无动机 dependency bump；顶层 eager import CUDA-only dependency |

## 10. 不应破坏的不变量

1. **Component independence**：`components/*` 不应互相依赖；共享逻辑放 `shared` 或 recipe composition。
2. **HF public API compatibility**：`NeMoAutoModel*` wrapper 应维持 HF mental model、forward/generation/config/tokenizer 语义。
3. **HF-compatible checkpoint artifacts**：保存产物应支持 Transformers/vLLM/SGLang 等 downstream load；PEFT adapter keys 要兼容。
4. **`distributed:` fixed schema**：分布式配置不是 arbitrary `_target_`；strategy 必须可 validation。
5. **`MeshContext` typed source of truth**：YAML parsing 不应下沉到 distributed components。
6. **Lazy import / optional dependency**：顶层 import 不应触发重 CUDA/TE/DeepEP/bitsandbytes dependency。
7. **Secret redaction**：config repr/log 不得泄漏 env var secret。
8. **Safe deserialization**：dataset/cache/checkpoint 不应默认使用 unsafe pickle load。
9. **Capability validation**：TP/PP/CP/EP、sequence packing、model/backend capability 不应被绕过。
10. **Train/validation parity**：分布式路径的 padding、label mask、`no_grad`、CP/PP guards 要一致。
11. **CI/examples as contract**：examples 和 functional scripts 是用户-facing contract，不应轻易破坏。
12. **Small, reviewable PR**：大范围跨域重构需要 RFC/plan；普通 PR 应单意图。

## 11. 常见反模式

### 大而杂的 PR

表现：标题是“小修”，实际跨上千文件或多个无关系统。

为什么不符合本项目：项目维护多模型、多后端、多 CI recipe，巨大 diff 难以验证、回滚和 cherry-pick。

维护者或历史 PR 证据：PR #2036 closed；子代理记录为 1415 files / 291,776 additions，pre-check 未完成。

正确做法：拆成 model、config、checkpoint、docs/tests 独立 PR；每个 PR 有清晰 scope 和验证。

PR 自查问题：

- 这个 PR 是否能用一句话描述一个问题？
- 是否有无关格式化/生成文件/lockfile 混入？
- 是否能单独 revert？

### 破坏 component independence

表现：在 `components/datasets` 直接 import `components/checkpoint` 或在 `components/distributed` 直接依赖某 loss/dataset。

为什么不符合本项目：组件独立是文档与 import-linter contract 明确边界。

证据：`docs/repository-structure.md#Components Directory`; `pyproject.toml::tool.importlinter.contracts`。

正确做法：在 recipe 层组合，或把真正通用 helper 放 `shared`。

PR 自查问题：

- 新 import 是否从一个 component 指向另一个 component？
- 是否可以通过 `_target_` / recipe composition 解决？

### 模型专属 carve-out 写进通用 infra

表现：在 `parallelizer.py`、collate、checkpoint mapping、TP plan 中继续追加 model name/class name 分支。

为什么不符合本项目：Issue #2163 已把这类散落特例列为 model zoo/infra 混杂问题。

证据：Issue #2163 RFC。

正确做法：model-specific parallelizer/patch/preprocess/state-dict hook 尽量放在 `components/models/<name>/`，通用层只做注册/调度。

PR 自查问题：

- 这个分支是否只服务一个 model family？
- 是否可以通过 registry/adapter/hook 下沉到 model-local？

### 过宽自动匹配 / 魔法默认

表现：默认匹配所有 Linear、自动修改 `lm_head`、自动启用行为改变旧训练路径。

为什么不符合本项目：会破坏 vLLM/HF/PEFT 预期，且难以审查。

证据：PR #2022 从 `match_all_linear` 切到 `target_modules`；PR #2077 opt-in flag。

正确做法：显式 target list、opt-in flag、backward-compatible default。

PR 自查问题：

- 默认行为是否改变？
- 用户是否能精确指定 target？

### 无 parity / round-trip / GPU evidence 的核心改动

表现：修改 checkpoint、distributed、model dtype、PEFT merge，只跑了 import 或单元测试。

为什么不符合本项目：复杂改动风险在数值、外部 artifact、multi-rank interaction。

证据：PR #2039、#2125、#2117、Issue #2143、#2118。

正确做法：补充数值 parity、HF round-trip、loss/eval、multi-GPU functional 或明确未测试原因。

PR 自查问题：

- 是否证明了 HF/vLLM load？
- 是否证明了 train/val/multi-rank path？
- 是否说明未跑 GPU test 的硬件原因？

### 危险反序列化或 secret 泄漏

表现：新增 `torch.load` 默认读取外部 cache；打印 resolved env secret。

为什么不符合本项目：安全默认值已有明确测试和历史修复。

证据：PR #2045；`tests/unit_tests/config/test_loader.py::test_confignode_repr_uses_orig_value_for_oc_env`；`.github/workflows/detect-secrets.yml`。

正确做法：使用安全格式/weights-only/safetensors；config 输出走 redaction。

PR 自查问题：

- 外部输入是否可触发 pickle？
- 日志是否可能包含 token/host/path secret？

### 无上下文依赖 bump

表现：只改 `pyproject.toml`/`uv.lock`，没有说明 CUDA/NGC/HF/TE 影响与 install tests。

为什么不符合本项目：依赖与 CUDA/TE/NGC 兼容高度耦合。

证据：PR #2105 closed；`CONTRIBUTING.md` 提醒容器内 torch/uv sync CUDA mismatch；`.github/workflows/install-test.yml` 覆盖 no-CUDA/NGC CUDA install。

正确做法：写明动机、compat matrix、install test、optional import impact。

PR 自查问题：

- 是否影响 no-CUDA install？
- 是否影响 NGC CUDA / TE / bitsandbytes？

## 12. PR 设计说明模板

```markdown
# PR Design Explanation

## Problem

这个 PR 解决什么问题？为什么现在需要解决？请链接 Issue / failing test / upstream change。

## Scope

这个 PR 修改什么？明确不修改什么？

- In scope:
- Out of scope:

## Existing Design Followed

遵循了项目中的哪些既有设计模式、模块边界或 contributor 习惯？

证据：

- 源码：
- 测试：
- 文档：
- PR / Issue / Review：

## Alternatives Considered

考虑过哪些方案？为什么没有选择？

- 方案 A：拒绝原因
- 方案 B：拒绝原因

## Final Design

最终设计是什么？为什么符合项目设计哲学？

- 是否保持 component independence？
- 是否通过 recipe/config/registry/adapter 扩展？
- 是否保持 HF/checkpoint/PEFT/vLLM 兼容？
- 是否保持 explicit config / backward-compatible default？

## Compatibility

是否影响 public API、配置、数据格式、错误语义、性能或安全边界？

- Public API:
- YAML config:
- Checkpoint / safetensors / adapter keys:
- Distributed behavior:
- Performance:
- Security / deserialization / secrets:

## Tests

新增或修改了哪些测试？如何运行？

- Unit:
- Functional / GPU:
- HF round-trip / parity / benchmark:
- Not tested and why:

## Risk

维护者需要重点审查什么？

- Risk:
- Rollback plan:
```

## 13. 证据索引

### 仓库与文档

- 文档：`README.md`
- 文档：`CONTRIBUTING.md`
- 文档：`docs/repository-structure.md#Components Directory`
- 文档：`docs/repository-structure.md#Recipes Directory`
- 文档：`docs/guides/overview.md`
- 文档：`docs/guides/configuration.md`
- 文档：`docs/guides/huggingface-api-compatibility.md`
- 文档：`docs/guides/checkpointing.md`
- 文档：`docs/about/key-features.md`
- 文档：`SECURITY.md`

### 构建、CI、贡献流程

- 配置：`pyproject.toml::tool.importlinter.contracts`
- 配置：`pyproject.toml::tool.pytest.ini_options`
- 配置：`.github/PULL_REQUEST_TEMPLATE.md`
- 配置：`.github/CODEOWNERS`
- CI：`.github/workflows/semantic_pull_request.yml`
- CI：`.github/workflows/install-test.yml`
- CI：`.github/workflows/detect-secrets.yml`
- CI：`.github/workflows/cicd-main.yml`

### 核心源码

- 源码：`nemo_automodel/__init__.py::__all__`
- 源码：`nemo_automodel/__init__.py::__getattr__`
- 源码：`nemo_automodel/_transformers/auto_model.py::NeMoAutoModelForCausalLM`
- 源码：`nemo_automodel/_transformers/registry.py::MODEL_ARCH_MAPPING`
- 源码：`nemo_automodel/_transformers/registry.py::_ModelRegistry.register`
- 源码：`nemo_automodel/cli/app.py::main`
- 源码：`nemo_automodel/cli/utils.py::find_recipe_target`
- 源码：`nemo_automodel/components/config/loader.py::ConfigNode`
- 源码：`nemo_automodel/components/config/loader.py::_resolve_target`
- 源码：`nemo_automodel/components/distributed/mesh.py::MeshContext`
- 源码：`nemo_automodel/components/distributed/mesh_utils.py::create_mesh`
- 源码：`nemo_automodel/components/distributed/config.py`
- 源码：`nemo_automodel/recipes/_dist_setup.py`
- 源码：`nemo_automodel/recipes/llm/train_ft.py`
- 源码：`nemo_automodel/components/loss/masked_ce.py::MaskedCrossEntropy`
- 源码：`nemo_automodel/shared/import_utils.py`

### 测试

- 测试：`tests/unit_tests/config/test_loader.py`
- 测试：`tests/unit_tests/config/test_allowed_import_prefixes.py`
- 测试：`tests/unit_tests/_transformers/test_registry.py`
- 测试：`tests/unit_tests/test_lazy_imports.py`
- 测试：`tests/unit_tests/test_validation.py`
- 测试：`tests/functional_tests/conftest.py`
- 测试：`tests/functional_tests/*/L2_*.sh`

### gh CLI / git 命令

- gh CLI command：`gh repo view --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo`
- gh CLI command：`gh pr list --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions`
- gh CLI command：`gh pr view 2022 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels`
- gh CLI command：`gh pr view 2039 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels`
- gh CLI command：`gh pr view 2045 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels`
- gh CLI command：`gh pr view 2117 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels`
- gh CLI command：`gh pr view 2125 --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels`
- gh CLI command：`gh pr list --state closed --limit 100 --json number,title,author,closedAt,labels,comments,reviews`
- gh CLI command：`gh issue list --state all --limit 100 --json number,title,author,createdAt,closedAt,labels,comments`
- gh CLI command：`gh issue view 2163 --json number,title,body,comments,labels,state`
- gh CLI command：`gh issue view 2143 --json number,title,body,comments,labels,state`
- gh CLI command：`gh issue view 2118 --json number,title,body,comments,labels,state`
- git command：`git shortlog -sn --all`
- git command：`git log --author="<AUTHOR>" --stat --oneline --date=short`
- git command：`git log --author="<AUTHOR>" --name-only --pretty=format:"%h %ad %s" --date=short`

### PR / Issue / Review evidence

- PR：#2022 `fix: switch from match_all_linear to target_modules`
- PR：#2039 `feat: DeepSeek V4 Flash support`
- PR：#2045 unsafe `torch.load` dataset cache security fix
- PR：#2070 closed PyTorch DCP plan caching workaround
- PR：#2071 closed because root cause was fixed elsewhere
- PR：#2077 in-batch negatives opt-in flag
- PR：#2105 closed dependency bump without sufficient context
- PR：#2117 Qwen3.5 checkpoint HF compatibility
- PR：#2125 VLM context parallelism
- PR：#2167 sys.modules pollution in training fixtures
- Issue：#2163 RFC model zoo vs infra
- Issue：#2143 DeepSeek-V4 HC mixer dtype sensitivity
- Issue：#2118 LoRA/QLoRA regression investigation with loss/eval/merge artifacts
- Issue：#2115 Gemma4 CP with TE backend limitations
- Issue：#2164 MiniMax PEFT constructor compatibility

### 重要未知 / 需维护者确认

- 未知：`licenseInfo` 在 `gh repo view` 返回 `null`，但仓库含 `LICENSE`；是否为 GitHub metadata 配置问题需维护者确认。
- 弱推断：Issue #2163 是 RFC，说明方向强烈，但具体 model-local plugin/hook 方案尚未全部落地。
- 弱推断：文档中 Transformers v4/v5 描述与当前 `pyproject.toml` pins 存在时效差异；以代码/lockfile/CI 当前状态为准，文档需更新。
- 未知：部分内部 review 可能在 NVIDIA 内部系统，不完全反映在 GitHub comments。
