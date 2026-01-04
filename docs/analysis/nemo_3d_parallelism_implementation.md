# NeMo AutoModel 3D Parallelism Implementation Analysis

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [5D DeviceMesh 结构](#5d-devicemesh-结构)
4. [Pipeline Parallelism 实现](#pipeline-parallelism-实现)
5. [FSDP2 实现](#fsdp2-实现)
6. [Tensor Parallelism 集成](#tensor-parallelism-集成)
7. [三者组合机制](#三者组合机制)
8. [DTensor 与通信模式](#dtensor-与通信模式)
9. [代码示例与工作流](#代码示例与工作流)
10. [性能优化策略](#性能优化策略)
11. [总结](#总结)

---

## 概述

### 什么是 3D Parallelism?

在深度学习分布式训练中，**3D Parallelism** 指的是三种并行策略的组合：

1. **Pipeline Parallelism (PP)**: 垂直切分模型，将不同层分配到不同设备
2. **Data Parallelism (DP)**: 数据并行，每个设备复制完整模型但处理不同数据
3. **Tensor Parallelism (TP)**: 张量并行，水平切分模型参数

NeMo AutoModel 实现了 **Torch-native 3D Parallelism**，完全基于 PyTorch 原生 API：

- **PyTorch DeviceMesh**: 多维设备网格
- **PyTorch FSDP2**: Fully Sharded Data Parallel 第二代
- **PyTorch DTensor**: 分布式张量与 placement 策略
- **PyTorch Pipeline**: 原生 pipeline 实现

### 核心特性

1. **完全 Torch-native**: 无需自定义通信原语，全部使用 PyTorch 内置 API
2. **可组合性 (Composability)**: PP、DP、TP 可任意组合，通过 DeviceMesh 配置
3. **SPMD 模型**: 单程序多数据，同一脚本可运行在 1 GPU 或 1000+ GPU
4. **灵活的 DeviceMesh**: 实际使用 **5D mesh** (pp, dp_replicate, dp_shard, cp, tp) 实现丰富的并行策略
5. **自动化**: AutoPipeline 自动切分模型，FSDP2Manager 自动应用并行策略

### 关键文件路径

| 文件路径 | 功能描述 |
|---------|---------|
| `nemo_automodel/components/distributed/fsdp2.py` | FSDP2Manager - 管理 DeviceMesh 和 FSDP2 并行化 |
| `nemo_automodel/components/distributed/parallelizer.py` | 并行化策略模式，应用 FSDP2+TP |
| `nemo_automodel/components/distributed/pipelining/autopipeline.py` | AutoPipeline - 自动模型切分和 pipeline 管理 |
| `nemo_automodel/components/distributed/pipelining/functional.py` | Pipeline 底层函数，切分模型、构建 schedule |
| `nemo_automodel/recipes/llm/train_ft.py` | LLM 训练 recipe，展示 3D 并行的端到端使用 |

---

## 架构设计

### 整体架构

NeMo AutoModel 的 3D Parallelism 采用分层设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    Recipe Layer (训练脚本)                    │
│  - train_ft.py: LLM 训练                                     │
│  - 配置化启用 PP, DP, TP                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Components Layer (组件)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ AutoPipeline │  │ FSDP2Manager │  │ Parallelizer │      │
│  │   (PP切分)    │  │  (DeviceMesh) │  │  (FSDP2+TP)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PyTorch Native APIs (底层)                      │
│  - DeviceMesh: 多维设备网格                                   │
│  - fully_shard (FSDP2): 参数分片                             │
│  - parallelize_module (TP): 张量并行                         │
│  - PipelineStage, PipelineSchedule: Pipeline 调度            │
│  - DTensor: 分布式张量与 placement (Shard, Replicate)        │
└─────────────────────────────────────────────────────────────┘
```

### 设计原则

#### 1. Component Independence (组件独立性)

NeMo AutoModel 严格遵循组件独立原则：

- **Components 不能相互导入**: `components/distributed/`, `components/checkpoint/`, `components/loss/` 等模块互不依赖
- **Recipes 组合 Components**: `recipes/llm/train_ft.py` 导入并组合各组件
- **强制执行**: `import-linter` 在 CI 中检查组件依赖关系

#### 2. Torch-native Philosophy (Torch 原生哲学)

- **不重新发明轮子**: 使用 PyTorch 原生 `DeviceMesh`, `fully_shard`, `parallelize_module`
- **不自定义通信**: 所有集合通信 (all-reduce, all-gather) 由 PyTorch 管理
- **不绑定框架**: 可与任何 HuggingFace 模型无缝集成

#### 3. SPMD Programming Model (单程序多数据)

```python
# 同一脚本，不同设备配置自动适配
# 1 GPU: 无并行
FSDP2Manager(tp_size=1, dp_size=1, pp_size=1)

# 8 GPU: 2D 并行 (DP×TP)
FSDP2Manager(tp_size=2, dp_size=4, pp_size=1)

# 64 GPU: 3D 并行 (PP×DP×TP)
FSDP2Manager(pp_size=4, dp_size=4, tp_size=4)
```

#### 4. Strategy Pattern (策略模式)

`parallelizer.py` 使用策略模式支持不同模型的并行化：

```python
# 默认策略：适用于大多数 Transformer 模型
class DefaultParallelizationStrategy(ParallelizationStrategy)

# 特化策略：NemotronH (Mamba 混合架构)
class NemotronHParallelizationStrategy(ParallelizationStrategy)

# 特化策略：Diffusion Transformer (Wan 架构)
class WanParallelizationStrategy(ParallelizationStrategy)

# 策略注册表
PARALLELIZATION_STRATEGIES: Dict[str, ParallelizationStrategy]
```

---

## 5D DeviceMesh 结构

### DeviceMesh 概念

PyTorch `DeviceMesh` 是一个多维设备网格，用于描述分布式训练拓扑：

```python
from torch.distributed.device_mesh import init_device_mesh

# 2D mesh: [data_parallel=4, tensor_parallel=2]
mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
```

### NeMo AutoModel 的 5D Mesh

虽然称为"3D Parallelism"，但 NeMo AutoModel 实际使用 **5D DeviceMesh**：

```python
# 来自 fsdp2.py:216
mesh_shape = (self.pp_size, self.dp_replicate_size, self.dp_shard_size, self.cp_size, self.tp_size)
mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
```

#### 5 个维度详解

| 维度 | 名称 | 含义 | 用途 |
|-----|------|------|------|
| 0 | `pp` | Pipeline Parallel | 模型垂直切分，每个 pipeline stage 一个 rank |
| 1 | `dp_replicate` | Data Parallel Replicate | HSDP 的 replicate 维度 (跨节点复制) |
| 2 | `dp_shard` | Data Parallel Shard | HSDP 的 shard 维度 (节点内分片) |
| 3 | `cp` | Context Parallel | 序列维度切分 (长序列并行) |
| 4 | `tp` | Tensor Parallel | 张量维度切分 (参数并行) |

#### HSDP (Hierarchical Sharded Data Parallel)

HSDP 是 FSDP2 的分层版本，适用于多节点训练：

```
┌─────────── dp_size = 8 ──────────┐
│                                   │
│  ┌── dp_replicate_size = 2 ──┐   │
│  │                            │   │
│  │  ┌ dp_shard_size = 4 ┐    │   │
│  │  │ Node 0:           │    │   │
│  │  │ [GPU0 GPU1        │    │   │
│  │  │  GPU2 GPU3]       │    │   │
│  │  └───────────────────┘    │   │
│  │                            │   │
│  │  ┌ dp_shard_size = 4 ┐    │   │
│  │  │ Node 1:           │    │   │
│  │  │ [GPU4 GPU5        │    │   │
│  │  │  GPU6 GPU7]       │    │   │
│  │  └───────────────────┘    │   │
│  └────────────────────────────┘   │
└───────────────────────────────────┘

- 节点内 (dp_shard): all-gather 参数，共享梯度
- 节点间 (dp_replicate): all-reduce 梯度，复制参数
```

### DeviceMesh 初始化

源代码位置: `nemo_automodel/components/distributed/fsdp2.py:215-255`

```python
def _get_device_mesh(self):
    mesh_shape = (self.pp_size, self.dp_replicate_size, self.dp_shard_size, self.cp_size, self.tp_size)
    mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")

    # 断言所有维度 > 0
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), "Expected {} to be an int, but got {}".format(name, type(shape))
        assert shape > 0, "Expected {} > 0, {}".format(name, shape)

    # 构建 5D mesh
    self.device_mesh = init_device_mesh(
        device_type="cuda" if self.backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    # 创建 submesh 以初始化所有必需的 process group
    dp_mesh_dim_names = []
    dp_shard_cp_mesh_dim_names = []
    dp_cp_mesh_dim_names = []

    # dp_replicate 维度
    dp_mesh_dim_names.append("dp_replicate")
    dp_cp_mesh_dim_names.append("dp_replicate")

    # dp_shard 维度
    dp_mesh_dim_names.append("dp_shard")
    dp_shard_cp_mesh_dim_names.append("dp_shard")
    dp_cp_mesh_dim_names.append("dp_shard")

    # cp 维度
    dp_shard_cp_mesh_dim_names.append("cp")
    dp_cp_mesh_dim_names.append("cp")

    # 创建 flattened submesh
    # submesh for dp (用于数据加载，无通信)
    self.device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

    # submesh for dp_shard_cp (用于参数分片)
    self.device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")

    # submesh for dp_cp (用于 loss all-reduce)
    self.device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

    return self.device_mesh
```

### Submesh 的作用

DeviceMesh 支持切片 (slicing) 创建 submesh：

```python
# 完整 5D mesh: (pp=2, dp_replicate=2, dp_shard=2, cp=1, tp=2)
device_mesh = init_device_mesh("cuda", mesh_shape=(2, 2, 2, 1, 2),
                               mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"))

# Submesh 示例
tp_mesh = device_mesh["tp"]                              # 只有 TP 维度
dp_mesh = device_mesh[("dp_replicate", "dp_shard")]      # HSDP 维度
dp_shard_cp_mesh = device_mesh[("dp_shard", "cp")]       # 用于 FSDP 分片
```

#### 三种 Submesh 用途

1. **`dp` mesh** (`dp_replicate` + `dp_shard`):
   - 用于数据加载和分布
   - 每个 dp rank 处理不同的数据批次
   - 无通信，仅用于数据划分

2. **`dp_shard_cp` mesh** (`dp_shard` + `cp`):
   - 用于 FSDP2 参数分片
   - 参数在这个 mesh 的所有 ranks 上分片
   - all-gather 用于前向/反向

3. **`dp_cp` mesh** (`dp_replicate` + `dp_shard` + `cp`):
   - 用于 loss all-reduce
   - 梯度在这个 mesh 上平均
   - 保证训练一致性

### 维度推导逻辑

源代码位置: `nemo_automodel/components/distributed/fsdp2.py:167-209`

```python
# 默认值推导
if self.tp_size is None or self.tp_size <= 0:
    self.tp_size = 1

if self.cp_size is None or self.cp_size <= 0:
    self.cp_size = 1

if self.pp_size is None or self.pp_size <= 0:
    self.pp_size = 1

# dp_size 自动推导
if self.dp_size is None or self.dp_size <= 0:
    # dp_size = world_size / (tp_size * cp_size * pp_size)
    total_parallel_ranks = self.tp_size * self.cp_size * self.pp_size
    if self.world_size % total_parallel_ranks != 0:
        raise ValueError(
            f"world_size ({self.world_size}) must be divisible by (tp_size * cp_size * pp_size) "
            f"({self.tp_size} * {self.cp_size} * {self.pp_size} = {total_parallel_ranks})"
        )
    self.dp_size = self.world_size // total_parallel_ranks

# dp_replicate_size 默认为 1 (无 HSDP)
if self.dp_replicate_size is None or self.dp_replicate_size <= 0:
    self.dp_replicate_size = 1

# dp_shard_size 推导
assert self.dp_size % self.dp_replicate_size == 0, "dp_size must be a multiple of dp_replicate_size"
self.dp_shard_size = self.dp_size // self.dp_replicate_size
```

#### 配置示例

**示例 1: 纯 DP (8 GPUs)**

```python
FSDP2Manager(
    world_size=8,
    pp_size=1,      # 无 pipeline
    tp_size=1,      # 无 tensor parallel
    cp_size=1,      # 无 context parallel
    dp_size=8       # 自动推导
)
# Mesh shape: (1, 1, 8, 1, 1)
```

**示例 2: 2D 并行 DP×TP (8 GPUs)**

```python
FSDP2Manager(
    world_size=8,
    pp_size=1,
    tp_size=2,      # 2-way tensor parallel
    cp_size=1,
    dp_size=4       # 自动推导
)
# Mesh shape: (1, 1, 4, 1, 2)
```

**示例 3: 3D 并行 PP×DP×TP (64 GPUs)**

```python
FSDP2Manager(
    world_size=64,
    pp_size=4,      # 4-stage pipeline
    tp_size=4,      # 4-way tensor parallel
    cp_size=1,
    dp_size=4       # 自动推导
)
# Mesh shape: (4, 1, 4, 1, 4)
```

**示例 4: HSDP 多节点 (16 GPUs, 2 nodes)**

```python
FSDP2Manager(
    world_size=16,
    pp_size=1,
    tp_size=2,
    cp_size=1,
    dp_size=8,           # 总 DP size
    dp_replicate_size=2  # 2 个副本 (2 nodes)
)
# Mesh shape: (1, 2, 4, 1, 2)
# dp_shard_size = dp_size / dp_replicate_size = 8 / 2 = 4
```

---

## Pipeline Parallelism 实现

### Pipeline Parallelism 概念

Pipeline Parallelism 将模型垂直切分为多个 stage，每个 stage 在不同的设备上：

```
┌──────────────────────────────────────────────────────────┐
│ GPU 0: Stage 0                                           │
│   [Embedding + Layers 0-7]                               │
└──────────────────────────────────────────────────────────┘
                    ↓ (activation tensor)
┌──────────────────────────────────────────────────────────┐
│ GPU 1: Stage 1                                           │
│   [Layers 8-15]                                          │
└──────────────────────────────────────────────────────────┘
                    ↓ (activation tensor)
┌──────────────────────────────────────────────────────────┐
│ GPU 2: Stage 2                                           │
│   [Layers 16-23]                                         │
└──────────────────────────────────────────────────────────┘
                    ↓ (activation tensor)
┌──────────────────────────────────────────────────────────┐
│ GPU 3: Stage 3                                           │
│   [Layers 24-31 + LM Head]                               │
└──────────────────────────────────────────────────────────┘
```

### AutoPipeline 架构

源代码位置: `nemo_automodel/components/distributed/pipelining/autopipeline.py`

#### 核心类

```python
@dataclass
class PipelineInfo:
    """Pipeline 状态信息"""
    enabled: bool = False
    schedule: Optional[Any] = None
    has_first_stage: bool = False
    has_last_stage: bool = False
    model_parts: Optional[list[nn.Module]] = None
    stages: Optional[list[Any]] = None


class AutoPipeline:
    """自动化 Pipeline 并行管理器

    功能：
    1. 自动切分模型为多个 stage
    2. 为每个 stage 应用 FSDP2+TP 并行化
    3. 构建 pipeline schedule (GPipe, 1F1B, Interleaved 1F1B)
    4. 管理 pipeline 训练循环
    """

    def __init__(
        self,
        world_mesh: DeviceMesh,
        moe_mesh: Optional[DeviceMesh] = None,
        pp_axis_name: str = "pp",
        dp_axis_names: Union[tuple[str, ...], str] = ("dp_replicate", "dp_shard_cp"),
        cp_axis_name: str = "cp",
        tp_axis_name: str = "tp",
        ep_axis_name: str | None = None,
        ep_shard_axis_names: tuple[str, ...] | None = None,
        layers_per_stage: Optional[int] = None,
        pp_schedule: str = "1f1b",
        pp_schedule_csv: Optional[str] = None,
        pp_microbatch_size: int = 1,
        pp_batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        module_fqns_per_model_part: Optional[list[list[str]]] = None,
        patch_inner_model: bool = True,
        patch_causal_lm_model: bool = True,
        round_virtual_stages_to_pp_multiple: Optional[str] = None,
        scale_grads_in_schedule: bool = False,
        patch_stage_backward_maybe_with_nosync: bool = False,
    ):
        self.world_mesh = world_mesh
        self.moe_mesh = moe_mesh
        self.pp_axis_name = pp_axis_name
        self.dp_axis_names = dp_axis_names
        self.cp_axis_name = cp_axis_name
        self.tp_axis_name = tp_axis_name
        self.ep_axis_name = ep_axis_name
        self.ep_shard_axis_names = ep_shard_axis_names

        # Pipeline 配置
        self.layers_per_stage = layers_per_stage
        self.pp_schedule = pp_schedule
        self.pp_schedule_csv = pp_schedule_csv
        self.pp_microbatch_size = microbatch_size
        self.pp_batch_size = pp_batch_size

        # 其他配置
        self.module_fqns_per_model_part = module_fqns_per_model_part
        self.patch_inner_model = patch_inner_model
        self.patch_causal_lm_model = patch_causal_lm_model
        self.scale_grads_in_schedule = scale_grads_in_schedule
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Pipeline mesh
        self.pp_mesh: DeviceMesh = self.world_mesh[pp_axis_name]

        # Pipeline 状态
        self._info = PipelineInfo(
            enabled=False,
            schedule=None,
            has_first_stage=False,
            has_last_stage=False,
            model_parts=None,
            stages=None,
        )
```

#### build() 方法

源代码位置: `autopipeline.py:119-167`

```python
def build(
    self,
    model: nn.Module,
    *,
    loss_fn: Optional[Callable] = None,
    parallelize_fn: Optional[ParallelizeFnProtocol] = None,
):
    """构建 pipeline: 验证 -> 初始化 -> 切分 -> 应用并行化 -> 构建 schedule

    Args:
        model: 待切分的模型
        loss_fn: 损失函数 (必需)
        parallelize_fn: 并行化函数，应用于每个 stage (可选)
                       签名: (model, world_mesh, moe_mesh, pp_enabled, dp_axis_names,
                              cp_axis_name, tp_axis_name, ep_axis_name, ep_shard_axis_names)

    Returns:
        self (AutoPipeline 实例)
    """
    # 0. 验证
    assert loss_fn is not None, "loss_fn must be provided"
    assert isinstance(model, nn.Module), "model must be a PyTorch module"

    validate_hf_model_for_pipeline_support(model)

    # 1. 调用底层 pipeline_model 函数
    pp_schedule_obj, model_parts, pp_has_first_stage, pp_has_last_stage, stages = pipeline_model(
        model,
        world_mesh=self.world_mesh,
        moe_mesh=self.moe_mesh,
        pp_axis_name=self.pp_axis_name,
        dp_axis_names=self.dp_axis_names,
        cp_axis_name=self.cp_axis_name,
        tp_axis_name=self.tp_axis_name,
        ep_axis_name=self.ep_axis_name,
        ep_shard_axis_names=self.ep_shard_axis_names,
        layers_per_stage=self.layers_per_stage,
        pipeline_parallel_schedule_csv=self.pp_schedule_csv,
        pipeline_parallel_schedule=self.pp_schedule,
        microbatch_size=self.pp_microbatch_size,
        local_batch_size=self.pp_batch_size,
        device=self.device,
        loss_fn=loss_fn,
        parallelize_fn=parallelize_fn,
        module_fqns_per_model_part=self.module_fqns_per_model_part,
        patch_inner_model=self.patch_inner_model,
        patch_causal_lm_model=self.patch_causal_lm_model,
        scale_grads=self.scale_grads_in_schedule,
        round_to_pp_multiple=self.round_virtual_stages_to_pp_multiple,
        patch_stage_backward_maybe_with_nosync=self.patch_stage_backward_maybe_with_nosync,
    )

    # 2. 更新 PipelineInfo 状态
    self._info.enabled = True
    self._info.schedule = pp_schedule_obj
    self._info.has_first_stage = pp_has_first_stage
    self._info.has_last_stage = pp_has_last_stage
    self._info.model_parts = model_parts
    self._info.stages = stages

    return self
```

### 模型切分逻辑

源代码位置: `nemo_automodel/components/distributed/pipelining/functional.py:449-539`

#### pipeline_model 函数

```python
def pipeline_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh,
    *,
    pp_axis_name: str,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    layers_per_stage: int | None,
    pipeline_parallel_schedule_csv: str | None,
    pipeline_parallel_schedule: str | None,
    microbatch_size: int,
    local_batch_size: int,
    device: torch.device,
    loss_fn: Callable = None,
    parallelize_fn: Callable | None = None,
    module_fqns_per_model_part: list[list[str]] | None = None,
    patch_inner_model: bool = True,
    patch_causal_lm_model: bool = True,
    scale_grads: bool = False,
    round_to_pp_multiple: str | None = None,
    patch_stage_backward_maybe_with_nosync: bool = False,
) -> tuple[_PipelineSchedule, list[torch.nn.Module], bool, bool, list[PipelineStage]]:
    """HF-specific pipeline 模型切分

    流程:
    1. 切分模型为多个 stage
    2. 为每个 stage 应用 parallelize_fn (FSDP2+TP)
    3. 构建 pipeline schedule
    4. 返回 schedule 和 model parts
    """
    pp_size = world_mesh[pp_axis_name].size()
    assert pp_size > 1, "Pipeline parallelism is not enabled"

    # Step 1: 使用 HF-specific pipeline split
    stages, model_parts = split_model_into_stages(
        model,
        world_mesh[pp_axis_name],
        pp_axis_name,
        pipeline_parallel_schedule,
        device,
        module_fqns_per_model_part,
        layers_per_stage=layers_per_stage,
        patch_inner_model=patch_inner_model,
        patch_causal_lm_model=patch_causal_lm_model,
        round_to_pp_multiple=round_to_pp_multiple,
    )

    # Step 2: 应用并行化 (FSDP2+TP) 到每个 stage
    for i, m in enumerate(model_parts):
        if parallelize_fn is not None:
            parallelize_fn(
                m,
                world_mesh=world_mesh,
                moe_mesh=moe_mesh,
                pp_enabled=True,
                dp_axis_names=dp_axis_names,
                cp_axis_name=cp_axis_name,
                tp_axis_name=tp_axis_name,
                ep_axis_name=ep_axis_name,
                ep_shard_axis_names=ep_shard_axis_names,
            )
            model_parts[i] = m
            stages[i].submod = m

    # Step 3: 构建 pipeline schedule
    pp_schedule = build_pipeline_schedule(
        pipeline_parallel_schedule_csv,
        pipeline_parallel_schedule,
        microbatch_size,
        local_batch_size,
        stages,
        loss_fn,
        scale_grads=scale_grads,
    )

    # Step 4: 可选的 MoE-aware FSDP backward patch
    if patch_stage_backward_maybe_with_nosync:
        from nemo_automodel.components.moe.fsdp_mixin import patched_backward_maybe_with_nosync

        for stage in stages:
            stage.backward_maybe_with_nosync = types.MethodType(patched_backward_maybe_with_nosync, stage)

        logger.info("Patched pipeline stages with MoE-aware FSDP backward logic")

    # Step 5: 确定当前 rank 是否有 first/last stage
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage, stages
```

#### split_model_into_stages 函数

这个函数负责实际的模型切分工作，将一个完整的模型切分为多个 pipeline stage。

关键逻辑：

1. **计算每个 stage 的层数**：
   ```python
   # 如果未指定 layers_per_stage，自动均分
   if layers_per_stage is None:
       num_layers = len(model.model.layers)  # HuggingFace 模型
       layers_per_stage = (num_layers + pp_size - 1) // pp_size
   ```

2. **生成每个 stage 的模块 FQN (Fully Qualified Name)**：
   ```python
   if module_fqns_per_model_part is None:
       module_fqns_per_model_part = generate_hf_model_fqn_per_model_part(
           model, pp_size, layers_per_stage
       )
   ```

3. **使用 PyTorch Pipeline API 创建 PipelineStage**：
   ```python
   from torch.distributed.pipelining import PipelineStage

   stages = []
   model_parts = []
   for stage_idx, module_fqns in enumerate(module_fqns_per_model_part):
       # 每个 stage 包含特定的模块
       stage = PipelineStage(
           submod=model,
           stage_index=stage_idx,
           num_stages=pp_size,
           device=device,
           group=pp_mesh.get_group(),
       )
       stages.append(stage)
       model_parts.append(stage.submod)
   ```

### Pipeline Schedule

NeMo AutoModel 支持多种 pipeline schedule 策略：

#### 1. GPipe (Gradient Pipeline)

```
Time →
     ┌───────┬───────┬───────┬───────┐
GPU0 │ F0    │ F1    │ F2    │ F3    │ B0    B1    B2    B3
     └───────┴───────┴───────┴───────┘
           ┌───────┬───────┬───────┬───────┐
GPU1       │ F0    │ F1    │ F2    │ F3    │ B0    B1    B2    B3
           └───────┴───────┴───────┴───────┘
                 ┌───────┬───────┬───────┬───────┐
GPU2             │ F0    │ F1    │ F2    │ F3    │ B0    B1    B2    B3
                 └───────┴───────┴───────┴───────┘
                       ┌───────┬───────┬───────┬───────┐
GPU3                   │ F0    │ F1    │ F2    │ F3    │ B0    B1    B2    B3
                       └───────┴───────┴───────┴───────┘

F = Forward, B = Backward
缺点: Pipeline bubble 大 (空闲时间多)
```

#### 2. 1F1B (One Forward One Backward)

```
Time →
     ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
GPU0 │ F0    │ F1    │ F2    │ F3    │ B0    │ B1    │ B2    │ B3    │
     └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
           ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐
GPU1       │ F0    │ F1    │ F2    │ F3    │ B0    │ B1    │ B2    │ B3
           └───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                 ┌───────┬───────┬───────┬───────┬───────┬───────┐
GPU2             │ F0    │ F1    │ F2    │ F3    │ B0    │ B1    │ B2    │ B3
                 └───────┴───────┴───────┴───────┴───────┴───────┘
                       ┌───────┬───────┬───────┬───────┬───────┐
GPU3                   │ F0    │ F1    │ F2    │ F3    │ B0    │ B1    │ B2    │ B3
                       └───────┴───────┴───────┴───────┴───────┘

优点: 更小的 pipeline bubble，更好的内存效率
```

#### 3. Interleaved 1F1B (Virtual Pipeline)

将每个 GPU 分配多个 stage (virtual stages)，进一步减少 bubble：

```python
# 示例: 4 GPU, 8 virtual stages
# GPU 0: Stage 0, 4
# GPU 1: Stage 1, 5
# GPU 2: Stage 2, 6
# GPU 3: Stage 3, 7
```

#### build_pipeline_schedule 函数

源代码位置: `functional.py:400-446`

```python
def build_pipeline_schedule(
    pp_schedule_csv: str | None,
    pp_schedule: str | None,
    microbatch_size: int,
    local_batch_size: int,
    stages: list[PipelineStage],
    loss_fn: Callable,
    scale_grads: bool = False,
):
    """构建 pipeline schedule

    Args:
        pp_schedule_csv: CSV 文件路径 (自定义 schedule)
        pp_schedule: Schedule 类型 ("gpipe", "1f1b", "interleaved_1f1b")
        microbatch_size: Microbatch 大小
        local_batch_size: 本地 batch 大小
        stages: Pipeline stages 列表
        loss_fn: 损失函数
        scale_grads: 是否在 schedule 中缩放梯度

    Returns:
        PipelineSchedule 对象
    """
    n_microbatches = local_batch_size // microbatch_size

    # 根据 stage 数量选择 schedule 类型
    if len(stages) == 1:
        # 单 stage: 使用 PipelineScheduleSingle
        from torch.distributed.pipelining import PipelineScheduleSingle

        schedule = PipelineScheduleSingle(
            stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )
    else:
        # 多 stage: 使用 PipelineScheduleMulti
        from torch.distributed.pipelining import PipelineScheduleMulti

        schedule = PipelineScheduleMulti(
            stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )

    # 设置梯度缩放
    if scale_grads:
        for stage in stages:
            stage.scale_grads(n_microbatches)

    # 加载自定义 CSV schedule (如果提供)
    if pp_schedule_csv is not None:
        schedule._load_csv(pp_schedule_csv)

    return schedule
```

### HuggingFace 模型适配

NeMo AutoModel 对 HuggingFace 模型进行了特殊处理，以支持 pipeline 切分。

#### 模型结构识别

HuggingFace Transformer 模型通常有以下结构：

```python
# 标准 HF Causal LM
model = XXXForCausalLM(
    model=XXXModel(
        embed_tokens=Embedding(...),
        layers=ModuleList([
            XXXDecoderLayer(...),
            XXXDecoderLayer(...),
            ...
        ]),
        norm=LayerNorm(...),
    ),
    lm_head=Linear(...),
)
```

#### generate_hf_model_fqn_per_model_part

这个函数生成每个 pipeline stage 应包含的模块 FQN：

```python
def generate_hf_model_fqn_per_model_part(
    model: nn.Module,
    pp_size: int,
    layers_per_stage: int,
) -> list[list[str]]:
    """生成每个 stage 的模块 FQN

    示例输出 (4 stages, 32 layers):
    [
        # Stage 0
        ["model.embed_tokens", "model.layers.0", "model.layers.1", ..., "model.layers.7"],
        # Stage 1
        ["model.layers.8", "model.layers.9", ..., "model.layers.15"],
        # Stage 2
        ["model.layers.16", "model.layers.17", ..., "model.layers.23"],
        # Stage 3
        ["model.layers.24", "model.layers.25", ..., "model.layers.31", "model.norm", "lm_head"],
    ]
    """
    module_fqns_per_model_part = []
    num_layers = len(model.model.layers)

    for stage_idx in range(pp_size):
        module_fqns = []

        # Stage 0: 包含 embed_tokens
        if stage_idx == 0:
            module_fqns.append("model.embed_tokens")

        # 计算当前 stage 的 layer 范围
        start_layer = stage_idx * layers_per_stage
        end_layer = min((stage_idx + 1) * layers_per_stage, num_layers)

        # 添加 transformer layers
        for layer_idx in range(start_layer, end_layer):
            module_fqns.append(f"model.layers.{layer_idx}")

        # Last stage: 包含 norm 和 lm_head
        if stage_idx == pp_size - 1:
            module_fqns.append("model.norm")
            module_fqns.append("lm_head")

        module_fqns_per_model_part.append(module_fqns)

    return module_fqns_per_model_part
```

---

## FSDP2 实现

### FSDP2 概念

FSDP (Fully Sharded Data Parallel) 是 PyTorch 的参数分片技术：

- **传统 DDP**: 每个 GPU 复制完整模型，仅同步梯度
- **FSDP**: 参数在多个 GPU 上分片，仅在需要时 all-gather

FSDP2 是 FSDP 的第二代，基于 DTensor 实现。

### FSDP2Manager 类

源代码位置: `nemo_automodel/components/distributed/fsdp2.py:33-318`

```python
@dataclass
class FSDP2Manager:
    """FSDP2 并行化管理器

    功能:
    1. 初始化 DeviceMesh (5D mesh)
    2. 为模型应用 FSDP2 + TP 并行化
    3. 支持 mixed precision, CPU offload, activation checkpointing

    Attributes:
        dp_size: Data parallel size
        dp_replicate_size: HSDP replicate size
        tp_size: Tensor parallel size
        cp_size: Context parallel size
        pp_size: Pipeline parallel size
        ep_size: Expert parallel size (MoE)
        sequence_parallel: 是否启用 sequence parallelism
        use_hf_tp_plan: 是否使用 HF TP plan
        custom_tp_plan: 自定义 TP plan
        mp_policy: Mixed precision policy
        offload_policy: CPU offload policy
        backend: 分布式后端 ("nccl" or "gloo")
        activation_checkpointing: 是否启用 activation checkpointing
        defer_fsdp_grad_sync: 是否延迟 FSDP 梯度同步
    """

    dp_size: Optional[int] = field(default=None, metadata={"help": "Data-parallel group size"})
    dp_replicate_size: Optional[int] = field(default=None, metadata={"help": "DP replicate group size (HSDP)"})
    tp_size: Optional[int] = field(default=1, metadata={"help": "Tensor-parallel group size"})
    cp_size: Optional[int] = field(default=1, metadata={"help": "Context-parallel group size"})
    pp_size: Optional[int] = field(default=1, metadata={"help": "Pipeline-parallel group size"})
    ep_size: Optional[int] = field(default=1, metadata={"help": "Expert-parallel group size"})

    sequence_parallel: Optional[bool] = field(default=False, metadata={"help": "Enable sequence parallelism"})
    use_hf_tp_plan: Optional[bool] = field(default=False, metadata={"help": "Use HF TP plan"})
    custom_tp_plan: Optional[dict] = field(default=None, metadata={"help": "Custom TP plan"})

    mp_policy: Optional[MixedPrecisionPolicy] = field(
        default=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        metadata={"help": "Mixed precision policy"},
    )
    offload_policy: Optional[CPUOffloadPolicy] = field(default=None, metadata={"help": "CPU offload policy"})
    backend: Optional[str] = field(default="nccl", metadata={"help": "Distributed backend"})
    world_size: Optional[int] = field(default=None, metadata={"help": "Total number of processes"})

    activation_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Enable activation checkpointing"})
    defer_fsdp_grad_sync: Optional[bool] = field(
        default=True,
        metadata={"help": "Defer FSDP gradient sync to final micro-batch"},
    )

    def __post_init__(self):
        """Post-initialization: 设置分布式环境"""
        if get_world_size_safe() == 1:
            return None
        return self._setup_distributed()
```

### parallelize() 方法

源代码位置: `fsdp2.py:272-318`

```python
def parallelize(self, model):
    """为模型应用 FSDP2 + TP 并行化

    流程:
    1. 如果 tp_size > 1, 选择 TP plan
    2. 调用 fsdp2_strategy_parallelize (在 parallelizer.py)
    3. 返回并行化后的模型

    Args:
        model: 待并行化的模型

    Returns:
        并行化后的模型 (wrapped by FSDP2)
    """
    # 单 GPU: 跳过并行化
    if get_world_size_safe() == 1:
        logger.info("World size is 1, skipping parallelization.")
        if self.activation_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            else:
                logger.error("Model does not support gradient checkpointing.")
        return model

    # TP > 1: 获取 TP plan
    if self.device_mesh["tp"].size() > 1:
        # 委托给 _get_parallel_plan (在 parallelizer.py)
        tp_shard_plan = _get_parallel_plan(
            model,
            sequence_parallel=bool(self.sequence_parallel),
            tp_shard_plan=self.custom_tp_plan,
            use_hf_tp_plan=self.use_hf_tp_plan,
        )
    else:
        tp_shard_plan = None

    # 应用 FSDP2 + TP 并行化
    fsdp2_strategy_parallelize(
        model,
        device_mesh=self.device_mesh,
        mp_policy=self.mp_policy,
        tp_shard_plan=tp_shard_plan,
        offload_policy=self.offload_policy,
        activation_checkpointing=self.activation_checkpointing,
    )

    return model
```

### fully_shard API

PyTorch FSDP2 的核心 API 是 `fully_shard`：

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# 应用 FSDP2 到模块
module = fully_shard(
    module,
    mesh=dp_mesh,                          # DeviceMesh for sharding
    mp_policy=MixedPrecisionPolicy(...),   # Mixed precision
    reshard_after_forward=True,            # 前向后是否重新分片
    offload_policy=None,                   # CPU offload (可选)
)
```

#### FSDP2 工作原理

1. **参数分片 (Sharding)**:
   ```python
   # 假设 dp_mesh.size() = 4
   # 参数 tensor shape: [1024, 1024]
   # 每个 rank 持有: [256, 1024] (在第0维分片)
   ```

2. **前向传播**:
   ```python
   # Step 1: all-gather 参数
   # Rank 0-3: 各自 all-gather 获得完整参数 [1024, 1024]

   # Step 2: 计算
   output = layer(input)

   # Step 3: reshard (如果 reshard_after_forward=True)
   # 释放完整参数，仅保留分片 [256, 1024]
   ```

3. **反向传播**:
   ```python
   # Step 1: all-gather 参数 (再次)
   # Step 2: 计算梯度
   # Step 3: reduce-scatter 梯度
   # 每个 rank 获得自己分片的梯度
   ```

### apply_fsdp2_sharding_recursively

源代码位置: `parallelizer.py:408-458`

```python
def apply_fsdp2_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[OffloadPolicy] = None,
) -> None:
    """递归应用 FSDP2 分片

    优化: 对于 ModuleList (通常是 transformer layers)
    - 最后一层不 reshard_after_forward
    - 因为 FSDP2 会立即 prefetch 它用于反向

    Args:
        module: 待分片的模块
        mesh: FSDP2 分片的 DeviceMesh
        mp_policy: Mixed precision policy
        offload_policy: CPU offload policy
    """
    if isinstance(module, nn.ModuleList):
        for layer_id, child_module in enumerate(module):
            # 如果子模块也是 ModuleList (嵌套结构), 递归
            if isinstance(child_module, nn.ModuleList):
                apply_fsdp2_sharding_recursively(child_module, mesh, mp_policy, offload_policy)
            else:
                # 优化: 最后一层不 reshard
                reshard_after_forward = int(layer_id) < len(module) - 1
                fully_shard(
                    child_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    offload_policy=offload_policy,
                )
                module[layer_id] = child_module
    else:
        # 非 ModuleList: 递归处理子模块
        for name, sub_module in module.named_children():
            apply_fsdp2_sharding_recursively(sub_module, mesh, mp_policy, offload_policy)
```

### Mixed Precision Policy

源代码位置: `fsdp2.py:107-115`

```python
mp_policy: Optional[MixedPrecisionPolicy] = field(
    default=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,      # 参数存储为 bfloat16
        reduce_dtype=torch.bfloat16,     # 梯度 reduce 使用 bfloat16
        output_dtype=torch.bfloat16,     # 输出为 bfloat16
        cast_forward_inputs=True,        # 自动转换输入
    ),
    metadata={"help": "MixedPrecisionPolicy for FSDP2"},
)
```

PyTorch FSDP2 支持灵活的混合精度策略：

```python
# 高精度 reduce (更稳定)
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,    # 梯度 all-reduce 使用 FP32
    output_dtype=torch.float32,
)

# 全 BF16 (更快)
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
)
```

---

## Tensor Parallelism 集成

### Tensor Parallelism 概念

Tensor Parallelism 将模型参数在张量维度上切分：

```python
# 原始 Linear: [hidden_size, 4*hidden_size]
gate_proj = nn.Linear(4096, 16384)

# TP=4: 每个 rank 持有
gate_proj_shard = nn.Linear(4096, 4096)  # [4096, 16384/4]
```

### PyTorch TP API

PyTorch 提供 `parallelize_module` API 用于 TP：

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)

# 定义 TP plan
tp_plan = {
    "layers.*.self_attn.q_proj": ColwiseParallel(),   # 列切分
    "layers.*.self_attn.k_proj": ColwiseParallel(),
    "layers.*.self_attn.v_proj": ColwiseParallel(),
    "layers.*.self_attn.o_proj": RowwiseParallel(),   # 行切分
    "layers.*.mlp.gate_proj": ColwiseParallel(),
    "layers.*.mlp.up_proj": ColwiseParallel(),
    "layers.*.mlp.down_proj": RowwiseParallel(),
}

# 应用 TP
parallelize_module(model, tp_mesh, tp_plan)
```

### Parallel Styles

NeMo AutoModel 定义了多种 parallel styles：

源代码位置: `parallelizer.py:582-603`

```python
@lru_cache
def translate_to_torch_parallel_style(style: str):
    """将字符串描述转换为 PyTorch parallel style

    在模型配置中，使用中性类型 (string) 指定 parallel style
    这里将它们转换为 torch.distributed tensor-parallel 类型
    """
    assert isinstance(style, str), f"parallel style type should be str, but got {type(style)}"

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")
```

#### ColwiseParallel

列切分：在输出维度切分

```python
# 原始: Linear(4096, 16384)
# TP=4: 每个 rank Linear(4096, 4096)
# 输出: Shard(-1) - 在最后一个维度分片

gate_proj = ColwiseParallel()

# 前向:
# Input: [batch, seq, 4096] - Replicate (每个 rank 相同)
# Weight: [4096, 4096] - Shard(1) (第1维分片)
# Output: [batch, seq, 4096] - Shard(-1) (第-1维分片)
```

#### RowwiseParallel

行切分：在输入维度切分

```python
# 原始: Linear(16384, 4096)
# TP=4: 每个 rank Linear(4096, 4096)
# 输入: Shard(-1) - 在最后一个维度分片

down_proj = RowwiseParallel()

# 前向:
# Input: [batch, seq, 4096] - Shard(-1) (从 ColwiseParallel 输出)
# Weight: [4096, 4096] - Shard(0) (第0维分片)
# Local output: [batch, seq, 4096]
# Final output: [batch, seq, 4096] - Replicate (all-reduce)
```

#### SequenceParallel

序列切分：在序列维度切分 (配合 TP 使用)

```python
# LayerNorm 在序列维度切分
layer_norm = SequenceParallel()

# 前向:
# Input: [batch, seq, hidden] - Shard(1) (序列维度分片)
# Output: [batch, seq, hidden] - Shard(1) (保持序列维度分片)
```

### TP Plan 生成

源代码位置: `parallelizer.py:825-915`

```python
def _get_parallel_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
    use_hf_tp_plan: bool = False,
) -> Dict[str, ParallelStyle]:
    """选择 tensor-parallel plan

    优先级:
    1. 如果提供 tp_shard_plan (dict 或 import path), 使用它
    2. 如果 use_hf_tp_plan=True, 使用 HF plan
    3. 如果模型在 PARALLELIZE_FUNCTIONS, 使用优化的 plan
    4. 否则, 使用默认 base plan
    """

    model_parallel_plan = None
    model_cls = type(model)

    # 1. 使用自定义 parallel plan
    if isinstance(tp_shard_plan, dict):
        model_parallel_plan = tp_shard_plan
        logger.info(f"Using parallel plan (dictionary). {tp_shard_plan}")

    elif tp_shard_plan is not None:
        # 从 import path 加载
        try:
            plan_obj = import_class_from_path(tp_shard_plan)
            if isinstance(plan_obj, FunctionType):
                model_parallel_plan = plan_obj()
            else:
                model_parallel_plan = plan_obj
            assert isinstance(model_parallel_plan, dict)
            logger.info(f"Using provided parallel plan (from path). {tp_shard_plan}")
        except Exception as e:
            raise ValueError(f"Custom parallel plan '{tp_shard_plan}' is not valid. Error: {e}")

    # 2. 使用 HF TP plan
    elif use_hf_tp_plan:
        assert not sequence_parallel, "sequence_parallel is not supported in HF tp plan."
        model_parallel_plan = get_hf_tp_shard_plan(model)

    # 3. 使用优化的 parallel plan
    elif model_cls in PARALLELIZE_FUNCTIONS:
        try:
            func = PARALLELIZE_FUNCTIONS[model_cls]
            model_parallel_plan = func(model, sequence_parallel)
            logger.info("Using optimized parallel plan.")
        except Exception as e:
            logger.info(f"Optimized parallel plan is not available: {e}. Falling back to the HF tp plan.")
            assert not sequence_parallel
            model_parallel_plan = get_hf_tp_shard_plan(model)

    # 4. 使用默认 base plan
    else:
        base_model_tp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        }

        if sequence_parallel:
            base_model_sp_plan = {
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": SequenceParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "model.layers.*.post_attention_layernorm": SequenceParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
            }
            base_model_tp_plan.update(base_model_sp_plan)

        model_parallel_plan = base_model_tp_plan
        logger.info("Using default base TP plan. Compatible with huggingface llama3-style models.")

    return model_parallel_plan
```

### HuggingFace TP Plan

源代码位置: `parallelizer.py:460-544`

```python
def get_hf_tp_shard_plan(model):
    """从 HuggingFace 模型获取 TP plan

    HF 模型可以在类级别或实例级别定义 _tp_plan 属性
    这个函数提取并转换这些 TP plan
    """
    model_cls = type(model)

    # 处理 VL 模型结构
    if model_cls in [Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"
    elif model_cls == Gemma3ForConditionalGeneration:
        inner_model = model.language_model
        model_prefix = "language_model"
    elif model_cls == Llama4ForConditionalGeneration:
        inner_model = model.language_model.model
        model_prefix = "language_model.model"
    elif model_cls in [
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextVideoForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
    ]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"
    elif model_cls == Mistral3ForConditionalGeneration:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"
    else:
        inner_model = model.model
        model_prefix = "model"

    hf_tp_plan = {}

    # 从类和实例收集 TP plan
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        hf_tp_plan.update(model_cls._tp_plan)

    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update({f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()})

    assert len(hf_tp_plan) > 0, (
        f"Hugging Face tp plan is not supported for {model_cls}, "
        "please set dtensor_cfg.tensor_parallel_size to 1 or provide a custom_parallel_plan."
    )

    # 添加 embed_tokens (HF plan 通常不包含)
    if f"{model_prefix}.embed_tokens" not in hf_tp_plan:
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    # 转换字符串 style 为 ParallelStyle 对象
    for k, v in hf_tp_plan.items():
        # 优化 lm_head
        if (k == "lm_head" or k == "language_model.lm_head") and v == "colwise_rep":
            hf_tp_plan[k] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
        else:
            hf_tp_plan[k] = translate_to_torch_parallel_style(v)

    logger.info(f"Hugging Face tp plan: {hf_tp_plan}")
    return hf_tp_plan
```

### 优化的 TP Plans

NeMo AutoModel 为特定模型提供了优化的 TP plans：

源代码位置: `nemo_automodel/components/distributed/optimized_tp_plans.py`

```python
# PARALLELIZE_FUNCTIONS 是一个字典，映射模型类到 TP plan 生成函数
PARALLELIZE_FUNCTIONS: Dict[Type[nn.Module], Callable] = {
    # LLM models
    LlamaForCausalLM: get_llama_tp_plan,
    Qwen2ForCausalLM: get_qwen2_tp_plan,
    MistralForCausalLM: get_mistral_tp_plan,
    # VLM models
    Qwen2VLForConditionalGeneration: get_qwen2_vl_tp_plan,
    # ... 更多模型
}

def get_llama_tp_plan(model, sequence_parallel: bool = False):
    """为 Llama 模型生成优化的 TP plan"""
    tp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Replicate()),
    }

    if sequence_parallel:
        # 添加 Sequence Parallel 配置
        sp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        }
        tp_plan.update(sp_plan)

    return tp_plan
```

---

## 三者组合机制

### 组合架构

3D Parallelism 的核心是如何将 PP、DP (FSDP2)、TP 组合在一起：

```
┌────────────────────────────────────────────────────────────────┐
│                   完整模型 (Full Model)                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Pipeline Split (PP)          │
              │  切分为 4 个 stage             │
              └───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌────────┐           ┌────────┐           ┌────────┐
   │Stage 0 │           │Stage 1 │           │Stage 2 │
   │GPU 0-3 │           │GPU 4-7 │           │GPU 8-11│
   └────────┘           └────────┘           └────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────────────────────────────────────────────┐
   │  parallelize_fn (FSDP2 + TP)                    │
   │  应用到每个 stage                                │
   └─────────────────────────────────────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ TP mesh │          │ TP mesh │          │ TP mesh │
   │ [0,1]   │          │ [4,5]   │          │ [8,9]   │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │DP mesh  │          │DP mesh  │          │DP mesh  │
   │ [0,2]   │          │ [4,6]   │          │ [8,10]  │
   │ [1,3]   │          │ [5,7]   │          │ [9,11]  │
   └─────────┘          └─────────┘          └─────────┘
```

### 关键函数: parallelize_for_pp

源代码位置: `nemo_automodel/recipes/llm/train_ft.py:830-846`

```python
def parallelize_for_pp(
    model: nn.Module,
    *,
    world_mesh: DeviceMesh,
    moe_mesh: Optional[DeviceMesh] = None,
    pp_enabled: bool = False,
    dp_axis_names: Union[tuple[str, ...], str] = ("data_parallel",),
    cp_axis_name: Optional[str] = None,
    tp_axis_name: Optional[str] = None,
    ep_axis_name: Optional[str] = None,
    ep_shard_axis_names: Optional[tuple[str, ...]] = None,
    model_wrapper: Optional[Any] = None,
) -> nn.Module:
    """为 pipeline stage 应用并行化

    这是传递给 AutoPipeline.build() 的 parallelize_fn

    Args:
        model: 单个 pipeline stage 模型
        world_mesh: 完整的 5D DeviceMesh
        moe_mesh: MoE mesh (如果使用 EP)
        pp_enabled: 是否启用 PP
        dp_axis_names: DP 轴名称
        tp_axis_name: TP 轴名称
        cp_axis_name: CP 轴名称
        ep_axis_name: EP 轴名称
        model_wrapper: FSDP2Manager 实例

    Returns:
        并行化后的 model (wrapped by FSDP2 and TP)
    """
    if model_wrapper is not None:
        if callable(getattr(model_wrapper, "parallelize", None)):
            # 调用 FSDP2Manager.parallelize()
            model = model_wrapper.parallelize(model)
    return model
```

### 端到端工作流

源代码位置: `nemo_automodel/recipes/llm/train_ft.py:145-318`

```python
def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_optimizer,
    peft_config,
    model_wrapper,  # FSDP2Manager 实例
    has_packed_sequence,
    seed,
    tp_size,
    cp_size,
    cfg_fp8,
    cfg_compile,
    cfg_quantization,
    cfg_qat,
    autopipeline,  # AutoPipeline 实例
    loss_fn,
    parallelize_fn,  # parallelize_for_pp
    load_base_model,
    checkpointer,
):
    """构建模型和优化器

    流程:
    1. 加载模型到 meta device (如果使用 FSDP2)
    2. 应用 PEFT (如果配置)
    3. 分支 A: 使用 AutoPipeline (PP 启用)
       - autopipeline.build() 自动切分模型
       - 为每个 stage 应用 parallelize_fn (FSDP2+TP)
    4. 分支 B: 不使用 Pipeline
       - 直接调用 model_wrapper.parallelize() (FSDP2+TP)
    5. 构建优化器
    """

    # 1. 确定是否使用 meta device
    is_meta_device = not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)) and not cfg_model.get(
        "pretrained_model_name_or_path_is_sharded", False
    )

    # 2. 加载模型
    with set_default_device(device, enabled=is_meta_device):
        model = build_model(
            cfg_model,
            has_packed_sequence=has_packed_sequence,
            seed=seed,
            tp_size=tp_size,
            cp_size=cp_size,
            cfg_fp8=cfg_fp8,
            cfg_compile=cfg_compile,
            cfg_quantization=cfg_quantization,
            cfg_qat=cfg_qat,
            load_base_model=load_base_model,
        )

    # 3. 应用 PEFT (如果配置)
    if peft_config is not None:
        model = apply_peft(model, peft_config)

    # 4. 并行化
    optimizer = None

    if autopipeline is not None:
        # 分支 A: 使用 AutoPipeline (PP + FSDP2 + TP)
        if model_wrapper is not None:
            # 构建 pipeline, 为每个 stage 应用 FSDP2+TP
            model = autopipeline.build(
                model,
                world_mesh=model_wrapper.device_mesh,
                moe_mesh=getattr(model_wrapper, "moe_mesh", None),
                loss_fn=loss_fn,
                parallelize_fn=parallelize_fn,  # parallelize_for_pp
                dp_axis_names=(
                    ("dp_replicate", "dp_shard_cp")
                    if "dp_replicate" in model_wrapper.device_mesh.mesh_dim_names
                    and "dp_shard_cp" in model_wrapper.device_mesh.mesh_dim_names
                    else ("dp",)
                ),
                tp_axis_name="tp",
                cp_axis_name="cp",
            )
        else:
            # 无 model_wrapper: 仅 PP
            model = autopipeline.build(model, loss_fn=loss_fn, parallelize_fn=None)

    elif callable(getattr(model_wrapper, "parallelize", None)):
        # 分支 B: 无 Pipeline, 直接 FSDP2+TP
        if isinstance(model_wrapper, MegatronFSDPManager):
            # MegatronFSDP 需要 optimizer
            optimizer = build_optimizer(cfg_optimizer, model)
            model, optimizer = model_wrapper.parallelize(model, optimizer)
        else:
            # FSDP2Manager
            model = model_wrapper.parallelize(model)

    # 5. 构建优化器 (如果尚未构建)
    if optimizer is None:
        optimizer = build_optimizer(cfg_optimizer, model)

    return model, optimizer
```

### 策略模式: fsdp2_strategy_parallelize

源代码位置: `parallelizer.py:920-984`

```python
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[OffloadPolicy] = None,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
    dp_replicate_mesh_name: str = "dp_replicate",
    dp_shard_cp_mesh_name: str = "dp_shard_cp",
    tp_mesh_name: str = "tp",
):
    """应用并行化策略到模型

    使用策略模式支持不同模型的并行化

    流程:
    1. 获取模型的并行化策略 (DefaultParallelizationStrategy 或特化策略)
    2. 委托给策略的 parallelize() 方法

    Args:
        model: 待并行化的模型
        device_mesh: DeviceMesh (5D)
        mp_policy: Mixed precision policy
        offload_policy: CPU offload policy
        sequence_parallel: 是否启用 sequence parallelism
        activation_checkpointing: 是否启用 activation checkpointing
        tp_shard_plan: TP plan (dict 或 import path)
        dp_replicate_mesh_name: DP replicate mesh 名称
        dp_shard_cp_mesh_name: DP shard + CP mesh 名称
        tp_mesh_name: TP mesh 名称

    Returns:
        并行化后的模型
    """
    # 获取并行化策略
    strategy = get_parallelization_strategy(model)

    # 委托给策略
    return strategy.parallelize(
        model=model,
        device_mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        sequence_parallel=sequence_parallel,
        activation_checkpointing=activation_checkpointing,
        tp_shard_plan=tp_shard_plan,
        dp_replicate_mesh_name=dp_replicate_mesh_name,
        dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
        tp_mesh_name=tp_mesh_name,
    )
```

### DefaultParallelizationStrategy

源代码位置: `parallelizer.py:109-205`

```python
class DefaultParallelizationStrategy(ParallelizationStrategy):
    """默认并行化策略，适用于大多数 Transformer 模型"""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        use_hf_tp_plan: bool = False,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        """应用默认并行化流程

        流程:
        1. 提取 TP mesh 和 DP mesh
        2. 应用 Tensor Parallelism (如果 tp_size > 1)
        3. 应用 Activation Checkpointing (如果启用)
        4. 应用 FSDP2 到所有 transformer layers
        5. 应用 FSDP2 到 root model
        """
        # Step 1: 提取 mesh
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh_dim_names = (dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        dp_mesh = device_mesh[dp_mesh_dim_names]

        # Step 2: 提取模型 layers
        layers = _extract_model_layers(model)

        # Step 3: 应用 Tensor Parallelism
        if tp_mesh.size() > 1:
            # 验证 attention heads 可被 TP size 整除
            validate_tp_mesh(model, tp_mesh)

            # 生成或使用 tensor parallel plan
            model_parallel_plan = {
                k: translate_to_lora(v)
                for k, v in _get_parallel_plan(
                    model,
                    sequence_parallel,
                    tp_shard_plan,
                    use_hf_tp_plan=use_hf_tp_plan,
                ).items()
            }

            # 应用 tensor parallelism
            if model_parallel_plan:
                parallelize_module(model, tp_mesh, model_parallel_plan)

        # Step 4: 应用 Activation Checkpointing
        if activation_checkpointing:
            # 禁用 KV caching (训练时需要确定性 shape)
            if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not False:
                try:
                    model.config.use_cache = False
                except Exception:
                    pass

            # 对 linear layers 应用 checkpoint
            for i, layer in enumerate(layers):
                if hasattr(layer, "mlp"):
                    layers[i].mlp = checkpoint_wrapper(layer.mlp)
                if hasattr(layer, "self_attn"):
                    layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)
                if hasattr(layer, "input_layernorm"):
                    layers[i].input_layernorm = checkpoint_wrapper(layers[i].input_layernorm)
                if hasattr(layer, "post_attention_layernorm"):
                    layers[i].post_attention_layernorm = checkpoint_wrapper(layers[i].post_attention_layernorm)

        # Step 5: 设置 mixed precision policy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        # Step 6: 递归应用 FSDP2 到所有 transformer layers
        apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)

        # Step 7: 应用 FSDP2 到 root model
        # 不 reshard_after_forward，因为参数会立即在 backward 中使用
        model = fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
            offload_policy=offload_policy,
        )

        return model
```

### 组合顺序的重要性

**关键点**: TP 必须在 FSDP2 之前应用

```python
# 正确顺序:
# 1. parallelize_module (TP)
# 2. fully_shard (FSDP2)

# 原因:
# - TP 修改模型结构 (替换 Linear 为 TP 版本)
# - FSDP2 wrap 模型参数
# - 如果先 FSDP2 后 TP, TP 无法正确识别参数
```

源代码验证: `parallelizer.py:138-156`

```python
# TP sharding with enhanced plan generation
if tp_mesh.size() > 1:
    # Validate that attention heads are divisible by TP size
    validate_tp_mesh(model, tp_mesh)

    # Generate or use tensor parallel plan
    model_parallel_plan = {...}

    # Apply tensor parallelism
    if model_parallel_plan:
        parallelize_module(model, tp_mesh, model_parallel_plan)  # TP 先

# Apply activation checkpointing to linear layers if requested
if activation_checkpointing:
    # ... checkpoint wrapping ...

# Find transformer layers and apply parallelisms
apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)  # FSDP2 后

# Apply FSDP to the root model
model = fully_shard(model, mesh=dp_mesh, ...)
```

---

## DTensor 与通信模式

### DTensor 概念

DTensor (Distributed Tensor) 是 PyTorch 的分布式张量抽象：

- **统一接口**: 分布式张量与本地张量 API 相同
- **Placement 策略**: 描述张量如何分布在设备上
  - `Shard(dim)`: 在 dim 维度分片
  - `Replicate()`: 在所有设备复制
- **自动通信**: PyTorch 自动插入必要的集合通信

```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# 创建 DTensor
dtensor = DTensor.from_local(
    local_tensor,
    device_mesh=mesh,
    placements=[Shard(0)],  # 在第0维分片
)

# DTensor 操作与普通 tensor 相同
output = dtensor @ weight  # 自动处理分布式矩阵乘法
```

### Placement 策略

#### Shard(dim)

在指定维度分片：

```python
# 假设 mesh.size() = 4
# 原始 tensor: [8, 1024]
# Placement: Shard(0)

# Rank 0: [2, 1024]
# Rank 1: [2, 1024]
# Rank 2: [2, 1024]
# Rank 3: [2, 1024]
```

#### Replicate()

在所有设备复制：

```python
# 假设 mesh.size() = 4
# 原始 tensor: [8, 1024]
# Placement: Replicate()

# Rank 0: [8, 1024]
# Rank 1: [8, 1024]
# Rank 2: [8, 1024]
# Rank 3: [8, 1024]
```

### TP 中的 DTensor

#### ColwiseParallel

```python
# gate_proj: Linear(4096, 16384)
# TP=4

# Weight placement: Shard(1) - 在输出维度分片
# Weight on Rank 0: [4096, 4096]
# Weight on Rank 1: [4096, 4096]
# Weight on Rank 2: [4096, 4096]
# Weight on Rank 3: [4096, 4096]

# Input placement: Replicate() - 每个 rank 相同
# Input on Rank 0-3: [batch, seq, 4096]

# Output placement: Shard(-1) - 在最后一维分片
# Output on Rank 0: [batch, seq, 4096]  # indices [0:4096]
# Output on Rank 1: [batch, seq, 4096]  # indices [4096:8192]
# Output on Rank 2: [batch, seq, 4096]  # indices [8192:12288]
# Output on Rank 3: [batch, seq, 4096]  # indices [12288:16384]
```

#### RowwiseParallel

```python
# down_proj: Linear(16384, 4096)
# TP=4

# Weight placement: Shard(0) - 在输入维度分片
# Weight on Rank 0: [4096, 4096]
# Weight on Rank 1: [4096, 4096]
# Weight on Rank 2: [4096, 4096]
# Weight on Rank 3: [4096, 4096]

# Input placement: Shard(-1) - 从 ColwiseParallel 输出
# Input on Rank 0-3: [batch, seq, 4096]

# Local output: [batch, seq, 4096]

# Final output placement: Replicate() - all-reduce
# Output on Rank 0-3: [batch, seq, 4096] (相同)
```

### FSDP2 中的 DTensor

FSDP2 使用 DTensor 实现参数分片：

```python
# 参数: [1024, 1024]
# DP mesh size = 4

# 分片后每个 rank 的参数
# Placement: Shard(0)
# Rank 0: [256, 1024]
# Rank 1: [256, 1024]
# Rank 2: [256, 1024]
# Rank 3: [256, 1024]
```

### 通信模式

#### 1. TP 通信

**ColwiseParallel 无通信** (输入已 replicate):

```python
# Input: Replicate() - 每个 rank 相同
# Local compute: y_local = x @ W_local
# Output: Shard(-1) - 无需通信
```

**RowwiseParallel 需要 all-reduce**:

```python
# Input: Shard(-1) - 每个 rank 不同
# Local compute: y_local = x_local @ W_local
# Output: all-reduce(y_local) -> Replicate()

# 通信量: O(batch * seq * hidden)
```

#### 2. FSDP2 通信

**前向传播**:

```python
# Step 1: all-gather 参数
# 通信量: O(model_size / dp_size)
# 每个 rank 从 [256, 1024] 获得完整 [1024, 1024]

# Step 2: 计算
output = layer(input)

# Step 3: reshard (如果 reshard_after_forward=True)
# 释放完整参数，仅保留分片
```

**反向传播**:

```python
# Step 1: all-gather 参数 (再次)
# Step 2: 计算梯度
# Step 3: reduce-scatter 梯度
# 通信量: O(model_size / dp_size)
# 每个 rank 获得自己分片的梯度 [256, 1024]
```

#### 3. Pipeline 通信

**点对点 (P2P) 通信**:

```python
# Stage 0 -> Stage 1: send activation
# 通信量: O(batch * seq * hidden)

# Stage 1 -> Stage 0: send gradient
# 通信量: O(batch * seq * hidden)
```

### 多维并行的通信模式

假设配置: PP=2, DP=4, TP=2 (16 GPUs)

```
Mesh shape: (2, 1, 2, 1, 2)
           (pp, dp_replicate, dp_shard, cp, tp)

GPU 布局:
┌─────────────────────────────────────┐
│ PP Stage 0 (GPU 0-7)                │
│  ┌────────────┬────────────┐        │
│  │ TP group 0 │ TP group 1 │        │
│  │ (GPU 0,1)  │ (GPU 2,3)  │  DP 0  │
│  ├────────────┼────────────┤        │
│  │ TP group 2 │ TP group 3 │        │
│  │ (GPU 4,5)  │ (GPU 6,7)  │  DP 1  │
│  └────────────┴────────────┘        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ PP Stage 1 (GPU 8-15)               │
│  ┌────────────┬────────────┐        │
│  │ TP group 4 │ TP group 5 │        │
│  │ (GPU 8,9)  │ (GPU 10,11)│  DP 2  │
│  ├────────────┼────────────┤        │
│  │ TP group 6 │ TP group 7 │        │
│  │ (GPU 12,13)│ (GPU 14,15)│  DP 3  │
│  └────────────┴────────────┘        │
└─────────────────────────────────────┘

通信模式:
1. TP 通信: 在 TP group 内 (GPU 0-1, 2-3, ...)
   - all-reduce (RowwiseParallel)

2. FSDP2 通信: 在 DP group 内 (GPU 0,2,4,6 或 1,3,5,7 或 ...)
   - all-gather 参数
   - reduce-scatter 梯度

3. PP 通信: Stage 之间 (GPU 0-7 <-> GPU 8-15)
   - send/recv activation
   - send/recv gradient
```

---

## 代码示例与工作流

### 配置文件示例

#### YAML 配置 (3D Parallelism)

```yaml
# examples/llm_finetune/llama3_2/llama3_2_1b_squad_3d.yaml

# 模型配置
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# 分布式配置
distributed:
  _target_: nemo_automodel.components.distributed.FSDP2Manager
  world_size: 64       # 64 GPUs
  pp_size: 4           # 4-stage pipeline
  dp_size: 4           # 4-way data parallel (自动推导)
  dp_replicate_size: 2 # 2-way replicate (HSDP, 假设 8 nodes)
  tp_size: 4           # 4-way tensor parallel
  cp_size: 1           # 无 context parallel
  sequence_parallel: false
  use_hf_tp_plan: false
  activation_checkpointing: true
  mp_policy:
    param_dtype: bfloat16
    reduce_dtype: float32
    output_dtype: float32

# AutoPipeline 配置
autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.AutoPipeline
  layers_per_stage: 8        # 每个 stage 8 层
  pp_schedule: 1f1b          # 1F1B schedule
  pp_microbatch_size: 4      # Microbatch size
  pp_batch_size: 16          # 总 batch size

# 数据配置
dataset:
  _target_: nemo_automodel.components.datasets.llm.HFDataset
  path: squad
  split: train

# 训练配置
step_scheduler:
  local_batch_size: 16
  max_steps: 1000
  validation_freq: 100
  checkpoint_freq: 500

# 优化器配置
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-5
  betas: [0.9, 0.999]
  weight_decay: 0.01
```

#### Mesh 推导

基于上述配置，DeviceMesh 推导：

```python
# 输入:
world_size = 64
pp_size = 4
tp_size = 4
cp_size = 1
dp_replicate_size = 2

# 推导:
total_parallel_ranks = tp_size * cp_size * pp_size = 4 * 1 * 4 = 16
dp_size = world_size / total_parallel_ranks = 64 / 16 = 4
dp_shard_size = dp_size / dp_replicate_size = 4 / 2 = 2

# 最终 mesh shape:
mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
           = (4, 2, 2, 1, 4)

# Mesh dimensions:
("pp", "dp_replicate", "dp_shard", "cp", "tp")
```

#### GPU 分配示例 (64 GPUs, 8 nodes)

```
Node 0 (GPU 0-7):
  PP Stage 0:
    DP Replicate 0, DP Shard 0, TP Group 0: GPU 0, 1, 2, 3
    DP Replicate 0, DP Shard 1, TP Group 1: GPU 4, 5, 6, 7

Node 1 (GPU 8-15):
  PP Stage 0:
    DP Replicate 1, DP Shard 0, TP Group 2: GPU 8, 9, 10, 11
    DP Replicate 1, DP Shard 1, TP Group 3: GPU 12, 13, 14, 15

Node 2 (GPU 16-23):
  PP Stage 1:
    DP Replicate 0, DP Shard 0, TP Group 4: GPU 16, 17, 18, 19
    DP Replicate 0, DP Shard 1, TP Group 5: GPU 20, 21, 22, 23

Node 3 (GPU 24-31):
  PP Stage 1:
    DP Replicate 1, DP Shard 0, TP Group 6: GPU 24, 25, 26, 27
    DP Replicate 1, DP Shard 1, TP Group 7: GPU 28, 29, 30, 31

... (类似 Node 4-7 for PP Stage 2-3)
```

### 训练循环工作流

#### 非 Pipeline 训练循环

```python
# train_ft.py: train_loop()

for step_idx in range(max_steps):
    # 1. 获取数据批次
    batch = next(dataloader_iter)
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # 2. 前向传播
    # FSDP2 自动 all-gather 参数
    # TP 自动分布式计算
    outputs = model(inputs)

    # 3. 计算损失
    loss = loss_fn(outputs.logits, labels)

    # 4. 反向传播
    # FSDP2 自动 reduce-scatter 梯度
    loss.backward()

    # 5. 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # 6. 优化器步进
    optimizer.step()
    optimizer.zero_grad()

    # 7. 日志
    if step_idx % log_freq == 0:
        logger.info(f"Step {step_idx}, Loss {loss.item()}")
```

#### Pipeline 训练循环

```python
# train_ft.py: train_loop() with AutoPipeline

for step_idx in range(max_steps):
    # 1. 获取数据批次
    batch = next(dataloader_iter)

    # 2. Pipeline schedule 执行
    # - 自动切分为 microbatch
    # - 执行 1F1B schedule
    # - 处理 activation/gradient 的 send/recv
    losses = []
    pp_schedule.step(
        **batch,
        losses=losses,
        device=device,
    )

    # 3. 计算平均损失 (仅在 last stage)
    if pp.info.has_last_stage:
        loss = sum(losses) / len(losses)

    # 4. 梯度裁剪
    if pp.info.has_first_stage or pp.info.has_last_stage:
        torch.nn.utils.clip_grad_norm_(
            [p for part in model_parts for p in part.parameters()],
            max_grad_norm
        )

    # 5. 优化器步进
    optimizer.step()
    optimizer.zero_grad()

    # 6. 日志 (仅在 rank 0)
    if dist.get_rank() == 0 and step_idx % log_freq == 0:
        logger.info(f"Step {step_idx}, Loss {loss.item()}")
```

### 完整示例: 从配置到训练

```python
# 1. 从 YAML 加载配置
from omegaconf import OmegaConf

cfg = OmegaConf.load("llama3_2_1b_squad_3d.yaml")

# 2. 初始化分布式环境
import torch.distributed as dist

dist.init_process_group(backend="nccl")

# 3. 创建 FSDP2Manager (自动构建 DeviceMesh)
from nemo_automodel.components.distributed import FSDP2Manager

fsdp2_manager = FSDP2Manager(
    world_size=dist.get_world_size(),
    pp_size=cfg.distributed.pp_size,
    tp_size=cfg.distributed.tp_size,
    cp_size=cfg.distributed.cp_size,
    dp_replicate_size=cfg.distributed.dp_replicate_size,
    sequence_parallel=cfg.distributed.sequence_parallel,
    use_hf_tp_plan=cfg.distributed.use_hf_tp_plan,
    activation_checkpointing=cfg.distributed.activation_checkpointing,
    mp_policy=cfg.distributed.mp_policy,
)

# DeviceMesh 已创建: fsdp2_manager.device_mesh

# 4. 创建 AutoPipeline
from nemo_automodel.components.distributed.pipelining import AutoPipeline

autopipeline = AutoPipeline(
    world_mesh=fsdp2_manager.device_mesh,
    pp_axis_name="pp",
    dp_axis_names=("dp_replicate", "dp_shard_cp"),
    tp_axis_name="tp",
    layers_per_stage=cfg.autopipeline.layers_per_stage,
    pp_schedule=cfg.autopipeline.pp_schedule,
    pp_microbatch_size=cfg.autopipeline.pp_microbatch_size,
    pp_batch_size=cfg.autopipeline.pp_batch_size,
)

# 5. 加载模型到 meta device
from nemo_automodel import NeMoAutoModelForCausalLM

with torch.device("meta"):
    model = NeMoAutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_model_name_or_path
    )

# 6. 定义 loss function
def loss_fn(logits, labels):
    from torch.nn import CrossEntropyLoss
    return CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

# 7. 定义 parallelize_fn
from functools import partial

def parallelize_for_pp(model, **kwargs):
    return fsdp2_manager.parallelize(model)

parallelize_fn = partial(parallelize_for_pp)

# 8. 构建 pipeline (自动切分 + 并行化)
autopipeline.build(
    model,
    loss_fn=loss_fn,
    parallelize_fn=parallelize_fn,
)

# model 现在是 AutoPipeline 实例
# autopipeline.parts 包含所有 pipeline stages (已并行化)

# 9. 构建优化器
from torch.optim import AdamW

optimizer = AdamW(
    [p for part in autopipeline.parts for p in part.parameters()],
    lr=cfg.optimizer.lr,
    betas=cfg.optimizer.betas,
    weight_decay=cfg.optimizer.weight_decay,
)

# 10. 构建 dataloader
from nemo_automodel.components.datasets.llm import HFDataset

dataset = HFDataset(path=cfg.dataset.path, split=cfg.dataset.split)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.step_scheduler.local_batch_size,
    shuffle=True,
)

# 11. 训练循环
device = torch.device("cuda")
pp_schedule = autopipeline.info.schedule

for step_idx in range(cfg.step_scheduler.max_steps):
    batch = next(iter(dataloader))

    losses = []
    pp_schedule.step(
        input_ids=batch["input_ids"].to(device),
        labels=batch["labels"].to(device),
        losses=losses,
        device=device,
    )

    if autopipeline.info.has_last_stage:
        loss = sum(losses) / len(losses)

    optimizer.step()
    optimizer.zero_grad()

    if dist.get_rank() == 0 and step_idx % 10 == 0:
        print(f"Step {step_idx}, Loss {loss.item()}")
```

---

## 性能优化策略

### 1. Reshard After Forward 优化

源代码位置: `parallelizer.py:446`

```python
# 优化: 最后一个 transformer layer 不 reshard_after_forward
# 原因: FSDP2 会立即 prefetch 它用于反向传播
reshard_after_forward = int(layer_id) < len(module) - 1
fully_shard(
    child_module,
    mesh=mesh,
    mp_policy=mp_policy,
    reshard_after_forward=reshard_after_forward,
    offload_policy=offload_policy,
)
```

**收益**: 减少一次 all-gather 通信

### 2. Defer FSDP Grad Sync

源代码位置: `fsdp2.py:131-134`

```python
defer_fsdp_grad_sync: Optional[bool] = field(
    default=True,
    metadata={"help": "Defer FSDP gradient sync to only the final micro-batch"},
)
```

**用途**: 在 Pipeline 中，仅在最后一个 microbatch 后同步 FSDP 梯度

**收益**: 减少梯度同步次数，提升 pipeline 吞吐

### 3. Activation Checkpointing

源代码位置: `parallelizer.py:158-181`

```python
if activation_checkpointing:
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp"):
            layers[i].mlp = checkpoint_wrapper(layer.mlp)
        if hasattr(layer, "self_attn"):
            layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)
```

**原理**: 前向时不保存 activation，反向时重新计算

**收益**: 大幅减少显存占用 (2-3x)，但增加计算量 (~30%)

### 4. 1F1B Pipeline Schedule

**优势对比 GPipe**:

- **GPipe bubble**: `(pp_size - 1) * microbatch_time`
- **1F1B bubble**: `(pp_size - 1) * microbatch_time / pp_size`

**示例** (PP=4, 8 microbatches):

```
GPipe bubble: 3 * 1 = 3 time units
1F1B bubble: 3 * 1 / 4 = 0.75 time units

效率提升: (3 - 0.75) / 3 = 75% bubble reduction
```

### 5. HSDP (Hierarchical Sharded Data Parallel)

**适用场景**: 多节点训练

**优势**:

- **节点内**: all-gather 参数 (高带宽 NVLink/NVSwitch)
- **节点间**: all-reduce 梯度 (低带宽 InfiniBand)

**通信量对比** (假设 8 nodes, 8 GPUs/node):

```
# 纯 FSDP (dp_size=64):
# 每次 all-gather: model_size / 64
# 每次 reduce-scatter: model_size / 64
# 跨节点通信: 高

# HSDP (dp_replicate=8, dp_shard=8):
# 节点内 all-gather: model_size / 8
# 节点间 all-reduce 梯度: model_size
# 跨节点通信: 低 (仅梯度, 可异步)
```

### 6. Mixed Precision

源代码位置: `fsdp2.py:108-113`

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,      # 参数 BF16
    reduce_dtype=torch.float32,      # 梯度 reduce FP32 (更稳定)
    output_dtype=torch.float32,      # 输出 FP32
)
```

**收益**:

- **显存**: ~50% reduction (BF16 vs FP32)
- **通信**: ~50% reduction (BF16 gradient all-reduce)
- **计算**: 2-3x speedup (on Tensor Cores)

### 7. Sequence Parallel

**原理**: 在序列维度切分 activation，配合 TP 使用

```python
# 无 SP:
# Activation shape: [batch, seq, hidden]
# 每个 TP rank 复制完整 activation
# 显存: O(batch * seq * hidden)

# 有 SP:
# Activation shape: [batch, seq/tp_size, hidden]
# 每个 TP rank 持有部分 sequence
# 显存: O(batch * seq * hidden / tp_size)
```

**收益**: 长序列训练显存大幅减少

### 8. Scale Grads in Schedule

源代码位置: `autopipeline.py:106`

```python
scale_grads_in_schedule: bool = False
```

**用途**: 在 pipeline schedule 中按 microbatch 数量缩放梯度

**收益**: 避免手动缩放，数值稳定性更好

---

## 总结

### 核心设计

NeMo AutoModel 的 3D Parallelism 实现完全基于 **PyTorch 原生 API**：

1. **DeviceMesh**: 5D mesh (pp, dp_replicate, dp_shard, cp, tp) 描述拓扑
2. **Pipeline Parallelism**: `torch.distributed.pipelining.PipelineStage` + `PipelineSchedule`
3. **FSDP2**: `torch.distributed.fsdp.fully_shard` 实现参数分片
4. **Tensor Parallelism**: `torch.distributed.tensor.parallel.parallelize_module` 应用 TP
5. **DTensor**: `Shard` 和 `Replicate` placement 自动处理通信

### 可组合性 (Composability)

**关键机制**: `parallelize_fn` 回调

```python
# AutoPipeline.build() 流程:
for stage in pipeline_stages:
    parallelize_fn(
        stage,
        world_mesh=mesh,
        pp_enabled=True,
        ...
    )

# parallelize_fn 实现 (默认):
def parallelize_for_pp(model, **kwargs):
    return fsdp2_manager.parallelize(model)  # 应用 FSDP2 + TP

# 结果: 每个 pipeline stage 都被 FSDP2+TP 并行化
```

### 工作流总结

```
┌──────────────────────────────────────────────────────────┐
│ 1. 初始化分布式环境 (torch.distributed)                   │
└──────────────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│ 2. 创建 FSDP2Manager                                     │
│    - 构建 5D DeviceMesh                                  │
│    - 推导 dp_size, dp_shard_size                         │
└──────────────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│ 3. 加载模型到 meta device                                 │
│    - NeMoAutoModelForCausalLM.from_pretrained()          │
└──────────────────────────────────────────────────────────┘
                       ↓
           ┌───────────┴───────────┐
           ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│ 4a. 无 Pipeline      │  │ 4b. 使用 AutoPipeline│
│                     │  │                     │
│ model_wrapper.      │  │ autopipeline.build()│
│ parallelize(model)  │  │ - 切分模型          │
│                     │  │ - 为每个 stage 应用  │
│ - 应用 TP           │  │   parallelize_fn    │
│ - 应用 FSDP2        │  │   (FSDP2 + TP)      │
└─────────────────────┘  └─────────────────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ 5. 构建优化器                                             │
│    - AdamW, SGD, etc.                                    │
└──────────────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│ 6. 训练循环                                               │
│    - 无 PP: 标准 forward -> backward -> optimizer.step   │
│    - 有 PP: pp_schedule.step() 自动处理 microbatches      │
└──────────────────────────────────────────────────────────┘
```

### 关键优势

1. **完全 Torch-native**: 无自定义通信，易于调试和维护
2. **SPMD 模型**: 同一脚本适配 1-1000+ GPU
3. **灵活可组合**: PP, DP, TP 可任意组合，配置化
4. **HuggingFace 兼容**: 任何 HF 模型开箱即用
5. **性能优化**: HSDP, 1F1B, Activation Checkpointing, Mixed Precision
6. **策略模式**: 支持自定义并行化策略

### 源代码统计

| 模块 | 文件 | 行数 | 核心功能 |
|-----|------|------|---------|
| FSDP2Manager | `fsdp2.py` | 318 | DeviceMesh 初始化, FSDP2+TP 应用 |
| Parallelizer | `parallelizer.py` | 1120 | 并行化策略, TP plan 生成, FSDP2 递归应用 |
| AutoPipeline | `autopipeline.py` | ~300 | Pipeline 管理, 自动切分接口 |
| Functional | `pipelining/functional.py` | ~600 | Pipeline 切分, schedule 构建 |
| Train Recipe | `train_ft.py` | ~1600 | 端到端训练流程, 组合所有组件 |

### 适用场景

| 场景 | 推荐配置 | 示例 (64 GPUs) |
|-----|---------|---------------|
| 小模型 (<10B) | DP only | dp=64, tp=1, pp=1 |
| 中模型 (10B-100B) | DP + TP | dp=8, tp=8, pp=1 |
| 大模型 (100B-1T) | DP + TP + PP | dp=4, tp=4, pp=4 |
| 超长序列 | DP + TP + CP | dp=8, tp=4, cp=2, pp=1 |
| 多节点 | HSDP + TP + PP | dp_replicate=8, dp_shard=2, tp=4, pp=1 |

### 未来方向

1. **Virtual Pipeline Stages**: Interleaved 1F1B 进一步减少 bubble
2. **Zero Bubble Pipeline**: Microsoft ZeRO++风格的 pipeline 优化
3. **Context Parallelism 增强**: Ulysses, Ring Attention 集成
4. **Expert Parallelism**: MoE 模型的专家并行优化
5. **自动调优**: 根据模型大小和 GPU 数量自动选择最优并行配置

---

## 参考资料

### 源代码位置

- **FSDP2**: `nemo_automodel/components/distributed/fsdp2.py`
- **Parallelizer**: `nemo_automodel/components/distributed/parallelizer.py`
- **AutoPipeline**: `nemo_automodel/components/distributed/pipelining/autopipeline.py`
- **Functional**: `nemo_automodel/components/distributed/pipelining/functional.py`
- **Train Recipe**: `nemo_automodel/recipes/llm/train_ft.py`
- **Optimized TP Plans**: `nemo_automodel/components/distributed/optimized_tp_plans.py`

### PyTorch 文档

- **DeviceMesh**: https://pytorch.org/docs/stable/distributed.tensor.parallel.html
- **FSDP2**: https://pytorch.org/docs/stable/fsdp.html
- **DTensor**: https://pytorch.org/docs/stable/distributed.tensor.html
- **Pipeline Parallelism**: https://pytorch.org/docs/stable/distributed.pipelining.html

### 论文

- **GPipe**: [Huang et al., 2019] - Google's Pipeline Parallelism
- **PipeDream (1F1B)**: [Narayanan et al., 2019] - Microsoft's Pipeline Schedule
- **Megatron-LM**: [Shoeybi et al., 2019] - NVIDIA's Tensor Parallelism
- **ZeRO**: [Rajbhandari et al., 2020] - Microsoft's Data Parallelism
- **FSDP**: [Zhao et al., 2023] - PyTorch's Fully Sharded Data Parallel

---

**文档版本**: v1.0
**最后更新**: 2026-01-04
**分析基于**: NeMo AutoModel commit `a8d9ca3`
**分析者**: Claude (Anthropic)
