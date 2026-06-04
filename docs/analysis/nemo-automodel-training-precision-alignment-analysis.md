# NeMo AutoModel 训练精度对齐体系源码与 Commit History 深度分析

## 0. Executive Summary

| 维度 | 结论 |
|------|------|
| **项目名称** | NeMo AutoModel (NVIDIA) |
| **总体判断** | **较完整** — 在 checkpoint 精度保持、TP/CP correctness 验证、MoE 精度控制方面有体系化能力；在 golden loss 自动回归、PP/EP/DP correctness 验证、NaN 检测、tensor dump 对齐工具方面存在显著缺口 |
| **最强能力** | (1) Checkpoint robustness 六阶段测试框架（KL 散度 + loss 续训一致性）；(2) TP=1 vs TP=2 KL 散度 parity 测试；(3) CP=1 vs CP=2 forward/grad/param 三层对齐测试；(4) 34 组 golden loss JSONL 基线；(5) MoE 路由/辅助损失的 fp32 精度控制体系 |
| **最大短板** | (1) 无 `torch.use_deterministic_algorithms` 支持；(2) 无 per-TP/PP rank RNG 隔离（无 CudaRNGStatesTracker）；(3) Golden JSONL 存在但未接入自动化 CI 断言；(4) PP/EP/DP 无 correctness parity 测试；(5) FSDP2 路径无 NaN/Inf 检测；(6) 无 tensor dump / activation compare 基础设施（仅有 example 级别工具） |
| **最值得借鉴的源码模块** | `tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py`；`tests/functional_tests/context_parallel/run_attention_cp.py`；`tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py`；`nemo_automodel/components/training/rng.py`；`nemo_automodel/components/moe/layers.py`（Gate 精度控制）；`examples/convergence/tulu3/model-verification/compare_activations.py` |
| **最值得研究的 commits / PR** | `6a1274d8` (#1606 checkpoint robustness)；`c59a96cc` (#1647 golden values)；`86d8c0fa` (#2248 mixed-precision optimizer)；`9b2b06bd` (#2201 MoE fp32 routing)；`15c3364e` (#1365 EP collective deadlock)；`0adca682` (#1412 MoE aux_loss gradient scaling)；`ec157f91` (#1985 NaN loss guard)；`0327814d` (#2083 MoE aux-loss dtype mismatch) |
| **是否适合作为训练精度对齐基础设施参考** | **是** — 在 checkpoint round-trip、parallel parity 测试、MoE 精度控制等方面有可直接复用的设计模式；但缺少 tensor dump 基础设施和 golden loss 自动化断言，需要补齐 |

---

## 1. 项目训练流程与精度相关架构总览

### 1.1 训练主入口

| 层级 | 文件 | 入口 |
|------|------|------|
| CLI | `nemo_automodel/cli/app.py:95` | `main()` — 解析 YAML，dispatch 到 recipe |
| Recipe | `nemo_automodel/recipes/llm/train_ft.py:892` | `TrainFinetuneRecipeForNextTokenPrediction(BaseRecipe)` |
| Setup | `train_ft.py:907` | `setup()` — 构建 model, optimizer, dataloader, checkpointer, scheduler |
| 训练循环 | `train_ft.py:1287` | `run_train_validation_loop()` — epoch/batch 外层循环 |
| 训练步 | `train_ft.py:1472` | `_run_train_optim_step()` — grad accum 循环 + optimizer step |
| Forward/Backward | `train_ft.py:1349` | `_forward_backward_step()` — device transfer → CP ctx → FP8 ctx → forward → loss → backward |

### 1.2 配置系统精度传递路径

```
YAML 配置文件
 ├── model.torch_dtype (bf16/fp32) → AutoModel.from_pretrained → 权重存储 dtype
 ├── distributed.mp_policy → FSDP2Config → MixedPrecisionPolicy
 │    ├── param_dtype (计算 dtype)
 │    ├── reduce_dtype (通信 dtype，默认 fp32)
 │    └── output_dtype (输出 dtype)
 ├── distributed.autocast_dtype → torch.autocast 上下文
 ├── loss_fn.fp32_upcast → MaskedCrossEntropy.fp32_upcast (默认 True)
 ├── fp8.enabled → FP8Config → torchao/TE FP8 转换
 ├── optimizer._target_ → Adam/AdamW/FusedAdam 选择
 │    ├── master_weight_dtype
 │    ├── exp_avg_dtype
 │    └── exp_avg_sq_dtype
 └── seed → StatefulRNG.seed (默认 42)
```

**关键设计特征**：精度控制分散在 4 个独立配置对象（model, distributed, loss, optimizer），无统一 `PrecisionConfig` 协调验证。

### 1.3 Training Step 精度流

```
_forward_backward_step():
  1. batch → device (dtype 不变)
  2. CP context (可选)
  3. FP8 autocast context (可选，仅 TE FP8 路径)
  4. model.forward() → FSDP2 MixedPrecisionPolicy 自动 cast
  5. calculate_loss() → MaskedCrossEntropy(fp32_upcast=True) → logits.float() → F.cross_entropy
  6. loss *= dp_group_size → .backward() → FSDP2 reduce_dtype=fp32 all-reduce
  7. scale_grads_and_clip_grad_norm() → fp32 范数计算 → clip
  8. optimizer.step() → AdamW (bf16 moments) 或 FusedAdam (fp32 master weights)
```

### 1.4 随机性控制

**核心模块**：`nemo_automodel/components/training/rng.py`

| 组件 | 实现 | 精度对齐意义 |
|------|------|-------------|
| `init_all_rng(seed, ranked)` | `random.seed` + `np.random.seed` + `torch.manual_seed` + `cuda.manual_seed_all` | 全局 RNG 初始化 |
| `StatefulRNG` | 封装 seed + `state_dict()`/`load_state_dict()` | 支持 checkpoint resume 时恢复 RNG 状态 |
| `ScopedRNG` | context manager，保存/恢复 RNG 状态 | 用于 `build_model()`、`build_dataloader()`、`_run_validation_epoch()` |
| Ranked seeding | `seed + dist.get_rank()` | 不同 rank 使用不同 seed |
| Dataset sampler seed | `g.manual_seed(self.seed + self.epoch)` | 每个 epoch 确定性 shuffle |

**缺失**：无 `CudaRNGStatesTracker` 式 per-TP/PP rank RNG 隔离；无 `torch.use_deterministic_algorithms` / `cudnn.deterministic` / `cudnn.benchmark` 控制。

### 1.5 Mixed Precision 架构

| 层级 | 机制 | 默认值 | 文件 |
|------|------|--------|------|
| FSDP2 MP Policy | `MixedPrecisionPolicy(param=bf16, reduce=fp32, output=bf16)` | `config.py:122-131` | `distributed/config.py` |
| Loss fp32 upcast | `MaskedCrossEntropy(fp32_upcast=True)` | True | `loss/masked_ce.py:25` |
| Autocast | `FSDP2Config.autocast_dtype` | None (不使用) | `distributed/config.py:111` |
| FP8 (torchao) | `FP8Config` → `convert_to_float8_training` | disabled | `quantization/fp8.py:28` |
| FP8 (TE) | `TEFp8Config` → `te.pytorch.quantization.autocast` | disabled | `models/common/utils.py:130` |
| GradScaler | **不存在** | — | — |
| TF32 控制 | **不存在** | PyTorch 默认 (Ampere+ 自动启用) | — |

---

## 2. 精度对齐能力矩阵

| 能力项 | 是否具备 | 源码证据 | commit/PR 证据 | 成熟度 | 备注 |
|--------|----------|----------|----------------|--------|------|
| 配置一致性扫描 | 间接存在 | `precision_warnings.py` 检测 bf16+AdamW 组合 | `86d8c0fa` (#2248) | 1 | 仅 warning，无系统性 config diff/validation |
| 随机种子/RNG 控制 | 明确存在 | `training/rng.py`: `StatefulRNG`, `ScopedRNG`, `init_all_rng` | — | 3 | 有 checkpoint RNG 状态保存；缺 per-TP/PP RNG 隔离 |
| 数据加载顺序确定性 | 明确存在 | `datasets/*/sampler.py`: `g.manual_seed(seed + epoch)`, ranked seed | `756863c9` (configurable shuffle_seed) | 3 | epoch-seeded 确定性 shuffle；Megatron dataset 有 `random_seed` |
| 初始权重一致性 | 明确存在 | `ScopedRNG(seed, ranked=True)` wraps `build_model()` | — | 2 | 确保同一 seed 同一权重；但无跨框架权重比较工具 |
| 单步 forward loss 对齐 | 间接存在 | `test_masked_ce.py`: `allclose(actual, expected, atol=1e-6)` | — | 2 | 单元测试级别；无 end-to-end 单步 golden loss 断言 |
| activation dump/compare | 间接存在 | `examples/convergence/tulu3/model-verification/compare_activations.py` | `d509e78b` (#1554) | 1 | 仅 example 级别，非测试/CI 集成 |
| gradient dump/compare | 明确存在 | `run_attention_cp.py`: CP=1 vs CP=2 gradient 比较 with `assert_close` | — | 3 | 仅限 CP 场景；无通用 gradient dump 工具 |
| optimizer state 对齐 | 间接存在 | `test_dcp.py`: optimizer state `allclose` after checkpoint round-trip | — | 3 | 有 round-trip 测试；bf16 Adam moments 是已知精度限制 |
| scheduler/lr curve 对齐 | 间接存在 | golden JSONL 中记录 `lr` 每步值 | `c59a96cc` (#1647) | 2 | 有记录但无自动化比较 |
| loss curve golden regression | 部分具备 | `tests/ci_tests/golden_values/` 34 组 JSONL 文件 | `c59a96cc` (#1647) | 2 | **JSONL 存在但未接入自动化 CI 断言** |
| mixed precision 对齐 | 明确存在 | `FSDP2Config.mp_policy`, `fp32_upcast`, `reduce_dtype=fp32` | `86d8c0fa` (#2248 reduce_dtype 改为 fp32), `6ff3cdc7` (#298) | 3 | 有系统配置；缺 bf16 vs fp32 parity 测试 |
| FP16/BF16/FP8 数值稳定性 | 部分具备 | MoE gate fp32 upcast, SwiGLU fp32 clamp, lm_head fp32 upcast | `9b2b06bd` (#2201), `8cb1f5ce` (#2173), `917b75e1` (#767) | 2 | 按需修复式，非系统性验证 |
| TF32 控制 | 未发现 | — | — | 0 | 无 `allow_tf32` 设置 |
| NaN/Inf/overflow 检测 | 部分具备 | `MegatronFSDPConfig.check_for_nan_in_grad=True`; `masked_ce.py` 零 token guard | `ec157f91` (#1985), `46b74fbb` (#2259) | 1 | 仅 MegatronFSDP 路径；FSDP2 路径无 NaN guard |
| checkpoint resume 一致性 | 明确存在 | `test_checkpoint_robustness_llm.py` Phase 6: baseline vs resume loss diff < 5e-3 | `6a1274d8` (#1606) | 3 | 有系统性功能测试 |
| data parallel correctness | 未发现 | — | — | 0 | 无 DP=1 vs DP=N parity 测试 |
| tensor parallel correctness | 明确存在 | `run_tp_output_parity_minified.py`: TP=1 vs TP=2 KL < 2e-6，覆盖 6 个模型族 | — | 4 | 体系化，多模型，多 SP 模式 |
| pipeline parallel correctness | 未发现 | 有 PP 实现但无 PP vs non-PP 数值 parity 测试 | — | 0 | 仅有单元测试验证 stage 切分逻辑 |
| sequence parallel correctness | 明确存在 | 通过 TP parity 测试 (`sequence_parallel=True`) 覆盖 | — | 4 | |
| expert parallel / MoE correctness | 部分具备 | 单 GPU 单元测试；无 EP=1 vs EP=N 分布式 parity 测试 | — | 1 | |
| collective communication correctness | 明确存在 | `run_clip_grad_norm_correctness.py`: EP+FSDP2 gradient norm 分布式验证 | — | 3 | 覆盖 EP, FSDP2, EP+FSDP2, Replicate, inf-norm |
| CI 精度回归测试 | 部分具备 | checkpoint robustness + golden JSONL + TP parity + CP parity | `6a1274d8` (#1606), `c59a96cc` (#1647) | 2 | golden JSONL 未自动化断言 |
| 自动化二分定位能力 | 未发现 | — | — | 0 | 无 bisect 工具 |
| 跨硬件/跨后端对齐能力 | 未发现 | golden JSONL 标注 `_h100` 但无 cross-hardware 比较 | — | 0 | |

---

## 3. 源码证据地图

### 3.1 RNG 与 Determinism

| 文件 | 关键函数/类 | 功能 |
|------|------------|------|
| `components/training/rng.py` | `init_all_rng()`, `StatefulRNG`, `ScopedRNG`, `RNGState` | 全局 RNG 管理，checkpoint 状态保存/恢复 |
| `components/datasets/llm/megatron/sampler.py:274` | `g.manual_seed(self.seed + self.epoch)` | 确定性 epoch shuffle |
| `components/datasets/diffusion/sampler.py:163` | `g.manual_seed(self.seed + self.epoch)` | 同上 |
| `components/moe/layers.py:175-182` | per-input seed from tensor content | MoE routing noise 确定性 |
| `components/moe/parallelizer.py:227/231` | `preserve_rng_state=True` | activation checkpointing RNG 保持 |
| `tests/unit_tests/training/test_rng.py` | 完整测试 | 验证 reproducibility, uniqueness, ranked mode, ScopedRNG |

### 3.2 Loss 函数精度控制

| 文件 | 关键类 | 精度相关机制 |
|------|--------|-------------|
| `components/loss/masked_ce.py:75` | `MaskedCrossEntropy` | `fp32_upcast=True` → `logits.float()` before CE |
| `components/loss/masked_ce.py:98-99` | 零 token guard | `num_label_tokens == 0` → return 0.0 (防 NaN) |
| `components/loss/kd_loss.py:69-74` | `_kl_forward_tp` | 分布式 softmax: global max subtraction → fp32 exp → all-reduce denom |
| `components/loss/chunked_ce.py:104` | `ChunkedCrossEntropy` | 分块计算降低内存；Python float accumulator (torch.compile 风险) |
| `components/loss/mtp.py:104-109` | `calculate_mtp_loss` | MTP 辅助损失 scaling |

### 3.3 MoE 精度控制

| 文件 | 行号 | 机制 |
|------|------|------|
| `components/moe/layers.py:276` | `e_score_correction_bias` buffer | 强制 fp32 — "small quantization errors in bf16 can cause completely different expert routing" |
| `components/moe/layers.py:326-339` | softmax 路径 | 显式 `dtype=self.gate_precision or torch.float32` |
| `components/moe/layers.py:361` | sqrtsoftplus 路径 | 显式 `.float()` |
| `components/moe/layers.py:370/390` | sigmoid 路径 | **无显式 fp32 upcast** (设计不一致) |
| `components/moe/layers.py:506` | `e_score_correction_bias_master` | fp32 master weight 模式 |
| `components/moe/layers.py:566-567` | `_compute_aux_loss` | 显式 `.float()` 防 activation checkpointing dtype 漂移 |

### 3.4 Checkpoint 精度保持

| 文件 | 关键函数 | 机制 |
|------|---------|------|
| `checkpoint/checkpointing.py:1630` | `_load_hf_checkpoint_preserving_dtype` | 读取 safetensors 保持原始 dtype |
| `checkpoint/checkpointing.py:1383-1411` | `_load_full_state_dict_into_model` | post-load dtype fixup |
| `checkpoint/stateful_wrappers.py:411-431` | `OptimizerState.state_dict()` | FSDP-aware optimizer state 序列化 |
| `checkpoint/_backports/hf_utils.py` | `DTYPE_MAP` | 扩展 FP8 dtype 支持 (`F8_E4M3`, `F8_E5M2`, `F8_E8M0`) |
| `models/deepseek_v3/state_dict_adapter.py:375` | `dequantize_from_fp8` | FP8→bf16 checkpoint 转换 |

### 3.5 分布式精度验证测试

| 测试文件 | 验证内容 | 比较方法 | 阈值 |
|----------|----------|----------|------|
| `functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py` | TP=1 vs TP=2 logits | KL divergence | 2e-6 (bf16), 1e-4 (Baichuan) |
| `functional_tests/context_parallel/run_attention_cp.py` | CP=1 vs CP=2 output/grad/param | `torch.testing.assert_close` | atol=1e-2, rtol=1e-2 (可配) |
| `functional_tests/context_parallel/run_hybrid_nemotron_v3_cp.py` | hybrid CP/Mamba | 同上 | atol=5e-2, rtol=1e-2 |
| `functional_tests/context_parallel/run_mamba_cp.py` | Mamba CP | 同上 | atol=0.01, rtol=1e-2 |
| `functional_tests/llm_pretrain_and_kd/run_clip_grad_norm_correctness.py` | EP+FSDP2 gradient norm | reference vs impl, `atol=1e-4` | 1e-4 |
| `functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py` | checkpoint round-trip + resume | KL divergence + loss diff | KL: 0~1e-5; loss: 5e-3 |
| `functional_tests/llm_pretrain_and_kd/loss/run_te_parallel_ce_dtensor.py` | TE parallel CE vs reference | `torch.allclose(rtol=1e-2, atol=1e-2)` | 1e-2 |

---

## 4. Commit / PR / Issue 历史演进时间线

### 时间线概览

```
2025-11   ┃ 917b75e1 — configurable router precision for MoEs (#767)
           ┃  ↳ 首次引入 gate_precision 参数，MoE softmax 可配 fp32
2025-12   ┃ e9813626 / 6ff3cdc7 — Simplify mixed_precision logic (#298)
           ┃  ↳ 统一 FSDP2 mixed precision / offload / device_mesh 配置
2026-01   ┃ cf4a2942 — fp8 dequant kernels (#1110)
           ┃  ↳ FP8 checkpoint 反量化，Triton kernel + torch 回退
2026-02   ┃ 15c3364e — EP collective deadlock fix (#1365)
           ┃  ↳ DTensor 可变长度 token 导致 NCCL 死锁
           ┃ 2952aa6f — MoE mixed precision policy configurable (#1392)
           ┃  ↳ MoE parallelizer 精度策略可从 recipe 配置
2026-03   ┃ 0adca682 — correct MoE aux_loss gradient scaling (#1412)
           ┃  ↳ 辅助损失梯度未正确回传，gate 权重实际无梯度
           ┃ 9c3f9439 — rms_norm numerical instability (#1410)
           ┃  ↳ bf16 下 RMSNorm 数值不稳定
           ┃ fb47c5d0 — native fp8 checkpoint + peft (#1459)
           ┃  ↳ FP8 + PEFT 联合 checkpoint 修复
           ┃ d509e78b — Tulu-3 E2E convergence pipeline (#1554)
           ┃  ↳ 端到端收敛验证流水线 + 激活比较工具
2026-04   ┃ c59a96cc — Add recipe golden values (#1647) ★★★
           ┃  ↳ 34 组 golden loss JSONL 基线文件
           ┃ 6a1274d8 — checkpoint robustness functional tests (#1606) ★★★
           ┃  ↳ 六阶段 checkpoint robustness 测试框架
           ┃ 8f0f38f1 — align EP expert weight dtype (#1913)
           ┃  ↳ FSDP2 跨 mesh EP 权重 dtype 不一致
           ┃ ec157f91 — guard NaN loss from zero label tokens (#1985)
           ┃  ↳ VLM 训练零标签 token → NaN loss
           ┃ da6061bb — gradient clip with torch_mm + EP (#2012)
           ┃  ↳ EP + torch.compile gradient clip 修复
2026-05   ┃ 0327814d — MoE aux-loss dtype mismatch under AC (#2083)
           ┃  ↳ activation checkpointing 下 aux_loss dtype 不一致
           ┃ 9b2b06bd — MoE routing + attention softmax in fp32 (#2201)
           ┃  ↳ DeepSeek-V4 路由 bf16 精度丢失 + attention softmax bf16
           ┃ 8cb1f5ce — SwiGLU clamp in fp32 (#2173)
           ┃  ↳ DeepSeek-V4 shared expert SwiGLU 溢出
           ┃ 0fb6535c — preserve reference fp32 parameters (#2216)
           ┃  ↳ DeepSeek-V4 fp32 参数在 FSDP2 下被覆盖
           ┃ 86d8c0fa — mixed-precision optimizer-state setup (#2248) ★★★
           ┃  ↳ reduce_dtype 默认改为 fp32 + bf16 optimizer 稳定性警告
```

### 详细条目

#### 1. `917b75e1` — feat: configurable router precision for moes (#767)
- **时间**: 2025-11-06
- **涉及文件**: `moe/layers.py` (+40), `moe/utils.py` (+7), `tests/unit_tests/moe/test_layers.py` (+336)
- **问题背景**: MoE gate 的 softmax 默认在 bf16 下运行，小模型训练中观察到路由不稳定
- **修改内容**: 引入 `gate_precision` 参数，可通过配置强制 softmax 在 fp32 执行
- **新增测试**: 完整的 `test_layers.py` gate precision 测试套件 (336 行)
- **对精度对齐体系的启发**: **Gate 精度应作为独立可配置项**。路由决策的微小精度差异会被 expert selection 放大为完全不同的计算路径

#### 2. `15c3364e` — fix: EP collective deadlock with variable-length token counts (#1365)
- **时间**: 2026-02-24
- **涉及文件**: `moe/experts.py`
- **问题背景**: `DTensor.from_local(x, [Shard(0)]).full_tensor()` 假设所有 rank token 数量一致。变长序列 packing 打破此假设，导致 NCCL 死锁
- **修改内容**: 三阶段修复 — pad+all_gather+trim → all_reduce+narrow → gradient anchor (`y + x*0.0`)
- **新增测试**: 无专门测试
- **对精度对齐体系的启发**: **分布式 collective 的 size 假设是隐含的精度/正确性风险**。gradient anchor (`y + x*0.0`) 确保所有 rank 进入 backward collective，即使某些 rank 无实际 token

#### 3. `0adca682` — fix: correct MoE auxiliary loss gradient scaling (#1412)
- **时间**: 2026-03-02
- **涉及文件**: `moe/layers.py`, `recipes/llm/train_ft.py`
- **问题背景**: 两个 bug 导致 MoE 辅助损失梯度完全失效：(1) `MoEAuxLossAutoScaler.apply()` 返回值未捕获，梯度无法反传；(2) `main_loss_backward_scale` 从未设置，梯度被错误除以 `dp_group_size`
- **修改内容**: 修复 autograd function return value capture + 在 recipe 中正确设置 backward scale
- **新增测试**: 未发现专门的回归测试
- **对精度对齐体系的启发**: **辅助损失的梯度缩放在分布式训练中极易出错**。FSDP2 的 gradient averaging 和 PP 的 post-hoc scaling 会隐式改变梯度尺度，辅助损失需要显式 counter-scaling

#### 4. `c59a96cc` — ci: Add recipe golden values (#1647)
- **时间**: 2026-04-02
- **涉及文件**: 34 个 JSONL 文件 in `tests/ci_tests/golden_values/`
- **问题背景**: 需要可参考的基线 loss 曲线
- **修改内容**: 为 24 个 LLM + 10 个 VLM recipe 添加 100 步 H100 training JSONL（包含 loss, grad_norm, lr, tps, mfu, num_label_tokens）
- **新增测试**: **无自动化断言** — JSONL 作为参考文件存在
- **对精度对齐体系的启发**: **Golden values 的价值 = 数据 × 自动化**。仅有数据而无自动化比较，golden values 是文档而非测试

#### 5. `6a1274d8` — test: add checkpoint robustness functional tests (#1606)
- **时间**: 2026-04-06
- **涉及文件**: `test_checkpoint_robustness_llm.py` (新增), `test_checkpoint_robustness_biencoder.py` (新增)
- **问题背景**: checkpoint save/load 后 logits 偏差需要系统性验证
- **修改内容**: 六阶段测试框架：
  - Phase 1: 训练 N 步并保存 checkpoint
  - Phase 2: 捕获参考 logits
  - Phase 3: 从 consolidated checkpoint 重载 AutoModel，断言 KL=0 (或 KL<1e-5 for TP>1)
  - Phase 4: 重载到 vanilla HF `AutoModelForCausalLM`，断言 KL<5e-3
  - Phase 5: 不同 TP size 重载 (cross-TP resharding)
  - Phase 6: checkpoint resume + 3 步续训，断言 `|loss_baseline - loss_resume| < 5e-3`
- **新增测试**: 完整的功能测试套件
- **对精度对齐体系的启发**: **多阶段 checkpoint robustness 测试是可直接复用的设计模式**。Phase 3 (same-framework reload) 和 Phase 4 (cross-framework reload) 分离了不同层次的精度保证

#### 6. `86d8c0fa` — fix(training): clarify mixed-precision optimizer-state setup (#2248)
- **时间**: 2026-05-28
- **涉及文件**: `distributed/config.py`, `training/precision_warnings.py` (新增), `recipes/llm/train_ft.py`, 多个 YAML 配置
- **问题背景**: FSDP2 默认 `reduce_dtype=bfloat16` 导致 gradient all-reduce 精度损失；用户使用 vanilla AdamW + bf16 参数未意识到 optimizer moments 也是 bf16
- **修改内容**: (1) `reduce_dtype` 默认改为 `torch.float32`；(2) 新增 `precision_warnings.py` 检测 bf16+AdamW 组合并 warning
- **新增测试**: `test_precision_warnings.py` 完整测试
- **对精度对齐体系的启发**: **reduce_dtype 是最容易被忽略的精度配置**。bf16 reduce 在 large-scale training 中累积误差显著。主动 warning 系统是低成本高收益的精度防护

#### 7. `9b2b06bd` — fix(deepseek-v4): keep MoE routing scores and attention softmax in fp32 (#2201)
- **时间**: 2026-05-10
- **涉及文件**: `models/deepseek_v4/model.py`, `models/deepseek_v4/layers.py`
- **问题背景**: MTP parity testing (#2191) 发现 DeepSeek-V4 与参考实现输出偏差。根因：(1) sqrtsoftplus Gate 计算 fp32 后立即 cast 回 bf16 丢精度；(2) eager attention softmax 在 bf16 下运行
- **修改内容**: 移除 routing scores 的 `.to(scores.dtype)` cast；强制 attention softmax fp32
- **新增测试**: 无专门测试
- **对精度对齐体系的启发**: **"计算完 fp32 后立即 cast 回 bf16"是经典的精度陷阱**。routing 和 attention softmax 是精度敏感操作，必须全程 fp32

#### 8. `ec157f91` — fix: guard against zero label tokens causing NaN loss (#1985)
- **时间**: 2026-04-23
- **涉及文件**: `loss/masked_ce.py` (+2), `recipes/vlm/finetune.py` (+1), `tests/unit_tests/loss/test_masked_ce.py` (+13)
- **问题背景**: VLM 训练中某些 batch 所有 label 为 -100（无监督 token），`num_label_tokens=0`，除以零 → NaN loss → 训练崩溃
- **修改内容**: `masked_ce.py` return 0.0 when `num_label_tokens == 0`
- **新增测试**: 回归测试 `test_empty_supervision_returns_zero_loss`
- **对精度对齐体系的启发**: **边界条件的 NaN 防护是 bug→fix→regression test 闭环的典型案例**

#### 9. `0327814d` — fix: MoE aux-loss dtype mismatch under activation checkpointing (#2083)
- **时间**: 2026-05-06
- **涉及文件**: `moe/layers.py` (+10), `moe/megatron/moe_utils.py` (+5)
- **问题背景**: activation checkpointing recompute 在不同 autocast context 下执行，导致 aux_loss 的 `original_scores` 和 `expert_load` dtype 不一致（首次 fp32，recompute bf16）
- **修改内容**: `_compute_aux_loss` 中显式 `.float()` 强制 fp32
- **新增测试**: 无专门测试
- **对精度对齐体系的启发**: **Activation checkpointing 的 recompute path 是精度不一致的高发区**。recompute 在不同 autocast context 下执行，必须显式控制关键运算的 dtype

#### 10. `8f0f38f1` — fix(moe): align EP expert weight dtype with activation dtype (#1913)
- **时间**: 2026-04-21
- **涉及文件**: `moe/experts.py`
- **问题背景**: FSDP2 的 `MixedPrecisionPolicy` 仅在其 wrap mesh 上 cast param，跨 mesh DTensor（EP-sharded MoE experts）保持 fp32，而激活已被 cast 为 bf16 → `grouped_mm` dtype 不匹配报错
- **修改内容**: 在 `GroupedExperts.forward` 中将本地 expert weights cast 到输入 activation dtype
- **新增测试**: 无专门测试
- **对精度对齐体系的启发**: **FSDP2 的 MixedPrecisionPolicy 不覆盖跨 mesh DTensor**。这是 FSDP2 + EP 交叉场景的固有限制

---

## 5. 典型精度问题案例复盘

### Case 1: Checkpoint resume 后 loss 不一致

- **现象**: 从 checkpoint resume 后训练 loss 与连续训练偏差较大
- **根因**: (1) LR scheduler 的 `lr_decay_steps` 在 resume 时重新计算，与原始训练不同；(2) RNG 状态未正确恢复
- **如何定位**: `test_checkpoint_robustness_llm.py` Phase 6 — baseline vs resume loss 比较
- **如何修复**: Phase 6 中显式 pin `lr_decay_steps = original_max_steps`（`test_checkpoint_robustness_llm.py:671-672`）；RNG 通过 `StatefulRNG.state_dict()` 保存恢复
- **是否新增测试**: 是 — Phase 6 本身即为回归测试
- **借鉴**: **Resume test 必须 pin LR schedule 参数**，否则 scheduler state 恢复后参数不同导致 curve 偏移

### Case 2: BF16 下 loss drift — optimizer moments 精度不足

- **现象**: 长期训练 bf16 + vanilla AdamW，loss 在后期出现 drift/不收敛
- **根因**: AdamW 的 `exp_avg` 和 `exp_avg_sq` 与 param 同 dtype (bf16)，bf16 仅 7 位尾数不足以精确追踪 moment 的微小更新
- **如何定位**: `precision_warnings.py` 主动检测 bf16+AdamW 组合并 warning
- **如何修复**: 使用 TE FusedAdam with `master_weights=True`，或设 `model.torch_dtype=float32` with FSDP mp_policy
- **是否新增测试**: 是 — `test_precision_warnings.py`
- **借鉴**: **主动 warning 系统成本极低但价值极高**。在问题发生前提醒用户

### Case 3: MoE routing scores bf16 精度丢失 (DeepSeek-V4)

- **现象**: DeepSeek-V4 与参考实现的 MTP parity 测试中 logits 偏差显著
- **根因**: `sqrtsoftplus` Gate 计算 `sqrt(softplus(x.float()))` 后立即 `.to(scores.dtype)` cast 回 bf16，丢失路由精度
- **如何定位**: MTP parity testing (#2191) 中逐层激活比较发现 MoE 层偏差
- **如何修复**: 移除 `.to(scores.dtype)` cast，保持 routing scores 在 fp32 — commit `9b2b06bd`
- **是否新增测试**: 否
- **借鉴**: **"计算完 fp32 后 cast 回原 dtype"是最常见的精度陷阱模式**

### Case 4: EP collective deadlock (非精度问题但影响正确性)

- **现象**: 变长序列 packing 下 EP 训练 hang/NCCL deadlock
- **根因**: `DTensor.from_local().full_tensor()` 假设 uniform token count。变长 packing → rank 间 buffer size 不一致 → NCCL 死锁
- **如何定位**: NCCL FlightRecorder 诊断
- **如何修复**: 三阶段 — pad+all_gather+trim + all_reduce+narrow + gradient anchor
- **是否新增测试**: 否
- **借鉴**: **分布式 collective 的隐含 size 假设是正确性风险**

### Case 5: MoE 辅助损失梯度完全失效

- **现象**: MoE load-balancing loss 梯度为零，expert 负载不均衡未被纠正
- **根因**: `MoEAuxLossAutoScaler.apply()` 返回值未捕获 → autograd graph 断开；`main_loss_backward_scale` 从未设置 → 梯度被隐式除以 dp_group_size
- **如何定位**: 手动检查 gate 权重梯度为零
- **如何修复**: 捕获 apply() 返回值 + 在 recipe 中设置 backward scale — commit `0adca682`
- **是否新增测试**: 否
- **借鉴**: **自定义 autograd function 的返回值必须被使用**。这类 bug 无 NaN/Inf 信号，仅表现为训练效果下降

### Case 6: Activation checkpointing 下 aux_loss dtype 漂移

- **现象**: MoE aux_loss 在 activation checkpointing 下 dtype 不一致，导致 backward 报错或 loss 数值不正确
- **根因**: recompute path 在不同 autocast context 下执行，`original_scores` 首次 fp32，recompute bf16
- **如何定位**: dtype mismatch 运行时报错
- **如何修复**: `_compute_aux_loss` 中显式 `.float()` — commit `0327814d`
- **是否新增测试**: 否
- **借鉴**: **activation checkpointing 的 recompute 是 dtype 不一致的高发区**

### Case 7: EP expert weight dtype 与 activation dtype 不匹配

- **现象**: `grouped_mm` 报错 `Expected b.scalar_type() == torch::kBFloat16 to be true, but got false`
- **根因**: FSDP2 `MixedPrecisionPolicy` 仅在 wrap mesh 上 cast param；EP-sharded expert weights 在不同 mesh，保持 fp32
- **如何定位**: 运行时 `grouped_mm` 报错
- **如何修复**: `GroupedExperts.forward` 中显式 `.to(input.dtype)` — commit `8f0f38f1`
- **是否新增测试**: 否
- **借鉴**: **FSDP2 + 多 mesh 场景下 MixedPrecisionPolicy 覆盖范围有限**

### Case 8: VLM 训练零标签 token → NaN loss

- **现象**: VLM 训练偶发 NaN loss → 训练崩溃
- **根因**: 某些 batch 所有 label 为 -100，`num_label_tokens=0` → 除以零 → NaN
- **如何定位**: loss 日志出现 NaN
- **如何修复**: `masked_ce.py` return 0.0 when `num_label_tokens == 0` — commit `ec157f91`
- **是否新增测试**: 是 — `test_empty_supervision_returns_zero_loss`
- **借鉴**: **最佳闭环案例**: bug → fix → regression test → CI 固化

### Case 9: Gradient clip 在 EP + torch.compile 下失败

- **现象**: GPT-OSS 120B recipe 在 EP + `torch._inductor.config._micro_pipeline_tp` 下 gradient clip 报错
- **根因**: `torch.compile` 下 DTensor gradient 的 norm 计算路径与 eager mode 不同
- **如何定位**: NCCL FlightRecorder + SIGABRT stack trace
- **如何修复**: 调整 gradient clip 实现 — commit `da6061bb`
- **是否新增测试**: 已有 `run_clip_grad_norm_correctness.py` 覆盖
- **借鉴**: **torch.compile + distributed 是精度/正确性问题的高发组合**

### Case 10: RMSNorm 数值不稳定

- **现象**: bf16 训练下 RMSNorm 输出不稳定
- **根因**: bf16 精度下 RMSNorm 的 variance 计算可能出现数值不稳定
- **如何定位**: 逐层 activation 比较
- **如何修复**: 修复 combined projection bias 加载 + RMSNorm 计算 — commit `9c3f9439`
- **是否新增测试**: 否
- **借鉴**: **归一化层（LayerNorm/RMSNorm）在低精度下是常见的数值不稳定源**

---

## 6. 三阶段精度对齐流程映射

### 阶段一：训练前准备与基础对齐

| 检查项 | 项目覆盖 | 证据 | 评估 |
|--------|----------|------|------|
| 配置一致性 | 部分 | `precision_warnings.py` 检测 bf16+AdamW | 无系统性 config diff/validation |
| 环境一致性 | 未发现 | — | 无 CUDA/driver/library 版本检查 |
| seed/RNG | 覆盖 | `StatefulRNG`, `ScopedRNG`, ranked seed | 缺 per-TP/PP RNG 隔离 |
| 数据顺序 | 覆盖 | epoch-seeded sampler, shuffle_seed configurable | 良好 |
| 模型结构 | 间接 | `validate_tp_mesh` 检查 head 数可分性 | 无模型结构 hash/fingerprint |
| 初始化权重 | 覆盖 | `ScopedRNG` wraps `build_model()` | 同 seed 同权重，但无跨框架比较 |
| dropout/正则 | 部分 | MoE `preserve_rng_state=True` | 无全局 dropout 确定性控制 |
| deterministic flags | **缺失** | 无 `torch.use_deterministic_algorithms` | **主要缺口** |

**评估**: 阶段一 **部分覆盖**。RNG 和 data order 做得好；deterministic flags 和环境一致性完全缺失。

### 阶段二：单卡/单步对齐

| 检查项 | 项目覆盖 | 证据 | 评估 |
|--------|----------|------|------|
| forward loss | 部分 | `test_masked_ce.py` allclose; 无端到端单步 golden loss | 单元测试级别 |
| activation | 间接 | `compare_activations.py` (example, non-CI) | 有工具但未集成 |
| backward gradient | 部分 | CP 测试中 gradient 比较 | 无通用 gradient dump 工具 |
| optimizer update | 间接 | checkpoint round-trip 验证 optimizer state | 无单步 optimizer update 精度验证 |
| scheduler | 间接 | golden JSONL 记录 lr | 无自动比较 |
| loss scaling | N/A | 项目使用 bf16 不需 GradScaler | — |
| tensor dump | **缺失** | 无通用 tensor dump hook 基础设施 | **主要缺口** |
| operator-level compare | **缺失** | — | **主要缺口** |

**评估**: 阶段二 **局部具备**。loss 函数有单元测试，CP/TP 测试有 gradient 比较；但无通用 tensor dump 和 operator-level 对齐工具。

### 阶段三：多步/分布式/长稳对齐

| 检查项 | 项目覆盖 | 证据 | 评估 |
|--------|----------|------|------|
| loss curve | 部分 | golden JSONL 存在但无自动化断言 | 需要补齐自动化 |
| checkpoint resume | 覆盖 | Phase 6 baseline vs resume loss diff | 良好 |
| DP correctness | **缺失** | 无 DP=1 vs DP=N parity 测试 | |
| TP correctness | 覆盖 | TP=1 vs TP=2 KL divergence, 6 个模型族 | **标杆** |
| PP correctness | **缺失** | 有实现无 parity 测试 | |
| SP correctness | 覆盖 | 通过 TP parity 测试覆盖 | |
| EP correctness | **缺失** | 无 EP=1 vs EP=N parity 测试 | |
| gradient accumulation | 部分 | state machine mock 测试；无数值等价验证 | |
| communication collectives | 覆盖 | gradient norm 分布式正确性测试 | |
| mixed precision stability | 部分 | reduce_dtype=fp32 默认；有 precision_warnings | 无 bf16 vs fp32 parity 测试 |
| NaN/Inf monitoring | 部分 | MegatronFSDP 有 NaN check；FSDP2 无 | |
| CI regression | 部分 | checkpoint robustness CI；golden JSONL 非自动化 | |

**评估**: 阶段三 **不均匀**。TP 和 CP correctness 是标杆级别；PP/DP/EP correctness 完全缺失；loss curve regression 有数据无自动化。

---

## 7. 可复用设计模式

### 7.1 六阶段 Checkpoint Robustness 测试框架

- **设计目标**: 系统性验证 checkpoint save/load/resume/cross-framework 的精度保持
- **源码位置**: `tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py`
- **工作流程**: Train → Checkpoint → Capture logits → Reload AutoModel (KL=0) → Reload HF (KL<5e-3) → Cross-TP reload → Resume loss continuity
- **优点**: 多层次精度保证（same-framework vs cross-framework vs cross-TP vs resume）；KL 散度是正确的 output-level 比较指标
- **局限**: 不验证 per-tensor dtype 保持（仅验证 logit 分布）；不覆盖 optimizer state 精度
- **迁移建议**: 直接复用六阶段结构；补充 per-tensor dtype assertion（参考 `test_hf_consolidated_llm.py:1166` 模式）

### 7.2 TP=1 vs TP=2 KL 散度 Parity 测试

- **设计目标**: 验证 tensor parallel 不改变模型输出分布
- **源码位置**: `tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py`
- **工作流程**: 构建 minified 模型 (2 层 thin) → TP=1 baseline → TP=2 with 同权重同输入 → KL divergence < 2e-6
- **优点**: minified 模型快速运行（秒级）；覆盖 6 个模型族；同时测 SP=True 和 SP=False
- **局限**: 仅验证 forward logits，不验证 backward gradient parity；阈值是 ad hoc 校准的
- **迁移建议**: 复用 minified 模型 + KL 比较模式；扩展到 PP 和 EP

### 7.3 CP=1 vs CP=2 三层对齐测试

- **设计目标**: 验证 context parallel 的 forward 输出、input gradient、parameter gradient 三者一致
- **源码位置**: `tests/functional_tests/context_parallel/run_attention_cp.py:91-153`
- **工作流程**: 用 `torch.testing.assert_close` 比较 `(output_atol, grad_atol, param_atol)` 三组阈值
- **优点**: **三层比较是最彻底的 parallel correctness 验证模式** — 不仅验证 forward，还验证 backward gradient 和 parameter gradient
- **局限**: 阈值跨配置不一致（科学记数法 vs 十进制混用）
- **迁移建议**: 将三层比较模式推广到 PP、EP、DP correctness 测试

### 7.4 Golden Loss JSONL 机制

- **设计目标**: 记录参考训练 loss 曲线作为基线
- **源码位置**: `tests/ci_tests/golden_values/` — 34 组 JSONL 文件
- **工作流程**: 运行参考训练 → 记录 `{step, loss, grad_norm, lr, tps, mfu, num_label_tokens}` → 存储为 JSONL
- **优点**: 每步完整指标；hardware-specific (`_h100`)；覆盖 LLM + VLM
- **局限**: **仅作为参考文件，无自动化比较逻辑**
- **迁移建议**: 补充 pytest fixture 自动加载 golden JSONL + assert `abs(actual_loss - golden_loss) < threshold` per step

### 7.5 MoE Gate 精度控制体系

- **设计目标**: 确保 MoE routing 在 fp32 精度下执行
- **源码位置**: `nemo_automodel/components/moe/layers.py` — `Gate` class
- **工作流程**: `gate_precision` 参数 → softmax/sqrtsoftplus fp32 → `e_score_correction_bias` fp32 buffer → `_compute_aux_loss` fp32 upcast
- **优点**: 多层防护（routing + bias + aux_loss）
- **局限**: sigmoid 路径遗漏 fp32 upcast
- **迁移建议**: 将 Gate 精度控制模式推广到所有精度敏感的 routing/attention score 计算

### 7.6 StatefulRNG + ScopedRNG 体系

- **设计目标**: 可 checkpoint、可恢复、scope-aware 的 RNG 管理
- **源码位置**: `nemo_automodel/components/training/rng.py`
- **工作流程**: `StatefulRNG(seed)` 初始化 → `ScopedRNG(seed)` 包裹 model/dataloader 构建 → `state_dict()` checkpoint → `load_state_dict()` resume
- **优点**: 干净的 context manager 接口；完整的 state 保存/恢复
- **局限**: 无 per-TP/PP rank RNG 隔离
- **迁移建议**: 在此基础上添加 `CudaRNGStatesTracker` 式 per-group RNG

### 7.7 Precision Warning 主动检测

- **设计目标**: 在训练开始前主动检测已知的精度配置风险
- **源码位置**: `nemo_automodel/components/training/precision_warnings.py`
- **工作流程**: `warn_if_torch_adam_with_bf16_params()` → 检测 optimizer 类型 + param dtype → rank-0 warning
- **优点**: 零运行时开销；仅 rank-0 输出避免日志重复
- **局限**: 仅检测一种组合 (bf16+AdamW)；无配置一致性全扫描
- **迁移建议**: 扩展为通用 `PrecisionConfig.validate()` 覆盖所有已知风险组合

### 7.8 层级式激活比较工具

- **设计目标**: 逐层比较 NeMo 与 HF 的 hidden states 和 logits
- **源码位置**: `examples/convergence/tulu3/model-verification/compare_activations.py`
- **工作流程**: HF 主进程 + NeMo torchrun 子进程 → forward hook 捕获每层 hidden states → cosine similarity ≥ 0.99 + max abs diff
- **优点**: 支持 `--gate-precision`、`--lm-head-precision` CLI 覆盖；进程隔离避免内存冲突
- **局限**: 在 examples/ 中，非 CI 集成；需要真实模型权重；仅推理时比较
- **迁移建议**: 提升为 `tests/functional_tests/precision/` 级别工具；适配训练时 activation dump

---

## 8. 缺口分析与改造建议

### P0: 必须补齐

#### 8.1 Golden Loss 自动化 CI 断言

- **问题**: 34 组 golden JSONL 存在但无自动化比较逻辑
- **为什么重要**: Golden values 无断言等于无测试。loss 回归可以在发布前被捕获
- **当前部分实现**: JSONL 文件已存在 (`tests/ci_tests/golden_values/`)
- **建议设计**: pytest fixture 加载 golden JSONL → 运行 50-100 步训练 → per-step `assert abs(actual_loss - golden_loss) < atol` → 失败时输出逐步 diff
- **涉及模块**: `tests/ci_tests/`, `tests/ci_tests/scripts/`
- **预期收益**: 每次 CI 自动捕获 loss 回归

#### 8.2 PP/DP/EP Correctness Parity 测试

- **问题**: 仅 TP 和 CP 有 correctness parity 测试；PP、DP、EP 完全缺失
- **为什么重要**: PP micro-batch 调度、DP gradient averaging、EP expert routing 均可能引入数值差异
- **当前部分实现**: 有 TP parity (`run_tp_output_parity_minified.py`) 和 CP parity (`run_attention_cp.py`) 可作为模板
- **建议设计**: 
  - PP: non-PP baseline vs PP=2 stages，比较 logits KL divergence
  - EP: EP=1 vs EP=2，比较 expert output parity
  - DP: DP=1 (batch×2) vs DP=2 (batch×1)，比较 gradient norm
- **涉及模块**: `tests/functional_tests/`
- **预期收益**: 完整的并行策略 correctness matrix

#### 8.3 FSDP2 路径 NaN/Inf 检测

- **问题**: 仅 MegatronFSDP 有 `check_for_nan_in_grad=True`；标准 FSDP2 路径无任何 NaN guard
- **为什么重要**: bf16 训练中 gradient overflow 不触发 GradScaler 的 skip-step，NaN 无声传播可能在很多步后才被发现
- **当前部分实现**: `MegatronFSDPConfig.check_for_nan_in_grad=True`
- **建议设计**: 在 `_run_train_optim_step` 的 gradient clip 后添加 `if total_norm.isnan(): log.error + skip step`
- **涉及模块**: `recipes/llm/train_ft.py`, `components/training/utils.py`
- **预期收益**: 第一时间发现 gradient explosion，避免无声 NaN 传播

### P1: 强烈建议补齐

#### 8.4 统一 PrecisionConfig 抽象

- **问题**: 精度控制分散在 4 个独立配置对象，无统一验证
- **为什么重要**: 组合式精度错误（如 bf16 param + fp32 reduce + bf16 optimizer moments）难以发现
- **当前部分实现**: `precision_warnings.py` 检测一种组合
- **建议设计**: `PrecisionConfig(storage_dtype, compute_dtype, reduce_dtype, optimizer_dtype)` + `validate()` 检查已知风险组合
- **涉及模块**: `components/distributed/config.py` 或新建 `components/training/precision_config.py`
- **预期收益**: 配置错误在训练前被拦截

#### 8.5 Tensor Dump / Activation Compare 基础设施

- **问题**: 无通用的 tensor dump hook 和 structured diff 工具
- **为什么重要**: 精度问题定位需要逐层对比；当前需要手动添加 hook
- **当前部分实现**: `compare_activations.py` (example 级别)
- **建议设计**: `PrecisionDumpHook(model, output_dir)` — register forward/backward hooks → dump per-layer tensors to disk → `PrecisionDiff(dir_a, dir_b)` → structured report (per-layer cosine sim, max abs diff, dtype)
- **涉及模块**: 新建 `components/training/precision_dump.py`
- **预期收益**: 精度问题定位时间从"天级"降到"分钟级"

#### 8.6 `torch.use_deterministic_algorithms` 支持

- **问题**: 完全缺失 deterministic algorithm enforcement
- **为什么重要**: 某些 CUDA 算子（conv, atomicAdd, scatter）在非 deterministic 模式下有不同实现
- **当前部分实现**: 无
- **建议设计**: 在 config 中添加 `deterministic: true/false` → 调用 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- **涉及模块**: `components/training/rng.py` 或新建 `components/training/determinism.py`
- **预期收益**: 精度对齐时可确保 bit-for-bit 可复现（代价是性能下降）

#### 8.7 Per-TP/PP Rank RNG 隔离

- **问题**: 仅有 `seed + rank` 粗粒度 RNG 区分，无 per-TP-group / per-PP-stage RNG
- **为什么重要**: TP 下 dropout 应在同一 TP group 内一致（否则 all-reduce 对应的 tensor 不同）；PP 下不同 stage 应独立
- **当前部分实现**: `ranked=True` in `init_all_rng`
- **建议设计**: 类似 Megatron-LM 的 `CudaRNGStatesTracker` — `tracker.set_states({'tp': seed_tp, 'pp': seed_pp, 'dp': seed_dp})` → `with tracker.fork('tp'):` context manager
- **涉及模块**: `components/training/rng.py`
- **预期收益**: TP/PP 下 dropout 精确一致

### P2: 长期优化项

#### 8.8 跨硬件精度对齐框架

- **问题**: Golden JSONL 标注 `_h100` 但无 A100/H200 对比
- **建议设计**: CI 在多 GPU 类型上运行 → 比较 loss 曲线 → 标注 hardware-specific 阈值

#### 8.9 自动化精度二分定位

- **问题**: 无 bisect 工具
- **建议设计**: `precision_bisect(model, ref_output, suspect_layers)` → 逐层替换 → 定位首个偏差层

#### 8.10 Checkpoint dtype 全面审计

- **问题**: 仅 `test_hf_consolidated_llm.py` 有 per-tensor dtype assertion
- **建议设计**: 推广 dtype assertion 到所有 checkpoint round-trip 测试

#### 8.11 TF32 显式控制

- **问题**: TF32 在 Ampere+ 默认启用但无显式控制
- **建议设计**: config 中添加 `tf32: true/false` → `torch.backends.cuda.matmul.allow_tf32 = value`

---

## 9. 推荐学习路线

### 第 1 步：读文档（0.5 天）

1. `docs/guides/mixed-precision-training.md` — 理解项目的 mixed precision 设计理念
2. `docs/repository-structure.md` — 理解目录结构
3. `.claude/rules/code-style.md` — 理解代码规范
4. `.claude/rules/distributed.md` — 理解分布式约束
5. `.claude/rules/testing.md` — 理解测试策略

### 第 2 步：跑 examples/tests（1 天）

1. `uv run pytest tests/unit_tests/loss/ -v` — 理解 loss 函数精度测试
2. `uv run pytest tests/unit_tests/training/test_rng.py -v` — 理解 RNG 管理
3. `uv run pytest tests/unit_tests/components/training/test_precision_warnings.py -v` — 理解精度 warning 系统
4. `uv run pytest tests/unit_tests/moe/test_layers.py -v -k gate_precision` — 理解 MoE gate 精度
5. （需 GPU）`torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py` — 跑 TP parity 测试
6. （需 GPU）`torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py` — 跑 CP parity 测试

### 第 3 步：读源码（2 天）

按以下顺序：

1. **RNG 体系**: `components/training/rng.py` (134 行) → 完整理解 StatefulRNG/ScopedRNG
2. **Training step 精度流**: `recipes/llm/train_ft.py` — 重点读 `_forward_backward_step()`(1349-1470) 和 `_run_train_optim_step()`(1472-1570)
3. **Loss 函数精度**: `components/loss/masked_ce.py` (fp32 upcast) → `components/loss/kd_loss.py` (分布式 softmax)
4. **FSDP2 mixed precision**: `components/distributed/config.py` (FSDP2Config) → `components/distributed/parallelizer.py` (mp_policy 传递)
5. **MoE 精度控制**: `components/moe/layers.py` — Gate class 全部
6. **Checkpoint 精度**: `components/checkpoint/checkpointing.py` — `_load_hf_checkpoint_preserving_dtype`, `_load_full_state_dict_into_model`
7. **FP8**: `components/quantization/fp8.py` → FP8Config + apply_fp8_to_model

### 第 4 步：复现 commit/PR 中的问题（1 天）

1. `9b2b06bd` — 手动在 MoE gate 中添加 `.to(scores.dtype)` 后跑 TP parity，观察 KL 变化
2. `ec157f91` — 构造全 -100 label batch，验证 masked_ce 是否返回 NaN
3. `0327814d` — 在 activation checkpointing 下检查 aux_loss dtype
4. `86d8c0fa` — 将 `reduce_dtype` 改回 bf16，跑多步训练观察 loss drift

### 第 5 步：抽象设计模式（0.5 天）

重点提炼：
1. 六阶段 checkpoint robustness 测试模式
2. minified 模型 + KL 散度 parallel parity 测试模式
3. 三层对齐（output + grad + param）correctness 测试模式
4. Golden JSONL 基线机制
5. Gate 精度 fp32 upcast 模式
6. Precision warning 主动检测模式

### 第 6 步：迁移到自己的训练系统（2-3 天）

1. 移植 `StatefulRNG`/`ScopedRNG` → 确保 RNG checkpoint resume
2. 移植 checkpoint robustness 六阶段框架 → 适配自己的 checkpoint 格式
3. 移植 TP/CP parity 测试模式 → 覆盖自己的并行策略
4. 实现 golden JSONL + 自动化断言
5. 添加 `PrecisionConfig` + `validate()`
6. 添加 NaN/Inf 检测 + gradient skip

---

## 10. 对我自研分布式训练系统的迁移建议

### 10.1 第一优先级：直接复用的设计模式

| 模式 | 源码参考 | 迁移难度 | 预期收益 |
|------|---------|----------|---------|
| StatefulRNG/ScopedRNG | `training/rng.py` | 低 (134 行) | 可 checkpoint RNG 状态 |
| Precision warning 系统 | `training/precision_warnings.py` | 低 (100 行) | 主动发现配置风险 |
| 六阶段 checkpoint robustness | `test_checkpoint_robustness_llm.py` | 中 | 系统性 checkpoint 精度验证 |
| TP=1 vs TP=N KL parity | `run_tp_output_parity_minified.py` | 中 | TP correctness 自动化验证 |
| MoE gate fp32 控制 | `moe/layers.py` Gate class | 低 | MoE routing 精度保证 |

### 10.2 第二优先级：需要扩展的能力

| 能力 | NeMo AutoModel 现状 | 需要补充 | 建议方案 |
|------|---------------------|----------|---------|
| Golden loss 自动化 | JSONL 存在无断言 | 加 pytest assert | 50 步 canary test + per-step loss threshold |
| NaN 检测 | 仅 MegatronFSDP 路径 | FSDP2 路径 | gradient clip 后检查 `total_norm.isnan()` |
| Tensor dump | 仅 example 级别 | 通用 hook 工具 | `register_forward_hook` + structured diff |
| PP/EP parity | 完全缺失 | 全新实现 | 复用 TP parity 模式推广 |
| Deterministic mode | 完全缺失 | 配置项 | `torch.use_deterministic_algorithms(True)` |
| Per-TP RNG | 缺失 | CudaRNGStatesTracker | Megatron-LM 参考实现 |

### 10.3 建设路线图

```
Phase 1 (Week 1-2): 基础对齐能力
  ├── 移植 StatefulRNG/ScopedRNG
  ├── 添加 PrecisionConfig + validate()
  ├── 添加 precision_warnings
  └── 添加 NaN/Inf 检测到训练循环

Phase 2 (Week 3-4): 测试基础设施
  ├── 实现 golden loss JSONL + 自动化断言
  ├── 移植 checkpoint robustness 六阶段框架
  ├── 实现 TP parity 测试 (minified model)
  └── 实现 CP/SP parity 测试

Phase 3 (Week 5-6): 诊断工具
  ├── 实现 tensor dump hook 基础设施
  ├── 实现 activation compare 工具
  ├── 添加 torch.use_deterministic_algorithms 支持
  └── 添加 per-TP/PP RNG 隔离

Phase 4 (Week 7-8): 完整覆盖
  ├── PP parity 测试
  ├── EP parity 测试
  ├── DP parity 测试
  ├── 跨硬件精度对齐
  └── 自动化 bisect 工具
```

---

## Appendix A. 检索关键词与命令记录

### 实际执行的 git log 检索关键词

| 关键词 | 匹配 commit 数 | 发现高价值 commit |
|--------|---------------|------------------|
| `precision` | 22 | 是 — MoE router precision, mixed precision |
| `accuracy` | 8 | 部分 — tool-call accuracy (非训练精度) |
| `determin` | 25+ | 是 — checkpoint robustness, golden values |
| `golden` | 2 | 是 — `c59a96cc` golden values 核心 commit |
| `numerical` | 30+ | 是 — rms_norm instability, MoE routing, KD |
| `loss` | 50+ | 是 — NaN guard, dtype mismatch, scaling |
| `gradient` | 30+ | 是 — checkpointing, EP deadlock, clip |
| `bf16` | 30+ | 是 — mixed precision, dtype alignment |
| `fp8` | 30+ | 是 — fp8 checkpoint, dequant, PEFT |
| `tf32` | 0 | 否 — 完全缺失 |
| `checkpoint` | 50+ | 是 — robustness, resume, format conversion |
| `seed` | 15 | 部分 — shuffle_seed, tokenizer |
| `distributed` | 30+ | 是 — parallelizer, EP, TP |
| `tensor parallel` | 5 | 是 — tp plan fixes |
| `pipeline parallel` | 20+ | 是 — PP + TP, dynamic seq, KD |
| `all_reduce` | 6 | 是 — KD loss, EP deadlock |
| `nan` | 20+ | 是 — NaN loss guard, NaN check |
| `overflow` | 1 | 较少 |

### 实际执行的 git grep 检索关键词

| 关键词 | 匹配数 | 关键发现 |
|--------|--------|----------|
| `allclose` | 50+ | checkpoint tests, loss tests, PEFT tests |
| `rtol` | 40+ | CP tests (每个比较函数都有) |
| `atol` | 40+ | CP tests, checkpoint tests |
| `golden` | 3 | 仅数据集文本 (非 golden values 引用) |
| `expected_loss`/`baseline_loss` | 30+ | checkpoint robustness Phase 6 |
| `deterministic` | 30+ | dataset sampler, model comments (非 torch flag) |
| `manual_seed` | 50+ | tests, datasets, RNG module |
| `isnan`/`isinf` | 30+ | flow_matching, LoRA experts, checkpoint |
| `loss_scale`/`GradScaler` | 30+ (MTP, 非 AMP) | MTP loss_scaling_factor (非传统 loss scaling) |
| `mixed_precision` | 0 有效匹配 | 关键词在 config/policy 层而非直接搜索 |

### 实际检查的核心目录

| 目录 | 检查内容 |
|------|---------|
| `tests/unit_tests/loss/` | 5 个 loss 精度测试文件 |
| `tests/unit_tests/training/` | RNG, precision_warnings, timers |
| `tests/unit_tests/moe/` | gate precision, expert forward/backward |
| `tests/unit_tests/_peft/` | LoRA NaN check, allclose |
| `tests/unit_tests/checkpoint/` | dtype map, load path |
| `tests/functional_tests/checkpoint/` | DCP, HF consolidated/sharded, PEFT round-trip |
| `tests/functional_tests/checkpoint_robustness/` | 六阶段 robustness 测试 |
| `tests/functional_tests/context_parallel/` | CP=1 vs CP=2 parity |
| `tests/functional_tests/llm_pretrain_and_kd/` | TP parity, grad norm, TE parallel CE |
| `tests/ci_tests/golden_values/` | 34 组 JSONL |
| `tests/ci_tests/scripts/` | CI launcher, config resolver |
| `nemo_automodel/components/training/` | rng.py, precision_warnings.py, utils.py |
| `nemo_automodel/components/loss/` | 7 个 loss 实现 |
| `nemo_automodel/components/moe/` | layers.py, experts.py, parallelizer.py |
| `nemo_automodel/components/distributed/` | config.py, parallelizer.py, grad_utils.py |
| `nemo_automodel/components/checkpoint/` | checkpointing.py, stateful_wrappers.py |
| `nemo_automodel/components/quantization/` | fp8.py |
| `nemo_automodel/recipes/llm/` | train_ft.py |
| `examples/convergence/tulu3/` | compare_activations.py |
| `.github/workflows/` | cicd-main.yml 等 28 个 workflow 文件 |

---

## Appendix B. 关键文件清单

### 精度控制核心文件

| 文件 | 行数(约) | 精度相关性 |
|------|---------|-----------|
| `components/training/rng.py` | 134 | RNG 管理 (StatefulRNG, ScopedRNG) |
| `components/training/precision_warnings.py` | 100 | bf16+AdamW 精度风险检测 |
| `components/training/utils.py` | 230 | gradient clip, error_if_nonfinite |
| `components/distributed/config.py` | 200 | FSDP2Config, MixedPrecisionPolicy |
| `components/distributed/grad_utils.py` | 120 | gradient norm 分布式计算 |
| `components/loss/masked_ce.py` | 110 | fp32 upcast, 零 token guard |
| `components/loss/kd_loss.py` | 240 | 分布式 KL, inf guard |
| `components/moe/layers.py` | 600 | Gate precision, aux_loss fp32, bias master weight |
| `components/quantization/fp8.py` | 260 | FP8 config, conversion, verification |
| `components/checkpoint/checkpointing.py` | 1100 | dtype preservation, format conversion |
| `recipes/llm/train_ft.py` | 1700 | training step, mixed precision flow |

### 精度测试核心文件

| 文件 | 精度测试类型 |
|------|-------------|
| `functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py` | 六阶段 checkpoint KL + resume loss |
| `functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py` | TP=1 vs TP=2 KL divergence |
| `functional_tests/context_parallel/run_attention_cp.py` | CP=1 vs CP=2 三层对齐 |
| `functional_tests/llm_pretrain_and_kd/run_clip_grad_norm_correctness.py` | EP+FSDP2 gradient norm 正确性 |
| `functional_tests/llm_pretrain_and_kd/loss/run_te_parallel_ce_dtensor.py` | TE parallel CE vs reference |
| `unit_tests/loss/test_masked_ce.py` | loss fp32 upcast, 零 token guard |
| `unit_tests/loss/test_chunked_ce.py` | chunked CE vs standard CE |
| `unit_tests/training/test_rng.py` | RNG reproducibility |
| `unit_tests/components/training/test_precision_warnings.py` | precision warning 检测 |
| `unit_tests/moe/test_layers.py` | gate precision, aux_loss dtype |
| `examples/convergence/tulu3/model-verification/compare_activations.py` | 层级式激活比较 |

---

## Appendix C. 关键 commits / PR / issues 清单

| Commit | 日期 | PR | 类别 | 标题 | 对精度对齐的意义 |
|--------|------|-----|------|------|-----------------|
| `917b75e1` | 2025-11-06 | #767 | 精度控制 | configurable router precision for MoEs | MoE gate fp32 精度基础设施 |
| `6ff3cdc7` | — | #298 | 配置 | Simplify mixed_precision logic | FSDP2 mixed precision 配置统一 |
| `cf4a2942` | 2026-01-28 | #1110 | FP8 | fp8 dequant kernels for DSv3 | FP8 checkpoint 精度转换 |
| `15c3364e` | 2026-02-24 | #1365 | 正确性 | EP collective deadlock fix | 可变长度 token EP 正确性 |
| `2952aa6f` | 2026-02-26 | #1392 | 精度控制 | MoE mixed precision policy configurable | MoE 并行精度策略可配置 |
| `0adca682` | 2026-03-02 | #1412 | 梯度正确性 | correct MoE aux_loss gradient scaling | MoE 辅助损失梯度修复 |
| `9c3f9439` | 2026-03-06 | #1410 | 数值稳定性 | rms_norm numerical instability | bf16 归一化层稳定性 |
| `fb47c5d0` | 2026-03-05 | #1459 | FP8 | native fp8 checkpoint + peft | FP8+PEFT checkpoint 兼容 |
| `d509e78b` | 2026-03-26 | #1554 | 收敛验证 | Tulu-3 E2E convergence pipeline | 端到端收敛 + 激活比较工具 |
| `c59a96cc` | 2026-04-02 | #1647 | golden values | Add recipe golden values | **34 组 golden loss 基线** |
| `6a1274d8` | 2026-04-06 | #1606 | 测试框架 | checkpoint robustness tests | **六阶段 checkpoint robustness 框架** |
| `8f0f38f1` | 2026-04-21 | #1913 | dtype 对齐 | EP expert weight dtype alignment | FSDP2 跨 mesh dtype 修复 |
| `ec157f91` | 2026-04-23 | #1985 | NaN 防护 | guard NaN loss from zero labels | **bug→fix→test 闭环案例** |
| `da6061bb` | 2026-04-23 | #2012 | 梯度正确性 | gradient clip with torch_mm + EP | torch.compile + EP clip 修复 |
| `0327814d` | 2026-05-06 | #2083 | dtype 漂移 | MoE aux-loss dtype under AC | AC recompute dtype 不一致 |
| `8cb1f5ce` | 2026-05-08 | #2173 | 数值溢出 | SwiGLU clamp in fp32 | DSv4 shared expert fp32 clamp |
| `9b2b06bd` | 2026-05-10 | #2201 | 精度丢失 | MoE routing + softmax in fp32 | **经典 fp32→bf16 cast 精度陷阱** |
| `0fb6535c` | 2026-05-14 | #2216 | 精度保持 | preserve reference fp32 parameters | DSv4 fp32 参数 FSDP2 下保持 |
| `86d8c0fa` | 2026-05-28 | #2248 | 精度默认值 | mixed-precision optimizer-state setup | **reduce_dtype→fp32 + precision warning** |

---

*本报告基于 NeMo AutoModel 仓库 commit `0ab8634f` (2026-06-02 分支 `source_code_analysis`) 编写。所有源码证据均经由 git grep、git log、文件读取和子代理代码审查交叉验证。标记为"未发现"的能力项均经过至少 3 个关键词的全仓库检索确认。*
