# upstream/main 精度对齐与调试能力分析

生成日期：2026-06-02

分析对象：

- 上游仓库：`https://github.com/NVIDIA-NeMo/Automodel`
- 上游分支：`upstream/main`
- 上游提交：`ef4444cf79e7dd2935e88f5280d56d42e09b613e`
- 本地工作区：`/root/Automodel/.worktrees/source_code_analysis`

本报告满足两个问题：

1. 基于 `gh` CLI 分析 `upstream/main` 如何实现 feature 或修复 debug/精度对齐问题。
2. 深入盘点项目源码里已经存在的、可用于实现 feature 或 debug 精度/准确率对齐的模块、脚本、测试与缺口。

## 方法与边界

本次使用四个子代理并等待全部完成后再综合：

- Explore：用 `gh pr view`、`gh pr list/search`、`git log`、`git show` 分析上游 PR/commit。
- architect：分析训练、分布式、loss、checkpoint、quantization、logging 的源码架构。
- code-reviewer：只读审查精度对齐相关风险点。
- analyst：全仓盘点可运行脚本、测试、文档和配置。

本地补充校验：

```bash
git fetch upstream main --prune
git rev-parse upstream/main
git log --oneline --decorate -20 upstream/main
gh pr view 2248 -R NVIDIA-NeMo/Automodel --json number,title,url,mergeCommit,files,body
gh pr view 2201 -R NVIDIA-NeMo/Automodel --json number,title,url,mergeCommit,files,body
gh pr view 2338 -R NVIDIA-NeMo/Automodel --json number,title,url,mergeCommit,files,body
gh pr view 2314 -R NVIDIA-NeMo/Automodel --json number,title,url,mergeCommit,files,body
```

注意：当前工作区分支 `source_code_analysis` 与 `upstream/main@ef4444cf` 在若干源码、示例、测试和 specs 文件上存在差异。因此，上游 PR/commit 结论以上游提交为准；源码模块盘点则说明当前项目中能看到和使用的实现面。两者不混为同一个状态。

## 总体结论

`upstream/main` 近期围绕精度/准确率对齐形成了四类主线：

1. 混合精度合同显式化：把 resident parameter dtype、FSDP2 compute/reduce/output dtype、optimizer state dtype 分清楚，避免 bf16 AdamW state 和 bf16 reduction 引起长期 loss/grad drift。
2. reference-sensitive fp32 路径保留：DeepSeek V4、MoE routing、attention softmax、FSDP2 mixed dtype shard 等路径持续修复，目标是让关键 gate、softmax、HCA/compressor/indexer 等路径与参考实现保持 fp32 语义。
3. token/loss 对齐修复：左 padding loss mask、zero label token、KD distributed reduction、packed sequence boundary、validation loss weighted averaging 等问题都直接影响 loss 与任务准确率对齐。
4. 从 loss 扩展到任务准确率：tool-call evaluator 与 offline checkpoint eval 补上了 loss 以外的功能正确性信号，避免 val loss 降低但工具调用名称或 JSON 参数错误。

源码层面已经存在大量可复用能力：TP/CP/logits parity、HF vs NeMo activation comparison、KD/TE CE reference tests、checkpoint robustness、RNG/dataloader state、golden JSONL、FP8/mixed precision 文档和 warning。但它们比较分散，缺少一个 recipe 级统一的 `alignment_debug` 开关，把 data/model/loss/grad/checkpoint/quantization 的 fingerprint 串成同一次实验里的 rank-aware 调试轨迹。

## 上游 PR/commit 脉络

### 1. 混合精度与 optimizer state

#### PR #2248: mixed-precision optimizer-state setup

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2248
- Merge commit: `86d8c0fab263ad58a8c6b0627b37662a9f7f35a8`
- Title: `fix(training): clarify mixed-precision optimizer-state setup`
- 主要路径：
  - `nemo_automodel/components/distributed/config.py`
  - `nemo_automodel/components/training/precision_warnings.py`
  - `docs/guides/mixed-precision-training.md`
  - `nemo_automodel/recipes/llm/train_ft.py`
  - `nemo_automodel/recipes/vlm/finetune.py`
  - 多个 LLM/diffusion pretrain YAML

实现要点：

- FSDP2 默认 `reduce_dtype` 由 bf16 路径调整为更保守的 fp32 reduction，降低大 DP world size 下梯度通信舍入误差。
- 文档明确区分：
  - `model.torch_dtype`：resident parameter dtype 与 checkpoint dtype。
  - FSDP2 mixed precision policy：forward/backward compute、reduce、output dtype。
  - optimizer `_target_`：optimizer state 的 dtype 行为。
- 新增 `precision_warnings.py`，在 full-parameter training 中发现 torch `Adam/AdamW` 搭配 trainable bf16 参数时发出 warning。
- Recipe 层接入 warning，使 LLM、VLM、retrieval、diffusion 等入口能暴露这个风险。

为什么对精度对齐重要：

- `torch.optim.AdamW` 通常从参数 dtype 初始化 EMA state。如果 resident params 是 bf16，`exp_avg` / `exp_avg_sq` 也可能落在 bf16。长训时这会表现为 loss、grad norm 或最终准确率漂移，而不是单步立即失败。
- 把 fp32 optimizer state 与 bf16 compute 分开，是做 loss curve 和 checkpoint parity 的基础。

局限：

- PR 记录中说明 DeepSeek 路径仍受 DeepEP dispatch 问题限制，相关路径未达到同等验证深度。

#### PR #2271: fully_shard_by_dtype param dtype

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2271
- Merge commit: `b05fd4ef`
- Title: `fix: adjust param_dtype during fully_shard_by_dtype traversal`
- 主要路径：
  - `nemo_automodel/components/distributed/parallelizer_utils.py`
  - `tests/functional_tests/llm_pretrain_and_kd/run_fully_shard_by_dtype_param_dtype.py`

实现要点：

- FSDP2 `fully_shard_by_dtype` 遍历模型子树时，根据子树真实参数 dtype 复制/调整 `MixedPrecisionPolicy.param_dtype`。
- 避免某些 fp32 子模块在 FSDP2 包装时被错误 cast 到 bf16。

为什么对精度对齐重要：

- 对 reference-sensitive 参数，错误 cast 不一定产生 shape 或 runtime error，但会造成 logits、routing、attention 或 MTP parity 漂移。
- 这个修复让“保留 fp32 参数”的模型级意图能穿过 FSDP2 sharding 层。

### 2. DeepSeek V4 / MoE reference fp32 对齐

#### PR #2201: MoE routing scores and attention softmax in fp32

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2201
- Merge commit: `9b2b06bd9aad84ec9d5013db03af712b8c965f44`
- Title: `fix(deepseek-v4): keep MoE routing scores and attention softmax in fp32`
- 主要路径：
  - `nemo_automodel/components/models/deepseek_v4/layers.py`
  - `nemo_automodel/components/moe/layers.py`

实现要点：

- MoE `sqrtsoftplus` gate routing score 在 fp32 中完成，不再过早 cast 回 bf16。
- DeepSeek V4 attention sink softmax 强制使用 fp32 accumulator。

为什么对精度对齐重要：

- MoE expert top-k 对接近分数极敏感；bf16 舍入可能选错 expert。
- attention softmax 的小差异会跨 61 层累积，最终表现为 backbone parity 下降。
- PR 背景明确来自 MTP parity 测试，属于直接面向 reference parity 的数值修复。

#### PR #2216: preserve reference fp32 parameters

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2216
- Merge commit: `0fb6535c`
- Title: `fix(dsv4): preserve reference fp32 parameters`
- 主要路径：
  - `nemo_automodel/components/models/deepseek_v4/*`
  - `nemo_automodel/components/models/common/utils.py`
  - `nemo_automodel/components/moe/parallelizer.py`

实现要点：

- 对 DSV4 中 HCA、compressor、indexer、attention sink 等参考实现要求 fp32 的参数进行保留。
- 修复相关 state-dict key mapping，使 checkpoint 转换不会破坏这些参数语义。

为什么对精度对齐重要：

- 这类参数不是普通可降精度权重；它们参与索引、压缩或注意力偏置路径，错误降到 bf16 会导致结构性偏差。

#### PR #2173: shared-expert SwiGLU clamp in fp32

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2173
- Merge commit: `8cb1f5ce`
- Title: `fix(dsv4/moe): clamp shared-expert SwiGLU in fp32`
- 主要路径：
  - `nemo_automodel/components/moe/layers.py`

实现要点：

- shared expert 的 SwiGLU clamp 在 fp32 中完成，然后再按需要 cast 回原 dtype。

为什么对精度对齐重要：

- clamp 是非线性门控操作，bf16 路径容易在临界值附近产生不同激活模式。

#### PR #2083: MoE aux-loss dtype under activation checkpointing

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2083
- Merge commit: `0327814d`
- Title: `fix: MoE aux-loss dtype mismatch under activation checkpointing`
- 主要路径：
  - `nemo_automodel/components/moe/layers.py`
  - `nemo_automodel/components/moe/megatron/moe_utils.py`
  - `tests/unit_tests/moe/test_layers.py`

实现要点：

- MoE aux loss 的保存/重算张量固定为 fp32。
- 避免 activation checkpointing 的 forward 与 recompute 阶段 dtype metadata 不一致。

为什么对精度对齐重要：

- activation checkpointing 会重跑 forward。若保存路径和重算路径 dtype 不一致，loss 和梯度可能在开启 checkpointing 后漂移。

#### PR #2277: HCA backward graph under FSDP2

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2277
- Merge commit: `c17d56cc`
- Title: `fix(deepseek-v4): align HCA backward graph under FSDP2`
- 主要路径：
  - `nemo_automodel/components/models/deepseek_v4/fsdp.py`
  - `nemo_automodel/components/models/deepseek_v4/layers.py`

实现要点：

- 当不同 rank 上长短序列混合时，为短 rank 添加 fully masked synthetic HCA window。
- 保证同步同一组 HCA 参数的 ranks 具有一致 backward graph，避免 FSDP2 collective mismatch。

为什么对精度对齐重要：

- 这不是简单数值容差问题，而是分布式图结构一致性问题。图结构不一致时，精度对齐无法进入可比较阶段。

局限：

- PR 声明覆盖的是 1D PyTorch FSDP2 mesh；多维 HSDP/CP/TP/PP/EP 组合需要进一步验证。

### 3. Loss、mask、KD 与数据路径对齐

#### PR #2204: KD validation loss averaging

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2204
- Merge commit: `adc20e23`
- Title: `fix(recipes): correct validation loss averaging in LLM KD recipe`
- 主要路径：
  - `nemo_automodel/recipes/llm/kd.py`

实现要点：

- 修正 validation loss 的重复除法问题。
- 按 `num_label_tokens` 对 CE/KD loss 加权汇总。

为什么对精度对齐重要：

- 验证指标不正确会掩盖训练路径错误。loss 对齐首先要保证统计口径一致。

#### PR #2212 / #2215: VLM KD distributed step and correctness tests

- URLs:
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2212
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2215
- Merge commits:
  - `98719c4f`
  - `bb2fe709`
- 主要路径：
  - `nemo_automodel/recipes/vlm/kd.py`
  - `nemo_automodel/components/loss/kd_loss.py`
  - `tests/unit_tests/recipes/test_vlm_kd_*`

实现要点：

- KD TP all-reduce 改为 autograd-safe distributed all-reduce。
- 覆盖 TP/CP correctness。
- 对 CP pre-embed teacher/student hidden-size 做校验。

为什么对精度对齐重要：

- KD loss 对 teacher/student logits 与 hidden states 对齐极敏感；错误 reduction 会导致 loss 看似可跑但梯度不正确。

#### PR #2314: left-padding reasoning and assistant loss masks

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2314
- Merge commit: `447ae569438160b86964fca9da6add8c2399045e`
- Title: `fix(datasets): align reasoning + assistant loss masks for left padding`
- 主要路径：
  - `nemo_automodel/components/datasets/llm/formatting_utils.py`
  - `tests/unit_tests/datasets/llm/test_shift_mask_left_padding.py`

实现要点：

- 抽出 `_maybe_shift_mask_for_left_padding` helper。
- 对 assistant mask 和 reasoning mask 都应用 left-padding shift。
- 单测覆盖 right-padding no-op、left-padding shift、zero pad、missing attention mask、all-padding 等边界。

为什么对精度对齐重要：

- 对 left-padding tokenizer，未 padding 的文本 span 与 padded `input_ids` 的实际内容位置不同。mask 不右移会落在 padding 区域，再被 attention mask 清零，导致样本出现全零 loss mask。
- 这是典型“loss 下降异常/学习不到目标 token”的根因。

#### PR #1985 / #2259: zero label tokens and NaN guard

- URLs:
  - https://github.com/NVIDIA-NeMo/Automodel/pull/1985
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2259
- Merge commits:
  - `5876c8aa`
  - `46b74fbb`
- 主要路径：
  - `nemo_automodel/components/loss/masked_ce.py`
  - `nemo_automodel/recipes/vlm/finetune.py`
  - `nemo_automodel/components/speculative/eagle/core.py`

实现要点：

- VLM 空监督 batch 不再产生 NaN loss。
- EAGLE3 对非法 `ttt_steps` 提前配置报错，而不是后续产生 NaN。

为什么对精度对齐重要：

- NaN guard 能把“不可比较”变为“明确边界行为”，对 debug 很关键。
- 同时也暴露当前仅部分 loss 实现有 zero-token guard，后续需要统一。

#### PR #2147: packed-sample boundaries in GatedDeltaNet

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2147
- Merge commit: `c4a4c832`
- Title: `fix(qwen3_5): preserve packed-sample boundaries in GatedDeltaNet`
- 主要路径：
  - `nemo_automodel/components/models/common/packing.py`
  - `nemo_automodel/components/models/qwen3_5/decoder_layer.py`
  - `nemo_automodel/components/models/qwen3_5_moe/cp_linear_attn.py`

实现要点：

- 向 linear attention 传递 `cu_seqlens`、`seq_idx`、`indices`。
- 阻止 recurrent state 跨 packed samples 泄漏。

为什么对精度对齐重要：

- sequence packing 的样本边界如果泄漏，模型输出不是容差问题，而是语义错误。PR 记录的小复现中 max diff 从显著非零降到零，说明它是直接的 parity 修复。

### 4. Checkpoint / resume / HF compatibility

#### PR #2285 / #2310 / #2319: EAGLE checkpoint-resume 修复串

- URLs:
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2285
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2310
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2319
- Merge commits:
  - `f6e7c325`
  - `8f0da347`
  - `3d7ee461`
- 主要路径：
  - `nemo_automodel/recipes/llm/train_eagle1.py`
  - `nemo_automodel/recipes/llm/train_eagle3.py`
  - `nemo_automodel/components/checkpoint/utils.py`
  - EAGLE checkpoint 单测

实现要点：

- 恢复 optimizer、scheduler、RNG、epoch、vocab mapping。
- 补齐 `model_state_dict_keys`。
- 修复 vocab shrink 后 `lm_head` shape mismatch。

为什么对精度对齐重要：

- resume parity 不只需要模型权重，还需要 optimizer state、RNG、epoch/dataloader state 与 vocab mapping 一致。

#### PR #2096 / #2120 / #2268: HF/PEFT/VLM-MoE checkpoint compatibility

- URLs:
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2096
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2120
  - https://github.com/NVIDIA-NeMo/Automodel/pull/2268
- Merge commits:
  - `a8df0137`
  - `a321dcf0`
  - `bc375932`
- 主要路径：
  - `nemo_automodel/components/checkpoint/checkpointing.py`
  - `nemo_automodel/components/checkpoint/stateful_wrappers.py`
  - `nemo_automodel/components/checkpoint/utils.py`
  - `nemo_automodel/components/models/qwen3_vl_moe/state_dict_adapter.py`

实现要点：

- PP/EP consolidated safetensors 不丢全局 key。
- QLoRA adapter prefix 与 HF 格式兼容。
- Qwen3-VL-MoE 的 `lm_head` 不被错误 tied 或重命名。

为什么对精度对齐重要：

- checkpoint 转换路径的 key 丢失、prefix 错误或 tied weight 错误，会直接破坏 reload logits parity。

### 5. 任务准确率与 agent SFT eval

#### PR #2338: tool-call accuracy evaluator

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2338
- Merge commit: `07610c118ac409f586ede5b3ee4e6d04a40ab51e`
- Title: `feat(eval): add tool-call accuracy evaluator for agent SFT validation`
- 主要路径：
  - `nemo_automodel/components/eval/tool_call_parser.py`
  - `nemo_automodel/components/eval/tool_call_evaluator.py`
  - `nemo_automodel/components/datasets/llm/agent_chat.py`
  - `nemo_automodel/recipes/llm/train_ft.py`
  - `tests/unit_tests/eval/test_tool_call_*`

实现要点：

- 新增 permissive tool-call parser，覆盖 Qwen、Hermes、Llama 3.1、Mistral、GPT-OSS Harmony 等格式。
- 新增 `ToolCallAccuracyEvaluator`，对 eval samples 执行 generation、parse、metric aggregation。
- 将工具调用准确率指标接入现有 logger。
- DP group 下用 all-reduce 汇总 mean 与 count，确保分片后的 corpus metric 正确。
- 当模型没有 `.generate()` 时，回退到 manual greedy decode，适配 custom model/FSDP 场景。

为什么对精度/准确率对齐重要：

- loss-only validation 不能发现“格式看起来学到了，但工具名或 JSON 参数错误”的失败模式。
- 这个 feature 把准确率调试从 token loss 扩展到任务行为层。

#### PR #2368: offline tool-call eval for checkpoints

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2368
- Merge commit: `ec5de8bf`
- 主要路径：
  - `examples/llm_finetune/agent/evaluate_tool_call.py`

实现要点：

- 增加离线 checkpoint eval 示例，避免训练内 FSDP2 generation 的 OOM 或复杂性。

为什么对准确率对齐重要：

- 能把训练 checkpoint 与离线评估剥离，方便稳定复现某个 checkpoint 的任务行为。

#### PR #2367: agent SFT data path end-to-end test

- URL: https://github.com/NVIDIA-NeMo/Automodel/pull/2367
- Merge commit: `be5118cc`
- Title: `test(datasets): cover agent SFT recipe data path end-to-end`

价值：

- 覆盖 JSON/ShareGPT 到 dataset、collator、answer-only loss mask 的端到端数据路径。
- 对防止 agent SFT mask/format 回归有直接价值。

## 源码架构：数值 mismatch 进入路径

### Recipe 编排层

主要入口：

- `nemo_automodel/recipes/llm/train_ft.py`
- `nemo_automodel/recipes/vlm/finetune.py`
- `nemo_automodel/recipes/llm/kd.py`
- `nemo_automodel/recipes/base_recipe.py`

LLM 训练由 recipe 统一装配：

1. 解析 YAML/config。
2. 初始化 distributed mesh/strategy。
3. 构建 model/tokenizer/dataset/dataloader。
4. 应用 checkpoint load、PEFT、FP8/QAT、TE attention、FSDP2/MegatronFSDP、TP/CP/PP/EP、compile。
5. 构建 loss、optimizer、scheduler、logger。
6. 在 forward/backward/optimizer step 中处理 token normalization、grad scaling/clipping、metric logging 和 checkpoint save。

对齐含义：

- 如果要实现统一 `alignment_debug`，recipe 是正确层级。组件之间有 import 边界，跨 data/model/loss/checkpoint 的组合逻辑不应塞进单个 component。

### 数据与 mask 层

主要路径：

- `nemo_automodel/components/datasets/utils.py`
- `nemo_automodel/components/datasets/llm/formatting_utils.py`
- `nemo_automodel/components/datasets/llm/packed_sequence.py`
- `nemo_automodel/components/datasets/llm/agent_chat.py`
- `nemo_automodel/components/datasets/vlm/collate_fns.py`
- `nemo_automodel/components/datasets/vlm/datasets.py`

高风险点：

- `labels == -100` 的构造和 shift。
- answer-only loss、reasoning mask、assistant mask。
- left padding 和 attention mask 的位置对齐。
- packed sequence 的 `seq_lens`、`cu_seqlens`、sample boundary。
- VLM overlong/error retry 的替换样本。
- CP 改写前后的 labels、position ids、attention mask。

建议 debug 输出：

- 每 rank 的有效 label token 数。
- 全 mask batch 数量。
- padding side、pad length、attention mask sum。
- `position_ids` min/max。
- packed sequence boundary 摘要。
- VLM fallback/retry 是否发生。

### Model / distributed infrastructure 层

主要路径：

- `nemo_automodel/_transformers/infrastructure.py`
- `nemo_automodel/components/distributed/config.py`
- `nemo_automodel/components/distributed/mesh.py`
- `nemo_automodel/components/distributed/parallelizer.py`
- `nemo_automodel/components/distributed/parallelizer_utils.py`
- `nemo_automodel/components/distributed/cp_utils.py`
- `nemo_automodel/components/distributed/grad_utils.py`

高风险点：

- checkpoint 是 sharding 前加载还是 sharding 后加载。
- `MixedPrecisionPolicy` 的 param/reduce/output dtype。
- TP plan、sequence parallel、context parallel、pipeline parallel 的 batch 和 tensor layout 改写。
- CP 对 attention mask、position ids、labels 的 padding 和 sharding。
- PP 最后 stage loss 与 token count 的同步。
- EP/MoE aux loss scaling。

建议 debug 输出：

- mesh shape 与每个 rank 的 coordinate。
- FSDP2 policy 实际 dtype。
- 关键参数 dtype histogram。
- selected parameters 的 norm/hash/fingerprint。
- backward 后、grad scaling 前后、clip 前后的 selected grad norm。
- CP/TP/PP/EP 相关的 per-rank token count 与 tensor local shape。

### Loss 层

主要路径：

- `nemo_automodel/components/loss/utils.py`
- `nemo_automodel/components/loss/masked_ce.py`
- `nemo_automodel/components/loss/chunked_ce.py`
- `nemo_automodel/components/loss/linear_ce.py`
- `nemo_automodel/components/loss/te_parallel_ce.py`
- `nemo_automodel/components/loss/kd_loss.py`
- `nemo_automodel/components/loss/soft_ce.py`
- `nemo_automodel/components/loss/mtp.py`

已有能力：

- KD loss 在 TP vocab-shard DTensor 路径上使用 distributed softmax，并做 fp32 upcast。
- TE parallel CE 有 DTensor correctness 功能测试。
- 多种 CE 实现有单测或 reference comparison。

风险点：

- `MaskedCE` 有 zero-token guard，但 `ChunkedCE`、`FusedLinearCE`、`TEParallelCE` 的 zero-valid-token 行为不够统一。
- 部分 CE 会在传入 `loss_mask` 时原地修改 labels，调试脚本复用 labels 比较多个 loss 时可能互相污染。
- PP/non-PP 路径对 raw sum loss、normalized loss、token count 的统计口径需要统一暴露。

建议 debug 输出：

- loss class 名称。
- raw sum loss。
- normalized loss。
- `num_label_tokens` 本地值与全局 all-reduced 值。
- ignored token 数。
- PP/TP/CP flags。

### Checkpoint / resume 层

主要路径：

- `nemo_automodel/components/checkpoint/checkpointing.py`
- `nemo_automodel/components/checkpoint/stateful_wrappers.py`
- `nemo_automodel/components/checkpoint/_backports/hf_storage.py`
- `nemo_automodel/components/checkpoint/utils.py`
- `docs/guides/checkpointing.md`
- `specs/009-distributed-checkpoint-safetensors-analysis/README.md`

已有能力：

- DCP/SafeTensors/HF format load-save。
- consolidated / sharded checkpoint。
- state dict adapter。
- model/optimizer/scheduler/RNG/dataloader state。
- checkpoint robustness tests 比较 logits KL、cross-TP reload KL、resume loss。

风险点：

- missing/unexpected checkpoint keys 在 adapter/PP load 路径中可能只 warning 不 fail-fast。
- `_load_full_state_dict_into_model` 的 dtype contract 需要更明确：注释与实际是否保持 checkpoint dtype 要用测试锁住。
- 缺少统一 load/save tensor fingerprint。

建议 debug 输出：

- save/load 前后的 selected tensor fingerprint。
- dtype histogram。
- missing/unexpected keys 分类。
- adapter key conversion 摘要。
- optimizer state dtype histogram。
- RNG/dataloader state 恢复摘要。

### Quantization / FP8 / QAT 层

主要路径：

- `nemo_automodel/components/quantization/fp8.py`
- `nemo_automodel/components/quantization/qat.py`
- `nemo_automodel/components/quantization/qlora.py`
- `docs/guides/fp8-training.md`
- `tests/unit_tests/quantization/test_fp8.py`

已有能力：

- FP8Config 支持 tensorwise/rowwise recipe。
- 验证 Float8Linear conversion 覆盖。
- 检查硬件与 tensor dim divisibility。
- QAT delayed fake quant。
- QLoRA quantized module 检查。

风险点：

- FP8 conversion 失败后当前逻辑可能 warning 并返回原模型；训练配置声称 FP8 但实际没有 FP8，会污染吞吐和精度对齐实验。
- QAT delayed fake-quant toggle 失败只 warning，实际启用 step 可能与配置不一致。

建议 debug 输出：

- FP8 conversion 成功/失败。
- 未转换模块及原因。
- Float8Linear count。
- scale/amax 摘要。
- QAT fake quant 开启 step 与状态。
- QLoRA module coverage 与 compute dtype。

## 已有脚本、测试和文档清单

| 类别 | 路径 | 用途 | 对齐价值 |
| --- | --- | --- | --- |
| 混合精度 | `nemo_automodel/components/training/precision_warnings.py` | 检测 bf16 params + torch Adam/AdamW | 发现 optimizer state dtype 漂移风险 |
| 混合精度 | `tests/unit_tests/components/training/test_precision_warnings.py` | precision warning 单测 | 锁定 warning 触发/跳过规则 |
| 混合精度 | `docs/guides/mixed-precision-training.md` | dtype 合同文档 | 明确 storage/compute/reduce/optimizer state |
| FP8 | `nemo_automodel/components/quantization/fp8.py` | TorchAO FP8 conversion | 支持 FP8 训练路径和覆盖验证 |
| FP8 | `tests/unit_tests/quantization/test_fp8.py` | FP8 config/conversion 单测 | 检查硬件、shape、missing torchao 等边界 |
| FP8 | `docs/guides/fp8-training.md` | FP8 使用说明 | 说明 H100+、compile、padding、BF16 parity 预期 |
| FP8 examples | `examples/llm_finetune/*/*_fp8.yaml` | FP8 recipe | 可做 FP8 vs BF16 小模型实验 |
| TP logits parity | `tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py` | TP=2 vs TP=1 logits KL | 验证 TP plan 和 sequence parallel 对 logits 的影响 |
| TP logits parity | `tests/functional_tests/llm_pretrain_and_kd/L2_TP_Output_Parity_Minified.sh` | 启动脚本 | 默认 KL threshold，2 GPU 运行 |
| HF vs NeMo activation | `examples/convergence/tulu3/model-verification/compare_activations.py` | 比较 hidden states/logits | 定位从哪一层开始漂移 |
| HF vs NeMo activation | `examples/convergence/tulu3/model-verification/run.sh` | 抽取/比较一键入口 | 支持 gate/lm head precision 和 threshold |
| parity 指南 | `.agents/contributor-skills/parity-testing/SKILL.md` | parity debug 流程 | state dict、component、E2E logits 三层策略 |
| parity pitfalls | `.agents/contributor-skills/parity-testing/pitfalls.md` | 常见失配清单 | QKV/GateUp、RoPE、TE/SDPA、tied weights |
| 模型 parity | `tests/unit_tests/models/nemotron_v3/test_nemotron_v3_mtp_parity.py` | MTP parity | CPU fp32 state-dict roundtrip 与 logits/loss |
| 模型 parity | `tests/unit_tests/models/ling_v2/parity_ling_v2.py` | Ling-mini 手动 parity | state dict、RoPE/gate、GPU bf16 logits |
| KD loss | `nemo_automodel/components/loss/kd_loss.py` | KD forward KL | TP shard distributed softmax，fp32 upcast |
| KD loss test | `tests/unit_tests/loss/test_kd_loss.py` | KD reference 单测 | `allclose` 对比 reference |
| TE CE | `nemo_automodel/components/loss/te_parallel_ce.py` | TE parallel CE wrapper | DTensor logits、TP group、fp32 loss/grad |
| TE CE test | `tests/functional_tests/llm_pretrain_and_kd/loss/run_te_parallel_ce_dtensor.py` | CE correctness | 全量 logits CE vs TE parallel CE |
| grad | `nemo_automodel/components/distributed/grad_utils.py` | grad norm/clip | DTensor local grad 与 group all-reduce |
| grad test | `tests/unit_tests/distributed/test_grad_utils.py` | grad utils 单测 | 缩放、None grad、L2/inf norm |
| grad functional | `tests/functional_tests/llm_pretrain_and_kd/run_clip_grad_norm_correctness.py` | distributed grad correctness | seeded DTensor grads vs full tensor reference |
| CP parity | `tests/functional_tests/context_parallel/run_attention_cp.py` | CP attention parity | output/input grad/param grad 对齐 |
| CP parity | `tests/functional_tests/context_parallel/run_mamba_cp.py` | Mamba CP parity | output 和梯度对齐 |
| CP parity | `tests/functional_tests/context_parallel/run_hybrid_nemotron_v3_cp.py` | Hybrid CP parity | attention + Mamba 场景 |
| CP parity | `tests/functional_tests/context_parallel/run_qwen3_5_moe_linear_attn_cp.py` | MoE linear attention CP parity | CP=2 vs CP=1 forward/grad |
| checkpoint | `tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py` | checkpoint robustness | train-save-reload/resume logits KL |
| checkpoint | `tests/functional_tests/checkpoint/test_flashoptim_dcp_roundtrip.py` | FlashOptim DCP roundtrip | full vs resumed training trace |
| checkpoint docs | `docs/guides/checkpointing.md` | checkpoint guide | DCP/SafeTensors/resharding/RNG |
| RNG | `nemo_automodel/components/training/rng.py` | StatefulRNG/ScopedRNG | checkpointable RNG state |
| recipe RNG | `nemo_automodel/recipes/base_recipe.py` | dataloader/RNG state wiring | resume parity 基础 |
| golden traces | `tests/ci_tests/golden_values/**/*.jsonl` | training trace baseline | loss、grad_norm、lr、tokens、MFU |
| MoE metrics | `nemo_automodel/components/moe/load_balance_metrics.py` | routing/load balance | 定位 MoE routing 漂移 |
| retrieval accuracy | `tests/functional_tests/retrieval/compare_cross_encoder_models.py` | task accuracy check | baseline vs finetuned ranking accuracy |
| tool-call eval | `nemo_automodel/components/eval/tool_call_evaluator.py` | generation-based evaluator | loss 外的任务准确率指标 |
| tool-call parser | `nemo_automodel/components/eval/tool_call_parser.py` | tool-call 解析 | 名称、JSON、参数匹配 |
| offline tool eval | `examples/llm_finetune/agent/evaluate_tool_call.py` | checkpoint offline eval | 避开训练内 FSDP generation OOM |
| convergence | `examples/convergence/tulu3/**` | 训练、评估、推理、数据校验 | 长链路收敛与质量分析 |

## 推荐运行入口

CPU/轻量单测：

```bash
uv run pytest tests/unit_tests/components/training/test_precision_warnings.py -v
uv run pytest tests/unit_tests/quantization/test_fp8.py -v
uv run pytest tests/unit_tests/loss/test_kd_loss.py -v
uv run pytest tests/unit_tests/distributed/test_grad_utils.py -v
uv run pytest tests/unit_tests/models/nemotron_v3/test_nemotron_v3_mtp_parity.py -v
uv run pytest tests/unit_tests/eval/test_tool_call_evaluator.py tests/unit_tests/eval/test_tool_call_parser.py -v
```

多 GPU / distributed parity：

```bash
bash tests/functional_tests/llm_pretrain_and_kd/L2_TP_Output_Parity_Minified.sh
bash tests/functional_tests/llm_pretrain_and_kd/loss/L2_TEParallelCrossEntropy_DTENSOR_TP2.sh
bash tests/functional_tests/llm_pretrain_and_kd/L2_ClipGradNorm_Correctness_Test.sh
bash tests/functional_tests/context_parallel/L2_CP_NemotronV3_Attention_Test.sh
bash tests/functional_tests/context_parallel/L2_CP_NemotronV3_Mamba_Test.sh
bash tests/functional_tests/context_parallel/L2_CP_Qwen3_5MoE_LinearAttn_Test.sh
```

Checkpoint / resume：

```bash
bash tests/functional_tests/hf_dcp/L2_DCP_FSDP2_Checkpoint.sh
bash tests/functional_tests/hf_dcp/L2_FlashOptim_DCP_Roundtrip.sh
torchrun --nproc-per-node=<N> -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py --config <config.yaml>
```

HF vs NeMo activation：

```bash
CONFIG=<recipe.yaml> NPROC=8 THRESHOLD=0.99 bash examples/convergence/tulu3/model-verification/run.sh
```

Tool-call accuracy：

```bash
python examples/llm_finetune/agent/evaluate_tool_call.py --help
```

## Code-review 风险清单

### 高风险

1. Checkpoint key mismatch 对 adapter/PP load 只 warning 不 fail-fast。

   - 证据路径：`nemo_automodel/components/checkpoint/checkpointing.py`
   - 风险：adapter 漏转 key、reshard 后缺 key、PP 分段 key 漂移时，模型可能带着未加载或随机初始化参数继续训练。
   - 建议：增加 strict/fail-fast 配置；对非白名单 missing/unexpected keys 报错；补负向测试。

2. FP8 conversion 失败可能静默回退原模型。

   - 证据路径：`nemo_automodel/components/quantization/fp8.py`
   - 风险：配置声明 FP8，但实际按 BF16/FP16/FP32 跑，吞吐和精度实验记录失真。
   - 建议：训练/CI 默认 strict；日志和 metrics 记录实际 Float8Linear 覆盖率。

3. NanoGPT shard shuffle 不受 config seed 控制。

   - 证据路径：`nemo_automodel/components/datasets/llm/nanogpt_dataset.py`
   - 风险：同 YAML seed 下，不同 PID 或 worker 布局产生不同 shard 顺序，影响 pretrain loss/resume 对齐。
   - 建议：引入显式 dataset seed，补 `shuffle_files=True` determinism 测试。

### 中风险

4. VLM fallback/retry 使用全局随机替换样本。

   - 证据路径：`nemo_automodel/components/datasets/vlm/datasets.py`、`nemo_automodel/components/datasets/vlm/collate_fns.py`
   - 风险：坏样本或超长样本触发 retry 后，resume 样本序列可能漂移。
   - 建议：基于 `(base_seed, original_idx, attempt, rank, worker_id)` 创建局部 RNG。

5. Full-state-dict load dtype contract 不够清晰。

   - 证据路径：`nemo_automodel/components/checkpoint/checkpointing.py`
   - 风险：bf16 checkpoint 加载到 fp32 initialized model 后，最终参数 dtype 可能不符合预期。
   - 建议：明确 contract，并补 dtype preservation 测试。

6. QAT delayed fake-quant toggle 失败只 warning。

   - 证据路径：`nemo_automodel/recipes/llm/train_ft.py`
   - 风险：配置要求某 step 启用 fake quant，但实际启用状态与配置不一致。
   - 建议：训练模式 fail-fast；记录 fake-quant state。

### 低风险

7. 带 mask 的 CE 可能原地修改 labels。

   - 证据路径：`nemo_automodel/components/loss/masked_ce.py`、`chunked_ce.py`、`te_parallel_ce.py`
   - 风险：调试脚本复用 labels 比较多个 loss 时互相污染。
   - 建议：clone labels 或在 API 文档中明确 destructive behavior。

8. 单进程 MLflow 禁用影响本地对齐实验审计。

   - 证据路径：`nemo_automodel/components/loggers/mlflow_utils.py`
   - 风险：单 GPU/local precision debug run 难以统一记录远端 metrics。

## 建议新增的统一 debug feature

建议在 recipe 层增加可选配置：

```yaml
alignment_debug:
  enabled: true
  steps: 2
  output_dir: ${checkpoint.checkpoint_dir}/alignment-debug
  ranks: all
  collect:
    data: true
    model: true
    loss: true
    gradients: true
    checkpoint: true
    quantization: true
  tensors:
    max_elements: 4096
    names:
      - model.embed_tokens.weight
      - lm_head.weight
```

设计原则：

- 跨组件编排放在 recipe 或 shared utility，不破坏 component import boundary。
- 每个 component 只暴露轻量 collector 或 pure utility。
- 输出 rank-scoped JSONL，避免 rank0-only logger 吞掉 per-rank mismatch。
- fingerprint 不保存大 tensor；默认只保存 shape、dtype、device、placement、norm、finite count、min/max、mean/std、sha256 sampled hash。
- 支持前 N step 自动采集，避免长期训练成本。

建议采集点：

1. Collate 后、CP 改写前后：
   - input shape、attention mask sum、label token count、padding side、packed boundary。
2. Model load/shard 前后：
   - dtype histogram、trainable/frozen count、selected parameter fingerprint。
3. Forward/loss：
   - logits/final hidden state sampled fingerprint、raw sum loss、normalized loss、num_label_tokens。
4. Backward/optimizer：
   - selected grad fingerprint、global grad norm、clip coeff、optimizer state dtype。
5. Checkpoint save/load：
   - selected tensor fingerprint、missing/unexpected key classification、RNG/dataloader state digest。
6. Quantization：
   - FP8 conversion count、unconverted modules、scale/amax summary、QAT fake quant status。

优先级：

1. 先补通用 JSONL debug collector 和 loss/token count tracing。
2. 再接入 model/checkpoint fingerprint。
3. 最后接入 FP8/QAT/QLoRA 和 per-rank distributed shape tracing。

## 测试补强路线

建议优先补以下测试，因为它们能用小模型或短 step 捕获高价值问题：

1. Zero-valid-token loss 一致性：
   - `MaskedCE`、`ChunkedCE`、`FusedLinearCE`、`TEParallelCE` 对 0 label token 的行为统一。
2. 1 GPU vs FSDP2/TP/CP first-step parity：
   - 比较 first-step loss、selected logits fingerprint、selected grad fingerprint、updated weight fingerprint。
3. Checkpoint load dtype preservation：
   - bf16 checkpoint -> fp32 initialized model 的最终 dtype contract。
4. Checkpoint strict negative tests：
   - 故意删除/改名 key，adapter/PP load 必须失败或结构化报告。
5. FP8 strict conversion：
   - enabled=True 且 conversion 失败时 fail-fast；记录 Float8Linear count。
6. Dataset determinism：
   - NanoGPT `shuffle_files=True` 同 seed 顺序一致。
   - VLM retry/replacement 在 fixed seed + worker/rank 下可复现。
7. Golden comparator：
   - 对 `tests/ci_tests/golden_values/**/*.jsonl` 增加通用阈值比较器，比较 loss、grad_norm、num_label_tokens。

## 最终判断

`upstream/main` 已经不是简单地靠“跑通训练”保证精度，而是在多个层面持续修复 reference parity：

- dtype contract：PR #2248、#2271。
- model reference fp32：PR #2201、#2216、#2173、#2083。
- distributed graph/layout：PR #2277、#2147、#2212、#2215。
- token/loss accounting：PR #2204、#2314、#1985、#2259。
- checkpoint/resume：PR #2285、#2310、#2319、#2096、#2120、#2268。
- task accuracy：PR #2338、#2368、#2367。

现有源码和测试已经具备“局部对齐工具箱”，但缺少“统一实验级对齐调试面板”。下一步最有价值的 feature 不是再新增一个单点 parity 脚本，而是把现有 data/model/loss/grad/checkpoint/quantization 证据串成一个 recipe-level、rank-aware、可复现的 `alignment_debug` 工作流。
