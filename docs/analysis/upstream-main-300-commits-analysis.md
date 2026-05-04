# Analysis of the latest 300 commits on `upstream/main`

Generated: 2026-04-27 13:36:38Z  
Worktree: `/root/Automodel/.worktrees/source_code_analysis`  
Analyzed ref: `upstream/main` at `17ed5796` / `17ed5796bdc220c314c9fd6bd718a773a3642521`  
Comparison base: parent of the 300th first-parent commit, `19f6043b93a0`  
Commit window: `30fbb000` (2026-03-10) through `17ed5796` (2026-04-27)  
Method: `git fetch --all --prune --tags`, then `git log --first-parent --max-count=300 upstream/main` and cumulative diff `19f6043b93a0..17ed5796`.

> Scope note: the prompt said “upstream origin main”. This repository has both `origin` (`PraMamba/Automodel`) and `upstream` (`NVIDIA-NeMo/Automodel`). I interpreted the request as the latest 300 first-parent commits on `upstream/main`, because that is the canonical upstream branch. At fetch time, `origin/main` was exactly one commit ahead of `upstream/main` (`feat(claude): add comprehensive .claude workspace configuration`), so the substantive upstream analysis below is unchanged if the intended branch was the fork's main except for that extra workspace-configuration commit.

## Executive synthesis

The 300-commit window is not a narrow patch train; it is a concentrated expansion of Automodel from a mostly LLM fine-tuning/training framework into a broader multimodal, diffusion, retrieval, cloud-launch, and model-onboarding platform. The most distinctive trend is **breadth plus hardening**: many new model families and recipes were added, but nearly as much energy went into checkpoint robustness, CI triage, HuggingFace/Transformers compatibility, and distributed-training edge cases.

Key observations:

1. **Model-surface expansion is the dominant feature.** New/custom model implementations and recipes cover DeepSeek V4 Flash, Gemma4 VLM/MoE, Mistral4, Baichuan, GLM MoE DSA, LLaVA-OneVision, Qwen3/Qwen3.5/Qwen3-VL/Qwen3-Omni variants, Nemotron Nano/Super/Flash variants, MiniMax, GPT-OSS, Step-3.5 Flash, and diffusion models such as Flux, HunyuanVideo, Wan2.1, and Qwen-Image.
2. **Distributed training moved from basic parallel modes to model-specific HPC plumbing.** The window adds or repairs FSDP2, tensor parallelism, pipeline parallelism, context parallelism, expert parallelism, UCCL-based expert parallel infrastructure, Mamba/linear-attention CP paths, dynamic sequence lengths for PP, and TransformerEngine DotProductAttention injection into HuggingFace models.
3. **Checkpointing became a product-quality focus.** PEFT/LoRA/QLoRA, safetensors, HF consolidated export, async/final saves, reload compatibility, vLLM deployment checks, and checkpoint robustness tests are recurring throughout the window.
4. **The dataset layer was substantially widened.** There is new VLM collation/sampling/neat-packing support, dLLM corruption/collation, retrieval bi-encoder/cross-encoder flows, ChatDataset tool-calling and reasoning-content support, length grouping, mock datasets, and safer dataset cache handling.
5. **CI evolved into a recipe-aware release-control system.** The repository gained per-recipe CI config, golden values, known-issue/allow-failure triage, nightlies for LLM/VLM/diffusion, deployment tests, benchmark artifact collection, and release-freeze workflow support.
6. **Docs and onboarding were rebuilt around coverage.** Model coverage moved from coarse pages to many per-model pages; guides were added for DeepSeek V4 Flash, diffusion, dLLM, Gemma4 VLM, launcher backends, large MoE fine-tuning, and skill-based contributor workflows.

## Quantitative overview

### Commit and diff metrics

| Metric | Value |
| --- | --- |
| Commits analyzed | 300 |
| Date span | 2026-03-10 to 2026-04-27 |
| Newest commit | `17ed5796` feat: DeepSeek V4 Flash support (#2039) |
| Oldest included commit | `30fbb000` feat: TP-aware KDLoss with distributed softmax and T² scaling (#1499) |
| Cumulative shortstat | 1116 files changed, 113723 insertions(+), 11919 deletions(-) |
| Parsed additions/deletions from numstat | 113,723 additions / 11,919 deletions |
| Unique changed files in cumulative diff | 1116 |
| Commit PR references | 299 distinct PR numbers, range #1065–#2055 |


### Commit type distribution

| Prefix | Commits |
| --- | --- |
| fix | 138 |
| feat | 64 |
| ci | 51 |
| docs | 22 |
| chore | 8 |
| test | 5 |
| cp | 3 |
| other | 3 |
| build | 2 |
| perf | 2 |
| style | 1 |
| refactor | 1 |


Interpretation: `fix` is the largest prefix, which matches the hardening/compatibility character of the window. `feat` still accounts for a large share and tends to be concentrated around new model families, recipes, launchers, and training capabilities. `ci` is unusually high for a feature window, indicating that the repository was scaling its testing/release machinery in parallel with feature growth.

### Overlapping thematic signal from commit messages

The following categories are overlapping keyword groups, not mutually exclusive buckets; one commit may count in multiple rows.

| Theme | Matching commits | Representative commits |
| --- | --- | --- |
| CI / tests / robustness | 92 | `f9e20e91` ci: add LoRA nightly tests for Wan, Hunyuan, Flux diffusion recipes (#2048); `d21d45de` ci: triage vllm_deploy rc9 failures (#2047); `6e1ae303` ci: triage rc9 finetune failures (#2043) |
| Distributed performance / parallelism | 51 | `f4155ad4` ci: triage pipeline benchmark failures (#2040); `da6061bb` fix: gradient clip with torch_mm + EP (gpt-oss 120b recipe) (#2012); `616064ba` feat: inject TransformerEngine DotProductAttention into HF models (#2011) |
| Checkpointing / PEFT / export | 41 | `f9e20e91` ci: add LoRA nightly tests for Wan, Hunyuan, Flux diffusion recipes (#2048); `838e176e` fix: Unsafe deserialization via `torch.load` on dataset cache files (#2045); `901a2074` fix: lora checkpointing (#2037) |
| VLM / multimodal | 38 | `5876c8aa` fix: guard against zero label tokens causing NaN loss in VLM training (#1985); `b3f97727` fix(vlm): qwen3_5_4b_neat_packing OOM - reduce seqlen to 4096 (#1975); `14f14cd2` fix: AC silently skipped on all registered VLMs — flatten ModuleList  (#1941) |
| Docs / tutorials / model coverage | 34 | `c9919746` docs: Update README with new finetuning support details (#2055); `2f979fb1` docs(llm): drop validate-yaml reference from DeepSeek V4 Flash guide (#2054); `52fd3bb1` docs(llm): add DeepSeek V4 Flash fine-tuning guide (#2053) |
| Dependencies / compatibility | 30 | `d21d45de` ci: triage vllm_deploy rc9 failures (#2047); `5b372e48` fix: regression in tokenizer+auto_map with transformers 5.5.0 (#2025); `cd901973` fix: transformers v5.5.0 validation (#2010) |
| Diffusion / dLLM / media generation | 17 | `f9e20e91` ci: add LoRA nightly tests for Wan, Hunyuan, Flux diffusion recipes (#2048); `3b21c621` feat: Add diffusion finetuning CI pipeline for nightly runs (#1728); `99a80359` fix: Add changes for QwenImage Training (#1976) |
| Data / retrieval / packing | 18 | `838e176e` fix: Unsafe deserialization via `torch.load` on dataset cache files (#2045); `d8be5c12` test: add dataloader checkpoint integration test for retrieval recipes (#1800); `5b372e48` fix: regression in tokenizer+auto_map with transformers 5.5.0 (#2025) |
| Security / vulnerability hardening | 6 | `838e176e` fix: Unsafe deserialization via `torch.load` on dataset cache files (#2045); `e34a12c6` fix: Address pillow CVE (#1994); `9c4c5c8d` docs: add SECURITY.md (#1996) |
| Launchers / cloud execution | 6 | `08da90f8` docs: add SkyPilot Kubernetes tutorial (#1667); `b0456bc0` fix(ci): retry apt-get and Azure CLI installs to handle mirror sync failures (#1872); `134d3066` fix: launcher option from being consumed as a config override. (#1766) |
| DeepSeek V4 Flash | 3 | `17ed5796` feat: DeepSeek V4 Flash support (#2039); `2f979fb1` docs(llm): drop validate-yaml reference from DeepSeek V4 Flash guide (#2054); `52fd3bb1` docs(llm): add DeepSeek V4 Flash fine-tuning guide (#2053) |


### Cumulative changed-file status

| Status | Files |
| --- | --- |
| Modified | 541 |
| Added | 502 |
| Renamed | 47 |
| Deleted | 26 |


### Top-level churn by cumulative diff

| Top-level path | Files | Additions | Deletions | Total line churn |
| --- | --- | --- | --- | --- |
| tests | 371 | 44662 | 3677 | 48339 |
| nemo_automodel | 212 | 30238 | 4053 | 34291 |
| examples | 346 | 21074 | 1735 | 22809 |
| docs | 122 | 10595 | 1083 | 11678 |
| skills | 17 | 3574 | 0 | 3574 |
| uv.lock | 1 | 868 | 551 | 1419 |
| docker | 6 | 767 | 447 | 1214 |
| .github | 15 | 407 | 218 | 625 |
| scripts | 2 | 490 | 0 | 490 |
| tools | 10 | 303 | 38 | 341 |
| AGENTS.md | 1 | 205 | 0 | 205 |
| README.md | 1 | 70 | 99 | 169 |
| BREAKING_CHANGES.md | 1 | 104 | 0 | 104 |
| tutorials | 2 | 99 | 0 | 99 |
| slurm.sub | 1 | 94 | 0 | 94 |
| pyproject.toml | 1 | 58 | 14 | 72 |
| app.py | 1 | 59 | 0 | 59 |
| SECURITY.md | 1 | 26 | 0 | 26 |
| CONTRIBUTING.md | 1 | 22 | 2 | 24 |
| .gitignore | 1 | 5 | 0 | 5 |
| .pre-commit-config.yaml | 1 | 2 | 1 | 3 |
| codecov.yml | 1 | 1 | 1 | 2 |
| {examples | 1 | 0 | 0 | 0 |


### Second-level hotspots

| Path group | Files | Additions | Deletions | Total line churn |
| --- | --- | --- | --- | --- |
| tests/unit_tests | 182 | 32741 | 2967 | 35708 |
| nemo_automodel/components | 169 | 23698 | 2912 | 26610 |
| examples/llm_finetune | 155 | 7396 | 279 | 7675 |
| docs/model-coverage | 81 | 7005 | 183 | 7188 |
| tests/functional_tests | 127 | 5913 | 698 | 6611 |
| examples/convergence | 43 | 5299 | 0 | 5299 |
| tests/ci_tests | 56 | 4779 | 0 | 4779 |
| nemo_automodel/_transformers | 18 | 3740 | 409 | 4149 |
| examples/vlm_finetune | 54 | 3415 | 233 | 3648 |
| docs/guides | 28 | 2457 | 504 | 2961 |
| nemo_automodel/recipes | 15 | 2352 | 340 | 2692 |
| examples/diffusion | 23 | 1211 | 1049 | 2260 |
| examples/llm_benchmark | 19 | 2115 | 0 | 2115 |
| skills/model-onboarding | 4 | 1739 | 0 | 1739 |
| uv.lock | 1 | 868 | 551 | 1419 |
| docs/launcher | 7 | 952 | 252 | 1204 |
| docker/common | 4 | 663 | 427 | 1090 |
| skills/parity-testing | 2 | 597 | 0 | 597 |
| skills/distributed-training | 1 | 572 | 0 | 572 |
| tests/test_media_token_consistency.py | 1 | 510 | 0 | 510 |
| examples/vlm_benchmark | 4 | 469 | 0 | 469 |
| .github/workflows | 12 | 353 | 114 | 467 |
| examples/dllm_generate | 1 | 414 | 0 | 414 |
| nemo_automodel/_cli | 1 | 0 | 346 | 346 |
| tools/diffusion | 10 | 303 | 38 | 341 |
| tests/integration | 1 | 340 | 0 | 340 |
| scripts/precompute_tokens.py | 1 | 309 | 0 | 309 |
| skills/recipe-development | 1 | 270 | 0 | 270 |
| nemo_automodel/cli | 2 | 240 | 0 | 240 |
| docs/index.md | 1 | 129 | 106 | 235 |
| skills/developer-guide | 1 | 231 | 0 | 231 |
| tests/test_meta_dataset_all.py | 1 | 211 | 0 | 211 |
| AGENTS.md | 1 | 205 | 0 | 205 |
| examples/{benchmark | 18 | 190 | 8 | 198 |
| examples/retrieval | 2 | 183 | 0 | 183 |


### Highest-churn individual files

| File | Additions | Deletions | Total line churn |
| --- | --- | --- | --- |
| tests/unit_tests/datasets/vlm/test_collate_fns.py | 1446 | 74 | 1520 |
| nemo_automodel/components/moe/uccl_ep/_buffer.py | 1464 | 0 | 1464 |
| uv.lock | 868 | 551 | 1419 |
| nemo_automodel/components/datasets/vlm/collate_fns.py | 1083 | 39 | 1122 |
| docker/common/uv-pytorch.lock | 626 | 423 | 1049 |
| tests/unit_tests/datasets/vlm/test_datasets.py | 1006 | 1 | 1007 |
| tests/unit_tests/models/mistral4/test_mistral4_model.py | 979 | 0 | 979 |
| docs/guides/llm/finetune.md | 689 | 285 | 974 |
| nemo_automodel/components/datasets/vlm/datasets.py | 947 | 2 | 949 |
| nemo_automodel/components/models/deepseek_v4/layers.py | 897 | 0 | 897 |
| tests/unit_tests/models/qwen3_5/test_cp_linear_attn_patch.py | 889 | 0 | 889 |
| nemo_automodel/components/models/mistral4/model.py | 820 | 0 | 820 |
| tests/unit_tests/_transformers/test_auto_model.py | 791 | 17 | 808 |
| tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py | 749 | 0 | 749 |
| examples/convergence/tulu3/inference/analyze_quality.py | 731 | 0 | 731 |
| nemo_automodel/components/models/deepseek_v4/state_dict_adapter.py | 727 | 0 | 727 |
| nemo_automodel/components/utils/flops_utils.py | 701 | 26 | 727 |
| tests/unit_tests/distributed/pipelining/test_functional.py | 671 | 52 | 723 |
| nemo_automodel/components/datasets/vlm/neat_packing_vlm.py | 698 | 0 | 698 |
| tests/unit_tests/moe/test_uccl_ep_integration.py | 695 | 0 | 695 |
| nemo_automodel/_transformers/model_init.py | 653 | 38 | 691 |
| tests/unit_tests/models/mistral4/test_mistral4_state_dict_adapter.py | 656 | 0 | 656 |
| nemo_automodel/components/moe/uccl_ep/_utils.py | 653 | 0 | 653 |
| nemo_automodel/components/models/gemma4_moe/model.py | 649 | 0 | 649 |
| tests/functional_tests/context_parallel/run_attention_cp.py | 575 | 74 | 649 |
| tests/unit_tests/models/gemma4/test_gemma4_model.py | 645 | 0 | 645 |
| nemo_automodel/components/distributed/parallelizer.py | 493 | 147 | 640 |
| tests/unit_tests/_transformers/test_te_attention.py | 630 | 0 | 630 |
| tests/unit_tests/models/qwen3_vl_moe/test_qwen3_vl_moe_model.py | 612 | 17 | 629 |
| tests/unit_tests/distributed/test_parallelizer.py | 612 | 15 | 627 |
| nemo_automodel/recipes/llm/kd.py | 545 | 79 | 624 |
| tests/unit_tests/_transformers/test_model_init.py | 609 | 0 | 609 |
| nemo_automodel/components/models/baichuan/model.py | 603 | 0 | 603 |
| tests/functional_tests/context_parallel/run_hybrid_nemotron_v3_cp.py | 601 | 0 | 601 |
| tests/unit_tests/distributed/test_mamba_cp.py | 600 | 0 | 600 |
| nemo_automodel/components/models/deepseek_v4/model.py | 599 | 0 | 599 |
| nemo_automodel/_transformers/te_attention.py | 582 | 0 | 582 |
| tests/functional_tests/context_parallel/run_mamba_cp.py | 579 | 0 | 579 |
| skills/distributed-training/SKILL.md | 572 | 0 | 572 |
| nemo_automodel/components/models/qwen3_5_moe/cp_linear_attn.py | 551 | 0 | 551 |


## Most distinctive feature themes

### 1. DeepSeek V4 Flash lands as a first-class custom model

The head commit, `17ed5796`, adds DeepSeek V4 Flash support. This is not just a YAML recipe addition: the cumulative diff includes a new custom model package under `nemo_automodel/components/models/deepseek_v4/` with config, layers, model code, and a state-dict adapter. The implementation captures several unusual V4-specific architecture traits: grouped-query attention with Q-LoRA/O-LoRA style projections, all-MoE FFNs, hash-routing layers, HyperConnection-style multi-copy hidden states, compressed/sliding-window attention parameters, and multi-token prediction configuration.

Evidence files:

- `nemo_automodel/components/models/deepseek_v4/config.py`
- `nemo_automodel/components/models/deepseek_v4/layers.py`
- `nemo_automodel/components/models/deepseek_v4/model.py`
- `nemo_automodel/components/models/deepseek_v4/state_dict_adapter.py`
- `examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml`
- `docs/guides/llm/dsv4-flash.md`
- `docs/model-coverage/llm/deepseek-ai/dsv4-flash.md`
- `tests/unit_tests/models/deepseek_v4/*`

Distinctive implication: Automodel is no longer only wrapping standard HF classes for new architectures; it is carrying bespoke model implementations and checkpoint conversion logic for frontier MoE variants where upstream HF behavior, routing, precision, and state-dict layout need tight control.

### 2. Diffusion and discrete-diffusion support become real workflows

The window adds a diffusion training/fine-tuning track and then hardens it. There are new and modified configs for Flux, HunyuanVideo, Wan2.1, and Qwen-Image; a generic diffusion generation entrypoint replaces model-specific generation scripts; LoRA variants are added for major diffusion recipes; diffusion safetensors and stateful dataloading become default expectations; and CI starts nightly diffusion fine-tuning coverage.

Evidence commits include `b60ab9ee` (Wan multi-resolution), `45f08465` (diffusion safetensors), `9ab601ba` (Stateful Dataloader), `924d6fac` (LoRA for diffusion), `c9d1aa03` / `99a80359` (Qwen-Image training), `3b21c621` (diffusion CI pipeline), and `f9e20e91` (LoRA nightly tests for Wan/Hunyuan/Flux).

Evidence files:

- `examples/diffusion/finetune/*_lora.yaml`
- `examples/diffusion/finetune/qwen_image_t2i_flow.yaml`
- `examples/diffusion/generate/generate.py`
- `nemo_automodel/_diffusers/auto_diffusion_pipeline.py`
- `nemo_automodel/components/datasets/diffusion/*`
- `nemo_automodel/components/flow_matching/adapters/qwen_image.py`
- `tools/diffusion/processors/qwen_image.py`
- `tests/unit_tests/diffusion_processors/*`
- `tests/ci_tests/configs/diffusion_finetune/nightly_recipes.yml`

Distinctive implication: image/video generation support is now a product lane with recipes, data processing, adapters, generation, LoRA, checkpointing, and CI—not a side example.

### 3. dLLM fine-tuning and generation are introduced

Discrete diffusion LLM work appears as a new recipe family: dLLM supervised fine-tuning support, LLaDA SFT support, generation support, docs, losses, strategy code, dataset corruption/collation, examples, and tests.

Evidence commits include `10a3b034`, `34097e46`, `15fb1210`, and `55b26a49`.

Evidence files:

- `nemo_automodel/recipes/dllm/train_ft.py`
- `nemo_automodel/recipes/dllm/strategy.py`
- `nemo_automodel/components/datasets/dllm/collate.py`
- `nemo_automodel/components/datasets/dllm/corruption.py`
- `nemo_automodel/components/loss/dllm_loss.py`
- `examples/dllm_sft/*`
- `examples/dllm_generate/generate.py`
- `docs/guides/dllm/finetune.md`
- `tests/unit_tests/datasets/dllm/*`
- `tests/unit_tests/recipes/dllm/*`

Distinctive implication: Automodel is extending beyond autoregressive next-token training into diffusion-style language modeling, including different loss/data mechanics rather than just recipe variants.

### 4. VLM and multimodal training become substantially more mature

VLM work is one of the heaviest clusters. The window adds LLaVA-OneVision-1.5 integration, Gemma4 VLM/MoE recipes and custom model support, Qwen3/Qwen3.5 VLM recipes, Qwen3.5 VLM TP+PP support, mock VLM datasets, pretokenized/neat-packing pipelines, richer collation utilities, media consistency tests, and many CI fixes around pipeline-parallel VLM recipes.

Evidence commits include `d73a22fb` (custom VLM chat templates), `6264a707` (VLM pretokenized data pipeline with neat packing), `9bb69cf9` (mock VLM dataset and Gemma4 pretokenize), `c4c5131a` (LLaVA-OneVision), `2dfad74b` (Gemma4 VLM TP+PP), `3cb42fe7` (Qwen3.5 VLM TP+PP), `14f14cd2` (activation checkpointing for VLMs), and `5876c8aa` (zero-label-token NaN guard).

Evidence files:

- `nemo_automodel/components/datasets/vlm/collate_fns.py`
- `nemo_automodel/components/datasets/vlm/datasets.py`
- `nemo_automodel/components/datasets/vlm/neat_packing_vlm.py`
- `nemo_automodel/components/datasets/vlm/samplers.py`
- `nemo_automodel/components/models/gemma4_moe/*`
- `nemo_automodel/components/models/llava_onevision/*`
- `nemo_automodel/recipes/vlm/finetune.py`
- `examples/vlm_finetune/gemma4/*`
- `examples/vlm_finetune/llava_onevision/*`
- `examples/vlm_finetune/qwen3_5*/*`
- `tests/test_media_token_consistency.py`
- `tests/unit_tests/datasets/vlm/*`

Distinctive implication: multimodal training is being treated as a first-class distributed-training problem, not merely as a dataset formatting variation over LLM SFT.

### 5. Distributed training and model-parallel infrastructure expand sharply

The codebase now contains substantially more fine-grained distributed machinery. The window touches FSDP2, TP, PP, CP, expert parallelism, device-mesh helpers, optimized TP plans, pipeline utility functions, mamba/linear attention context parallel paths, and MoE token dispatch.

Standout changes:

- TransformerEngine DotProductAttention injection into HF models (`616064ba`) via `nemo_automodel/_transformers/te_attention.py`.
- FSDP2 weight prefetching and async TP optimization (`2013a4dd`).
- Dynamic sequence length support for pipeline parallelism (`da3ec826`).
- Context-parallel tests for attention, hybrid Nemotron, Mamba, and Qwen3.5 MoE linear attention.
- UCCL expert-parallel implementation files under `nemo_automodel/components/moe/uccl_ep/` plus setup script `scripts/setup_uccl_ep.sh`.
- TP-aware KDLoss with distributed softmax and T² scaling (`30fbb000`, the oldest included commit).
- Gradient clipping fixes for `torch_mm + EP` in GPT-OSS 120B (`da6061bb`).

Evidence files:

- `nemo_automodel/components/distributed/parallelizer.py`
- `nemo_automodel/components/distributed/mamba_cp.py`
- `nemo_automodel/components/distributed/mesh_utils.py`
- `nemo_automodel/components/distributed/pipelining/*`
- `nemo_automodel/components/moe/megatron/*`
- `nemo_automodel/components/moe/uccl_ep/*`
- `nemo_automodel/_transformers/te_attention.py`
- `tests/functional_tests/context_parallel/*`
- `tests/unit_tests/distributed/*`
- `tests/unit_tests/moe/test_uccl_ep_integration.py`

Distinctive implication: the framework is pushing toward large MoE/VLM training where architecture-specific parallel plans, communication backends, and precision-aware dispatch are central capabilities.

### 6. Checkpointing, PEFT, and HF export robustness become a major quality gate

Checkpointing is the most repeated reliability concern. The commits touch PEFT/LoRA save/load, QLoRA loading, safetensors consolidation, HF DCP, fused QKV PEFT behavior, checkpoint resume, asynchronous final saves, offline HF behavior, KL thresholds for robustness tests, and vLLM deployment validation. The test suite adds a full checkpoint-robustness area.

Evidence commits include `cadbba77` (checkpoint robustness tests), `3a3f6858` (vLLM deployment tests), `901a2074` (LoRA checkpointing), `03227873` and `785ab086` (batch robustness fixes), `a4a73022` (QLoRA checkpoint loading), `c97722a8` (HF_HUB_OFFLINE), `45f08465` / `f09ff4c0` (diffusion safetensors), and `80e81b65` (save-every-epoch and max-steps support).

Evidence files:

- `nemo_automodel/components/checkpoint/checkpointing.py`
- `nemo_automodel/components/checkpoint/addons.py`
- `nemo_automodel/components/checkpoint/utils.py`
- `tests/functional_tests/checkpoint_robustness/*`
- `tests/functional_tests/checkpoint/test_hf_consolidated_gptoss_mxfp4.py`
- `tests/unit_tests/checkpoint/test_checkpointing.py`
- `tests/unit_tests/checkpoint/test_consolidate_safetensors.py`

Distinctive implication: persistence/export compatibility is now a first-order acceptance criterion, especially for PEFT and downstream serving/deployment.

### 7. Data pipelines shift toward packing, retrieval, safety, and richer chat semantics

The dataset changes are broad and practical. LLM/VLM neat packing, length grouping, VLM media-aware collation, mock datasets, retrieval dataset refactors, cross-encoder training, dataloader checkpoint tests, ChatDataset reasoning-content/tool-calling, and safer dataset cache deserialization all appear in this window.

Evidence commits include `1c5714f8` (greedy knapsack neat packing), `6264a707` (VLM pretokenized packing), `e9bee96a` (reasoning_content/tool-calling), `3eb5bc07` (skip malformed jsonl), `d8be5c12` (retrieval dataloader checkpoint), and `838e176e` (unsafe deserialization fix).

Evidence files:

- `nemo_automodel/components/datasets/llm/neat_packing.py`
- `nemo_automodel/components/datasets/llm/length_grouped_sampler.py`
- `nemo_automodel/components/datasets/llm/chat_dataset.py`
- `nemo_automodel/components/datasets/vlm/*`
- `nemo_automodel/_transformers/retrieval.py`
- `nemo_automodel/recipes/retrieval/*`
- `examples/retrieval/*`
- `tests/functional_tests/retrieval/*`
- `tests/unit_tests/datasets/llm/test_neat_packing.py`
- `tests/unit_tests/datasets/vlm/test_neat_packing_vlm.py`

Distinctive implication: training efficiency and data correctness are being handled at the framework level, including multimodal sequence packing and recoverable dataloader state.

### 8. Recipe coverage explodes across LLM, VLM, benchmark, convergence, and diffusion cases

The `examples/` tree accounts for the largest file count and a major share of line churn. New/updated recipes span:

- LLM fine-tuning: DeepSeek V4, GLM 5/5.1, GPT-OSS 20B/120B, Llama 3.1/3.2/3.3, MiniMax M2, Mistral/Ministral/Mistral4, Nemotron Nano/Super/Flash, Phi, Qwen/Qwen3/Qwen3 MoE/Qwen3 Next, Step-3.5 Flash, Baichuan, Falcon, Granite, Seed, StarCoder.
- VLM fine-tuning: Gemma3/3n/4, InternVL, Kimi VL, LLaVA-OneVision, Mistral/Ministral/Mistral4, Nemotron Parse, Phi4-MM, Qwen2.5/Qwen3/Qwen3.5/Qwen3.5 MoE.
- Benchmarks: LLM/VLM benchmark directories, DeepEP/TE/FP8 configs, LoRA benchmark variants.
- Convergence: Tulu3 examples with model-specific configs, chat templates, evaluation scripts, activation comparison, inference quality analysis, and plots.
- Diffusion: finetune/pretrain/generation configs for Flux, Hunyuan, Wan, Qwen-Image.

Distinctive implication: this window is as much about codifying validated recipes as it is about adding core library primitives.

### 9. CI becomes release-aware, recipe-aware, and failure-triage-aware

The test/CI layer grows aggressively. The commits add per-recipe env vars, known issue / allow-failure keys, recipe scopes, nightlies, benchmark artifact collection, deployment environments, codecov base SHA handling, shorter pytest tracebacks, AWS ephemeral runners for external contributors, release-freeze workflow, and many golden-value files.

Evidence files:

- `.github/workflows/cicd-main.yml`
- `.github/workflows/release-freeze.yml`
- `tests/ci_tests/configs/*`
- `tests/ci_tests/golden_values/*`
- `tests/ci_tests/scripts/*`
- `tests/ci_tests/utils/generate_ci_tests.py`
- `tests/ci_tests/utils/transformers_version_check.py`
- `docker/Dockerfile.deploy`

Distinctive implication: CI is now a metadata-driven validation matrix for a large recipe catalog, not a small fixed test suite.

### 10. Launching and operations move into the framework

The repository adds SkyPilot and NeMo-Run launchers, launcher docs, local/interactive launcher tests, and config handling for recipe targets and launcher options. The old Slurm-only component layout is replaced by a more general launcher abstraction with cloud and managed-executor backends.

Evidence commits include `e766d71a` (SkyPilot backend), `44e86dc5` (NeMo-Run launcher), `08da90f8` (SkyPilot Kubernetes tutorial), and `134d3066` (launcher option/config override fix).

Evidence files:

- `nemo_automodel/components/launcher/base.py`
- `nemo_automodel/components/launcher/skypilot/*`
- `nemo_automodel/components/launcher/nemo_run/*`
- `nemo_automodel/components/launcher/interactive.py`
- `docs/launcher/*`
- `tests/unit_tests/launcher/*`

Distinctive implication: Automodel is becoming a runnable training platform with explicit submission backends rather than only a Python package of recipes.

### 11. Documentation and contributor onboarding are structurally upgraded

Docs changed heavily: model coverage was split into many per-family pages; new guides were added for DeepSeek V4 Flash, diffusion, dLLM, Gemma4 VLM, launcher backends, large MoE fine-tuning, and CI summaries. The `skills/` tree adds detailed workflow guidance for developer onboarding, distributed training, launcher configuration, model onboarding, parity testing, and recipe development.

Evidence files:

- `docs/model-coverage/**`
- `docs/guides/llm/dsv4-flash.md`
- `docs/guides/diffusion/*`
- `docs/guides/dllm/finetune.md`
- `docs/guides/vlm/gemma4.md`
- `docs/launcher/*`
- `skills/*/SKILL.md`
- `skills/model-onboarding/*-patterns.md`

Distinctive implication: maintainers are encoding tacit model-porting and validation knowledge into docs and skill files, likely to support rapid onboarding of new architectures without repeatedly rediscovering pitfalls.

### 12. Security and dependency hygiene is not incidental

Several commits address concrete vulnerability and safety concerns: unsafe dataset cache deserialization, replacing pickle with safe `torch.load(..., weights_only=True)`, Pillow CVEs, container/source CVEs, WandB-core Go dependency CVEs, SECURITY.md, secrets-baseline updates, and dependency lock refreshes. There are also compatibility fixes for Transformers 5.x, PyTorch 2.9+/2.11+, DeviceMesh private APIs, quantization configs, and optional flash-linear-attention placement.

Distinctive implication: the larger model/recipe surface is paired with active supply-chain and runtime-safety maintenance.

## Architectural impact by subsystem

### `nemo_automodel/_transformers`

This area becomes the framework's compatibility and model-resolution brain. It handles model registry and initialization, AutoModel/AutoTokenizer behavior, TransformerEngine attention injection, capability metadata, retrieval models, MFU utilities, V4 patches, tokenization, and compatibility wrappers for new Transformers releases.

High-impact files: `model_init.py`, `auto_model.py`, `registry.py`, `te_attention.py`, `capabilities.py`, `retrieval.py`, `utils.py`, and `tokenization/*`.

### `nemo_automodel/components/models`

This area gains several custom implementations and adapters. The new or heavily modified families include DeepSeek V4, Gemma4 MoE, Mistral4, Baichuan, GLM MoE DSA, LLaVA-OneVision, Qwen3.5 MoE linear attention/CP support, Qwen3 VL/MoE adapters, GPT-OSS improvements, Nemotron V3 improvements, and common packing utilities.

### `nemo_automodel/components/distributed` and `components/moe`

These subsystems absorb most of the large-scale training work: context parallel helpers, Mamba CP, optimized TP plans, pipelining utilities, device mesh helpers, MoE token dispatch, load-balance metrics, fused all-to-all, UCCL EP buffers/utilities, and expert dtype/gradient sync fixes.

### `nemo_automodel/components/checkpoint`

Checkpointing evolves from a utility to a central reliability subsystem: memory usage, safetensors consolidation, HF storage/backports, add-ons, stateful wrappers, config fields, and broader save/load semantics all changed.

### `nemo_automodel/components/datasets`

Datasets now cover LLM, VLM, diffusion, dLLM, retrieval, packing, length grouping, mock data, media utilities, samplers, and safer cache behavior.

### `nemo_automodel/recipes`

Recipes are no longer thin examples only. New recipe families for dLLM and retrieval appear, while LLM/VLM/diffusion recipes gain distributed, checkpointing, progress, logging, and config-target improvements.

## Release and integration risks

1. **Huge surface area change.** More than one thousand files changed in the cumulative diff. Even with extensive tests, integration risk is broad because model implementations, recipes, distributed primitives, docs, and CI changed together.
2. **Many fixes are compatibility-driven.** Repeated Transformers/PyTorch/HF changes suggest upstream dependencies are moving quickly. Future upgrades should expect breakage around config fields, tokenizer auto_map, meta-device init, DeviceMesh APIs, and quantization configs.
3. **Checkpoint robustness is improved but still fragile by nature.** The volume of checkpoint-specific fixes implies that PEFT/LoRA/QLoRA, HF consolidation, vLLM deployment, and custom state-dict adapters are high-risk seams.
4. **Distributed VLM/MoE paths are complex.** TP+PP+CP+EP combinations, dynamic sequence length, root mesh preservation, MoE gate behavior, and gradient sync knobs are easy to regress without dedicated recipe coverage.
5. **CI has triage mechanisms.** `known_issue_id` and `allow_failure` are useful for release operations, but they can hide quality debt if not actively monitored.
6. **Security posture improved.** The unsafe deserialization fixes and CVE updates are meaningful, but the expansion into dataset caches, remote launchers, and optional dependencies increases future attack surface.

## Recommended follow-up review focus

If this branch will be integrated or audited, prioritize:

1. Run or inspect checkpoint-robustness suites for the exact model families you rely on: PEFT, QLoRA, HF consolidated export, and vLLM deployment.
2. Validate distributed combinations used in production: FSDP2+TP, TP+PP VLM, CP for Mamba/linear attention, and EP/UCCL paths.
3. Review dataset cache serialization/deserialization and remote launcher inputs for trust-boundary assumptions.
4. Check CI configs for `allow_failure` and `known_issue_id` entries before treating green CI as fully green.
5. For model onboarding, prefer the new `skills/model-onboarding` and parity-testing docs rather than ad hoc adapter work.
6. Treat `upstream/main` and `origin/main` as almost identical for this analysis, but include the fork's single `origin/main`-only workspace config commit if your target is the fork main branch exactly.

## Appendix A: All 300 analyzed commits

| # | Commit | Date | Author | Subject |
| --- | --- | --- | --- | --- |
| 1 | 17ed5796 | 2026-04-27 | khazzz1c | feat: DeepSeek V4 Flash support (#2039) |
| 2 | c9919746 | 2026-04-25 | Huiying | docs: Update README with new finetuning support details (#2055) |
| 3 | 2f979fb1 | 2026-04-26 | khazzz1c | docs(llm): drop validate-yaml reference from DeepSeek V4 Flash guide (#2054) |
| 4 | 52fd3bb1 | 2026-04-25 | khazzz1c | docs(llm): add DeepSeek V4 Flash fine-tuning guide (#2053) |
| 5 | f9e20e91 | 2026-04-24 | Pranav Thombre | ci: add LoRA nightly tests for Wan, Hunyuan, Flux diffusion recipes (#2048) |
| 6 | d21d45de | 2026-04-24 | Dong Hyuk Chang | ci: triage vllm_deploy rc9 failures (#2047) |
| 7 | 6e1ae303 | 2026-04-24 | Dong Hyuk Chang | ci: triage rc9 finetune failures (#2043) |
| 8 | 838e176e | 2026-04-24 | tomaioo | fix: Unsafe deserialization via `torch.load` on dataset cache files (#2045) |
| 9 | 901a2074 | 2026-04-24 | linnan wang | fix: lora checkpointing (#2037) |
| 10 | f4155ad4 | 2026-04-24 | Dong Hyuk Chang | ci: triage pipeline benchmark failures (#2040) |
| 11 | d8be5c12 | 2026-04-24 | Oliver Holworthy | test: add dataloader checkpoint integration test for retrieval recipes (#1800) |
| 12 | 18560fa8 | 2026-04-24 | ooo oo | fix: add required YAML frontmatter to skills (#2032) |
| 13 | da6061bb | 2026-04-23 | Alexandros Koumparoulis | fix: gradient clip with torch_mm + EP (gpt-oss 120b recipe) (#2012) |
| 14 | 4b754485 | 2026-04-23 | Dong Hyuk Chang | ci: add known_issue_id / allow_failure keys + triage (#2028) |
| 15 | 616064ba | 2026-04-24 | khazzz1c | feat: inject TransformerEngine DotProductAttention into HF models (#2011) |
| 16 | 5dcc9abe | 2026-04-23 | Abhishree Thittenamane | fix: Propagate torch_dtype to sub-configs correctly (#2027) |
| 17 | 5b372e48 | 2026-04-23 | Alexandros Koumparoulis | fix: regression in tokenizer+auto_map with transformers 5.5.0 (#2025) |
| 18 | 3b21c621 | 2026-04-23 | Pranav Thombre | feat: Add diffusion finetuning CI pipeline for nightly runs (#1728) |
| 19 | 9df2e140 | 2026-04-23 | Alexandros Koumparoulis | fix: add discover pp seq len (#2024) |
| 20 | 6d0c37bc | 2026-04-23 | Alexandros Koumparoulis | fix: switch from match_all_linear to target_modules (#2022) |
| 21 | d03d3516 | 2026-04-23 | Dong Hyuk Chang | ci: add --tb=short to pytest invocations in CI test scripts (#2018) |
| 22 | 9eccbb61 | 2026-04-23 | Alexandros Koumparoulis | chore: bump pyt (#2003) |
| 23 | 753b517b | 2026-04-23 | oliver könig | ci: add base_sha to codecov/codecov-action upload step (#2016) |
| 24 | 3e333a6d | 2026-04-22 | Alexandros Koumparoulis | fix: change drop_long_samples to True by default (#2009) |
| 25 | cd901973 | 2026-04-22 | Alexandros Koumparoulis | fix: transformers v5.5.0 validation (#2010) |
| 26 | 708ed84d | 2026-04-22 | Huiying | test: add test to 1985 (#2006) |
| 27 | 785ab086 | 2026-04-22 | Adil | fix: batch Flash 1B + Super-49B PEFT + qwen2.5-7B ckpt-robustness (#1984) |
| 28 | 5876c8aa | 2026-04-23 | khazzz1c | fix: guard against zero label tokens causing NaN loss in VLM training (#1985) |
| 29 | 99a80359 | 2026-04-22 | Pranav Thombre | fix: Add changes for QwenImage Training (#1976) |
| 30 | 8bd335e7 | 2026-04-22 | Dong Hyuk Chang | ci: Update test recipe list (#2001) |
| 31 | b8867610 | 2026-04-22 | Dong Hyuk Chang | ci: Support per-recipe env_vars in CI config (#1999) |
| 32 | e34a12c6 | 2026-04-22 | Dong Hyuk Chang | fix: Address pillow CVE (#1994) |
| 33 | 9c4c5c8d | 2026-04-22 | Charlie Truong | docs: add SECURITY.md (#1996) |
| 34 | db1843e4 | 2026-04-22 | Dong Hyuk Chang | fix: Address ci timeout test from rc8 (#1991) |
| 35 | 96b00f5d | 2026-04-22 | Huiying | feat: add qwen3.6 27B config (#1992) |
| 36 | 731f8c13 | 2026-04-22 | Dong Hyuk Chang | fix: Move benchmark recipe out of llm_finetune nightly (#1989) |
| 37 | ad461c84 | 2026-04-22 | Dong Hyuk Chang | fix: vllm deploy test should fail if vllm is not present (#1987) |
| 38 | e6b75ccf | 2026-04-22 | khazzz1c | feat: add tqdm progress bar to all training recipe loops (#1983) |
| 39 | e744250e | 2026-04-22 | Huiying | fix(devstral): point 24B Squad recipes at official FP8 model (#1980) |
| 40 | 03227873 | 2026-04-22 | Adil | fix: batch ckpt-robustness fixes for pipeline 48953745 (supersedes 9 PRs) (#1971) |
| 41 | a494d095 | 2026-04-21 | Alexandros Koumparoulis | fix: nemotron flash (#1973) |
| 42 | b3f97727 | 2026-04-21 | Huiying | fix(vlm): qwen3_5_4b_neat_packing OOM - reduce seqlen to 4096 (#1975) |
| 43 | fed190f2 | 2026-04-21 | Alexandros Koumparoulis | fix: ministral tp plan (#1963) |
| 44 | 8bce05da | 2026-04-21 | Huiying | feat: add LoRA recipes for GLM-5.1, MiniMax-M2.7, and Qwen3.6-35B-A3B (#1970) |
| 45 | a966a5a4 | 2026-04-21 | Huiying | chore: add tests to 1941 (#1959) |
| 46 | c4ba9bf2 | 2026-04-21 | Alexandros Koumparoulis | chore: add @zyzhou5 and @athitten to codeowners (#1968) |
| 47 | f1bc9f57 | 2026-04-21 | Dong Hyuk Chang | fix: Update gemm4 26b ci timeout (#1962) |
| 48 | defff747 | 2026-04-21 | Charlie Truong | docs: Add container version to docs version picker (#1965) |
| 49 | 99483394 | 2026-04-21 | Alexandros Koumparoulis | fix: update docs (#1961) |
| 50 | da12cd93 | 2026-04-21 | Dong Hyuk Chang | fix: Patch wandb-core Go CVEs: bump otel SDK, add go-jose (#1957) |
| 51 | 14f14cd2 | 2026-04-22 | khazzz1c | fix: AC silently skipped on all registered VLMs — flatten ModuleList  (#1941) |
| 52 | 388845e7 | 2026-04-21 | oliver könig | ci(feat): use AWS ephemeral runners for external contributors (#1892) |
| 53 | c7098b31 | 2026-04-21 | Dong Hyuk Chang | fix: Update recipe test time based on release test run (#1955) |
| 54 | 556d6872 | 2026-04-21 | Dong Hyuk Chang | ci: Add test_recipes for custom test scope (#1915) |
| 55 | 3cb42fe7 | 2026-04-21 | Alexandros Koumparoulis | feat: Qwen3.5 VLM TP+PP support with per-microbatch grad reduce-scatter knob (#1859) |
| 56 | 79ce7b20 | 2026-04-21 | jQizhang | fix(moe): align EP expert weight dtype with activation dtype (#1913) |
| 57 | a1dc3a67 | 2026-04-21 | Hemil Desai | fix: Step-3.5-Flash layer_types mismatch and related recipe fixes (#1916) |
| 58 | d82a831e | 2026-04-21 | Adil | fix: disable packed sequences for nemotron_nano_4b_squad (#1929) |
| 59 | 83dfbc7c | 2026-04-20 | Alexandros Koumparoulis | fix: make _get_logits pp aware in ckpt robustness (#1923) |
| 60 | fc46ae53 | 2026-04-20 | Alexandros Koumparoulis | fix: chat dataset (#1921) |
| 61 | 822977bb | 2026-04-20 | Alexandros Koumparoulis | fix: update defer_fsdp_grad_sync in recipes (#1919) |
| 62 | cea9fd21 | 2026-04-21 | Abhishree Thittenamane | fix: Update recipe_owner for gemma4 (#1925) |
| 63 | a4a73022 | 2026-04-20 | Alexandros Koumparoulis | fix: qlora ckpt loading (#1549) |
| 64 | c9d1aa03 | 2026-04-20 | Harsha Pasham | fix: support Qwen-Image finetune (T2I) (#1704) |
| 65 | d0874b3e | 2026-04-21 | alexchiu | build: move flash-linear-attention back to optional-dependencies (#1894) |
| 66 | c2b7f37f | 2026-04-20 | khazzz1c | fix: add embed_vision to MULTIMODAL_SUFFIXES and set lbs=2 for Gemma4 PP2 recipe (#1911) |
| 67 | 45537f96 | 2026-04-20 | ooo oo | chore: update mask helpers for Transformers inputs_embeds rename (#1782) |
| 68 | 08da90f8 | 2026-04-19 | Zeel Desai | docs: add SkyPilot Kubernetes tutorial (#1667) |
| 69 | 8d193015 | 2026-04-18 | zyzhou5 | fix: baichuan dynamic cache (#1865) |
| 70 | 2dfad74b | 2026-04-19 | khazzz1c | feat: TP+PP support for Gemma4 VLM (with tied lm_head fix and 31B recipes) (#1904) |
| 71 | c72b9310 | 2026-04-18 | Huiying | fix: restore Qwen3.5 + Phi-4-MM nightly CI after transformers v5.5 update (#1906) |
| 72 | ad529835 | 2026-04-19 | jQizhang | fix(gemma4_moe): vision-aware mask when use_bidirectional_attention==vision (#1905) |
| 73 | fb62eb48 | 2026-04-18 | sharonyu-115 | fix: pass unnormalized residual to MoE gate in Gemma4 decoder layer (#1895) |
| 74 | 306b4bd7 | 2026-04-17 | Alexandros Koumparoulis | fix: test in mesh_utils (#1898) |
| 75 | 83888bec | 2026-04-17 | Dong Hyuk Chang | ci: Add Dockerfile.deploy for deploy test environment (#1804) |
| 76 | 74cc6428 | 2026-04-17 | Pranav Thombre | fix: Fix bug in diffusion generation (#1850) |
| 77 | aef0b4b7 | 2026-04-17 | Hemil Desai | fix: relax KL thresholds and remove invalid kwargs in Qwen3Next linear attn (#1867) |
| 78 | e4581201 | 2026-04-17 | Zhiyu Li | chore: move recipes to have perf CI/CD coverage (#1885) |
| 79 | bd942f20 | 2026-04-17 | oliver könig | ci(action): surface launch info and pass/fail banner; fix exit_code capture (#1887) |
| 80 | 905d39c6 | 2026-04-17 | khazzz1c | fix: preserve root mesh for multi-node Gemma4 TP4 FSDP2 runs (#1868) |
| 81 | 55b26a49 | 2026-04-16 | zyzhou5 | docs: Add dLLM SFT fine-tuning and generation guide (#1806) |
| 82 | 809965e1 | 2026-04-16 | Huiying | fix: resolve VLM CI failures for PP recipes and collate_fn (#1799) |
| 83 | f3d11b77 | 2026-04-16 | Alexandros Koumparoulis | fix: handle transformers.FineGrainedFP8Config quantization config (#1864) |
| 84 | 6038b7b0 | 2026-04-16 | Dong Hyuk Chang | fix: Create diffusion_kernels group to fix HF_HUB_OFFLINE compatibility (#1842) |
| 85 | ea9ef345 | 2026-04-17 | khazzz1c | fix(metric_logger): handle non-scalar tensor metrics without crashing (#1871) |
| 86 | 7276c3f4 | 2026-04-17 | khazzz1c | fix(collate): auto-derive assistant turn markers for non-Qwen models (#1862) |
| 87 | db82563c | 2026-04-16 | Huiying | cp: 1813 fix: FSDP2 meta-device crash for Qwen3.5 GatedDeltaNet fp32 params (#1869) |
| 88 | 62668321 | 2026-04-16 | Alexandros Koumparoulis | fix: lint in llava_onevision (#1884) |
| 89 | c4c5131a | 2026-04-17 | vaibhav gauraha | feat: LLaVA-OneVision-1.5 integration #1783 (#1790) |
| 90 | b0456bc0 | 2026-04-16 | oliver könig | fix(ci): retry apt-get and Azure CLI installs to handle mirror sync failures (#1872) |
| 91 | 9bb69cf9 | 2026-04-16 | Huiying | feat: add mock VLM dataset and Gemma4 pretokenize support (#1682) |
| 92 | 906ecaed | 2026-04-16 | Huiying | feat: add Qwen3.6-35B-A3B VLM finetune recipe (#1882) |
| 93 | c97722a8 | 2026-04-16 | Dong Hyuk Chang | fix: Skip snapshot_download when HF_HUB_OFFLINE=1 (#1834) |
| 94 | c77b2c5f | 2026-04-16 | Dong Hyuk Chang | fix: Setup vllm testing with uv --no-config (#1875) |
| 95 | 611f4183 | 2026-04-16 | VM-IPA | fix: gradient checkpointing broken for MoE models on single GPU (ep_size=1) (#1873) |
| 96 | 860725b0 | 2026-04-16 | Alexandros Koumparoulis | fix: gpt oss ci (#1877) |
| 97 | dfd58656 | 2026-04-16 | Dong Hyuk Chang | ci: Reduce default finetune step count from 100 to 50 (#1874) |
| 98 | ca93d12c | 2026-04-16 | oliver könig | chore: bump FW-CI-templates to v0.80.2 (#1585) |
| 99 | ea8f5c0d | 2026-04-16 | khazzz1c | fix: PyTorch 2.9.x compatibility for DeviceMesh private APIs (#1825) |
| 100 | d8543990 | 2026-04-16 | ooo oo | style: run `prek run --all-files` to format all files (#1065) |
| 101 | fc6fe198 | 2026-04-15 | Adil | ci: add NMP customizer contract test configs (#1712) |
| 102 | 307c2ab0 | 2026-04-15 | Adil | fix: trust_remote_code guard in robustness test (#1845) |
| 103 | 384adcf7 | 2026-04-15 | Adil | fix: pre-cache HF dynamic modules to prevent filesystem race in robustness test (#1840) |
| 104 | 9f785040 | 2026-04-15 | Adil | fix: relax checkpoint robustness HF KL threshold for nemotron_nano_8b_v1 (#1839) |
| 105 | 3fbf0338 | 2026-04-15 | Abhishree Thittenamane | ci: Update to transformers v5.5 (#1734) |
| 106 | e0950014 | 2026-04-14 | Alexandros Koumparoulis | fix: mute unsupported field attribute warning on startup (#1773) |
| 107 | 5d311934 | 2026-04-14 | Alexandros Koumparoulis | fix: rotary embeddings for v4 (#1821) |
| 108 | c27347ce | 2026-04-14 | Huiying | fix: install ffmpeg and rebuild torchcodec for phi4mm audio decoding (#1826) |
| 109 | 0ccbd903 | 2026-04-14 | Dong Hyuk Chang | fix: Re-apply PyTorch dependency overrides after full COPY in Dockerfile (#1847) |
| 110 | 8dc00dcb | 2026-04-14 | Adil | fix: gpt_oss_20b_single_gpu_peft CI crash with nproc_per_node override (#1835) |
| 111 | ec5cb469 | 2026-04-15 | khazzz1c | fix: stop resolve_yaml_env_vars from scanning runtime data in instantiate() (#1827) |
| 112 | 8f1e8166 | 2026-04-15 | Taishi Nakamura | docs: fix GLM-5.1 attention mechanism name in README (#1833) |
| 113 | 844f0efe | 2026-04-14 | Dong Hyuk Chang | fix: Align benchmark TEST_LEVEL check with generate_ci_tests scope (#1831) |
| 114 | 9179da61 | 2026-04-14 | stanley1208 | fix: in-place state dict conversion to reduce peak VRAM by ~50% (#1742) |
| 115 | f3266513 | 2026-04-14 | Alexandros Koumparoulis | fix: tie weights outside _init_model (#1817) |
| 116 | 2633a395 | 2026-04-13 | Alexandros Koumparoulis | fix: meta init with force_hf=True (#1810) |
| 117 | a52e2805 | 2026-04-13 | Alexandros Koumparoulis | fix: enable dequantization for ministral3 and dataset limit  (#1807) |
| 118 | 835f7a47 | 2026-04-13 | Dong Hyuk Chang | ci: Increase benchmark timeout for GLM and Qwen3.5 MoE LoRA recipes (#1818) |
| 119 | a3a57176 | 2026-04-13 | Dong Hyuk Chang | ci: RC6 timeout fixes for release test recipes (#1801) |
| 120 | 038ff16e | 2026-04-13 | Hemil Desai | chore: Update GPT-OSS and Qwen3 recipe configs (#1811) |
| 121 | b2ee68f8 | 2026-04-13 | Dong Hyuk Chang | fix: Restrict auto-discovery scopes in generate_ci_tests.py (#1805) |
| 122 | e4b45412 | 2026-04-13 | Adil | fix: Coerce plain-dict backend to BackendConfig in model init (#1784) |
| 123 | ea971779 | 2026-04-12 | Harsha Pasham | fix: `NotImplementedError: aten::equal` on meta tensors during multi-GPU init (#1769) |
| 124 | 36ebdf34 | 2026-04-13 | Abhishree Thittenamane | fix: Add per-tensor conversion in gemma4 state_dict_adapter.py (#1764) |
| 125 | 79994aa1 | 2026-04-12 | Dong Hyuk Chang | feat: Enable benchmark CI testing with llm_benchmark and vlm_benchmark (#1793) |
| 126 | 75da116a | 2026-04-12 | Alexandros Koumparoulis | docs: update index (#1788) |
| 127 | 12c835ee | 2026-04-12 | Dong Hyuk Chang | docs: Add nightly CI test summary for LLM and VLM finetune configs (#1791) |
| 128 | 23f416f4 | 2026-04-11 | Huiying | feat: minimax m27 (#1785) |
| 129 | eb1e1c2b | 2026-04-10 | Alexandros Koumparoulis | fix: update yamls for vllm_deploy (#1780) |
| 130 | ac28e560 | 2026-04-10 | Hemil Desai | fix: add THD logit unsqueeze for GPT-OSS model (#1757) |
| 131 | fd74fd51 | 2026-04-10 | Dong Hyuk Chang | ci: Resolve cve and remove uv cache (#1774) |
| 132 | e45f6324 | 2026-04-10 | Alexandros Koumparoulis | ci: add missing recipe owners (#1775) |
| 133 | 2013a4dd | 2026-04-10 | Zhiyu Li | feat: FSDP2 w weight prefetching and async TP optimization (#1711) |
| 134 | 134d3066 | 2026-04-10 | Alexandros Koumparoulis | fix: launcher option from being consumed as a config override. (#1766) |
| 135 | 13893501 | 2026-04-10 | Alexandros Koumparoulis | fix: skip embedding[padding_idx] = 0 with TP (#1675) |
| 136 | b1268ac9 | 2026-04-10 | Hemil Desai | fix: MoE gate bias defaults and configurable gate_bias_update_factor (#1768) |
| 137 | 897ebedf | 2026-04-10 | Abhishree Thittenamane | fix: Allow use_cache when activation_checkpointing is True (#1726) |
| 138 | bb86cf55 | 2026-04-09 | Dong Hyuk Chang | ci: Update test timeout and add ci_tests readme (#1752) |
| 139 | f444fd2a | 2026-04-09 | Dong Hyuk Chang | ci: Address container and source code cve (#1753) |
| 140 | 9e65291e | 2026-04-09 | oliver könig | build: drop rc0 pre-release tag and add dynamic git versioning (#1729) |
| 141 | 1a1f68d3 | 2026-04-09 | Adil | fix: Baichuan2 checkpoint robustness test CI failures (#1727) |
| 142 | 4e28aed4 | 2026-04-09 | Huiying | docs: update brev links (#1751) |
| 143 | 27855ae0 | 2026-04-09 | Krishna Kalyan | docs: minor changes to tutorials (#1747) |
| 144 | f33d2d52 | 2026-04-09 | Abhishree Thittenamane | fix: Update lora configs for gemma4 (#1748) |
| 145 | 3a3f6858 | 2026-04-09 | Adil | test: add vLLM deployment tests for checkpoint robustness (#1656) |
| 146 | c7ba3045 | 2026-04-09 | Abhishree Thittenamane | feat: Add lora recipes for gemma4 (#1731) |
| 147 | 70c161d6 | 2026-04-08 | Dong Hyuk Chang | test: Checkpoint robustness skips atexit-registered destroy_process_group() (#1730) |
| 148 | 71b5ad19 | 2026-04-08 | Dong Hyuk Chang | ci: Address timeout is ci tests (#1733) |
| 149 | 6ba40748 | 2026-04-08 | Huiying | fix: Qwen3.5 dense CP support and FSDP mixed-dtype fix (#1710) |
| 150 | 9482d7db | 2026-04-09 | ooo oo | fix(docker): replace deprecated pynvml with nvidia-ml-py (#1725) |
| 151 | 104fcc08 | 2026-04-08 | Alexandros Koumparoulis | fix: propagate brev link to docs (#1735) |
| 152 | bef1b019 | 2026-04-08 | Dong Hyuk Chang | ci: Update install test scope (#1697) |
| 153 | 9be6dafe | 2026-04-08 | Dong Hyuk Chang | ci: Remove relative path in codcov with explicit path (#1695) |
| 154 | 44588063 | 2026-04-08 | rnyak | fix: fixing the pooling error for non-llama models for biencoder training (#1645) |
| 155 | bde81d11 | 2026-04-08 | Krishna Kalyan | feat: nemotron Parse fine-tuning notebook and assets (#1655) |
| 156 | 80e81b65 | 2026-04-07 | Pranav Thombre | feat: add save_checkpoint_every_epoch flag and max_steps support for … (#1723) |
| 157 | ea6691ac | 2026-04-07 | Alexandros Koumparoulis | fix: mute warning spam (#1721) |
| 158 | da3ec826 | 2026-04-07 | Hemil Desai | feat: add dynamic sequence length support for pipeline parallelism (#1689) |
| 159 | 193e325f | 2026-04-07 | Huiying | docs: update model coverage/docs for glm5.1 (#1720) |
| 160 | 33d5ee5f | 2026-04-07 | Adil | fix: handle dict-typed chat_template in format_chat_template (#1696) |
| 161 | 786ee96c | 2026-04-07 | Hemil Desai | fix: resolve PT 2.11 DeviceMesh deprecation warnings and unify EP mesh (#1684) |
| 162 | 924d6fac | 2026-04-07 | linnan wang | feat: adding lora to diffusion (#1653) |
| 163 | 6cb5804c | 2026-04-06 | Hemil Desai | feat: MoE model benchmarks, LoRA configs, and flops calculators (#1676) |
| 164 | 44e86dc5 | 2026-04-06 | Hemil Desai | feat: integrate NeMo-Run launcher for managed job submission (#1668) |
| 165 | b9a2154e | 2026-04-06 | Dong Hyuk Chang | ci: Update version to 0.4.0 (#1703) |
| 166 | 3fadac96 | 2026-04-06 | Huiying | fix: freeze dead KV-sharing params to fix checkpoint resume (#1698) |
| 167 | e68cbe10 | 2026-04-06 | Dong Hyuk Chang | ci: Remove duplicate ci config (#1702) |
| 168 | cadbba77 | 2026-04-06 | Adil | test: add checkpoint robustness functional tests (#1606) |
| 169 | 62d2f8d3 | 2026-04-06 | Dong Hyuk Chang | ci: Associate recipe owners (#1690) |
| 170 | 50180392 | 2026-04-06 | Adil | fix: swap DTensor shard placements after transpose in Step3p5 state dict adapter (#1691) |
| 171 | 263fcb14 | 2026-04-06 | Huiying | feat: enable packed sequences for Qwen3.5-MoE with EP+PP (#1685) |
| 172 | 1c3944af | 2026-04-06 | stanley1208 | feat: implement NEFTune noisy embeddings for instruction fine-tuning (#1686) |
| 173 | 15fb1210 | 2026-04-06 | Pranav Thombre | feat: Add dllm generation support (#1692) |
| 174 | 3eb5bc07 | 2026-04-06 | Somshubra Majumdar | feat: Allow to conditionally skip malformed jsonl lines when loading dataset (#1694) |
| 175 | f7afa1f8 | 2026-04-06 | Dong Hyuk Chang | ci: Add code freeze workflow (#1688) |
| 176 | a3cac3da | 2026-04-05 | Alexandros Koumparoulis | docs: add per-model pages (#1683) |
| 177 | 91c6e411 | 2026-04-05 | Dong Hyuk Chang | ci: Set target version for ruff (#1636) |
| 178 | 4ecba761 | 2026-04-05 | stanley1208 | fix: add best_metric_key field to CheckpointingConfig dataclass (#1641) |
| 179 | 0dea775e | 2026-04-04 | Alexandros Koumparoulis | docs: update the finetune guide (#1678) |
| 180 | 34097e46 | 2026-04-04 | Pranav Thombre | feat: Add Llada SFT support (#1672) |
| 181 | e9bee96a | 2026-04-03 | Zeel Desai | feat: add reasoning_content and tool-calling support to ChatDataset (#1644) |
| 182 | 86c686b8 | 2026-04-03 | Alexandros Koumparoulis | fix: move .claude/skills to skills (#1673) |
| 183 | 6f2643a4 | 2026-04-03 | Alexandros Koumparoulis | fix: add tp plan for phi2 (#1674) |
| 184 | 90f625ff | 2026-04-03 | Hemil Desai | feat: add cp2 convergence configs and eval fixes (#1602) |
| 185 | cccf771e | 2026-04-03 | svcnvidia-nemo-ci | chore(beep boop 🤖): bump FW-CI-templates workflow pins to v0.88.0 (#1669) |
| 186 | 4ebde178 | 2026-04-03 | Adil | feat: context-parallel with nemotron v3 (#1441) |
| 187 | 10a3b034 | 2026-04-03 | zyzhou5 | feat: Add discrete diffusion LLM (dLLM) supervised fine-tuning support (#1665) |
| 188 | 7d9b3f78 | 2026-04-03 | Huiying | fix: update gemma4 configs and doc with correct model IDs (#1670) |
| 189 | ca68ef77 | 2026-04-02 | Zakir Jiwani | fix: Finetune DeepSeek V3 (issue #1496) (#1654) |
| 190 | 5792a9b3 | 2026-04-02 | Huiying | fix: Mistral4 FP8 dequant on multi-dim mesh (#1594) |
| 191 | cefa53f9 | 2026-04-02 | Hemil Desai | feat: add HybridEP example config for Qwen3-30B-A3B (#1666) |
| 192 | a3a59d7b | 2026-04-02 | Alexandros Koumparoulis | fix: link in readme. (#1664) |
| 193 | f7b98a29 | 2026-04-02 | Dong Hyuk Chang | ci: Add recipe golden values (#1647) |
| 194 | 6606a0f5 | 2026-04-02 | Huiying | feat: add gemma 4  (#1660) |
| 195 | bd9e79ce | 2026-04-02 | Alexandros Koumparoulis | fix: move skills to .claude/skills (#1662) |
| 196 | c1b78f14 | 2026-04-02 | Huiying | feat: add gemma4 configs (#1658) |
| 197 | a8bdc4fb | 2026-04-02 | Huiying | docs: add gemma4 tutorial (#1657) |
| 198 | 48d18d8c | 2026-04-01 | Hemil Desai | feat: GPT-OSS 20B and Moonlight 16B convergence results (#1577) |
| 199 | ec2f7240 | 2026-04-01 | Hemil Desai | fix: Float32RMSNorm torch.compile crash on PyTorch 2.11+ (#1650) |
| 200 | 980f23d5 | 2026-04-01 | Adil | feat: enable TE Linear layers for PEFT/LoRA (#1626) |
| 201 | 67b9bca1 | 2026-04-01 | Hemil Desai | feat: add UCCL-EP as alternative dispatcher for expert parallelism (#1635) |
| 202 | 784b1f84 | 2026-04-01 | Adil | fix: skip initialize_weights for Phi3ForCausalLM with TP sharding (#1648) |
| 203 | 9a0e0df3 | 2026-04-01 | Dong Hyuk Chang | ci: Update mistral4 medpix ci run time (#1646) |
| 204 | dbe0f65d | 2026-04-01 | Dong Hyuk Chang | ci: Update run time for nemotron super ci (#1614) |
| 205 | fedc9d2b | 2026-04-01 | Alexandros Koumparoulis | feat: add missing recipe in yaml (#1642) |
| 206 | 35b5ed9c | 2026-04-01 | Alexandros Koumparoulis | refactor: CLI app and launching (#1406) |
| 207 | 6264a707 | 2026-04-01 | Huiying | cp: feat: VLM pretokenized data pipeline with neat packing (#1618) |
| 208 | 84aee9bf | 2026-03-31 | stanley1208 | fix: remove redundant _keep_in_fp32_modules for layer norms in GptOssForCausalLM (#1633) |
| 209 | e0308022 | 2026-03-31 | Alexandros Koumparoulis | feat: add AGENTS.md (#1638) |
| 210 | 4e8649b2 | 2026-03-31 | Dong Hyuk Chang | ci: Add deleted files explicitly in coverage omit (#1637) |
| 211 | 54cbe820 | 2026-03-31 | Alexandros Koumparoulis | fix: tied embedding v4 to v5 (#1631) |
| 212 | 5d5bf848 | 2026-03-31 | Hemil Desai | feat: add hybridep (#1333) |
| 213 | 175e5edd | 2026-04-01 | alexchiu | fix: from_pretrained with nested kwargs (e.g. text_config) crashes on VLM models (#1623) |
| 214 | 21d97e48 | 2026-03-31 | Dong Hyuk Chang | ci: Pass argument automodel dir for transformer version check (#1617) |
| 215 | 45f08465 | 2026-03-30 | Pranav Thombre | feat: Ensure that diffusion training jobs use the safetensors checkpoint format (#1627) |
| 216 | 9fe278a2 | 2026-03-30 | David O'Neil | feat: add Nemotron Nano 4B SQuAD finetune recipe (#1624) |
| 217 | 9ab601ba | 2026-03-30 | Pranav Thombre | feat: Migrate diffusion recipe to use Stateful Dataloader (#1630) |
| 218 | 92635e74 | 2026-03-31 | Yuki Huang | fix: fix gradient_checkpointing overhead in transformers 5.3 (#1621) |
| 219 | b1696470 | 2026-03-31 | Yuki Huang | fix: fix NemotronHForCausalLM force_hf=True (#1625) |
| 220 | e5cd7b23 | 2026-03-30 | Adil | feat: add reranker training (#1449) |
| 221 | 136bd8c7 | 2026-03-30 | Huiying | docs: update coverage doc (#1609) |
| 222 | eb90feb7 | 2026-03-30 | Charlie Truong | ci: Enable CI variables for changing lint runner and container (#1619) |
| 223 | 177b4cf6 | 2026-03-27 | Dong Hyuk Chang | ci: Set upperbound for transformers (#1615) |
| 224 | e766d71a | 2026-03-27 | Aditya Saxena | feat: add SkyPilot as a cloud execution backend for AutoModel (#1590) |
| 225 | ededdc0a | 2026-03-26 | Hemil Desai | feat: add Tulu-3 E2E convergence pipeline (#1554) |
| 226 | ee4752ef | 2026-03-26 | Dong Hyuk Chang | ci: Update vllm_finetune ci config (#1611) |
| 227 | 9db8c1f5 | 2026-03-25 | Huiying | fix: resolve TP+PP for nemotron super 49B (#1607) |
| 228 | 3e338449 | 2026-03-25 | Dong Hyuk Chang | ci: Update llm_finetune recipes for ci (#1608) |
| 229 | c88d24e2 | 2026-03-25 | Dong Hyuk Chang | fix: Remove duplicate keys in recipes (#1605) |
| 230 | f09ff4c0 | 2026-03-24 | Adil | fix: resolve deadlock saving diffusion checkpoints in safetensors format (#1601) |
| 231 | 24fc504d | 2026-03-24 | Dong Hyuk Chang | ci: Add ci_tests to tests folder (#1596) |
| 232 | b2dcbb68 | 2026-03-24 | oliver könig | ci: upgrade GitHub Actions for Node.js 24 compatibility (#1593) |
| 233 | 8b11622f | 2026-03-24 | Yuki Huang | fix: fix tp plan lookup (#1600) |
| 234 | 492add84 | 2026-03-23 | Hemil Desai | fix: narrow model.to(device) skip to checkpoint-loaded path only (#1597) |
| 235 | 383ab839 | 2026-03-23 | Hemil Desai | fix: convert DTensor biases to local in MoE _forward_loop (#1565) |
| 236 | b858b949 | 2026-03-23 | Huiying | perf: simplify Qwen3.5-MoE state_dict_adapter + DTensor passthrough (#1589) |
| 237 | 63a5774a | 2026-03-23 | Alexandros Koumparoulis | ci: add @pthombre to codeowners (#1588) |
| 238 | 0ce86952 | 2026-03-24 | Yuki Huang | fix: remove in-place change model config (#1595) |
| 239 | 90962b6f | 2026-03-20 | Alexandros Koumparoulis | docs: merge tables (#1587) |
| 240 | 4becc007 | 2026-03-20 | Pranav Thombre | docs: Add docs about diffusion support in AM (#1495) |
| 241 | e66d1ed8 | 2026-03-19 | Hemil Desai | perf: simplify Qwen3-VL-MoE state_dict_adapter + use torch hf reader (#1570) |
| 242 | b21cdfaf | 2026-03-20 | Sepehr Sameni | feat: add pipeline parallelism support for knowledge distillation (#1500) |
| 243 | 222f4870 | 2026-03-19 | Huiying | fix: register kimi_k25 and kimi_vl configs eagerly in lazy registry (#1579) |
| 244 | 2297b1a1 | 2026-03-19 | Dong Hyuk Chang | ci: Move source install fla to dev group (#1580) |
| 245 | ea58f0f4 | 2026-03-19 | Alexandros Koumparoulis | fix: checkpointing for PEFT. (#1576) |
| 246 | a3987c94 | 2026-03-19 | Piotr Żelasko | fix: Nemotron v3 inputs_embeds generation (#1583) |
| 247 | a06f190b | 2026-03-19 | Dong Hyuk Chang | ci: Update coverage path and fix coverage upload (#1582) |
| 248 | 8138965f | 2026-03-19 | Alexandros Koumparoulis | docs: update finetune guide  (#1548) |
| 249 | e0016dea | 2026-03-19 | Dong Hyuk Chang | Revert "fix" |
| 250 | f3baba44 | 2026-03-19 | Alexandros Koumparoulis | fix |
| 251 | e43a3b30 | 2026-03-19 | alexchiu | feat: Add context parallel support for Qwen3.5 MoE (#1560) |
| 252 | 0d34829c | 2026-03-18 | Charlie Truong | ci: Fix sso user check (#1578) |
| 253 | 4ce0fab2 | 2026-03-18 | Pranav Thombre | feat: VDR feedback: Common inference utility (#1491) |
| 254 | 3cb68174 | 2026-03-18 | Huiying | fix: patch missing mock in meta-tensor retry test (#1575) |
| 255 | 4f883b48 | 2026-03-18 | Alexandros Koumparoulis | docs: add navigation table (#1573) |
| 256 | 51a05a03 | 2026-03-18 | Huiying | fix: enable Phi-4-multimodal-instruct VLM finetuning (#1552) |
| 257 | f46ea5e6 | 2026-03-18 | Alexandros Koumparoulis | fix: seq cls trainer (#1564) |
| 258 | ee495f3e | 2026-03-17 | Alexandros Koumparoulis | fix: kd inference mode (#1567) |
| 259 | cdd64282 | 2026-03-17 | Alexandros Koumparoulis | feat: input validation & model capability (#1542) |
| 260 | 64235427 | 2026-03-17 | Huiying | fix: fall back to HF for Mistral3 VLMs with non-Mistral4 text backbone (#1557) |
| 261 | 112c92bd | 2026-03-17 | Dong Hyuk Chang | ci: Update permissions for claude review workflow (#1562) |
| 262 | a70b138b | 2026-03-17 | Alexandros Koumparoulis | fix: lora test (#1561) |
| 263 | 6350332d | 2026-03-17 | Dong Hyuk Chang | ci: Add claude code review (#1545) |
| 264 | 2a4c2236 | 2026-03-17 | Hemil Desai | fix: GPT-OSS MoE aux_loss softmax and remove torch.compile from _apply_bias (#1559) |
| 265 | 4c9a0089 | 2026-03-17 | SwekeR | feat: MFU logging in train recipes (#1413) |
| 266 | c38ead48 | 2026-03-17 | Dong Hyuk Chang | ci: Updating testing path to /opt/Automodel, update codecov settings (#1544) |
| 267 | 20a91bfa | 2026-03-16 | Alexandros Koumparoulis | improve ux of peft |
| 268 | b6eb6599 | 2026-03-16 | Alexandros Koumparoulis | fix: add dynamic=True to Float32RMSNorm (#1555) |
| 269 | 89db21db | 2026-03-16 | Huiying | feat: add mistral4 recipe (#1556) |
| 270 | cc23e045 | 2026-03-17 | Rayen | fix: handle Nemotron V3 with force_hf=True in weight initialization skip logic (#1551) |
| 271 | 233b3776 | 2026-03-16 | Alexandros Koumparoulis | feat: add more example configs (#1553) |
| 272 | 8919251c | 2026-03-15 | Huiying | feat: model addition (#1550) |
| 273 | 385073c8 | 2026-03-16 | Zhiyu Li | fix: optimized TP plan lookup in NeMo-RL by qualname (#1547) |
| 274 | 53dcd406 | 2026-03-13 | Alexandros Koumparoulis | fix: replace pickle with torch.load(..., weights_only=True)  (#1546) |
| 275 | 3aff0b73 | 2026-03-12 | Logan Vegna | feat: Add native Comet ML experiment tracking (#1411) |
| 276 | b60ab9ee | 2026-03-12 | Pranav Thombre | feat: Integrate Wan with multi-resolution DL (#1475) |
| 277 | 25c24fa1 | 2026-03-12 | Huiying | docs: add large moe llm doc (#1541) |
| 278 | 983517c8 | 2026-03-12 | Dong Hyuk Chang | ci: Update uv lock codeowner and commit block (#1539) |
| 279 | 73834a5d | 2026-03-12 | Alexandros Koumparoulis | fix: baichuan .bin ckpt loading (#1515) |
| 280 | 3dc90a0e | 2026-03-12 | Alexandros Koumparoulis | feat: add v4_compatible ckpt (#1532) |
| 281 | f2543667 | 2026-03-12 | Charlie Truong | fix: Revert uv.lock to fix install test with NGC Cuda (#1534) |
| 282 | 1c5714f8 | 2026-03-11 | Huiying | cp: feat: add neat packing (greedy knapsack) for LLM and VLM datasets (#1485) |
| 283 | d73a22fb | 2026-03-11 | Bambuuai | feat: Enable custom chat_template override for VLM fine-tuning (#1525) |
| 284 | 24669610 | 2026-03-11 | Alexandros Koumparoulis | ci: add default env vars ala .github/actions/test-template/action.yml L120 (#1523) |
| 285 | aa149c61 | 2026-03-11 | Alexandros Koumparoulis | fix: de-pickle (#1517) |
| 286 | ab3a59a1 | 2026-03-11 | Alexandros Koumparoulis | ci: improve functional test msg (#1524) |
| 287 | 482c2c60 | 2026-03-11 | Alexandros Koumparoulis | feat: update readme (#1531) |
| 288 | 60b9b2a2 | 2026-03-11 | Hemil Desai | feat: Add GLM 5 implementation (#1372) |
| 289 | c0e4fe1b | 2026-03-11 | Adil | feat: Super V3 (#1522) |
| 290 | 722a54cf | 2026-03-10 | Hemil Desai | fix: construct rope_parameters fallback for MiniMaxM2 (#1518) |
| 291 | cd8d9d40 | 2026-03-10 | Alexandros Koumparoulis | fix: TP paralellizer with replicated qkvs (#1519) |
| 292 | 31e0fe76 | 2026-03-10 | Alexandros Koumparoulis | fix: gpt-oss ckpt saving (#1501) |
| 293 | 4605fada | 2026-03-10 | Hemil Desai | feat: add MoE expert diversity metrics (#1506) |
| 294 | 3ec2366d | 2026-03-10 | Huiying | feat: add new score func and pp microbatch pixel split handling (#1513) |
| 295 | 7108b8eb | 2026-03-10 | Hemil Desai | fix: attach CP attention-mask hooks for dense (non-TE) context parallelism (#1470) |
| 296 | 03b8f91e | 2026-03-10 | Hemil Desai | feat: add FlashOptim optimizer integration (#1492) |
| 297 | 0e9c11ec | 2026-03-10 | Abhishree Thittenamane | fix: Log exception and error in FirstRankPerNode before exiting (#1468) |
| 298 | 4d50a85d | 2026-03-09 | Huiying | fix: forward-compatible _patched_get_init_context for transformers v5.3.0 (#1504) |
| 299 | bd513636 | 2026-03-09 | Alexandros Koumparoulis | fix: make MistralCommonBackend inherit from PreTrainedTokenizerBase (#1505) |
| 300 | 30fbb000 | 2026-03-10 | Sepehr Sameni | feat: TP-aware KDLoss with distributed softmax and T² scaling (#1499) |


## Appendix B: Top authors by commit count

| Author | Commits |
| --- | --- |
| Alexandros Koumparoulis | 65 |
| Dong Hyuk Chang | 56 |
| Huiying | 38 |
| Hemil Desai | 25 |
| Adil | 20 |
| khazzz1c | 14 |
| Pranav Thombre | 12 |
| Abhishree Thittenamane | 8 |
| oliver könig | 7 |
| Charlie Truong | 5 |
| ooo oo | 4 |
| stanley1208 | 4 |
| Yuki Huang | 4 |
| alexchiu | 3 |
| zyzhou5 | 3 |


## Appendix C: Reproduction commands

```bash
git fetch --all --prune --tags
git log --first-parent --max-count=300 --format='%H %aI %an %s' upstream/main
git diff --shortstat 19f6043b93a0d70b4eed1118a12c85ecaa78e555..17ed5796bdc220c314c9fd6bd718a773a3642521
git diff --numstat --find-renames 19f6043b93a0d70b4eed1118a12c85ecaa78e555..17ed5796bdc220c314c9fd6bd718a773a3642521
git diff --name-status --find-renames 19f6043b93a0d70b4eed1118a12c85ecaa78e555..17ed5796bdc220c314c9fd6bd718a773a3642521
```
