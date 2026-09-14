# RoboTwin 节点 G05 判据 A 复验记录

2026-09-15。范围：仅 `RoboTwin_benchmark_Node/` 仓库（本文档所在仓库）；XPolicyLab、
RoboTwin 平台仓库、forge_runtime、RoboDojo 节点等一律未动。

## 一、结论

**判据 A 通过**：G05（Galaxea G0.5，g05-robotwin20 checkpoint）在官方 runner 与
Forge 两节点（本节点）同 checkpoint、同 task、同 episodes 口径下，逐 task Wilson
95% CI 重叠且 Fisher 双侧 p > 0.05；对齐子集 McNemar 配对 p = 1.0。

| task | 官方 runner | Forge 两节点 | Wilson 95% CI | Fisher p | 对齐子集 McNemar p |
|:---|:---|:---|:---|:---|:---|
| adjust_bottle | 50/50 = 100% | 50/50 = 100% | 官方 [92.87%, 100%] vs Forge [92.87%, 100%]（完全重叠） | 1.0000 | 1.0000（b=0, c=0） |
| beat_block_hammer | 48/50 = 96% | 50/50 = 100% | 官方 [86.54%, 98.90%] vs Forge [92.87%, 100%]（重叠） | 0.4949 | 1.0000（b=0, c=1） |
| **汇总** | **98/100** | **100/100** | — | — | — |

对齐子集（43 个共同种子，含失败集）：官方 42/43 vs Forge 43/43，Fisher p = 1.0000。
两侧种子集不完全一致（各独有 7 个）是专家预筛
不确定性的预期表现（同 seed 两次 play_once 可能一成一败），验收对齐以协议同构为准，
与 2026-09-12 X_VLA 验收口径一致。

判定工具：`tools/acceptance_stats.py --acc-dir ~/robotwin_deps/acceptance_g05
--forge-dir results/acceptance_g05`。

## 二、口径与环境

- 两侧共用同一份部署配置 `configs/g05_policy_deploy.yml`（`policy_name G05`、
  `protocol ws`、`action_type joint`、`action_dim 14`、`env_cfg_type aloha_agilex`、
  `eval_embodiment robotwin`、`action_source auto`→fm、`frequency 30`）。
- checkpoint：HF OpenGalaxea/G05 `g05-robotwin20`（11.44GB，
  sha256 `300d24c691f8d76846a218b9bd670312cf577b701ea5863f221af1b8d1c0d637`）；
  FM 连续头（discrete_action=false, continuous_action=true）。
- 官方侧：`setup_policy_server.py`（G05 py3.10 env，torch 2.7.1 cu126）+
  官方 `scripts/eval_policy_xpolicylab.py --task_config demo_clean --test_num 50
  --seed 0 --additional_info "ckpt_name=g05-robotwin20,action_type=joint"`。
- Forge 侧：同一 deploy yml 起独立 server（另一端口）+ `dora run` 两节点闭环，
  benchmark 配置 task_ids "0-1"、init_state_ids "0-49"、qpos、expert_check + seen、
  test_num 50。
- GPU：官方 server/eval 各一张，Forge server/benchmark 各一张，共 4 张。

## 三、冒烟与运行时核对

判据 A 前小规模冒烟（官方侧 2 task × 2 init-state）：adjust_bottle 2/2（首集 110
步成功）、beat_block_hammer 2/2（首集 103 步成功）。服务端加载日志确认：
`[G05] loaded ckpt=…g05-robotwin20 … embodiment=robotwin … action_source=fm
continuous_action=True`，run-dir 侧挂件（action_tokenizer.pt 软链、hf_processor、
dataset_stats.json）全部生效。

## 四、本次踩坑与修复（环境侧，仓库代码零改动）

1. G05 serve 路径缺 `gitpython`（`g05/utils/data/normalizer.py` 用）→ 最小依赖补装。
2. **`scripts` 包遮蔽**：XPolicyLab 与 RoboTwin 仓库根各有 `scripts/` 目录（无
   `serve_policy.py`），服务端 PYTHONPATH 若包含它们会抢先遮蔽 g05_root 的
   `scripts/serve_policy.py`（`ModuleNotFoundError: No module named
   'scripts.serve_policy'`）。修复：服务端 PYTHONPATH 只放
   `<g05_root>/src:<g05_root>`，XPolicyLab 靠 editable 安装提供。
3. `real_policy.py` 的 `robotwin_root` **不展开 `~`**（与 benchmark 节点 config.py
   不同），policy 运行时配置必须写绝对路径。README §3.2/§3.4 示例已加注修正。
4. 镜像节流（hf-mirror 漏桶）导致 11.4GB ckpt 下载多轮停滞：最终靠 aria2c 8 连接
   + 停滞即重启拿新突发窗口 + 落盘 sha256 校验完成；期间曾误判文件空洞（SEEK_HOLE
   探测逻辑写反），修正后确认实缺 112MB、续传补齐。

## 五、代码版本

- 代码与测试在 commit ①（`5f7ba33`，G05 ws 后端 + 契约测试 + 部署配置）之上**无
  代码改动**；本记录随 README §3.2/§3.4 的 `robotwin_root` 绝对路径加注与 deploy
  yml 注释同步一起提交。
- 提交前回归：pytest 88/88；`sha256sum -c code_manifest.txt` 28/28 OK；
  `_core/`、`bench_tool/` 零 diff。
