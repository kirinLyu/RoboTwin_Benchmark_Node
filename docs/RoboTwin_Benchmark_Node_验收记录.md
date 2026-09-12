# RoboTwin Benchmark Node 验收记录（判据 A）

> **2026-09-12 修订**：本文档 2026-09-10 的原始判定（✅ 通过）已被同事复查
> （`RoboTwin_检查验收报告912.md`，基于 commit `70ca263`）推翻并**撤回**；7 项
> 问题逐条核实与整改见「复查与整改」章。整改后的判据 A 正式复跑**通过**，见
> 「修复后判据 A 复跑」章。

## 基本信息（2026-09-10 原始验收）

| 项 | 内容 |
| :--- | :--- |
| Benchmark | RoboTwin 2.0（[官方仓库](https://github.com/RoboTwin-Platform/RoboTwin)，论文 arXiv:2506.18088） |
| 任务 / 集数 | `adjust_bottle`、`beat_block_hammer`；官方 runner 与 Forge 各 2 × 50 = 100 集 |
| 权重 | X_VLA co-train `ckpt-30000`（`TianxingChen/RoboTwin2.0`，model.safetensors sha256 `39d60eeb7c8b…`） |
| 论文已公开成绩（Easy） | adjust_bottle **100**、beat_block_hammer **92**（50 任务均值 68.0） |
| Robot / Action | aloha-agilex / ee（16D = 每臂位姿 + 夹爪） |
| task_config / seed | demo_clean / 0（两侧一致） |
| 场景是否同一批 | **否**（原记录"是"被复查推翻：init_state_id→seed 直映 + UnStableError 原地 +1 导致跨集种子碰撞；修复后改为官方顺序种子队列） |

判据 A：同一官方免训练 checkpoint、同两 task、同 episode 数，分别经官方 runner
（`scripts/eval_policy_xpolicylab.py` + XPolicyLab ws server）与 Forge 两节点链路
执行，比较成绩（Wilson 95% CI 重叠 + Fisher 双侧精确检验）。

## 原始成绩对照（判定已撤回）

| 路径 | 成功 / 总数 | 成功率 | Wilson 95% CI |
| :--- | ---: | ---: | :--- |
| 官方 runner | 99 / 100 | 99.0% | [94.55%, 99.82%] |
| Forge 节点 | 95 / 100 | 95.0% | [88.82%, 97.85%] |
| **差异** | 4 集 | 4.0 pp | **Fisher p = 0.2116** |

原始判定「✅ 通过」**撤回**。撤回原因（复查证实）：① 场景集口径与官方不同
（问题 1/2，见下章）；② 当时的 seed 对齐工具只保留成功集（失败集 metrics 为空、
无 seed），对齐子集恒 100%、p=1.0 属**结构性偏置假象**。原始批次的其余事实记录
（批次结果 `success_rate=0.95`、0 traceback、确定性对照、接入过程修复清单等）
压缩保留于文末「历史记录摘要」。

## 复查与整改（同事报告 7 项逐条）

7 项核实：**6 项属实、1 项部分成立**。

| # | 报告问题 | 处置 |
| :-- | :-- | :-- |
| 1 | `UnStableError` 原地 +1 重试与 init_state_id→seed 直映冲突，跨集种子碰撞 | ✅ 官方顺序种子队列：每 (task, run_index) 一个游标只增不减，接受即推进、跳过不回收 |
| 2 | 未实现官方 expert_check | ✅ `expert_check: true` 默认开：setup_demo → play_once → close_env 预筛，通过后重新 setup 同一 seed（官方同款两次 setup） |
| 3 | seed 对齐统计只保留成功集，证据无效 | ✅ 失败集 seed 经批次级 `metrics.scene_seed_ledger` 补全；对齐含失败集 + McNemar 配对检验 |
| 4 | 指令为 full_description，官方为 seen | ✅ `instruction_type: seen` 默认：play_once episode_info → 官方 `generate_episode_descriptions` 填充（导入路径与官方一致：description/utils） |
| 5 | policy 节点为仓库内新写件，非"未改动 Forge policy 节点" | ✅ README §9 明确来源与契约（通道别名/state_joints/JointCommand 四对齐）；本机无生产 Forge policy 节点，端到端验证留待 Forge 环境 |
| 6 | episode_seed 非真实场景种子；失败集无 seed | ✅ 失败集场景 seed 由台账补全（改 `_core` 直写违反共享核心零漂移约束，台账达到同等可定位性） |
| 7a | env 跨 episode 不 close_env | ✅ 每集一次 close_env（下一集指令阶段开头，官方每集 close 同口径） |
| 7b | real_policy chunk 边界可能用旧帧 | ✅ 逐通道帧计数（proprio_count/head_count）保证推理用同一帧 proprio + head；推理节奏改官方同款（每步 update_obs、chunk 耗尽才 get_action） |

对报告中有误/不可达之处的回应：

1. **"两次运行逐 seed 相同"官方自身也做不到**：报告 §3.4 实测（同 seed 两次
   play_once 一成一败、官方两次运行计入 seed 集不同）证明官方 eval 不 seed 任何
   RNG、expert 预筛不确定。整改目标定为**协议同构**（同样顺序种子队列、同样
   expert 跳过口径、同样 seen 生成器）——可达的最强口径。
2. **报告 §4 的自我纠错反而支持原成绩差是噪声**：同场景对照中 seen vs
   full_description 无系统性效应；原 4pp 差异来自场景集不同（问题 1/2，现已修）
   + bb 重尾随机。报告"n=50 功效不足"批评成立，原"统计等价"表述确属过度断言。
3. **问题 6 的 episode_seed 直写要求需改 `_core/result.py`**：`_core` 是共享调度
   核心（与 benchmark-libero 逐字一致），台账方案在不动核心前提下达到同等可定位性。

修复后验证：

| 项 | 结果 |
| :-- | :-- |
| 契约测试 | ✅ **73 passed**（新增 6 个协议回归：seed 队列无重复 + 每集 close / expert 预筛与两次 setup / seen 官方生成器 / 台账注入 / 配置解析） |
| manifest 复验 | ✅ 28/28 OK（见「最终代码版本」） |
| 判据 A 正式复跑 | ✅ 通过（见下章） |

## 修复后判据 A 复跑（2026-09-12，官方 vs Forge，协议同构）

16:26–17:10 同机并行复跑：各 2 task × 50 集，ckpt-30000 / demo_clean / seed 0 /
ee；6 卡 4 路并行，全程 0 traceback。复跑日志与合并结果 JSON 归档于交付包
`acceptance_rerun/`（官方日志、Forge 双批日志、`robotwin_merged.json`）；Forge
原始结果 JSON 在本机部署副本 `results/acceptance_t0|t1/`（results/ 为 gitignored
运行产物，不随仓库分发）。

成绩对照（expert_check + seen + 顺序种子队列）：

| task | 官方 | Forge | Wilson 95% CI（官方 / Forge） | Fisher p |
| :--- | ---: | ---: | :--- | ---: |
| adjust_bottle | 50/50 = 100.0% | 50/50 = 100.0% | [92.87%, 100.00%] / 同 | 1.0000 |
| beat_block_hammer | 48/50 = 96.0% | 45/50 = 90.0% | [86.54%, 98.90%] / [78.64%, 95.65%] | 0.4360 |
| **合计** | **98/100 = 98.0%** | **95/100 = 95.0%** | [93.00%, 99.45%] / [88.82%, 97.85%] | **0.4448** |

对齐子集比较（含失败集，台账补全真实 seed）：

| task | 共同种子 | 对齐成绩 | Fisher p | McNemar（b / c）p |
| :--- | ---: | :--- | ---: | ---: |
| adjust_bottle | 50 | 50/50 vs 50/50 | 1.0000 | b=0, c=0 → 1.0000 |
| beat_block_hammer | 45 | 43/45 vs 40/45 | 0.4340 | b=5, c=2 → 0.4531 |

- Forge bb 失败集真实 seed（台账补全）：100002 / 100034 / 100037 / 100046 /
  100063，均 400 步 step_lim 内未完成（官方侧亦失败 2 集），属重尾波动。
- bb 共同种子 45/50：两侧 expert 预筛各自跳过 5 个不同种子（官方独有
  [100016, 100025, 100039, 100060, 100061]；Forge 独有
  [100005, 100045, 100066, 100067, 100068]）。expert 预筛与推理均为随机过程，
  逐 seed 全等不可达（官方自身两次运行亦不同），协议同构 + 对齐子集为最强
  可验证口径。
- 协议同构日志实据：两侧种子队列同步推进、跳过量级一致（官方 bb 消费至 100064、
  Forge 至 100068）；Forge `skip unstable seed=… (expert setup)` / `seed accepted
  … retries=N` 与官方 `skip unstable seed=… (expert check)` 同语义；seen 指令出自
  官方 generate_episode_descriptions；每集 close_env 无遗漏。

复跑判定：

| 条件 | 结果 |
| :--- | :--- |
| 官方 baseline 明确非零（论文 Easy 100 / 92） | ✅ |
| 同 checkpoint（sha256 `39d60eeb…`） | ✅ |
| 同 task、同 episode 数（2 × 50） | ✅ |
| 同种子调度协议（顺序种子队列 + expert 预筛 + seen 指令，两侧同构） | ✅ |
| 两边 Wilson 95% CI 重叠 | ✅ |
| Fisher exact p > 0.05（合计 0.4448；分 task 1.0000 / 0.4360） | ✅ |
| 对齐子集（含失败集）Fisher p > 0.05 + McNemar 无方向性偏置 | ✅ |
| **最终判定** | ✅ **通过（判据 A，整改后复跑）** |

**结论**：同 checkpoint、同 task、同 episode 数、协议同构的种子调度下，官方
98.0% vs Forge 95.0%，差 3 集（全在 bb），Wilson CI 重叠、Fisher 双侧 p = 0.4448、
对齐子集含失败集 McNemar p = 0.4531——无统计可区分的系统性差异；整改前的结构性
偏置（场景集不同、对齐统计只含成功集）已消除。逐条回复与复跑细节见交付文档
`RoboTwin_整改回复.md` §六。

## 历史记录摘要（2026-09-10 原始验收，判定已撤回，事实保留）

- **批次结果**：Forge 100 集 status=succeeded，success_rate=0.95，
  elapsed 4281.4s，mean_policy_latency_ms=7.75；逐集
  `success_source = env.eval_success`；`metrics.resources` 峰值（cpu 556.0%、
  rss 3826 MB、gpu_util 84.0%、gpu_mem 9498 MB）；0 traceback。
- **交付前 gate**：契约测试 67 passed（整改后 73）、stub 时序闭环
  （obs=1600 act=1600 锁步）、真 policy 小规模 3/4=0.75 对论文基线、
  num_runs>1 差异 / 中途取消 / 资源记录。
- **确定性对照**（额外诊断证据，不影响判据 A）：同 seed 两次 reset 渲染逐位相同；
  异 seed / 异 run_index 渲染不同（51.6% 像素差，负对照可分辨）；跨 seed qpos
  相同（官方口径：每集回 homestate）；1:1 锁步 obs==act（真 policy 827/827、
  stub 1600/1600）。
- **接入过程发现并修复的 9 个问题**（websockets 依赖、资源采样器被 close 停表、
  dora canonicalize 弄丢 uv venv、envs 依赖 cwd=仓库根、embodiment 相对路径、
  policy_command queue_size、建景楔死取证等）已落入 README §5/§6。
- 与官方口径的 4 项历史差异（expert_check / 指令口径 / UnStableError 处理 /
  record_dir mp4）前 3 项已随整改消除，record_dir 仍为配置位保留（README §8）。

## 最终代码版本

交付件以 sha256 manifest 存档（28 个交付文件，仓库根 `code_manifest.txt`，
`sha256sum -c` 应全 OK）。2026-09-12 复验：整改（3 个文件哈希更新）与路径去
个性化（config.py `b60d4a32…`、tests/test_contracts.py `b5ce8fc2…` 二次更新）
后 manifest 已重算，**28/28 OK**，其余 26 个文件零漂移——判据 A 复跑批次运行的
就是该版本。

manifest 自身 sha256：`a0a9ae695a57a099bc8ea550a351d4c8e7f054539fc866fb197fb5d687de0b1e`
