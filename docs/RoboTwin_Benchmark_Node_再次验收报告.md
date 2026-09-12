# RoboTwin Benchmark Node 再次验收报告

## RoboTwin ✅ 通过（判据 A）

> 整改后的 RoboTwin Benchmark Node（commit `454cf57`）**满足判据 A**，RoboTwin 接入 Forge 的实现正确。
> 采用双方确认的判据 A 口径（协议同构 + 对齐子集含失败集 + McNemar）：
> 官方 runner **98/100 = 98.0%** vs Forge **95/100 = 95.0%**，Wilson 95% CI 重叠，
> Fisher 双侧精确检验 **p = 0.4448**，对齐子集（含失败集）官方 43/45 vs Forge 40/45、
> McNemar 配对检验 **p = 0.4531**——均无统计上可区分的系统性差异。

### 基本信息

| 项 | 内容 |
| :-- | :-- |
| Benchmark | RoboTwin 2.0 |
| 官方仓库 | https://github.com/RoboTwin-Platform/RoboTwin（上游 commit `96c1fea`） |
| 被验收仓库 | `RoboTwin_Benchmark_Node`（GitHub），commit `454cf57cbbe5dcdda9a3090fcc6d11a607d1af26`（整改版本） |
| 复核对象 | `RoboTwin_整改回复.md`（作者针对《RoboTwin 检查验收报告 912》意见的答复） |
| 任务 | `adjust_bottle`、`beat_block_hammer` |
| 权重 | X_VLA co-train `ckpt-30000` |
| Robot / Action Type | `aloha-agilex / ee`（16D = 每臂 xyz+wxyz 位姿 + 夹爪） |
| 该权重已公开成绩（论文 Easy） | `adjust_bottle` **100**、`beat_block_hammer` **92** |
| 集数（每边） | 50 / task × 2 task = 100 |
| Seed / task_config | `0` / `demo_clean` |
| 验收基准 | 《Benchmark 接入手册》v14 |
| 判据 A 口径 | 协议同构 + 对齐子集含失败集 + McNemar（双方确认） |
| 验收方式 | 拉取新版代码 + 静态审阅 + 单元/协议测试 + 独立端到端冒烟 + 对作者复跑数据独立离线重算 |
| 报告日期 | 2026-09-12 |

本次验收采用双方确认的**判据 A**：使用同一个官方免训练 checkpoint、同两个 RoboTwin task、同 episode 数，分别通过官方 runner（`scripts/eval_policy_xpolicylab.py` + XPolicyLab ws policy server）与 Forge 两节点链路执行，比较最终 benchmark 成绩。因官方 expert 预筛本身带随机性（同一 seed 两次 `play_once()` 可一成一败），场景集采用**协议同构**口径，并以**含失败集的对齐子集 + McNemar 配对检验**作补充判定。

Checkpoint SHA256（model.safetensors）：

```text
39d60eeb7c8b7e445b271f4c6aa7da5a1ee36bda5716bc6ae832a184468cad0b
```

### 成绩对照

| task | 官方 | Forge | Fisher(主) | 对齐共同种子(含失败集) | Fisher(对齐) | McNemar |
|:--|--:|--:|--:|:--|--:|--:|
| adjust_bottle | 50/50 = 100% | 50/50 = 100% | 1.0000 | 50/50 vs 50/50 | 1.0000 | b=0,c=0 → 1.0000 |
| beat_block_hammer | 48/50 = 96% | 45/50 = 90% | 0.4360 | **43/45 vs 40/45** | 0.4340 | b=5,c=2 → 0.4531 |
| **合计** | **98/100 = 98%** | **95/100 = 95%** | **0.4448** | — | — | — |

补充结果：

- Wilson 95% CI：官方 `[93.00%, 99.45%]` vs Forge `[88.82%, 97.85%]`，重叠。
- **场景唯一性（上轮问题 1 的回归验证）**：t0/t1 均为 **50 集 = 50 个唯一 scene seed，零重复**；上轮 adjust 只有 35 个唯一 seed。
- 协议日志证据：Forge 侧 `skip unstable seed=… (expert setup)`、`skip expert-fail seed=…`、`seed accepted … retries=N` 与官方 `skip unstable seed=… (expert check)` 同义；adjust 不稳定跳过 22 次（官方亦 22）；全程 0 traceback。
- 失败集真实 seed（台账补全，bb）：`100002 / 100034 / 100037 / 100046 / 100063`。

### 判定

| 条件 | 结果 |
|:--|:--:|
| 官方 baseline 明确非零（论文 Easy 100 / 92） | ✅ |
| 同 checkpoint（sha256 `39d60eeb…cad0b`） | ✅ |
| 同 task、同 episode 数（2 × 50） | ✅ |
| **协议同构**（顺序种子队列 + expert 预筛 + seen 指令生成器，两侧同规则） | ✅ |
| Wilson 95% CI 重叠 | ✅ |
| Fisher 双侧 `p > 0.05`（合计 0.4448） | ✅ |
| 对齐子集**含失败集**且配对检验无方向性偏置（McNemar p=0.4531） | ✅ |
| **最终判定** | **✅ 通过** |

**结论：RoboTwin Benchmark Node（整改后）通过正式验收（判据 A）。**

作者对上轮意见的整改经本次独立复核**成立**：

- 上轮三项硬伤（seed 冲突、未实现 expert_check、seed 对齐统计偏置）已根治；
- 指令口径、结果台账、每集 close、chunk 混帧等次要问题已修；
- 独立重算的判据 A 结果：官方 98/100 vs Forge 95/100，Wilson CI 重叠、Fisher `p=0.4448`、对齐子集（含失败集）官方 43/45 vs Forge 40/45、McNemar `p=0.4531`，无统计可区分的系统性差异。

### 为什么正确实现了 RoboTwin 接入 Forge

**（1）观测/动作接口与官方逐位一致（接口语义正确）**
上轮 Phase 3 的逐位对照（`artifacts/phase3_obs_parity.json`）已证：同 task/同 stable seed 下，
`to_state` 与官方 `convert_state`、`to_images` 与官方 `convert_vision`、16D ee 动作与官方 `xpolicylab_action_to_robotwin` **全部 bit-exact**；三路图像 240×320、不翻转；分辨率分层（原生 320×240 → policy 内缩 224）正确。本次整改未触碰这些转换逻辑。

**（2）场景调度与官方协议同构（复现语义正确）**
新版用官方 `now_seed` 同义游标：每 `(task, run_index)` 只增不减、接受即推进、跳过不回收；`expert_check` 严格照官方 `setup_demo → play_once → close_env → 重新 setup` 两步流程。回归结果 50 集 50 唯一 seed，与官方“接受过的 seed 永不重复”一致。

**（3）指令进入训练分布（口径正确）**
`instruction_type: seen` 默认，走官方 `generate_episode_descriptions`，导入路径与官方一致；实测下发的是场景化 seen 指令（如 “Grab the medium dark brown bottle head-up using the right arm…”），而非静态描述。

**（4）成功判定来自环境本身（判定正确）**
沿用 `env.eval_success`（`take_action` 内部逐 sim step 调 `check_success()`），逐集 `success_source=env.eval_success`；三态区分（`success` / `failed` / `timed_out`）保持。

**（5）结果可复现、可追溯（证据正确）**
每集真实场景 seed、跳过次数、指令写入批次级 `metrics.scene_seed_ledger`（含失败集），失败集 `metrics={}` 的三态契约不破；统计工具可按 `(task,task_init,run)` 补全失败集 seed——上轮“对齐子集只含成功集”的结构性偏置已消除。

**（6）统计结论稳健（结论正确）**
主检验（Fisher）与配对检验（McNemar）均 `p>0.05`，方向性不显著（bb：官方成/Forge 败 5 对，官方败/Forge 成 2 对）；Wilson CI 重叠。在 100 集规模下，**没有证据表明 Forge 节点系统性改变了 RoboTwin 的任务、场景、观测、动作与评分语义**。

**（7）分层与共享核心约束未破**
`_core/`、`bench_tool/`、`main.py` 未动，RoboTwin 全部差异收敛在 `adapters/robotwin_adapter.py`（符合手册 §8-12“只换 adapter”）；`episode_seed` 的共享核心字段未强改，改用台账达到同等可定位性。

### 最终候选代码 Runtime Smoke

用拉取后的 `main.py` + `stub_policy.py`，`expert_check: true` + `instruction_type: seen`，task 0/1 各 1 集：

```text
[robotwin_bench] skip unstable seed=100000 (expert setup)
[robotwin_bench] seed accepted task=adjust_bottle i0 r0 seed=100001 retries=1
[stub_policy] command=set_instruction instruction=
   'Grab the medium dark brown bottle head-up using the right arm from the table.'
[robotwin_bench] seed accepted task=beat_block_hammer i0 r0 seed=100000 retries=0
[stub_policy] command=set_instruction instruction='Take the hammer for nails and smash the block.'
批次 succeeded | obs=800 act=800 (1:1 锁步: True)
```

结果 JSON 台账：

```json
"scene_seed_ledger": [
 {"task_id":0,"init_state_id":0,"run_index":0,"scene_seed":100001,"seed_retries":1,
  "instruction":"Grab the medium dark brown bottle head-up using the right arm from the table."},
 {"task_id":1,"init_state_id":0,"run_index":0,"scene_seed":100000,"seed_retries":0,
  "instruction":"Take the hammer for nails and smash the block."}]
```

同时确认：失败集逐集 `metrics={}`（三态契约未破）、`success_source=env.eval_success`、台账补全场景 seed。

### 加分项：静态审阅与独立统计复核

**静态代码审阅**：逐行对照官方 `scripts/eval_policy_xpolicylab.py`（上游 commit `96c1fea`，未改动）审阅各项修复：

- `_advance_cursor()` / `episode_instruction()` 与官方 `now_seed` 游标、`while succ_seed < test_num` 循环（`eval_policy_xpolicylab.py:795-856`）同构；
- `expert_check` 的两次 setup（预筛一次 + 正式一次）与官方 `:805-843` 同款；
- `instruction_type` 的导入路径 `description/utils` 与官方 `:27` 一致；
- `_core/`、`bench_tool/`、`main.py` 未改动（`_core` 仍与 `benchmark-libero` 逐字一致，符合手册 §8-12 分层约束）。

**独立统计复核**：以作者复跑产物（`robotwin_deps/acceptance_rerun/`：官方日志 + `robotwin_merged.json`，及部署副本 `results/acceptance_t0|t1/`）为输入，**用验收方脚本与作者新版 `tools/acceptance_stats.py` 各算一遍**，结果一致（见「成绩对照」）。

### 接入过程中发现并修复的问题

| # | 上轮意见 | 整改 | 本次复核 |
|:--|:--|:--|:--:|
| 1 | `UnStableError` 原地重试与 `init_state_id→seed` 直映冲突，跨集场景重复 | 改官方顺序种子队列（每 `(task,run_index)` 游标只增不减） | ✅ |
| 2 | 未实现官方 `expert_check` | `expert_check: true` 默认，setup→`play_once`→close 预筛，通过后重新 setup | ✅ |
| 3 | seed 对齐统计只含成功集，证据无效 | 批次级 `metrics.scene_seed_ledger` 补全失败集 seed；对齐含失败集 + McNemar | ✅ |
| 4 | 指令为 `full_description`，官方为 `seen` | `instruction_type: seen` 默认，走官方 `generate_episode_descriptions` | ✅ |
| 5 | policy 节点为仓库内新写件，非「未改动 Forge policy 节点」 | README §9 澄清节点来源 + 四对齐契约文档化（生产 policy 节点端到端验证留待 Forge 环境） | ✅ |
| 6 | `episode_seed` 非真实场景 seed | 失败集场景 seed 走批次级台账（不动共享核心） | ✅ |
| 7a | env 跨 episode 不 close_env | 每集 close | ✅ |
| 7b | `real_policy` chunk 边界可能混用旧帧 | 逐通道帧计数 + 官方 update_obs/get_action 节奏 | ✅ |

### 最终代码版本

- `git pull --ff-only`：`70ca263 → 454cf57`（Fast-forward，11 文件）。
- `sha256sum -c code_manifest.txt`：**28/28 OK**（交付文件哈希与代码一致）。
- 将推送版关键文件与产生复跑数据的本机部署副本逐一哈希比对：

```text
adapters/robotwin_adapter.py  pushed=6a1a34672952a55e deployed=6a1a34672952a55e MATCH
config.py                     pushed=b60d4a3211e3097b deployed=b60d4a3211e3097b MATCH
real_policy.py                pushed=c7c81d54fed06b3d deployed=c7c81d54fed06b3d MATCH
tools/acceptance_stats.py     pushed=861cd3185dc31e95 deployed=861cd3185dc31e95 MATCH
tests/test_contracts.py       pushed=b5ce8fc217e4efb9 deployed=b5ce8fc217e4efb9 MATCH
```

→ **验收结论针对的代码 = 产生复跑数据的代码**（溯源闭环）。

Pre-commit gate：

```text
pytest -q                        →  73 passed（1 个 seen 生成测试依赖 ~/RoboTwin checkout；
                                    在含该 checkout 的环境下复跑亦 73 passed）
新增 6 个协议回归测试覆盖：顺序种子队列无重复 + 每集 close、expert 预筛与两次 setup、
seen 官方生成器、台账注入、expert_check/instruction_type 配置解析。
```

### 证据与产物路径

| 产物 | 路径 |
|:--|:--|
| 拉取后代码 | 本地验收副本（HEAD `454cf57`） |
| manifest 校验 | `sha256sum -c code_manifest.txt` → 28/28 OK |
| 单测 | `robotwin_acceptance/logs/`（73 passed） |
| 独立 stub 冒烟 | `robotwin_acceptance/logs/reaccept_stub_v2.log`、`results/stub_v2/robotwin_20260912_200709.json` |
| 作者新统计工具输出 | `robotwin_acceptance/reaccept/stats_tool_output.txt` |
| 作者复跑原始产物 | `robotwin_deps/acceptance_rerun/`（官方日志、`robotwin_merged.json`、forge_t0/t1 日志） |
| 上轮逐位对照证据 | `robotwin_acceptance/artifacts/phase3_obs_parity.json` |

### 最终结论

**RoboTwin Benchmark Node（commit `454cf57`）：✅ 通过，判据 A。**

正式证据：

```text
论文已公开成绩（Easy）：adjust_bottle 100、beat_block_hammer 92

官方 runner   ：98 / 100 = 98.0%，Wilson 95% CI [93.00%, 99.45%]
Forge         ：95 / 100 = 95.0%，Wilson 95% CI [88.82%, 97.85%]
Fisher 双侧   ：p = 0.4448
对齐子集含失败集：官方 43/45 vs Forge 40/45，Fisher p = 0.4340，McNemar p = 0.4531
场景唯一性    ：50 集 = 50 个唯一 scene seed（零重复）
```

在“协议同构 + 对齐子集含失败集 + McNemar”口径、100 集（2 task × 50）、同 checkpoint、同 robot/action type、同 seed=0、同 task_config 的测试条件下，没有证据表明 Forge Benchmark Node 系统性改变了 RoboTwin 的 benchmark 行为或成绩分布，满足判据 A。**Forge 两节点链路正确实现了 RoboTwin benchmark 的任务、场景、观测、动作与评分语义。**
