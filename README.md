# RoboTwin Benchmark 节点

把 RoboTwin 包一层的 Dora 节点。进程内持有 RoboTwin env，只做测评：批次展开 → 逐 episode 跑 → 结果落盘。不经过 XPolicyLab / policy server；判定、动作、重置全走 RoboTwin 官方接口，口径照抄官方 eval 脚本 `scripts/eval_policy_xpolicylab.py`。

```
benchmark 节点（本节点）                    policy 节点（不改）
├─ 进程内 import RoboTwin envs
├─ proprio_state / image/head / image/left_wrist / image/right_wrist ──观测──▶ 缓存 + tick 推理
├─ action ◀─────────────────────────────────────────────────动作── JointCommand 14D(qpos) / 16D(ee)
├─ policy_command ──────────────────────────────────────────命令──▶ start / set_instruction / reset_scene / stop
└─ 判定 env.eval_success；benchmark_status（流式）/ benchmark_result（批次末）
```

设计规范见《Benchmark 接入手册》v14，本节点是其参考实现的 RoboTwin 版本，与
`benchmark-libero` 同构（`_core/`、`bench_tool/`、`main.py` 除 adapter 与标识符外
一字未改）。

Dora 端口契约（拆仓不改变）：

- 输入：`action`（policy 的 `JointCommand`；qpos = 臂关节 + 夹爪 14D，ee = 位姿 + 夹爪 16D）；
- 输出：`proprio_state`（14D `JointState`）、`image/head`、`image/left_wrist`、
  `image/right_wrist`、`policy_command`、`benchmark_status`、`benchmark_result`。

---

## 0. 本仓库说明

独立测试仓库：把节点从 PhyAgentOS 环境拆出，供测试与复现。2026-09-12 验收复查
的 7 项问题已全部核实并修复：顺序种子队列（1）、expert_check 专家预筛（2）、
失败集种子台账 + 对齐统计修正（3/6）、seen 生成式指令（4）、policy 来源澄清（5）、
每集 close_env（7a）、policy 帧对齐（7b）。**修复后判据 A 正式复跑通过**（官方
98/100 vs Forge 95/100，Fisher p=0.4448，对齐子集含失败集 McNemar p=0.4531），
详见 `docs/RoboTwin_Benchmark_Node_验收记录.md`。

| 路径 | 内容 |
|:---|:---|
| `main.py` / `config.py` / `result.py` | 节点装配 / 配置模型 / 结果落盘 |
| `adapters/robotwin_adapter.py` | ★ RoboTwin 特有逻辑（换 benchmark 只改这里） |
| `_core/` `bench_tool/` `tests/` | 调度核心 / 服务模式 glue / 契约测试（照搬 benchmark-libero） |
| `config.example.yaml` `dataflow.example.yaml` | 运行配置与 dora 拓扑模板 |
| `stub_policy.py` `real_policy.py` `policy_backend.py` | 本地测试 policy（假 policy 冒烟 / X_VLA 真策略薄封装 / G05 ws 后端 + obs・action 口径转换） |
| `tools/` | 全 suite 跑分驱动、判据 A 统计工具 |
| `code_manifest.txt` | 28 个核心交付文件的 sha256 版本锚点（`sha256sum -c` 应全 OK） |

上游致谢：`_core/`、`bench_tool/`、`main.py` 骨架照搬
[benchmark-libero](https://github.com/Forgelab-Robotics/benchmark-libero)。

---

## 1. 环境

本节点进程内 `import envs`，运行环境须装好 RoboTwin 仿真依赖（sapien 3.0.0b1 /
torch cu121 / mplib / toppra / trimesh / opencv-headless / gymnasium 等，由评测环境
提供，**不写进本节点 pyproject**；`uv sync` 只装 dora-rs / forge-msgs / forge-tool /
numpy / pyarrow / pyyaml / websockets / msgpack / msgpack-numpy / pydantic——后四者
是 G05 ws 后端连 XPolicyLab policy server 的协议依赖，官方 eval 同款）。

两个环境分工：

| 角色 | 要求 |
|:---|:---|
| 节点运行环境 `$BENCH_PY` | 两节点共用：RoboTwin 全套仿真依赖 + dora / forge_msgs + policy 推理栈 + torch+CUDA |
| dora CLI `$DORA_BIN` | 只拉 dataflow；**两边 dora 必须同版本**（如 0.4.1 / message 0.7.0），否则 version mismatch |

```bash
export PATH=$DORA_BIN:$PATH
$BENCH_PY -c "import dora, forge_msgs, sapien, torch; print('ok')"   # 自检
```

任务资产（`assets/` 的 embodiments / objects / background_texture）来自 HuggingFace，
由 RoboTwin 仓库的 `scripts/_download_assets.sh` 下载到 `$ROBOTWIN_ROOT/assets/`，
**离线部署前先本地化下载完成**。

---

## 2. 整体配置：三个文件，一张对应表

```
bench.yaml          benchmark 节点配置（见 config.example.yaml）
policy_xxx.yaml     policy 节点配置（policy 推理栈的）
dataflow.yaml       dora 拓扑（见 dataflow.example.yaml）
```

字段对不上不报错、只掉分。必须对齐：

| 概念 | bench.yaml | policy_xxx.yaml | dataflow.yaml |
|:---|:---|:---|:---|
| 策略标识 | `policy_id` | `policy.policy_id` | — |
| 相机通道 | 硬编码在 adapter（见 §4） | `image_inputs` 的**左键** | `inputs:` 通道名 |
| proprio 关节名 | 动态读 embodiment 配置 | `state_joints`（对不上静默取空） | — |
| 动作维度 | `control_mode`（qpos/ee，由 embodiment 推导） | policy 动作空间 | — |
| G05 ws 后端 | `control_mode: qpos`（G05 硬性 joint 14D） | `backend: g05` + `policy_server_host/port`（§3.4） | policy 节点不占 GPU |
| 分辨率 | `_camera_config.yml`（D435 320×240） | policy 内部自己 resize | — |
| 语言指令 | `send_instruction: true` | `policy.instruction`（逐 episode 覆盖） | — |

关键配置项（`bench.yaml`，完整注释见 config.example.yaml）：

```yaml
suite: robotwin
task_ids: "0-1"                # all / "0-9" / "0,2-4,8"；留空=全部 50 任务
init_state_ids: "0-1"
num_runs: 1
max_steps: 1800                # 必须 ≥ 最大 step_lim（1700），见 §5 坑 2
seed: 0
robotwin_root: ~/RoboTwin      # 缺省取 ROBOTWIN_ROOT 环境变量；~ 自动展开
task_config: demo_clean        # demo_clean / demo_randomized
control_mode: qpos             # qpos / ee
expert_check: true             # 官方专家预筛（关闭则场景集口径与官方不同）
instruction_type: seen         # seen / unseen / full_description（官方默认 seen）
test_num: 100                  # 每 task 测试场景数
policy_id: robotwin_policy     # ← 必须与 policy 配置一致
send_instruction: true
result_dir: results/benchmark
record_dir: null
```

`policy_robotwin.yaml`（示意）与 `dataflow.yaml`（见 dataflow.example.yaml）要点：

- policy 侧 `state_joints` 与 embodiment 配置逐字一致（qpos 14D / ee 16D）、
  `image_inputs` 左键 = dora 通道名、`policy_id` 与 bench 一致；
- benchmark 节点 `inputs: {action: policy/action}`，policy 节点输入含
  `policy_command`（`queue_size: 32` **必须显式加深**，见 §5 坑 4）+ 四条观测通道；
- dora 的节点 `path` 必须是绝对路径（`~` 不展开；home 是软链时用 `readlink -f`）。

---

## 3. 怎么跑

### 3.1 stub 冒烟（验时序，无需策略权重）

```bash
export PATH=$DORA_BIN:$PATH
cd $ROBOTWIN_NODE_ROOT
cp config.example.yaml config.stub.yaml     # 预设：task_ids "0-1"、init_state_ids "0"、num_runs 1
cp dataflow.example.yaml dataflow_stub.yaml # policy 节点换成 ① 接线（见模板头注释）
dora run dataflow_stub.yaml
```

stub 是零动作假 policy，`success_rate=0` 是预期（episode 跑到 step_lim）。要确认
四件事：每集 `set_instruction` 内容不同；退出打印 `obs == act`（1:1 锁步）；
`t/i/r` 索引与配置一致；结果 JSON 条数 == `total_episodes`。

### 3.2 真策略两节点（验口径，出真分数）

```bash
cp config.example.yaml config.real.yaml      # control_mode: ee + checkpoint 声明
cp dataflow.example.yaml dataflow_real.yaml  # policy 节点换成 ② 接线
dora run dataflow_real.yaml
```

`real_policy.py` 的配置（env `REAL_POLICY_CONFIG` 指向）：

```yaml
policy_id: robotwin_policy     # 必须与 bench 配置一致
robotwin_root: /path/to/RoboTwin   # 绝对路径！real_policy 不展开 ~（与 benchmark 节点 config 不同）
model_path: <X_VLA ckpt-30000 解包目录>
processor_path: null           # 该目录缺 preprocessor_config.json 时填 X-VLA-Pt 基座目录
steps: 10                      # 去噪步数
device: cuda
instruction: ""
```

**第一次接新 checkpoint 必须先小规模对基线**（2 task × 2 init-state）。已验基线：
X_VLA 官方 co-train `ckpt-30000`（`TianxingChen/RoboTwin2.0` datasets，
`cotrain_ckpt/X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0.tar.gz`），推理栈 XPolicyLab
`policy/X_VLA/model.py` commit `3f7823726fd0dd80705f009a94f59d9042ced56a`，
steps=10 去噪 / chunk 30 步 / ee 16D / 喂原生 320×240（X_VLA 自带 resize 到 224）。
实测 3/4 = 0.75（adjust_bottle 2/2、bb 1/2），锁步 `obs=827 act=827`，与论文同量级。

接新 checkpoint 必核三项（错了只安静掉分）：proprio 语义（读 checkpoint 归一化
统计量，别信文字描述）、相机别名（`image_inputs` 左键）、动作空间（qpos/ee）。

### 3.3 复现判据 A 验收（官方 runner vs 本节点）

官方侧在 RoboTwin 仓库根目录跑（XPolicyLab ws server + eval 脚本，需
`pip install websockets`），本节点侧跑 `config.acceptance.yaml`（task_ids "0-1"、
init_state_ids "0-49"、ee、expert_check + seen，两节点拆两张 GPU）。统计判定：

```bash
python tools/acceptance_stats.py --acc-dir <官方日志目录> --forge-dir results/acceptance
```

输出 Wilson 95% CI、Fisher 双侧精确检验、seed 对齐比较（**含失败集**，失败集种子
由 `metrics.scene_seed_ledger` 补全）与 McNemar 配对检验。判据：CI 重叠且 Fisher
`p > 0.05`。本仓库验收当次实测见 `docs/RoboTwin_Benchmark_Node_验收记录.md`。

### 3.4 G05 真策略（ws 后端，qpos 口径）

G05（Galaxea G0.5）走 XPolicyLab websocket policy server（官方 eval 同架构）：
policy 节点只做 env client（`real_policy.py` 的 g05 分支，经 `policy_backend.py`），
推理在 server 进程的 GPU 上。benchmark 侧**零改动**——`control_mode: qpos` 即官方
默认 joint 14D 口径。

```bash
# 0) 起 G05 policy server（官方侧与 Forge 侧各起一个，配置共用同一份）
#    XPolicyLab/policy/G05/setup_eval_policy_server.sh --config_path \
#      configs/g05_policy_deploy.yml（host/port/ckpt 由 launcher 注入）
# 1) benchmark 配置：control_mode: qpos + checkpoint 声明（config.example.yaml
#    头注释 G05 预设：task_ids "0-1"、init_state_ids "0-1" 先小规模对基线）
cp config.example.yaml config.g05.yaml       # 改 control_mode: qpos
cp dataflow.example.yaml dataflow_g05.yaml   # policy 节点换成 ④ 接线（不占 GPU）
dora run dataflow_g05.yaml
```

`real_policy.py` 的 g05 配置（env `REAL_POLICY_CONFIG` 指向，policy_g05.yaml）：

```yaml
policy_id: robotwin_policy     # 必须与 bench 配置一致
robotwin_root: /path/to/RoboTwin   # 绝对路径！real_policy 不展开 ~（与 benchmark 节点 config 不同）
backend: g05                   # xvla（缺省）/ g05
policy_server_host: localhost
policy_server_port: 18091
frequency: 30                  # 与 deploy 的 frequency 同值（官方 eval 同口径）
request_timeout_s: 300         # 首次推理冷启动留足
instruction: ""
```

obs/action 口径照抄官方 `eval_policy_xpolicylab.py`（`policy_backend.py` 逐字段
同构）：三路相机 → `vision.{cam_head,cam_left_wrist,cam_right_wrist}`；proprio
14D → `state.{left_arm_joint_state(6),left_ee_joint_state(1),right_arm_joint_state(6),
right_ee_joint_state(1)}`；动作同序拍平 [左臂6, 左爪1, 右臂6, 右爪1]；节奏为每帧
`update_obs`、chunk 耗尽才 `get_action`、每集 `reset`。`configs/g05_policy_deploy.yml`
的 `eval_embodiment` 已按 g05-robotwin20 填好 **robotwin**（= G05 repo
`configs/data/robotwin.yaml` 的 embodiment_type，填错 filter_embodiment 直接 ValueError）。

### 3.5 G05 判据 A 复验（官方 runner vs 本节点，qpos）

官方侧在 RoboTwin 仓库根目录跑（G05 ws server + 官方
`scripts/eval_policy_xpolicylab.py`，`action_type` 缺省即 joint）：

```bash
python scripts/eval_policy_xpolicylab.py --task_name adjust_bottle \
  --policy_name G05 --protocol ws --port 18091 --task_config demo_clean \
  --test_num 50 --seed 0 --additional_info "ckpt_name=g05-robotwin20,action_type=joint"
```

本节点侧：`config.g05.yaml`（task_ids "0-1"、init_state_ids "0-49"、qpos、
expert_check + seen）+ `dataflow_g05.yaml`（④ 接线）挂**另一个** G05 server 实例。
统计判定与 §3.3 同款（`tools/acceptance_stats.py`，判 Wilson/Fisher/McNemar）。
两侧 `frequency` 必须同值、同 checkpoint、同 task、同 episodes；GPU 账：官方
server/eval 各一张，Forge server/benchmark 各一张。

---

## 4. 代码清单

```
main.py                      节点装配层（照搬 LIBERO 版，只改 adapter 装配与日志前缀）
config.py                    配置模型 + 校验（RoboTwin 特有字段）
result.py                    BenchmarkExecutionResultV1 组装与落盘（再导出 _core.result）
adapters/robotwin_adapter.py ★ RoboTwin 特有的全部逻辑
_core/  bench_tool/          调度核心 / 服务模式 glue（照搬 LIBERO 版，一字未改）
tests/                       契约测试（照搬 LIBERO 版并适配）
config.example.yaml          配置模板（全字段带注释）
dataflow.example.yaml        真策略 dataflow 模板
policy_backend.py            G05 ws 后端（XPolicyLab 官方 client 薄封装 + obs/action 口径转换）
configs/g05_policy_deploy.yml G05 policy server 部署配置（官方侧与 Forge 侧共用）
pyproject.toml               包声明；RoboTwin/sapien 刻意不写进依赖
```

adapter 关键单元：

| 单元 | 职责 |
|:---|:---|
| `num_tasks` / `num_init_states()` | `envs/` 自动发现 50 任务 / 每 init state = 一个测试场景 seed（`cfg.test_num`） |
| `state_joint_names()` / `step_limit()` | 动态读 embodiment `arm_joints_name`（不手写）/ 官方 step_lim（`_eval_step_limit.yml`） |
| `_ensure_env()` | 同 task 复用 env，换 task 才重建 |
| `episode_instruction()` | **官方协议场景扫描**（runner 每集 reset 前调用）：顺序种子队列取候选 seed → `setup_demo`（`UnStableError` 跳过）→ expert_check 时 `play_once()`+`close_env()` 预筛（`plan_success and check_success()` 不过则跳过）→ 通过后**重新 setup 同一 seed** 为正式场景（官方同款两次 setup）→ 用 play_once 的 episode_info 生成 seen 指令返回；episode 开头关闭上一集 env（官方每集 close_env） |
| `reset_episode()` | 消费就位场景；逐集 `(seed, seed_retries, instruction)` 追加进批次级台账 |
| `step()` | 维度校验 → `take_action(action, action_type=cfg.control_mode)` → 读 `eval_success` |
| `to_state()` / `to_images()` | `joint_action.vector`（14D）/ 三相机 rgb **不翻转** |
| `_ResourceSampler` | 后台采 CPU/GPU 峰值，批次末注入 `metrics.resources`（env 重建后重启） |
| `_task_args()` | 官方 `load_task_args()` 等价实现（不 import 官方脚本，避免顶层拉 XPolicyLab） |

`result.py`：`success_rate` 分母用计划总数；成功集 metrics 带 `scene_seed` /
`seed_retries`；**失败集 metrics 必须为空**（三态契约），场景种子由批次级
`metrics.scene_seed_ledger` 按 `(task_id, init_state_id, run_index)` 补全。

---

## 5. RoboTwin 特有的坑

1. **场景是 seed 的全量重建**，不是 init_state 恢复（对象/光照/背景全按 seed 重采样）。
   官方口径：从 `100000 * (1 + seed)` 起用**顺序种子队列**逐集分配——每
   (task, run_index) 一个游标只增不减，遇 `UnStableError` / expert 预筛失败跳过
   （seed+1 不回收），**接受过的 seed 永不重复**；`run_index>0` 偏移一个种子域。
   每集实际 seed 与重试数记入 `metrics.scene_seed_ledger`。官方 eval 不 seed 任何
   RNG、expert 预筛本身不确定（同 seed 两次 play_once 可能一成一败），"两次运行
   逐 seed 相同"连官方 vs 官方都做不到——验收对齐以**协议同构**为准，不是逐 seed 一致。
2. **`max_steps` 必须 ≥ 最大 step_lim（1700）**（`env_cfg/task_config/_eval_step_limit.yml`，
   400~1700）。设小了长任务被核心提前掐成 `timed_out`，口径与官方（跑到 step_lim）不一致。
3. **done & !success 时 metrics 必须为空**：核心把「done 且 metrics 非空」记 `partial`；
   RoboTwin 没有部分完成口径，step_lim 到点未成功就是失败（成功集带 seed、失败集空字典）。
4. **`policy_command` 的 queue_size 必须显式加深**：episode 边界连发
   `set_instruction` + `reset_scene`，默认深度挤掉前一条 → 首集拿上一句指令跑完，不报错。
5. **图像不翻转**（与 LIBERO 相反）：官方 eval 拿 `get_obs()` rgb 原样喂 policy，这里照做。
6. **embodiment file_path 必须转绝对路径**：官方 `load_task_args` 返回 `./assets/...`
   按 cwd 解析；dora 节点 cwd 是节点目录，adapter 已转绝对。
7. **expert_check 官方默认开启**：每集前规则专家 `play_once()` 预筛，专家都过不了的
   不计入分母；通过后重新 setup 同一 seed（官方同款两次 setup）。本 adapter 同款实现
   （`expert_check: true` 默认）。关闭则场景集口径与官方不同，只用于对照实验。
8. **关节名与动作维度都不手写**：动态读 embodiment（臂关节 + 夹爪，aloha-agilex 14D）。
   policy 配置的 `state_joints` 仍需自己核对逐字一致，对不上静默取空。
9. **envs 代码依赖 cwd = 仓库根**：`envs/utils/rand_create_cluttered_actor.py:17`
   在 import 期读 `assets/objects/objaverse/list.json`。adapter 构造时
   `os.chdir(robotwin_root)`；节点输出路径已在 `load_config` 转绝对，不受影响。
   症状 `FileNotFoundError: assets/objects/objaverse/list.json` = cwd 不对（先查这个）。
10. **dora canonicalize 会弄丢 uv venv**：uv venv 的 `bin/python` 是符号链接，
    canonicalize 后节点拿 base 解释器启动 → `import numpy` 直接挂（shell 手跑却正常）。
    做法：把 base 解释器 **copy** 成 venv 里的 `python-real`（venv 检测靠同目录
    `pyvenv.cfg`），dataflow 指向它。conda / 系统 python 不受影响。

另：episode 间用 `reset_scene` 而不是 `reset`（后者把 policy 打回 idle 相）。
时间预算（RTX3090，320×240 三相机）：节点启动 ~7s（import 50 个任务模块）；
首集建景 ~40s，之后每集重建 ~30s；单步 ~0.16s（adjust_bottle 400 步 ≈ 65s）。

---

## 6. 排错

| 症状 | 原因 |
|:---|:---|
| 跑起来没反应、卡住 | policy 没进 running（`policy_id` 不对 / 没收到 `start`）；本节点**无超时看门狗**，policy 卡死会静默悬停 |
| 首个 episode 用了上一句指令 | `policy_command` 的 `queue_size` 没加深 |
| 分数异常低但不报错 | 按顺序查：proprio 语义 → 相机别名 → 分辨率 → `control_mode` |
| `UnStableError` 反复刷屏 | 该 seed 段场景普遍不稳，跳过属正常；连续 200 次才放弃 |
| `envs/xxx.py 里没有 xxx 类` | `envs/` 混进非任务模块，发现逻辑要加黑名单 |
| `FileNotFoundError: assets/objects/objaverse/list.json` | cwd 不是 RoboTwin 仓库根（§5 坑 9）；资产没下全同报错 |
| `No module named 'numpy'`/`'dora'`，dora 打印解释器是 `~/.local/share/uv/...` | 节点 `path` 指到 uv venv 符号链接解释器（§5 坑 10） |
| `'function' object has no attribute 'get'` | `PolicyCommand.inputs` 是方法不是属性，写 `cmd.inputs()` |
| `robotwin_root 不存在` | config 没写对 / 仓库不在缺省位置；设 `ROBOTWIN_ROOT` |
| WARNING: max_steps 小于最大 step_lim | `max_steps` 调到 ≥1700 |
| version mismatch | dora-rs 与 dora-rs-cli 版本不一致 |
| `Could not initiate node from environment variable` | 对端节点先崩了，看它的 traceback |

---

## 7. 测试矩阵

| 阶段 | 结果 |
|:---|:---|
| 契约测试 + 协议回归（2026-09-12 整改） | ✅ 73 passed：通道集 / config 校验 / 结果 schema / seed 确定性 / 顺序种子队列无重复 + 每集 close_env / expert 预筛跳过与两次 setup / seen 指令官方生成器 / 台账注入 / expert_check・instruction_type 解析 |
| GPU 渲染冒烟 / 判定验证 | ✅ sapien 离屏出图；`setup_demo → play_once → check_success()` True |
| adapter 运行时 / 确定性验收 | ✅ 50 任务全过；同 seed 两次渲染逐位相同、异 seed 不同（51.6% 像素差） |
| stub 时序闭环 | ✅ `obs=1600 act=1600` 锁步；4 集指令交替；JSON 条数一致；4 集全 `failed`（step_lim，metrics 空） |
| timed_out/failed 分离 | ✅ `max_steps: 5` → 核心掐断 `timed_out`；正常 400 步 → env step_lim 判败 `failed` |
| 真策略小规模 | ✅ 3/4=0.75 对论文基线；锁步 827/827 |
| G05 ws 后端契约测试（2026-09-14） | ✅ 后端解析 / qpos 14D 四键切片 / 官方 obs schema 同构 / 动作拍平顺序与别名 / ws 协议假服务器端到端（真 `WsModelClient` 打假 server，msgpack 编码往返核对） |
| num_runs>1 差异 / 中途取消 / 资源记录 | ✅ 种子 100005 vs 200005 渲染不同；SIGINT 批次 `cancelled` 干净收尾；`metrics.resources` 注入 |

---

## 8. 已知限制

- `record_dir` mp4 落盘未实现（配置位保留）；wall-clock 看门狗未实现（policy 卡死
  静默悬停，靠 dora STOP 兜底）；真正的 Forge policy 节点端到端验证未做（本机没有
  该节点，见 §9）。
- 相机别名硬编码在 adapter（换 checkpoint 改代码）；`start` 与首帧观测无 ack（启动
  竞态靠通道投递顺序）；不打包 pyinstaller（资产 3G+，直接 `python main.py` 跑）。

---

## 9. policy 节点来源与契约

仓库里的 `stub_policy.py` / `real_policy.py` / `policy_backend.py` 都是**本地测试件**，
不是"未改动的 Forge policy 节点"：stub 是零动作假 policy（验时序）；real_policy 是
XPolicyLab X_VLA 的 dora 薄封装（本机出真分数用），按 Forge policy 通道契约实现
（§2 的 `image/*` 别名、`state_joints`、16D `JointCommand`、1:1 锁步、
`policy_command` 四命令、chunk 边界帧对齐——逐通道帧计数保证推理用同一帧的
proprio + head 图）；policy_backend 是其 G05 ws 后端的可测纯函数层（官方
`WsModelClient` + 官方 obs/action 口径，三路相机齐帧才推理）。两者都**不能充当
"外部 policy 节点零改动接入"的验收证据**。

接生产 Forge policy 节点时，`dataflow.example.yaml` 里替换 `policy` 节点即可
（benchmark 节点不动），对齐四件事：① `policy_id` 一致；② `image_inputs` 左键 =
`image/head` / `image/left_wrist` / `image/right_wrist`；③ `state_joints` 与
`state_joint_names()` 逐字一致（qpos 14D / ee 16D）；④ 动作 = `JointCommand`，
1:1 锁步（每帧一动作，多余丢弃）。该项端到端验证留到 Forge 环境完成。

---

## 10. 本机路径适配清单

换机器跑之前改这几处（均可用环境变量覆盖，见各文件头注释）：

| 文件 | 位置 | 内容 |
|:---|:---|:---|
| `config.example.yaml` 副本 | `robotwin_root` | RoboTwin 仓库根（`~` 自动展开；或设 `ROBOTWIN_ROOT`） |
| `dataflow.example.yaml` 副本 | 节点 `path:` | `$BENCH_PY` 解释器绝对路径（uv venv 需 copy 出的真二进制，§5 坑 10） |
| 同上 | `env.BENCHMARK_CONFIG` | 保持 `./config.*.yaml`，在仓库根执行 dora run |
| 同上 | `env.CUDA_VISIBLE_DEVICES` | 按机器 GPU 分配调整 |
| `policy_real.yaml`（自建，§3.2） | `model_path` | X_VLA ckpt-30000 解包目录 |
| `policy_g05.yaml`（自建，§3.4） | `policy_server_host/port` | G05 ws policy server 地址（与 deploy 的 host/port 一致） |
| `configs/g05_policy_deploy.yml` | `eval_embodiment` | g05-robotwin20 已填 **robotwin**（= G05 repo `configs/data/robotwin.yaml` 的 embodiment_type）；`G05_CKPT_PATH` 环境变量兜底 ckpt 路径 |
| `tools/run_full_suite.py` | 四个路径常量 | `ROBOTWIN_NODE_ROOT` / `ROBOTWIN_ROOT` / `ROBOTWIN_MODEL_PATH` / `BENCH_PY` 环境变量覆盖 |
