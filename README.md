# RoboTwin Benchmark 节点

把 RoboTwin 包一层的 Dora 节点。进程内持有 RoboTwin env，只做测评：批次展开 → 逐 episode 跑 → 结果落盘。

```
benchmark 节点（本节点）                     policy 节点（不改）
├─ 进程内 import RoboTwin envs，直接持有 env
├─ proprio_state / image/head / image/left_wrist / image/right_wrist ──观测──▶ 缓存 + tick 推理
├─ action ◀──────────────────────────────────────────────────动作──  JointCommand 14D(qpos) / 16D(ee)
├─ policy_command ───────────────────────────────────────────命令──▶ start / set_instruction / reset_scene / stop
├─ 判定 env.eval_success（take_action 内部逐 sim step 调 check_success()）
└─ benchmark_status（流式）/ benchmark_result（批次末）
```

不经过 XPolicyLab / policy server。判定、动作、重置全走 RoboTwin 官方接口，
口径照抄官方 eval 脚本 `scripts/eval_policy_xpolicylab.py`。
设计规范见《Benchmark 接入手册》v14（PhyAgentOS 文档库），本节点是其参考实现的
RoboTwin 版本，结构与 `benchmark-libero` 同构（除 adapter 外全部照搬）。

拆仓不改变 Dora 端口契约：

- 输入：`action`，来源为 policy 的 `JointCommand`
  （qpos 口径 = 每臂 `arm_joints_name` 的臂关节 + 1 夹爪，aloha-agilex = 14D；
  ee 口径 = 每臂 xyz + wxyz 位姿 + 1 夹爪，16D）；
- 输出：`proprio_state`（14D `JointState`）、`image/head`、`image/left_wrist`、
  `image/right_wrist`、`policy_command`、`benchmark_status`、`benchmark_result`。

---

## 0. 本仓库说明

独立测试仓库：把上述节点从 PhyAgentOS 环境里拆出来，供同事测试与复现。
2026-09-10 已通过验收**判据 A**（官方 runner 99/100 vs 本节点 95/100，
Fisher p = 0.2116；完整验收记录见 `docs/RoboTwin_Benchmark_Node_验收记录.md`）。

| 路径 | 内容 |
|:---|:---|
| `main.py` / `config.py` / `result.py` | 节点装配 / 配置模型 / 结果落盘 |
| `adapters/robotwin_adapter.py` | ★ RoboTwin 特有逻辑（换 benchmark 只改这里） |
| `_core/` `bench_tool/` `tests/` | 调度核心 / 服务模式 glue / 契约测试（照搬 benchmark-libero） |
| `config.example.yaml` `dataflow.example.yaml` | 运行配置与 dora 拓扑模板（其余按 §3 预设从模板另存；结构同 benchmark-libero） |
| `stub_policy.py` `real_policy.py` | 本地测试 policy（假 policy 冒烟 / X_VLA 真策略） |
| `tools/` | 全 suite 跑分驱动、判据 A 统计工具 |
| `code_manifest.txt` | 28 个核心交付文件的 sha256 版本锚点 |

**版本锚点**：`code_manifest.txt` 列出 28 个核心文件（`main.py`、`config.py`、
`result.py`、`adapters/`、`_core/`、`bench_tool/`、`tests/`）的 sha256，与验收
批次运行的代码逐字节一致（零漂移）。`sha256sum -c code_manifest.txt` 应全部 OK。

**上游致谢**：`_core/`、`bench_tool/`、`main.py` 的骨架照搬
[benchmark-libero](https://github.com/Forgelab-Robotics/benchmark-libero)（除
adapter 与标识符外一字未改），本仓库为其 RoboTwin 移植版。

---

## 1. 环境

### 1.1 RoboTwin 仓库与依赖

本节点进程内 `import envs`（RoboTwin 的 50 个任务环境），因此运行环境必须装好
RoboTwin 的仿真依赖。`config.example.yaml` 的 `robotwin_root` 指向 RoboTwin 仓库根目录
（缺省取 `ROBOTWIN_ROOT` 环境变量、再退 `~/RoboTwin`），adapter 把该目录挂进
`sys.path` 后 import `envs` / 读 `env_cfg` / 读 `description`。

RoboTwin 官方依赖（sapien 3.0.0b1 / torch cu121 / mplib / toppra / trimesh /
opencv-headless / gymnasium 等）由评测环境提供，**不写进本节点的 pyproject**，
避免把整套仿真依赖拖进基础节点环境。`uv sync` 只装节点基础依赖
（dora-rs / forge-msgs / forge-tool / numpy / pyarrow / pyyaml）。

```bash
uv sync --all-groups
uv run pytest -q
```

### 1.2 两个环境的分工

| 角色 | 要求 |
|:---|:---|
| **节点运行环境**（`$BENCH_PY`） | 两个节点的 python 都用它。须有 RoboTwin 全套仿真依赖 + dora(python 模块) / forge_msgs / policy 推理栈 / torch+CUDA |
| **`dora` CLI**（`$DORA_BIN`） | 只用来拉起 dataflow。**两边 dora 必须同版本**（如同为 0.4.1 / message 0.7.0），否则注册节点时报 version mismatch |

```bash
export PATH=$DORA_BIN:$PATH     # 拉起用它
# 节点里写 $BENCH_PY 的绝对路径（见 dataflow）
```

自检：

```bash
$BENCH_PY -c "import dora, forge_msgs, sapien, torch; print('ok')"
dora --version
```

### 1.3 任务资产

RoboTwin 的运行时资产（`assets/` 下的 embodiments / objects / background_texture）
来自 HuggingFace，由 RoboTwin 仓库的 `scripts/_download_assets.sh` 下载到
`$ROBOTWIN_ROOT/assets/`。**离线部署前先本地化下载完成**，节点跑起来后不再联网。

---

## 2. 整体配置：三个文件，一张对应表

跑一次评测要三个文件，**它们之间有字段必须逐字对齐，对不上不报错、只掉分**：

```
bench.yaml          benchmark 节点配置（本节点的，见 config.example.yaml）
policy_xxx.yaml     policy 节点配置（policy 推理栈的）
dataflow.yaml       dora 拓扑：两个节点怎么连
```

### 2.1 必须对齐的字段

| 概念 | bench.yaml | policy_xxx.yaml | dataflow.yaml |
|:---|:---|:---|:---|
| **策略标识** | `policy_id` | `policy.policy_id` | — |
| **相机通道** | （硬编码在 adapter，见 §4.3） | `image_inputs` 的**左键** | `inputs:` 里的通道名 |
| **proprio 关节名** | 动态读 embodiment 配置（见 §4.3） | `state_joints` | — |
| **动作维度** | `control_mode`（qpos/ee）；维度由 embodiment 推导（aloha-agilex 14D） | policy 输出的动作空间 | — |
| **渲染分辨率** | 由 `_camera_config.yml` 决定（D435 320×240） | （policy 内部自己 resize） | — |
| **语言指令** | `send_instruction: true` | `policy.instruction`（会被逐 episode 覆盖） | — |

`policy_id` 对不上 → policy 忽略我们所有命令，整批用初始指令跑完。
`image_inputs` 左键对不上通道名 → policy 收不到图。
关节名对不上 → `JointState` 按名字查表**静默取到空值**。

### 2.2 完整可用配置

`bench.yaml`（见 config.example.yaml，关键项）：

```yaml
suite: robotwin
task_ids: "0-1"                # all / "0-9" / "0,2-4,8" / [0,1,2]；留空=全部 50 任务
init_state_ids: "0-1"
num_runs: 1
max_steps: 1800                # 必须 ≥ 最大 step_lim（1700），见 §5 坑 2
seed: 0
robotwin_root: /data7/home/shenxin/RoboTwin
task_config: demo_clean        # demo_clean / demo_randomized
control_mode: qpos             # qpos / ee
test_num: 100                  # 每 task 的测试场景数，官方口径 100
policy_id: robotwin_policy     # ← 必须与 policy 配置一致
send_instruction: true
result_dir: results/benchmark
record_dir: null
```

`policy_robotwin.yaml`（示意，字段随推理栈）：

```yaml
state_joints: [fl_joint1, ..., fl_joint7, fr_joint1, ..., fr_joint7]   # 14 个，与 embodiment config 逐字一致
policy:
  policy_id: robotwin_policy                # ← 必须与 bench.yaml 一致
  auto_start: false                         # 由 benchmark 发 start
  instruction: "..."                        # 占位，会被逐 episode 覆盖
image_inputs:
  image/head: head                          # 左=dora 通道名，右=checkpoint 特征名
  image/left_wrist: left_wrist
  image/right_wrist: right_wrist
```

`dataflow.yaml`（见 dataflow.example.yaml）：

```yaml
nodes:
  - id: robotwin_benchmark
    path: $BENCH_PY
    args: -u main.py
    env:
      BENCHMARK_CONFIG: $WORK/bench.yaml
    inputs:
      action: policy/action
    outputs: [proprio_state, image/head, image/left_wrist, image/right_wrist,
              policy_command, benchmark_status, benchmark_result]

  - id: policy
    path: $BENCH_PY
    args: -u -m <policy_runner> infer --config $WORK/policy_robotwin.yaml
    inputs:
      tick: dora/timer/millis/20
      policy_command:
        source: robotwin_benchmark/policy_command
        queue_size: 32                # 必须显式加深，见 §5 坑 4
      proprio_state: robotwin_benchmark/proprio_state
      image/head: robotwin_benchmark/image/head
      image/left_wrist: robotwin_benchmark/image/left_wrist
      image/right_wrist: robotwin_benchmark/image/right_wrist
    outputs: [action]
```

**dora 的 `path` 必须是绝对路径**，`~` 不展开；若 home 是软链，用 `readlink -f` 取真实路径。

---

## 3. 怎么跑

### 3.1 stub 冒烟（验时序，不需要策略权重）

stub 是个假 policy，只回零动作。**`success_rate` 恒为 0 是预期行为**：零动作
不改变位形，episode 会一直跑到该任务的 `step_lim`（所以只开 1~2 个快任务，
`adjust_bottle` / `beat_block_hammer` 都是 400 步）。

```bash
export PATH=$DORA_BIN:$PATH
cd $ROBOTWIN_NODE_ROOT
cp config.example.yaml config.stub.yaml     # 预设：task_ids "0-1"、init_state_ids "0"、num_runs 1
cp dataflow.example.yaml dataflow_stub.yaml # policy 节点换成 ① 接线（见模板头注释）
dora run dataflow_stub.yaml    # 需要 RoboTwin 仿真依赖 + GPU，不需要策略权重
```

stub 预设把批次压到 2 episode（`task_ids: "0-1"`、`init_state_ids: "0"`、
`num_runs: 1`；task 0/1 的 step_lim 均 400）。

**要确认的四件事**（不是看跑完就算过）：

1. 每个 episode 的 `set_instruction` 内容**不一样**
2. stub 退出时打印的 `obs == act`（1:1 锁步）
3. `t/i/r` 索引与配置里的批次一致
4. 结果 JSON 落盘且 `episodes[]` 条数 == `total_episodes`

### 3.2 真策略两节点（验口径，出真分数）

```bash
export PATH=$DORA_BIN:$PATH
cd $ROBOTWIN_NODE_ROOT
cp config.example.yaml config.real.yaml      # control_mode: ee + checkpoint 声明（见 §2.2）
cp dataflow.example.yaml dataflow_real.yaml  # policy 节点换成 ② 接线（见模板头注释）
dora run dataflow_real.yaml
```

`real_policy.py` 的配置（env `REAL_POLICY_CONFIG` 指向它，如 `policy_real.yaml`）：

```yaml
policy_id: robotwin_policy        # 必须与 bench 配置一致
robotwin_root: /data7/home/shenxin/RoboTwin
model_path: /data7/home/shenxin/robotwin_deps/ckpt/RoboTwin-cotrain-aloha_agilex-ee-0/ckpt-30000
processor_path: null              # 该目录缺 preprocessor_config.json 时填 X-VLA-Pt 基座模型目录
steps: 10                         # 去噪步数
device: cuda
instruction: ""
```

**第一次接新 checkpoint 必须先小规模对基线**（2 task × 2 init-state），拿
`success_rate` 对 RoboTwin 2.0 论文数字。stub 验的是时序，真策略验的是口径，
两件事不能合并。

#### 已验基线：X_VLA（官方 co-train，ee 口径）——2026-09-10

checkpoint 版本记录（手册 §8-15）：

| 项 | 值 |
|:---|:---|
| 权重 | `TianxingChen/RoboTwin2.0`（datasets）`cotrain_ckpt/X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0.tar.gz` |
| 步数 | `ckpt-30000`（state.json `global_step=30000`，自包含：含 config/model.safetensors/preprocessor_config.json/tokenizer） |
| 推理栈 | XPolicyLab `policy/X_VLA/model.py`，commit `3f7823726fd0dd80705f009a94f59d9042ced56a` |
| 推理参数 | steps=10 去噪，action chunk 30 步，domain_id=0，device=cuda |
| 口径 | `control_mode: ee`；X_VLA 自带 CLIP resize 到 224（`resize_images_to: [224,224]`），benchmark 喂原生 320×240；proprio 16D = 每臂 xyz+wxyz 位姿 + 夹爪 |

结果（配置按上面两个 cp + 编辑得到；`results/real/`）：

| task | 官方 Easy | 本次 n=2 |
|:---|:---|:---|
| adjust_bottle | 100 | **2/2**（121 / 129 步） |
| beat_block_hammer | 92 | **1/2**（177 步成功；1 集 400 步走满未达成） |
| 合计 | 68.0（Easy） | **3/4 = 0.75** |

1:1 锁步 `obs=827 act=827`，耗时 334s。与论文同量级，口径（wxyz 四元数、ee
16D、head 相机）验证通过。

### 3.3 接一个新 checkpoint 前必核三项

错了都不报错，只会安静地掉分：

| 项 | 怎么核 |
|:---|:---|
| **proprio 语义** | **读 checkpoint 的归一化统计量**，别信配置里的文字描述。`observation.state` 的 min/mean/max 逐维量纲直接暴露它是关节角还是末端位姿 |
| **相机别名** | policy 配置 `image_inputs` 左键要能对上我们的输出通道名（`image/head` 等） |
| **动作空间** | `control_mode` 要与 checkpoint 的动作头一致：qpos（关节角）/ ee（末端位姿） |

### 3.4 复现判据 A 验收（官方 runner vs 本节点）

**官方侧**（在 RoboTwin 仓库根目录，具体参数以 `scripts/` 下的脚本为准）：

```bash
cd $ROBOTWIN_ROOT
python scripts/setup_policy_server.py \      # XPolicyLab ws policy server（需 pip install websockets）
    --ckpt_path <ckpt-30000 解包目录> --ckpt_name ckpt-30000 --action_type ee --port 18090
python scripts/eval_policy_xpolicylab.py \
    --test_num 50 --seed 0 --task_config demo_clean \
    --additional_info "ckpt_name=ckpt-30000,action_type=ee" 2>&1 | tee official_<task>.log
```

**本节点侧**（本仓库根目录）：

```bash
export PATH=$DORA_BIN:$PATH
cp config.example.yaml config.acceptance.yaml      # task_ids "0-1"、init_state_ids "0-49"、control_mode: ee
cp dataflow.example.yaml dataflow_acceptance.yaml  # policy 节点换成 ② 接线，两节点拆到两张 GPU（问题 #8）
dora run dataflow_acceptance.yaml
```

**统计判定**：

```bash
python tools/acceptance_stats.py --acc-dir <官方日志目录> --forge-dir results/acceptance
```

输出每 task 的 Wilson 95% CI、Fisher 双侧精确检验 p 值与 seed 对齐比较。
判据：两边 CI 重叠且 Fisher `p > 0.05` 即通过。本仓库验收当次的实测数据见
`docs/RoboTwin_Benchmark_Node_验收记录.md`。

---

## 4. 代码清单

```
main.py                      节点装配层（照搬 LIBERO 版，只改 adapter 装配与日志前缀）
config.py                    配置模型 + 校验（RoboTwin 特有字段）
result.py                    BenchmarkExecutionResultV1 组装与落盘（再导出 _core.result）
adapters/robotwin_adapter.py ★ RoboTwin 特有的全部逻辑
adapters/__init__.py         空
_core/                       调度核心（照搬 LIBERO 版，一字未改）
bench_tool/                  服务模式 glue（照搬 LIBERO 版）
config.example.yaml          配置模板（全字段带注释）
dataflow.example.yaml        真策略 dataflow 模板
tests/                       契约测试（照搬 LIBERO 版并适配）
pyproject.toml               包声明；RoboTwin/sapien 刻意不写进依赖，由评测环境提供
```

### 4.1 `_core/` — 批次调度核心

照搬 LIBERO 版一字未改：`BenchmarkRunner`（批次推进 + 1:1 锁步）、`run_node`
（autorun 事件循环）、`BenchSession`（serve 模式）、`BenchmarkResult`（结果
schema）、`EpisodeRecorder`（多相机横向拼接录像）。换 benchmark 不动核心。

### 4.2 `config.py` — 配置与校验

| 单元 | 职责 |
|:---|:---|
| `RobotwinConfig` | 共有字段继承 `BaseBenchmarkConfig`；特有：`control_mode` / `task_config` / `test_num` / `robotwin_root`；`max_steps` 默认 1800（≥ 最大 step_lim 1700） |
| `load_config()` | 读 YAML + 校验 `control_mode` 合法、`test_num ≥ 1`、`robotwin_root` 存在（fail fast） |
| `resolve_and_validate_plan()` | （来自 `_core`）回填 `all`、逐 task 校验 init_state 范围 |

### 4.3 `adapters/robotwin_adapter.py` — RoboTwin 特有逻辑

**换 benchmark 只改这一个文件。**

| 单元 | 职责 |
|:---|:---|
| `num_tasks` | `envs/` 目录自动发现的任务模块数（50），构造时验证每个模块都有同名类 |
| `num_init_states()` | 返回 `cfg.test_num`——每个"init state"是一个测试场景 seed |
| `task_language()` | `description/task_instruction/<task>.json` 的 `full_description` |
| `state_joint_names()` | **动态读 embodiment 配置**的 `arm_joints_name`（与官方口径逐字一致，不手写） |
| `step_limit()` | 每任务的官方 step_lim（`_eval_step_limit.yml`，缺省 1000） |
| `_ensure_env()` | **同 task 复用 env，换 task 才重建**（建 env + 场景都贵） |
| `reset_episode()` | `setup_demo(now_ep_num, seed, is_test=True, **task_args)`；scene seed = `100000 * (1 + seed + run_index) + init_state_id`（run_index=0 即官方公式），遇 `UnStableError` +1 重试（官方口径） |
| `step()` | 校验维度（`action_dim`，由 embodiment 推导）finite → `take_action(action, action_type=cfg.control_mode)` → 读 `eval_success` |
| `to_state()` | `joint_action.vector`（= left_jointstate + right_jointstate，含夹爪；aloha-agilex 14D） |
| `to_images()` | `{image/head, image/left_wrist, image/right_wrist}`，**不翻转** |
| `_ResourceSampler` | 后台线程采 CPU/GPU 峰值（psutil + nvidia-smi 降频链，缺失时降级为 None），批次末 `aggregate()` 停表注入 `metrics.resources`；`_ensure_env` 重建 env 后重启（close 只停表不终结批次） |
| `_task_args()` | 官方 `load_task_args()` 的等价实现（不 import 那个脚本——它顶层拉 XPolicyLab 协议层） |

### 4.4 `result.py` — 结果组装

照搬 LIBERO 版（再导出 `_core.result`）。`success_rate` **分母用计划总数**；
逐 episode 带 `episode_seed`、`success_source`（`env.eval_success`）、`metrics`
（成功集带 `scene_seed` / `seed_retries`）。

---

## 5. RoboTwin 特有的坑

1. **场景是 seed 的全量重建，不是 init_state 的恢复**。`setup_demo` 每次重建
   整场（对象/光照/背景全部按 seed 重采样），所以不存在 LIBERO 的"家具位姿在
   模型里恢复不到"问题；但复现性完全押在 seed 链上。官方口径：从
   `100000 * (1 + seed)` 起，用**顺序种子队列**逐 episode 分配，遇 `UnStableError`
   （物体停不稳）跳过换下一个 seed，跳过的不回收。adapter 改用
   `100000 * (1 + seed + run_index) + init_state_id` 直接映射（无跳过时与官方
   逐位相同；有跳过时该集种子比官方小一位，场景不同但统计等价）。`run_index=0`
   即官方公式；`>0` 偏移一个种子域——保证 num_runs>1 产生差异（手册完成标准），
   官方口径（每 task 100 init × 1 run）不受影响。

2. **`max_steps` 必须 ≥ 最大 step_lim（1700）**。RoboTwin 每任务自带步数上限
   （`env_cfg/task_config/_eval_step_limit.yml`，400~1700，最大是
   `put_bottles_dustbin` 的 1700），到点 env 自己
   done。核心的 `max_steps` 只是兜底护栏：设小了，长任务会被核心提前掐成
   `timed_out`，分数口径与官方 eval（跑到 step_lim 为止）不一致。
   adapter 构造时发现 `max_steps` 小于最大 step_lim 会打 WARNING。

3. **done & !success 时 metrics 必须为空**。核心把「done 且 metrics 非空」记成
   `partial`（长程任务语义）；RoboTwin 没有部分完成口径，step_lim 到点未成功
   按官方口径就是失败。adapter 已在 step 里处理（成功集 metrics 带 scene_seed，
   失败集为空字典）。

4. **`policy_command` 的 queue_size 必须显式加深**。episode 边界连发
   `set_instruction` + `reset_scene`，默认深度会把前一条挤掉，表现为首个
   episode 拿着上一句指令跑完——照样出分，不报错。

5. **图像不翻转**。与 LIBERO 相反：LIBERO 的 mujoco 离屏渲染上下颠倒要
   `[::-1,::-1]`；RoboTwin 官方 eval 拿 `get_obs()` 的 rgb **原样**喂 policy，
   这里照做。

6. **embodiment file_path 必须转绝对路径**。官方 `load_task_args` 返回
   `./assets/embodiments/...` 相对路径，按 **cwd** 解析（官方脚本在仓库根目录
   跑）；dora 节点的 cwd 是节点目录，adapter 已转成绝对路径。照抄官方脚本时
   别把这处漏回去。

7. **官方 eval 的 expert_check 预筛未实现**。官方在每集前用规则专家
   `play_once()` 预筛场景（试跑一遍，专家都过不了的不计入分母；无需 demo
   数据集）。本 adapter 只做了 `UnStableError` 重试，未做专家预筛（adapter
   范围选择）。对分数的系统性影响是单向的：官方分母里剔掉的是"专家都做不了
   的场景"，我们保留它们可能略微压低分数。跑基线对比时留意这一处口径差异。

8. **关节名与动作维度都不手写**。`state_joint_names()` 动态读 embodiment 配置：
   `arm_joints_name[i]` 的臂关节 + `gripper_name[i]["base"]` 的夹爪（aloha-agilex
   = `fl_joint1..6` + `fl_joint7` + `fr_joint1..6` + `fr_joint7` = **14D**）。
   **别按 16D 记**：`arm_joints_name` 里没有夹爪，官方 `get_left_arm_jointState()`
   是「臂关节驱动目标 + 末尾 1 个夹爪值」，维度随 embodiment 变。
   policy 配置里的 `state_joints` 仍要自己核对逐字一致，对不上是静默取空。

9. **RoboTwin 的 envs 代码依赖 cwd = 仓库根**。envs 里大量 `./assets/...` 相对
   路径，其中一部分在 **import 时**就执行（`envs/utils/rand_create_cluttered_actor.py:17`
   模块级读 `assets/objects/objaverse/list.json`），官方 eval 也是在仓库根目录
   跑的。dora 节点的 cwd 是数据流所在目录，所以 adapter 构造时
   `os.chdir(robotwin_root)`；节点自己的输出路径（`result_dir`/`record_dir`）已在
   `load_config` 里转成绝对，不受这次 chdir 影响。**症状**：
   `FileNotFoundError: assets/objects/objaverse/list.json`——看着像资产没下全，
   其实是 cwd 不对。

10. **dora 会把节点 `path` 做 canonicalize，而 uv venv 的解释器是符号链接**。
    `dora-core` 的 `resolve_path` 对 `path` 调 `canonicalize()`
    （`descriptor/mod.rs:221`），uv 建的 venv 里 `bin/python` 是指向 base 解释器
    的**符号链接**（`~/.local/share/uv/python/cpython-3.12.x/.../bin/python3.12`），
    canonicalize 之后 venv 就丢了：节点进程拿 base 解释器启动，`import numpy`
    直接 `ModuleNotFoundError`——而 shell 里手跑同一个路径一切正常，很容易往
    反方向排查。本机做法：把 base 解释器 **copy** 成
    `.venv-robotwin/bin/python-real`（venv 检测靠同目录的 `pyvenv.cfg`，copy 出来
    的真二进制照样进 venv），dataflow 指向它。conda / 系统 python 不是符号链接，
    不受影响。

另：episode 间用 `reset_scene` 而不是 `reset`——后者把 policy 打回 idle 相，
之后不再响应观测。

另：时间预算（RTX3090，320×240 三相机，实测 stub 批次）：节点启动 ~7s
（`_discover_tasks()` import 全部 50 个任务模块，fail fast：任务模块坏了要在建
dora 连接之前炸，而不是跑起来才发现）；首个 episode 场景搭建 ~40s（首次加载
资产/URDF + curobo 初始化），之后每集重建场景 ~30s；单步 ~0.16s
（`adjust_bottle` 400 步 ≈ 65s）。照此估算，单任务 100 集的官方口径大约
3~4 小时。

---

## 6. 排错

| 症状 | 原因 |
|:---|:---|
| 跑起来没反应、卡住不动 | policy 没进 running（`policy_id` 对不上，或没收到 `start`）；也可能 policy 卡死——本节点**没有超时看门狗**，会静默悬停 |
| 首个 episode 用了上一句指令 | dataflow 里 `policy_command` 的 `queue_size` 没加深 |
| 分数异常低但不报错 | 按顺序查：proprio 语义 → 相机别名 → 分辨率 → `control_mode` |
| `UnStableError` 反复刷屏 | 该 seed 段场景普遍不稳，重试属正常；连续 200 次才放弃 |
| `envs/xxx.py 里没有 xxx 类` | `envs/` 里混进了非任务模块，adapter 的发现逻辑需要加黑名单 |
| `FileNotFoundError: assets/objects/objaverse/list.json` | cwd 不是 RoboTwin 仓库根（§5 坑 9）；资产没下全也是同一句报错，先查 cwd |
| `No module named 'numpy'` / `'dora'`，且 dora 打印的解释器是 `~/.local/share/uv/python/...` | 节点 `path` 指到了 uv venv 的符号链接解释器（§5 坑 10） |
| `'function' object has no attribute 'get'` | `PolicyCommand.inputs` 是**方法**不是属性，写 `cmd.inputs()` |
| `robotwin_root 不存在` | config.yaml 没写对，或 RoboTwin 仓库不在缺省位置；设 `ROBOTWIN_ROOT` |
| WARNING: max_steps 小于最大 step_lim | 把 `max_steps` 调到 ≥1700（或具体任务的最大值） |
| version mismatch | `dora-rs` 与 `dora-rs-cli` 版本不一致 |
| `Could not initiate node from environment variable` | 对端节点先崩了，看它的 traceback，不是本节点的问题 |

---

## 7. 测试矩阵

| 阶段 | 内容 | 状态 |
|:---|:---|:---|
| 契约测试 | dataflow 通道集 / config 校验 / 结果 schema / seed 确定性 | ✅ 67 passed（`pytest -q`，0.5s） |
| GPU 渲染冒烟 | `test_render.py`：sapien 离屏渲染出图 | ✅ Render Well |
| 判定验证 | `setup_demo → play_once → check_success()` 返回 True | ✅ PASS（stack_blocks_two, seed 0） |
| adapter 运行时 | 真实 env：50 任务 / 14D / reset_episode / step / 维度与 NaN 校验 / env 复用 | ✅ 全部通过 |
| 确定性验收 | 同 seed 两次渲染逐位对比（同 env 复用 / 独立 env 各一次） | ✅ 图像与 qpos 逐位相同；异 seed 不同 |
| stub 时序闭环 | 2 task × 2 init-state，1:1 锁步 + 结果落盘 | ✅ `obs=1600 act=1600`；4 集指令交替；JSON `episodes=4=total`；4 集全 `failed`（400 步 env step_lim，metrics 空） |
| timed_out/failed 分离 | `max_steps: 5` 跑 1 集 vs 正常 400 步 | ✅ 同一任务两种终结：核心掐断 → `timed_out`(5 步)、env step_lim 判败 → `failed`(400 步)；启动 WARNING 正常触发 |
| 真策略小规模 | 2 task × 2 init-state 对 RoboTwin 2.0 论文基线（X_VLA） | ✅ 3/4=0.75；锁步 obs=827 act=827；adjust_bottle 2/2、beat_block_hammer 1/2 |
| num_runs>1 差异 | 同 init_state 不同 run_index 渲染对比 | ✅ 种子 100005 vs 200005，渲染图 51.6% 像素不同；同 run_index 两次 reset 逐位相同（PNG 见 `~/robotwin_deps/scratch/run_index_diff/`） |
| 中途取消 | 批次跑到 296/400 步时 dora SIGINT | ✅ 批次 `cancelled` 收尾，进行中的 episode 丢弃（`episodes=0`），JSON 照常落盘含 `metrics.resources`，节点干净退出 |
| 资源占用记录 | stub 批次末看 `metrics.resources` | ✅ `{cpu_percent: 584.5, rss_mb: 3695, gpu_util_percent: 69.0, gpu_mem_mb: 4618}`（取消批次实测值） |

---

## 8. 已知限制

**功能未实现**

- `record_dir` 的 mp4 落盘（配置项已留，无代码——照搬 LIBERO 版状态）
- wall-clock 看门狗——policy 卡死时节点静默悬停，只能靠 dora STOP 兜底
- 官方 eval 的 expert_check 专家预筛（见 §5 坑 7）
- 官方 eval 的 seen/unseen 指令口径（需要数据集里的场景参数做占位符填充），
  当前用 `full_description` 静态指令

资源占用已实现（adapter `_ResourceSampler`，批次末注入 `metrics.resources`）。

**设计上的欠缺**

- 相机别名（`image/head` 等）硬编码在 adapter，换 checkpoint 要改代码
- 启动竞态：`start` 命令与首帧观测走不同通道且无 ack，靠通道投递顺序侥幸成立
- 不打包 pyinstaller：LIBERO 版有 scripts/ 打包脚本，但 RoboTwin 资产 3G+、
  仿真依赖庞大，打包无意义，节点直接 `python main.py` 跑

---

## 9. 本机路径适配清单

仓库内的运行件保留了验收当次的本机绝对路径，在别的机器上跑之前改这几处：

| 文件 | 位置 | 内容 |
|:---|:---|:---|
| `config.example.yaml`（另存副本） | `robotwin_root` | RoboTwin 仓库根目录（或设 `ROBOTWIN_ROOT` 环境变量） |
| `dataflow.example.yaml`（另存副本） | 每个节点 `path:` | `$BENCH_PY` 解释器绝对路径（本机为 `/data7/home/shenxin/RoboTwin/.venv-robotwin/bin/python-real`；uv venv 需用 copy 出的真二进制，见 §5 坑 10） |
| `dataflow.example.yaml`（另存副本） | `env.BENCHMARK_CONFIG` | 相对节点目录的配置路径（保持 `./config.*.yaml`，在仓库根目录执行 dora run） |
| `dataflow.example.yaml`（另存副本） | `env.CUDA_VISIBLE_DEVICES` | 按机器 GPU 分配调整 |
| `policy_real.yaml`（自建，§3.2） | `model_path` | X_VLA ckpt-30000 解包目录 |
| `tools/run_full_suite.py` | 路径 | `ROBOTWIN_NODE_ROOT`（节点根）、`ROBOTWIN_ROOT`、`ROBOTWIN_MODEL_PATH`、`BENCH_PY`、`DORA_BIN` 均可用环境变量覆盖 |
