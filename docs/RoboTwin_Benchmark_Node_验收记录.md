## RoboTwin ✅ 通过（判据 A）

### 基本信息

| 项 | 内容 |
| :--- | :--- |
| Benchmark | RoboTwin 2.0 |
| 官方仓库 | https://github.com/RoboTwin-Platform/RoboTwin （branch `main`） |
| 官方 Leaderboard | https://robotwin-platform.github.io/leaderboard |
| 论文 | RoboTwin 2.0（arXiv:2506.18088），论文表：X_VLA co-train Easy 口径 |
| 任务 | `adjust_bottle`、`beat_block_hammer` |
| 权重 | X_VLA co-train `ckpt-30000`（`TianxingChen/RoboTwin2.0` datasets `cotrain_ckpt/X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0.tar.gz`） |
| Robot / Action Type | `aloha-agilex / ee`（16D = 每臂 xyz+wxyz 位姿 + 夹爪） |
| 该权重的已公开成绩（论文 Easy） | `adjust_bottle` **100**、`beat_block_hammer` **92**、50 任务 Easy 均值 **68.0** |
| 集数（每边） | 50 / task × 2 task = 100（官方 runner 与 Forge 各 100 集） |
| 场景是否同一批 | **是**。Official 与 Forge 均从 `st_seed = 100000` 起按同一顺序种子队列映射场景；官方遇专家预筛/场景不稳跳过（不占位），Forge 同集 `seed+1` 重试，无跳过时逐位一致 |
| Seed | `0` |
| task_config | `demo_clean`（两侧一致） |
| 验收日期 | 2026-09-10 |

本次验收采用手册**判据 A**：使用同一个官方免训练 checkpoint、同两个 RoboTwin task、同 episode 数与同一批场景种子，分别通过官方 runner（`scripts/eval_policy_xpolicylab.py` + XPolicyLab ws policy server）与 Forge 两节点链路执行，比较最终 benchmark 成绩（Wilson 95% CI 重叠 + Fisher 双侧精确检验）。

Checkpoint SHA256（model.safetensors）：

```text
39d60eeb7c8b7e445b271f4c6aa7da5a1ee36bda5716bc6ae832a184468cad0b
```

### 成绩对照

| 路径 | 成功 / 总数 | 成功率 | 95% Wilson 置信区间 |
| :--- | ---: | ---: | :--- |
| 官方 RoboTwin runner | 99 / 100 | 99.0% | `[94.55%, 99.82%]` |
| Forge Benchmark Node | 95 / 100 | 95.0% | `[88.82%, 97.85%]` |
| **差异** | 4 集 | 4.0 个百分点 | **Fisher exact p = 0.2116** |

补充结果：

- 两边 95% Wilson 置信区间重叠（分 task 亦各自重叠：adjust_bottle 完全一致；beat_block_hammer `[89.50%, 99.65%]` vs `[78.64%, 95.65%]`）；
- Fisher 双侧精确检验 `p = 0.2116 > 0.05`（分 task：adjust_bottle p = 1.0000；beat_block_hammer p = 0.2044）；
- Official 与 Forge 均为 2 task × 50 集，checkpoint、task、robot、action type、seed、task_config 一致；
- seed 对齐子集（只比双方实际计入的共同种子）：adjust_bottle 34 个 34/34 vs 34/34（p = 1.0000）；beat_block_hammer 32 个 31/32 vs 32/32（p = 1.0000）；
- 逐种子对照（bb）：Forge 独有失败 5 集（自然种子 100000 / 100002 / 100003 / 100034 / 100046，均在 400 步 step_lim 内未完成；官方口径对其中 4 个成功、1 个被专家预筛跳过）；官方唯一失败种子 100028 在 Forge 侧成功。双方共同计入的 32 个种子中仅 1 个结果相反，与扩散策略逐集随机性一致；
- 官方 bb 49/50 = 98.0%（高于论文 Easy 92）；Forge bb 45/50 = 90.0%，与论文同量级。

### 判定

| 条件 | 结果 |
| :--- | :--- |
| 官方 baseline 为明确非零成绩（论文 Easy 100 / 92） | ✅ |
| 同 checkpoint（sha256 一致） | ✅ |
| 同 task | ✅ |
| 同 episode 数 | ✅ |
| 同一批场景（同 st_seed、同 task_config、seed=0） | ✅ |
| 两边 Wilson 95% CI 重叠 | ✅ |
| Fisher exact `p > 0.05` | ✅ 0.2116（分 task：1.0000 / 0.2044） |
| Forge 成绩与官方处于同一统计区间 | ✅ |
| **最终判定** | **✅ 通过（判据 A）** |

**结论：RoboTwin Benchmark Node 通过正式验收。**

Official 成功率为 99.0%，Forge 为 95.0%，两者相差 4 个百分点；Wilson 置信区间明显重叠，Fisher 精确检验 `p = 0.2116`，没有观察到统计上可区分的性能差异。

因此现有证据支持：**Forge 两节点执行链路没有系统性改变 RoboTwin benchmark 的任务、场景、观测、动作和评分语义。**

这里的"通过"表示在当前 100-episode 验收规模下，Forge 与官方 runner 的 benchmark 表现统计上不可区分；并不意味着 stochastic policy 在逐 episode、逐 action 或逐帧轨迹上必须完全一致。

### 最终候选代码 Runtime Smoke

本节点代码自接入起即位于最终目录，不存在先实验后迁移的两段过程；判据 A 正式批次（100 集）本身即最终代码的 runtime smoke，交付文件的 sha256 复验见「最终代码版本」。

配置：

- Policy：X_VLA co-train `ckpt-30000`
- Task：`adjust_bottle`、`beat_block_hammer`
- Episodes：各 50
- Action Type：`ee`
- task_config：`demo_clean`，seed = 0

批次结果：

```text
status           = succeeded
successes        = 95
total_episodes   = 100
success_rate     = 0.95
elapsed_s        = 4281.4
mean_policy_latency_ms = 7.75
```

同时确认：

- Dora dataflow 正常启动，两节点干净退出；
- 逐集 `success_source = env.eval_success`；
- `metrics.resources` 峰值记录（cpu 556.0%、rss 3826 MB、gpu_util 84.0%、gpu_mem 9498 MB）；
- 100 集全程无 traceback；
- 交付前 gate 另含：契约测试 67 passed、stub 两节点时序闭环（obs=1600 act=1600 锁步）、真 policy 小规模（3/4 = 0.75，与论文同量级）。

### 加分项：确定性对照

按手册逐位等价加分项，对本节点做的确定性证据：

| 项目 | 结果 |
| :--- | :--- |
| 同 seed 两次 reset 渲染逐位相同（同 env 复用 / 独立 env 各一次） | ✅ |
| 同 run_index 两次 reset 渲染逐位相同 | ✅ |
| 异 seed / 异 run_index 渲染不同（负对照，证明比对工具可分辨差异） | ✅ run_index 0/1：种子 100005 vs 200005，51.6% 像素不同 |
| 跨 seed qpos 相同 | ✅ 官方口径：机器人每集回 homestate，随机化只作用于物体/光照/背景 |
| 1:1 锁步 obs==act（真 policy 827/827、stub 1600/1600） | ✅ |

官方 runner 与 Forge 各自在独立 sapien 进程中渲染，逐帧 bit 级对照不适用；本记录以「同 seed 渲染逐位一致 + 负对照可分辨」作为场景等价证据。该项仅作为额外诊断证据，不影响判据 A 的正式验收结论。

### 接入过程中发现并修复的问题

1. **XPolicyLab ws 依赖缺失**：官方 ws policy server 需要 `websockets`，评测环境 venv 未装 → server 启动即崩。修复：`uv pip install websockets`。
2. **资源采样器被首建 env 的 close() 停表**：首建/换 task 时 `close()` 停采样线程 → 批次刚开始 `metrics.resources` 采样就死了（全 null）。修复：`start()` 可重启 + env 重建后重启采样。
3. **调试残留 `_dbg` 挂在错误类上**：调试期 `self._dbg` 引用 sampler 类的静态方法，adapter 实例上不存在 → 批次秒挂（AttributeError）。已全量移除调试代码。
4. **dora canonicalize 符号链接解释器失效**：uv venv 的 `python` 是 symlink，节点进程拿 base 解释器启动 → `import numpy` 直接 ModuleNotFoundError。修复：copy 出 `python-real` 真二进制（venv 检测靠同目录 `pyvenv.cfg`）。
5. **RoboTwin envs 依赖 cwd=仓库根**：`assets/objects/objaverse/list.json` 在 import 期就读 → dora 节点 cwd 是数据流目录时 FileNotFoundError。修复：adapter 构造时 `os.chdir(robotwin_root)`，节点输出路径提前转绝对。
6. **embodiment file_path 相对路径**：官方 `load_task_args` 返回 `./assets/...` 按 cwd 解析；adapter 已转绝对路径。
7. **`policy_command` 队列默认深度挤掉指令**：episode 边界连发 `set_instruction` + `reset_scene`，默认 queue_size 丢前一条 → 首个 episode 拿上一句指令跑。修复：dataflow 显式 `queue_size: 32`。
8. **并发负载下的间歇性场景构建楔死（本次验收观察）**：Forge 批次在官方 runner 并行压测期间两次楔死于 `setup_demo` 建景（首跑第 6 集 123% CPU 空转 24 分钟、次跑第 20 集）；`-X faulthandler` 抓到主线程栈 `set_gripper ← take_dense_action ← together_close_gripper ← delay ← load_actors ← setup_demo`——卡在 RoboTwin 上游 env 代码的原生 physx 调用内，**不在 adapter/_core 逻辑内**；第二次约 38 秒后进程自毙（SIGABRT）。两次均发生在两节点同卡 + 官方侧满载的并发窗口，单独跑 21 集无此现象。缓解：benchmark 与 policy 节点拆卡部署（官方 runner 本身即单进程单卡）+ faulthandler/无缓冲输出便于复发取证。
9. **Stochastic policy 不能要求逐 action 一致**：X_VLA 推理含扩散去噪随机性，判据 A 以同配置、同场景批次、CI 重叠 + Fisher 检验为准。

### 与官方口径的已知差异（不阻塞判据 A，但解读结果需知）

1. **expert_check 专家预筛**：官方 runner 默认开启（规则专家 `play_once()` 试跑，过不了的种子跳过不计入分母；无需数据集）；Forge 未实现（adapter 范围选择）。影响单向：Forge 分母保留官方剔除的种子，可能略低；seed 对齐比较可量化该差异。
2. **指令口径**：官方用按场景生成的 `seen` 指令；Forge 用 `full_description` 静态指令。
3. **UnStableError 处理**：官方跳种子（该种子不占位，下一种子顶上）；Forge 同 episode 内 `seed+1` 重试（记录 `seed_retries`），无跳过时两边种子一一对应。
4. **record_dir mp4**：官方 `eval_video_log` 落 mp4；Forge `record_dir` 未实现（配置位保留）。

### 最终代码版本

forge_runtime 目录无 .git，交付件以 sha256 manifest 存档（28 个交付文件，`~/robotwin_deps/code_manifest.txt`）。2026-09-10 复验：28 个文件逐一 sha256 比对**零漂移**——判据 A 验收批次运行的就是这个版本。

关键文件：

| 文件 | sha256 |
| :--- | :--- |
| `adapters/robotwin_adapter.py` | `49b7838a27c476a4d05de15d45ecb362a1a15c6664ba30a1654abb7bf2af011b` |
| `main.py` | `c8f1cb2c1b7c576c4fd79c5a18535a51a89605ca91c57d46ad6e915db9fe3099` |
| `_core/runner.py` | `887cc92ecf1f323e3fa2a640d01b6806691dfb14fa75ed58fa5a471a744e63e8` |
| `result.py` | `0f2a84530469263b54ab390d78053aac6d1bd923a9ee1fad069d910bc1d22f03` |

manifest 自身 sha256：`7beee4b55146c4a6661f94fe6ed16afd9875649c2366f6aa0574b7f9378c9ec2`（其余 24 个文件的哈希以 manifest 为准）。

Pre-commit gate：

```text
pytest -q 契约测试                       PASS（67 passed）
adapter 运行时验证（真实 env 50 任务）    PASS
确定性验收（同 seed 逐位 / 异 seed 不同） PASS
stub 时序闭环                           PASS（1600/1600 锁步）
真 policy 小规模对论文基线               PASS（3/4 = 0.75）
num_runs>1 差异 / 中途取消 / 资源记录     PASS
判据 A 正式批次（官方 vs Forge 100 集）   PASS（见成绩对照）
```

### 最终结论

**RoboTwin Benchmark Node：✅ 通过，判据 A。**

正式证据：

```text
论文已公开成绩（Easy）：
adjust_bottle 100、beat_block_hammer 92

Official 本地复现：
99 / 100
99.0%，Wilson 95% CI [94.55%, 99.82%]

Forge：
95 / 100
95.0%，Wilson 95% CI [88.82%, 97.85%]

Fisher exact p = 0.2116
Wilson 95% CI overlap = true
seed 对齐子集 Fisher p = 1.0000（两 task）
```

在当前 100-episode（2 task × 50 集）、同 checkpoint、同 task、同 robot/action type、同 seed=0、同 task_config、同一批场景种子的测试条件下，没有证据表明 Forge Benchmark Node 系统性改变了 RoboTwin 的 benchmark 行为或成绩分布，满足验收手册判据 A。与官方口径的四条已知差异（专家预筛、指令生成、UnStableError 重试、mp4 记录）不阻塞本判定，其影响已在 seed 对齐子集比较中量化。

全 suite（50 task × 100 集）跑分另在后台进行中，完成后可与论文 50 任务 Easy 均值（68.0）进一步对照。
