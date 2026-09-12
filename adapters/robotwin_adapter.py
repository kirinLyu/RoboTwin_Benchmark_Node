"""RoboTwin adapter：env 生命周期、观测转换、成功判定。

benchmark 特有的东西全部收敛在这里；_core 的调度不认识 RoboTwin。
结构与 LIBERO adapter 对齐（手册 §5.2 / _core/adapter.py 协议），
benchmark 特有逻辑照抄官方 eval（scripts/eval_policy_xpolicylab.py）的口径：

- 场景重建 = setup_demo(now_ep_num, seed, is_test=True, **task_args)，每次重置整场；
- seed 口径 = 官方顺序种子队列：每 (task, run_index) 一个游标，从
  `100000 * (1 + seed + run_index)` 起只增不减，接受即推进、跳过不回收——
  与官方 `now_seed` 游标逐位同语义（run_index>0 偏移一个种子域，供
  num_runs>1 产生差异）。遇 UnStableError / expert 预筛失败 seed+1 重试；
- expert_check = 官方专家预筛：setup_demo → play_once() → close_env()，
  `plan_success and check_success()` 不过则该 seed 跳过；通过后**重新 setup**
  同一 seed 作为本集正式场景（官方同款两次 setup，eval_policy_xpolicylab.py:805-843）；
- 指令 = 官方 `seen` 生成式指令：用 play_once 返回的 episode_info 经
  `generate_episode_descriptions` 填充占位符（官方 build_instruction 同款）；
- 判定 = env.eval_success（take_action 内部逐 sim step 调 check_success() 置位）；
- episode 结束 = eval_success or take_action_cnt >= step_lim；
- env 生命周期 = 官方口径：每集一次 close_env（本实现放在下一集指令阶段开头），
  换 task 重建 env；
- 官方 policy server 协议层被 dora 通道替代，本 adapter 只做 env 驱动与转换；
- 资源记录：后台采样 CPU/GPU 峰值，批次末经 aggregate() 注入结果 JSON；
- 场景 seed 台账：每集实际使用的 seed（含重试数）以批次级 ledger 注入结果
  JSON 的 metrics.scene_seed_ledger——失败集 metrics 必须为空（三态契约），
  失败集的场景 seed 靠台账补全，供判据 A 的 seed 对齐统计使用。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import psutil
except ImportError:  # 评测环境可能没有，CPU 峰值缺失不影响跑分
    psutil = None

def _import_core(name):
    """导 _core.<name>；父目录只临时挂上，避免遮蔽同名仿真器包（见 main.py 注释）。"""
    import importlib

    try:
        return importlib.import_module("_core." + name)
    except ImportError:
        p = str(Path(__file__).resolve().parents[2])
        sys.path.insert(0, p)
        try:
            return importlib.import_module("_core." + name)
        finally:
            if p in sys.path:
                sys.path.remove(p)


_m = _import_core("adapter")
StepOutcome = _m.StepOutcome

# 官方相机名 → dora 通道名（eval_policy_xpolicylab.py 的 CAMERA_NAME_MAP 左侧是
# RoboTwin 观测键，右侧换成 dora 通道；policy 节点的 image_inputs 左键要对上这里）。
CAMERA_NAME_MAP = {
    "head_camera": "image/head",
    "left_camera": "image/left_wrist",
    "right_camera": "image/right_wrist",
}

# ee 模式 proprio 通道的关节名（与 to_state 的 16D 拼接顺序逐位对应）。
# 位姿四元数是 wxyz（env get_arm_pose 的 mat2quat 输出口径）。
EE_STATE_NAMES = [
    *(f"left_ee_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    "left_gripper",
    *(f"right_ee_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    "right_gripper",
]

# 官方 eval 的 seed 起点：st_seed = 100000 * (1 + seed)。官方用顺序种子队列
# （now_seed 游标只增不减：接受即推进，专家预筛/场景不稳跳过的 seed 永不回收）；
# 本 adapter 同语义，按 (task, run_index) 各维护一个游标，run_index 偏移由
# _seed_base 处理——修复「init_state_id 直映 + 原地重试」导致的跨集种子碰撞。
SEED_STRIDE = 100_000
# 官方 batch eval 默认 max_seed_attempts = test_num * 50；单进程串行用 200 足够。
MAX_SEED_RETRIES = 200

# 动作/状态维度**不是常数**，由 embodiment 配置推导：
#   qpos 模式：每臂 = len(arm_joints_name[i]) 个臂关节 + 1 个夹爪值
#   （get_left_arm_jointState 的末尾追加项）。aloha-agilex 是 6+1 + 6+1 = 14 维
#   （ALOHA 标准口径），**不是 16**——arm_joints_name 里没有夹爪，夹爪名在
#   gripper_name[i]["base"]。
#   ee 模式：每臂 = 7D 位姿（xyz + wxyz 四元数，与 take_action 的 quat2mat 口径
#   一致）+ 1 夹爪 = 16D（官方 eval 的 xpolicylab_action_to_robotwin 扁平口径）。


class _ResourceSampler:
    """后台采样进程 CPU/GPU 峰值（手册完成标准：资源占用已记录）。

    每 `interval` 秒采一次，stop() 停线程并返回峰值快照
    {cpu_percent, rss_mb, gpu_util_percent, gpu_mem_mb}，采不到的键为 None
    （psutil / nvidia-smi 缺失时降级，不影响跑分）。
    """

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peaks: dict[str, Any] = {
            "cpu_percent": None,
            "rss_mb": None,
            "gpu_util_percent": None,
            "gpu_mem_mb": None,
        }
        self._thread: threading.Thread | None = None
        # GPU 采样器链：nvidia-smi（尊重 CUDA_VISIBLE_DEVICES）→ pynvml →
        # torch 显存峰值（只有 mem，没有 util）。
        self._gpu_sampler = self._pick_gpu_sampler()

    # ---- GPU 采样器选择 ----

    @staticmethod
    def _pick_gpu_sampler():
        def via_nvidia_smi() -> tuple[float | None, float | None]:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None, None
            line = out.stdout.strip().splitlines()[0].strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                return None, None
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None, None

        try:
            subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, timeout=5, check=True
            )
            return via_nvidia_smi
        except Exception:  # noqa: BLE001
            pass
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            def via_pynvml() -> tuple[float | None, float | None]:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                    return float(util), float(mem)
                except Exception:  # noqa: BLE001
                    return None, None

            return via_pynvml
        except Exception:  # noqa: BLE001
            pass

        def via_torch() -> tuple[None, float | None]:
            try:
                import torch

                if torch.cuda.is_available():
                    return None, torch.cuda.max_memory_allocated() / 1024**2
            except Exception:  # noqa: BLE001
                pass
            return None, None

        return via_torch

    # ---- 生命周期 ----

    def start(self) -> None:
        """起采样线程。幂等，且 stop() 之后可再次 start()（换 env 重建场景时
        _ensure_env 会先 close() 停一次，重建完靠这里续上）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        # stop() 之后旧 Event 已置位，重开必须换新的。
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """停采样线程，返回峰值快照（幂等，可重复调用；峰值跨 start/stop 累积）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            self._thread = None
        with self._lock:
            return dict(self._peaks)

    def _run(self) -> None:
        proc = psutil.Process() if psutil is not None else None
        gpu_every = max(1, int(5.0 / max(self._interval, 0.1)))
        tick = 0
        while not self._stop.wait(self._interval):
            tick += 1
            cpu_pct = rss = None
            if proc is not None:
                try:
                    cpu_pct = float(proc.cpu_percent())
                    rss = float(proc.memory_info().rss) / 1024**2
                except Exception:  # noqa: BLE001 进程采样失败不致命
                    pass
            gpu_util = gpu_mem = None
            if tick % gpu_every == 1:  # nvidia-smi 有进程开销，降频采
                try:
                    gpu_util, gpu_mem = self._gpu_sampler()
                except Exception:  # noqa: BLE001
                    pass
            with self._lock:
                p = self._peaks
                if cpu_pct is not None:
                    p["cpu_percent"] = max(p["cpu_percent"] or 0.0, cpu_pct)
                if rss is not None:
                    p["rss_mb"] = max(p["rss_mb"] or 0.0, rss)
                if gpu_util is not None:
                    p["gpu_util_percent"] = max(p["gpu_util_percent"] or 0.0, gpu_util)
                if gpu_mem is not None:
                    p["gpu_mem_mb"] = max(p["gpu_mem_mb"] or 0.0, gpu_mem)


class RobotwinAdapter:
    """进程内持有 RoboTwin env，不经过 XPolicyLab / policy server。"""

    SUCCESS_SOURCE = "env.eval_success"

    # ee 模式动作 = 每臂 [x, y, z, qw, qx, qy, qz] + 1 夹爪 = 16D。
    EE_ACTION_DIM = 16

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.root = Path(cfg.robotwin_root).resolve()
        # envs/ 与 scripts/ 都是 RoboTwin 仓库内的包，必须进 sys.path 才能 import。
        # （资产路径**不是**都从模块位置推导：envs/_GLOBAL_CONFIGS.py 是，但 envs/utils
        #  里还有一批 `./assets/...`，依赖 cwd——见下面 chdir。）
        for p in (str(self.root), str(self.root / "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)

        # RoboTwin 的 envs 代码里有大量 `./assets/...` 相对路径，其中一部分在
        # **import 时**就执行（envs/utils/rand_create_cluttered_actor.py:17 模块级
        # 读 assets/objects/objaverse/list.json），官方 eval 也是在仓库根目录跑的。
        # 所以本进程必须把 cwd 切到仓库根，否则 import envs 直接 FileNotFoundError。
        # 节点自己的输出路径（result_dir/record_dir）已在 load_config 里转成绝对，
        # 不受这次 chdir 影响。
        os.chdir(self.root)

        self.task_names = self._discover_tasks()
        self._step_limits = self._load_step_limits()
        self._task_args_cache: dict[str, dict[str, Any]] = {}
        self._task_language_cache: dict[str, str] = {}
        self._joint_names: list[str] | None = None

        self._env = None
        self._env_task: str | None = None
        self._last_scene_seed: int | None = None
        self._last_seed_retries: int = 0
        # 官方顺序种子队列：{(task_name, run_index): 下一个候选 seed}。
        # 只增不减——接受与跳过都推进，接受过的 seed 永不重复（官方 now_seed 同语义）。
        self._seed_cursor: dict[tuple[str, int], int] = {}
        # 每集实际场景 seed 台账（含失败集；失败集 metrics 必须为空以维持三态
        # 契约，种子由批次级 ledger 补全）。append 顺序 = episode 计划顺序。
        self._scene_seed_ledger: list[dict[str, Any]] = []
        # 已就位的本集场景：(task_name, init_state_id, run_index, seed, retries,
        # instruction)。由 episode_instruction（官方协议扫描）就位，reset_episode
        # 消费——runner 先取指令再 reset，指令必须在场景就位后才能生成（seen
        # 依赖 play_once 的 episode_info），这个顺序是刻意的。
        self._standing: tuple[str, int, int, int, int, str] | None = None
        # 指令候选选择用 np.random.choice（官方 build_instruction 同款 API）。
        # 官方 eval 脚本不 seed 任何 RNG（其指令字符串逐 run 变化，见验收报告
        # §3.4 的官方自身不确定性）；本节点在构造时按 cfg.seed 固定一次起点，
        # 使本节点自身的指令选择逐 run 可复现（与官方同分布，实例不逐字一致——
        # 官方自己也无法逐字一致）。
        np.random.seed(int(self.cfg.seed))

        max_limit = max(self._step_limits.values())
        if self.cfg.max_steps < max_limit:
            print(
                f"[robotwin_bench] WARNING: max_steps={self.cfg.max_steps} 小于最大 "
                f"step_lim={max_limit}，对应任务的 episode 会被核心提前掐成 timed_out，"
                f"分数口径与官方 eval 不一致（官方跑到 step_lim 为止）",
                flush=True,
            )
        self._startup_selfcheck()

        # 资源峰值采样（手册完成标准：资源占用已记录）。批次末 aggregate()
        # 停表并注入结果 JSON；close() 兜底停一次。
        self._sampler = _ResourceSampler()
        self._sampler.start()

    def _startup_selfcheck(self) -> None:
        """启动自检：把本节点的观测口径打进日志，并与 cfg.checkpoint 声明的
        checkpoint 口径交叉核对（手册 §3.2 分辨率分层契约 / §6 第 3 步 / §8-9）。

        分辨率/维度/别名填错都不报错、只会安静掉分，只能靠这里拦住。
        """
        args0 = self._task_args(self.task_names[0])
        head_w, head_h = args0.get("head_camera_w"), args0.get("head_camera_h")
        names = self.state_joint_names()
        print("[robotwin_bench] === 启动自检：观测口径 ===", flush=True)
        print(
            f"[robotwin_bench] camera: head={head_w}x{head_h} "
            f"（{self.cfg.task_config} → _camera_config.yml）",
            flush=True,
        )
        print(f"[robotwin_bench] control_mode={self.cfg.control_mode}", flush=True)
        print(f"[robotwin_bench] proprio: {len(names)}D {names}", flush=True)
        print(f"[robotwin_bench] action: {self.action_dim}D", flush=True)
        print(
            "[robotwin_bench] image 通道: " + " ".join(CAMERA_NAME_MAP.values()),
            flush=True,
        )

        chk = dict(getattr(self.cfg, "checkpoint", None) or {})
        if not chk:
            print(
                "[robotwin_bench] 未声明 checkpoint 口径（cfg.checkpoint 为空），"
                "跳过交叉核对——正式接 checkpoint 时必须填写。",
                flush=True,
            )
            return
        # 分辨率分层契约（§3.2）：有 resize_images_to → policy 自带缩放，喂原生
        # 分辨率正确（benchmark 不提前缩）；没有且 input_features 与本节点原生
        # 分辨率不一致 → 双重重采样或形状不符，分数会安静掉。
        in_feat = chk.get("input_features")
        if in_feat:
            c, h, w = (int(x) for x in in_feat)
            resize = chk.get("resize_images_to")
            if resize:
                print(
                    f"[robotwin_bench] checkpoint: input_features={in_feat}, "
                    f"resize_images_to={resize} → policy 自带缩放，喂原生 "
                    f"{head_w}x{head_h} 正确（benchmark 不提前缩）",
                    flush=True,
                )
            elif (h, w) == (int(head_h), int(head_w)):
                print(
                    f"[robotwin_bench] checkpoint: input_features={in_feat} 与"
                    f"本节点原生分辨率一致",
                    flush=True,
                )
            else:
                print(
                    f"[robotwin_bench] WARNING: checkpoint input_features="
                    f"{in_feat} 与本节点原生 {head_w}x{head_h} 不一致，且没有 "
                    f"resize_images_to——分辨率口径不符！",
                    flush=True,
                )
        if "state_dim" in chk and int(chk["state_dim"]) != len(names):
            print(
                f"[robotwin_bench] WARNING: checkpoint state_dim="
                f"{chk['state_dim']} 与本节点 {len(names)}D 不一致",
                flush=True,
            )
        if chk.get("state_joints"):
            theirs = [str(n) for n in chk["state_joints"]]
            if theirs != names:
                print(
                    f"[robotwin_bench] WARNING: checkpoint state_joints 与本节点"
                    f"关节名不一致：\n  theirs={theirs}\n  ours  ={names}",
                    flush=True,
                )
        print("[robotwin_bench] === 启动自检结束 ===", flush=True)

    # ---- 给结果文件的复现口径 ----

    def reproducibility(self) -> dict[str, Any]:
        return {
            "task_config": self.cfg.task_config,
            "control_mode": self.cfg.control_mode,
            "test_num": self.cfg.test_num,
            "expert_check": bool(self.cfg.expert_check),
            "instruction_type": self.cfg.instruction_type,
            "scene_seed": (
                f"官方顺序种子队列：每 (task, run_index) 一个游标，从 "
                f"100000 * (1 + {self.cfg.seed} + run_index) 起只增不减；"
                "UnStableError / expert 预筛失败跳过（seed+1 不回收），接受即推进。"
                "与官方 eval 的 now_seed 游标同语义（run_index=0 时种子域与官方"
                "逐位相同）；每集实际 seed 与重试数记入结果 JSON 的 "
                "metrics.scene_seed_ledger（含失败集）"
            ),
            "step_limits": "env_cfg/task_config/_eval_step_limit.yml（每任务 400~1700）",
            "action_layout": (
                f"qpos：每臂 arm_joints_name + 1 夹爪（gripper_name.base，aloha-agilex "
                f"14D）；ee：每臂 xyz + wxyz 位姿 + 1 夹爪（16D）。"
                f"当前 control_mode={self.cfg.control_mode}，{self.action_dim}D："
                f"{', '.join(self.state_joint_names())}"
            ),
            "camera": "D435（env_cfg/task_config/_camera_config.yml，320x240）",
            "robotwin_root": str(self.root),
            "instruction": (
                f"官方生成式指令（instruction_type={self.cfg.instruction_type}）："
                "play_once 的 episode_info → generate_episode_descriptions 填充"
                "占位符 → np.random.choice 选候选（官方 build_instruction 同款；"
                "官方 eval 脚本不 seed 任何 RNG，其指令字符串逐 run 变化，本节点按 "
                "cfg.seed 固定自身选择、逐 run 可复现）。生成失败或 instruction_type="
                "full_description 时回退 description/task_instruction/<task>.json 的"
                "full_description"
            ),
        }

    def state_joint_names(self) -> list[str]:
        """proprio 通道的关节名：qpos 模式 = embodiment 关节名（aloha-agilex 14D）；
        ee 模式 = 每臂 xyz+wxyz 位姿 7 名 + 1 夹爪 = 16D。"""
        if self.cfg.control_mode == "ee":
            return EE_STATE_NAMES
        if self._joint_names is None:
            self._joint_names = self._load_joint_names()
        return self._joint_names

    @property
    def action_dim(self) -> int:
        """动作维度随 control_mode 变：qpos = 关节名个数（aloha-agilex 14D）；
        ee = EE_ACTION_DIM（16D）。"""
        if self.cfg.control_mode == "ee":
            return self.EE_ACTION_DIM
        return len(self.state_joint_names())

    # ---- 批次展开需要的元信息 ----

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    def num_init_states(self, task_id: int) -> int:
        # RoboTwin 没有 LIBERO 式的预存 init state：每个"init state"= 一个
        # 测试场景 seed（官方 eval 100 个），场景在 reset 时按 seed 重建。
        return self.cfg.test_num

    def task_language(self, task_id: int) -> str:
        name = self.task_names[task_id]
        if name not in self._task_language_cache:
            self._task_language_cache[name] = self._load_language(name)
        return self._task_language_cache[name]

    def step_limit(self, task_id: int) -> int:
        return self._step_limits[self.task_names[task_id]]

    # ---- env 生命周期 ----

    def _ensure_env(self, task_id: int):
        """同一个 task 复用 env；换 task 才重建（建 env 与场景都贵）。"""
        name = self.task_names[task_id]
        if self._env is not None and self._env_task == name:
            return self._env

        self.close()
        import importlib

        env_module = importlib.import_module(f"envs.{name}")
        env_class = getattr(env_module, name)
        self._env = env_class()
        self._env_task = name
        # close() 把采样线程停掉了（首个 env 也一样），这里续上——close 的语义
        # 是"释放 env"，不是"结束批次"，采样必须覆盖整个批次。
        self._sampler.start()
        return self._env

    def _require_env(self):
        if self._env is None:
            raise RuntimeError("env 尚未创建，先调用 reset_episode")
        return self._env

    # ---- episode ----

    @staticmethod
    def scene_seed(cfg_seed: int, init_state_id: int, run_index: int) -> int:
        """官方公式 + run_index 偏移：100000 * (1 + seed + run_index) + init_state_id。

        仅作种子域基线公式保留（无跳过时的映射），调度不再用直映——顺序种子
        队列见 _seed_cursor / episode_instruction。run_index=0 时与官方公式
        逐位相同；run_index>0 偏移到另一个种子域——手册完成标准要求
        num_runs > 1 必须产生差异，否则多次 run 退化成重复计算。
        """
        return SEED_STRIDE * (1 + int(cfg_seed) + int(run_index)) + int(init_state_id)

    def _seed_base(self, task_name: str, run_index: int) -> int:
        """(task, run_index) 的种子域起点：100000 * (1 + seed + run_index)。"""
        return SEED_STRIDE * (1 + int(self.cfg.seed) + int(run_index))

    def _advance_cursor(self, task_name: str, run_index: int) -> int:
        """返回下一个候选 seed 并推进游标（只增不减，接受与跳过同口径）。"""
        key = (task_name, run_index)
        cur = self._seed_cursor.get(key, self._seed_base(task_name, run_index))
        self._seed_cursor[key] = cur + 1
        return cur

    def _build_instruction(self, task_name: str, episode_info: dict[str, Any]) -> str:
        """官方 build_instruction 同款：episode_info → seen/unseen 模板填充。

        episode_info 来自 play_once() 的返回（{"info": {占位符: 场景参数}}，
        官方 eval_policy_xpolicylab.py:982-986 的 episode_info_list 口径）。
        候选为空（expert_check 关闭时无 episode_info）回退 full_description。
        """
        if self.cfg.instruction_type in ("seen", "unseen") and episode_info:
            try:
                # 官方 eval 的导入路径是 description/utils（eval_policy_xpolicylab.py:27）
                sys.path.insert(0, str(self.root / "description" / "utils"))
                try:
                    from generate_episode_instructions import (  # noqa: PLC0415
                        generate_episode_descriptions,
                    )
                finally:
                    sys.path.remove(str(self.root / "description" / "utils"))
                # 官方 build_instruction 口径：只取 play_once 返回的 "info" 内层字典
                results = generate_episode_descriptions(
                    task_name, [episode_info.get("info", {})], self.cfg.test_num
                )
                candidates = results[0].get(self.cfg.instruction_type) if results else []
                if candidates:
                    return str(np.random.choice(candidates))
            except Exception as exc:  # noqa: BLE001 生成失败退回静态指令，不致命
                print(
                    f"[robotwin_bench] WARNING: 生成 {self.cfg.instruction_type} 指令失败"
                    f"（{type(exc).__name__}: {exc}），回退 full_description",
                    flush=True,
                )
        return self._load_language(task_name)

    def episode_instruction(self, task_id: int, init_state_id: int, run_index: int) -> str:
        """官方协议的整段场景扫描 + 本集指令生成（runner 在 reset 前调用）。

        对应官方 eval 的 while succ_seed < test_num 循环（eval_policy_xpolicylab.py
        :795-856）：
          1. 顺序种子队列取候选 seed；
          2. setup_demo；UnStableError → 跳过（seed+1，不回收）；
          3. expert_check：play_once() + close_env()，plan_success and
             check_success() 不过 → 跳过；通过则**重新 setup 同一 seed** 作为
            本集正式场景（官方同款两次 setup）；
          4. 正式 setup 再 UnStableError → 跳过；
          5. 接受：推进游标，场景**留在就位状态**由 reset_episode 消费，
             用 play_once 的 episode_info 生成 seen 指令。
        episode 边界关 env（官方每集 close_env，eval_policy_xpolicylab.py:943）。
        """
        env = self._ensure_env(task_id)
        name = self.task_names[task_id]
        args = self._task_args(name)

        # 每集一次的 env 关闭（官方口径；第一集之前没有旧场景，跳过）。
        if self._last_scene_seed is not None:
            try:
                env.close_env()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[robotwin_bench] WARNING: close_env 失败（忽略）："
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        from envs.utils.create_actor import UnStableError  # noqa: PLC0415

        attempts = 0
        while True:
            seed = self._advance_cursor(name, run_index)
            retries = seed - self._seed_base(name, run_index)
            try:
                env.setup_demo(now_ep_num=int(init_state_id), seed=int(seed), is_test=True, **args)
            except UnStableError:
                print(f"[robotwin_bench] skip unstable seed={seed} (expert setup)", flush=True)
                attempts += 1
                if attempts >= MAX_SEED_RETRIES:
                    raise RuntimeError(
                        f"task {name} init_state {init_state_id}：连续 "
                        f"{MAX_SEED_RETRIES} 个候选 seed 未通过，放弃"
                    ) from None
                continue

            episode_info: dict[str, Any] = {}
            if self.cfg.expert_check:
                try:
                    episode_info = env.play_once()
                    expert_ok = bool(env.plan_success and env.check_success())
                except Exception as exc:  # noqa: BLE001 官方对 expert 期异常同样跳过
                    print(
                        f"[robotwin_bench] expert check error seed={seed}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    expert_ok = False
                env.close_env()
                if not expert_ok:
                    print(f"[robotwin_bench] skip expert-fail seed={seed}", flush=True)
                    attempts += 1
                    if attempts >= MAX_SEED_RETRIES:
                        raise RuntimeError(
                            f"task {name} init_state {init_state_id}：连续 "
                            f"{MAX_SEED_RETRIES} 个候选 seed 未通过，放弃"
                        ) from None
                    continue

            # 正式场景：官方在 expert 通过后重新 setup 同一个 seed（两步都过才算接受）
            try:
                env.setup_demo(now_ep_num=int(init_state_id), seed=int(seed), is_test=True, **args)
            except UnStableError:
                print(f"[robotwin_bench] skip unstable seed={seed} (eval setup)", flush=True)
                attempts += 1
                if attempts >= MAX_SEED_RETRIES:
                    raise RuntimeError(
                        f"task {name} init_state {init_state_id}：连续 "
                        f"{MAX_SEED_RETRIES} 个候选 seed 未通过，放弃"
                    ) from None
                continue
            except Exception as exc:  # noqa: BLE001 官方对正式 setup 的通用异常也跳过
                print(f"[robotwin_bench] skip seed={seed} setup error: {type(exc).__name__}: {exc}", flush=True)
                attempts += 1
                if attempts >= MAX_SEED_RETRIES:
                    raise RuntimeError(
                        f"task {name} init_state {init_state_id}：连续 "
                        f"{MAX_SEED_RETRIES} 个候选 seed 未通过，放弃"
                    ) from None
                continue
            break

        instruction = self._build_instruction(name, episode_info)
        self._last_scene_seed = int(seed)
        self._last_seed_retries = int(retries)
        self._standing = (name, int(init_state_id), int(run_index), int(seed), int(retries), instruction)
        print(
            f"[robotwin_bench] seed accepted task={name} i{init_state_id} "
            f"r{run_index} seed={seed} retries={retries}",
            flush=True,
        )
        return instruction

    def reset_episode(
        self, task_id: int, init_state_id: int, run_index: int, episode_seed: int
    ) -> dict[str, Any]:
        env = self._ensure_env(task_id)
        name = self.task_names[task_id]

        # runner 先调 episode_instruction 取指令、再调本方法 reset（顺序保证），
        # 场景此时已就位；就位信息不匹配时（异常路径）现场重扫一遍兜底。
        standing = self._standing
        if standing is None or standing[:3] != (name, int(init_state_id), int(run_index)):
            self.episode_instruction(task_id, init_state_id, run_index)
            standing = self._standing
        assert standing is not None
        _, _, _, seed, retries, instruction = standing
        self._standing = None
        self._last_scene_seed = int(seed)
        self._last_seed_retries = int(retries)
        # 场景 seed 台账（含失败集；失败集 metrics 为空，种子由批次级 ledger 补全）。
        self._scene_seed_ledger.append(
            {
                "task_id": int(task_id),
                "init_state_id": int(init_state_id),
                "run_index": int(run_index),
                "scene_seed": int(seed),
                "seed_retries": int(retries),
                "instruction": instruction,
            }
        )
        return env.get_obs()

    def step(self, action: np.ndarray) -> StepOutcome:
        env = self._require_env()
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = self.action_dim
        if act.size != expected or not np.all(np.isfinite(act)):
            raise ValueError(
                f"RoboTwin action must contain {expected} finite values, "
                f"got size={act.size}"
            )
        env.take_action(act, action_type=self.cfg.control_mode)
        obs = env.get_obs()
        success = bool(env.eval_success)
        done = bool(success or env.take_action_cnt >= env.step_lim)
        # metrics 决定核心的 termination 分类：done & !success 时必须给空字典，
        # 非空会被记成 partial——RoboTwin 没有部分完成口径（官方按失败计）。
        if success:
            metrics = {
                "success": True,
                "scene_seed": self._last_scene_seed,
                "seed_retries": self._last_seed_retries,
            }
        elif done:
            metrics = {}
        else:
            metrics = {"success": False}
        return StepOutcome(
            obs=obs,
            reward=1.0 if success else 0.0,
            done=done,
            success=success,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close_env()
            except Exception:  # noqa: BLE001 关闭失败不该影响批次收尾
                pass
        self._env = None
        self._env_task = None
        # 兜底停采样线程（批次末 aggregate 已停过一次，这里幂等；_ensure_env
        # 重建场景后会重新 start，见那边注释）。
        self._sampler.stop()

    def aggregate(self, episode_metrics: list[dict[str, Any]]) -> tuple[dict, dict]:
        """默认成功率（分母 = 计划总数，与核心 default_aggregate 同口径）+ 资源
        峰值快照 + 场景 seed 台账。覆盖 aggregate 钩子只为把资源记录与台账带进
        结果 JSON 的 metrics。

        台账是失败集场景 seed 的唯一载体：失败集 episode metrics 必须为空（三态
        契约里 done & !success & 空 metrics = failed），所以不能在逐集 metrics 里
        带 seed；台账按 episode 计划顺序逐条记录，统计工具按
        (task_id, init_state_id, run_index) 匹配回各集（含失败集）。
        """
        successes = sum(
            1 for m in episode_metrics if isinstance(m, dict) and m.get("success")
        )
        try:
            planned = len(self.cfg.episode_plan())
        except Exception:  # noqa: BLE001 计划不可得时退回已完成数
            planned = len(episode_metrics)
        primary = {
            "name": "success_rate",
            "value": successes / (planned or 1),
            "max": 1.0,
        }
        extra: dict[str, Any] = {"resources": self._sampler.stop()}
        if self._scene_seed_ledger:
            extra["scene_seed_ledger"] = list(self._scene_seed_ledger)
        return primary, extra

    # ---- 观测转换 ----

    def to_state(self, obs: dict[str, Any]) -> np.ndarray | None:
        """qpos 模式：proprio = [左臂关节…, 左夹爪, 右臂关节…, 右夹爪]（14D）。

        官方口径直接读 joint_action.vector（= left_jointstate + right_jointstate），
        顺序与 state_joint_names() 逐位对应。维度随 embodiment 变，不硬编码。

        ee 模式：proprio = [left_endpose(7), left_gripper(1), right_endpose(7),
        right_gripper(1)] = 16D，照官方 eval 的 convert_state 口径——位姿取
        endpose（xyz + wxyz 四元数），夹爪取 joint_action。
        """
        if self.cfg.control_mode == "ee":
            endpose = obs.get("endpose") or {}
            joint_action = obs.get("joint_action") or {}
            parts = [
                endpose.get("left_endpose"),
                joint_action.get("left_gripper"),
                endpose.get("right_endpose"),
                joint_action.get("right_gripper"),
            ]
            if any(p is None for p in parts):
                return None
            return np.concatenate(
                [np.asarray(p, dtype=np.float32).reshape(-1) for p in parts]
            ).astype(np.float64)
        joint_action = obs.get("joint_action") or {}
        vector = joint_action.get("vector")
        if vector is not None:
            return np.asarray(vector, dtype=np.float32).reshape(-1).astype(np.float64)
        # 兜底：vector 缺失时按官方数据口径手工拼接。
        parts = [
            joint_action.get("left_arm"),
            joint_action.get("left_gripper"),
            joint_action.get("right_arm"),
            joint_action.get("right_gripper"),
        ]
        if any(p is None for p in parts):
            return None
        state = np.concatenate(
            [np.asarray(p, dtype=np.float32).reshape(-1) for p in parts]
        )
        return state.astype(np.float64)

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """官方口径直接送 rgb，**不翻转**。

        LIBERO 的 [::-1,::-1] 是 mujoco 离屏渲染特有问题；RoboTwin 官方 eval
        拿 get_obs() 的 rgb 原样喂 policy，这里照做。
        """
        source = obs.get("observation") or {}
        frames: dict[str, np.ndarray] = {}
        for camera_name, output_id in CAMERA_NAME_MAP.items():
            entry = source.get(camera_name)
            if entry is None or "rgb" not in entry:
                continue
            frames[output_id] = np.ascontiguousarray(
                np.asarray(entry["rgb"], dtype=np.uint8)
            )
        if not frames:
            raise RuntimeError("观测里没有任何相机图像（检查 task_config 的 camera 开关）")
        return frames

    # ---- RoboTwin 仓库内的静态配置 ----

    def _discover_tasks(self) -> list[str]:
        """任务清单 = envs/ 下的任务模块（文件名 = 类名 = 官方 eval 的 task_name）。

        构造时验证每个模块都有同名类（fail fast：缺类要留到建 env 才炸的话，
        批次已经开跑了）。目录（robot/camera/utils）与下划线开头的私有模块
        被 glob("*.py") 自动排除。
        """
        names = sorted(
            p.stem
            for p in (self.root / "envs").glob("*.py")
            if not p.stem.startswith("_")
        )
        if not names:
            raise RuntimeError(f"envs/ 下没有发现任务模块：{self.root / 'envs'}")
        import importlib

        for name in names:
            module = importlib.import_module(f"envs.{name}")
            if not isinstance(getattr(module, name, None), type):
                raise RuntimeError(f"envs/{name}.py 里没有 {name} 类")
        return names

    def _load_step_limits(self) -> dict[str, int]:
        """每任务的 step_lim（官方 _init_task_env_ 同口径：缺省 1000）。"""
        path = self.root / "env_cfg" / "task_config" / "_eval_step_limit.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {name: int(data.get(name, 1000)) for name in self.task_names}

    def _load_language(self, task_name: str) -> str:
        """静态完整指令（full_description）——seen/unseen 生成失败或
        instruction_type=full_description 时的回退口径。
        """
        path = self.root / "description" / "task_instruction" / f"{task_name}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            desc = str(data.get("full_description", "")).strip()
            if desc:
                return desc
        except Exception as exc:  # noqa: BLE001
            print(
                f"[robotwin_bench] WARNING: 读任务指令失败 {path.name}："
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return task_name

    def _load_joint_names(self) -> list[str]:
        """proprio 的关节名：直接读 embodiment 配置，与官方数据口径逐字一致。

        官方 `get_left_arm_jointState()` = [arm_joints_name[0] 里每个关节的驱动目标]
        + [夹爪值]，所以一臂 = len(arm_joints_name[i]) + 1 维，夹爪名取
        `gripper_name[i]["base"]`（如 fl_joint7；mimic 关节不单独出现在状态里）。
        aloha-agilex → fl_joint1..6 + fl_joint7 + fr_joint1..6 + fr_joint7 = 14 维。

        JointState 按名字查表，名字对不上会静默取到空值，所以必须和 policy 配置的
        `state_joints` 逐字一致。
        """
        try:
            args = self._task_args(self.task_names[0])
            left_cfg = args["left_embodiment_config"]
            right_cfg = args["right_embodiment_config"]
            left = [str(n) for n in left_cfg["arm_joints_name"][0]]
            right = [str(n) for n in right_cfg["arm_joints_name"][1]]
            left_grip = str(left_cfg["gripper_name"][0]["base"])
            right_grip = str(right_cfg["gripper_name"][1]["base"])
            return left + [left_grip] + right + [right_grip]
        except Exception as exc:  # noqa: BLE001
            print(
                f"[robotwin_bench] WARNING: 读 embodiment 关节名失败："
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return (
                [f"left_joint{i}" for i in range(1, 7)]
                + ["left_gripper"]
                + [f"right_joint{i}" for i in range(1, 7)]
                + ["right_gripper"]
            )

    def _task_args(self, task_name: str) -> dict[str, Any]:
        """官方 load_task_args() 的等价实现（scripts/eval_policy_xpolicylab.py）。

        不 import 那个脚本本身——它顶层拉 XPolicyLab 协议层，我们不需要。
        与官方只有一处差异：embodiment file_path 从相对路径转成绝对路径
        （官方脚本在仓库根目录跑，相对 cwd 解析；dora 节点的 cwd 是节点目录）。
        """
        if task_name in self._task_args_cache:
            return self._task_args_cache[task_name]

        cfg_root = self.root / "env_cfg" / "task_config"
        args = yaml.safe_load(
            (cfg_root / f"{self.cfg.task_config}.yml").read_text(encoding="utf-8")
        ) or {}
        args["task_name"] = task_name
        args["task_config"] = self.cfg.task_config
        args["policy_name"] = "forge"
        args["ckpt_setting"] = "forge"

        # 官方 ensure_xpolicylab_observation_flags：三条观测路必须开。
        data_type = args.setdefault("data_type", {})
        data_type["rgb"] = True
        data_type["qpos"] = True
        data_type["endpose"] = True

        # eval 口径：无 viewer、无官方视频（录像由核心 EpisodeRecorder 负责）、
        # 不写数据目录。官方 eval 循环也是 render_freq=0 + eval_mode=True。
        args["eval_mode"] = True
        args["render_freq"] = 0
        args["eval_video_log"] = False
        args["save_data"] = False

        # embodiment（与官方 load_task_args 逐字同构）
        embodiment_types = yaml.safe_load(
            (cfg_root / "_embodiment_config.yml").read_text(encoding="utf-8")
        ) or {}

        def get_embodiment_file(embodiment: str) -> str:
            robot_file = embodiment_types[embodiment]["file_path"]
            if robot_file is None:
                raise RuntimeError("No embodiment files")
            # 官方返回相对路径（依赖 cwd=仓库根）；转绝对，摆脱 cwd 依赖。
            return str((self.root / robot_file).resolve())

        embodiment_type = args.get("embodiment")
        if embodiment_type is None:
            raise ValueError(f"{self.cfg.task_config}.yml 没有 embodiment 字段")
        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        # 相机尺寸（官方同口径）
        camera_config = yaml.safe_load(
            (cfg_root / "_camera_config.yml").read_text(encoding="utf-8")
        ) or {}
        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        args["left_embodiment_config"] = self._embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = self._embodiment_config(args["right_robot_file"])

        self._task_args_cache[task_name] = args
        return args

    @staticmethod
    def _embodiment_config(robot_file: str) -> dict[str, Any]:
        with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
