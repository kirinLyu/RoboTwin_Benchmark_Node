"""RoboTwin 节点配置：共有字段来自 _core，这里只加 RoboTwin 特有的。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

def _import_core(name):
    """导 _core.<name>；父目录只临时挂上，避免遮蔽同名仿真器包（见 main.py 注释）。"""
    import importlib

    try:
        return importlib.import_module("_core." + name)
    except ImportError:
        p = str(Path(__file__).resolve().parents[1])
        sys.path.insert(0, p)
        try:
            return importlib.import_module("_core." + name)
        finally:
            if p in sys.path:
                sys.path.remove(p)


_m = _import_core("config")
BaseBenchmarkConfig = _m.BaseBenchmarkConfig
fill_base = _m.fill_base
read_yaml = _m.read_yaml
derive_episode_seed = _m.derive_episode_seed
resolve_and_validate_plan = _m.resolve_and_validate_plan

# action 口径（官方 take_action 的 action_type）
CONTROL_MODES = ("qpos", "ee")
# 指令口径（官方 eval_instruction 的取值 + 本节点扩展的静态回退）
INSTRUCTION_TYPES = ("seen", "unseen", "full_description")


@dataclass
class RobotwinConfig(BaseBenchmarkConfig):
    benchmark: str = "robotwin"
    suite: str = "robotwin"
    # qpos = 关节角（官方默认 joint）；ee = 末端位姿。维度由 embodiment 推导
    # （每臂 arm_joints_name + 1 夹爪，aloha-agilex = 14D），不是常数。
    control_mode: str = "qpos"
    # 场景配置：demo_clean（官方 eval 默认）/ demo_randomized（域随机化更激进）。
    task_config: str = "demo_clean"
    # 官方专家预筛（eval_policy_xpolicylab.py 默认 True）：setup_demo →
    # play_once() → close_env()，plan_success and check_success() 不过则跳过
    # 该 seed；通过后重新 setup 同一 seed 作为正式场景。
    expert_check: bool = True
    # 指令口径：seen / unseen = 官方生成式指令（play_once 的 episode_info 填充
    # 占位符，官方 build_instruction 同款）；full_description = 静态完整描述回退。
    instruction_type: str = "seen"
    # 每 task 的测试场景数（= 核心的 init_state 数），官方口径 100。
    test_num: int = 100
    # RoboTwin 仓库根目录（envs/、assets/、env_cfg/ 都在这里）。
    robotwin_root: str = ""
    # 步数兜底护栏：RoboTwin 每任务自带 step_lim（_eval_step_limit.yml，最大
    # 1700），到点 env 自己 done；这里必须 ≥ 最大 step_lim，否则对应任务的
    # episode 会被核心提前掐成 timed_out，分数口径会变（见 README §5 坑 2）。
    max_steps: int = 1800
    # 场景布局：RoboTwin 的 reset 按 seed 全量重建场景（对象/光照/背景），
    # 天然就是 per_episode 口径，固定不变，仅作结果文件里的复现口径记录。
    layout_mode: str = "per_episode"
    policy_id: str = "robotwin_policy"
    # 可选：目标 checkpoint 的口径声明，adapter 构造时做启动自检并打进日志
    # （手册 §3.2 分辨率分层契约 / §6 第 3 步）。分辨率/维度填错不报错、只会
    # 安静掉分，只能靠这里的交叉核对拦住。支持的键：
    #   input_features: [C, H, W]      # checkpoint 模型输入形状
    #   resize_images_to: [H, W]       # checkpoint 自带缩放目标（有则喂原生分辨率）
    #   state_dim: int                 # checkpoint proprio 维度
    #   state_joints: [names]          # checkpoint proprio 关节名（逐字核对）
    checkpoint: dict = field(default_factory=dict)


def load_config(path: str | Path) -> RobotwinConfig:
    data = read_yaml(path)
    cfg = RobotwinConfig()
    fill_base(cfg, data)

    cfg.control_mode = str(data.get("control_mode", cfg.control_mode)).strip().lower()
    if cfg.control_mode not in CONTROL_MODES:
        raise ValueError(
            f"control_mode 必须是 {CONTROL_MODES} 之一，收到 {cfg.control_mode!r}"
        )
    cfg.task_config = str(data.get("task_config", cfg.task_config))
    cfg.expert_check = bool(data.get("expert_check", cfg.expert_check))
    cfg.instruction_type = str(
        data.get("instruction_type", cfg.instruction_type)
    ).strip().lower()
    if cfg.instruction_type not in INSTRUCTION_TYPES:
        raise ValueError(
            f"instruction_type 必须是 {INSTRUCTION_TYPES} 之一，收到 {cfg.instruction_type!r}"
        )
    cfg.test_num = int(data.get("test_num", cfg.test_num))
    if cfg.test_num < 1:
        raise ValueError(f"test_num 必须 ≥1，收到 {cfg.test_num}")

    # 统一展开 ~：config.yaml 可以写 ~/RoboTwin 这类可移植路径，不必暴露本机绝对路径。
    cfg.robotwin_root = os.path.expanduser(
        str(data.get("robotwin_root") or os.environ.get("ROBOTWIN_ROOT") or "~/RoboTwin")
    )
    if not Path(cfg.robotwin_root).is_dir():
        raise ValueError(
            f"robotwin_root 不存在：{cfg.robotwin_root!r}"
            "（在 config.yaml 里指定或设 ROBOTWIN_ROOT）"
        )

    # 输出路径以"加载配置时的 cwd"为基准钉成绝对：adapter 构造时会把进程 cwd
    # 切到 robotwin_root（RoboTwin 的 envs 代码依赖 cwd=仓库根，见 adapter 注释），
    # 相对路径会跟着漂到仓库根下的 results/ 里去。
    for field in ("result_dir", "record_dir"):
        value = getattr(cfg, field)
        if value:
            setattr(cfg, field, os.path.abspath(str(value)))
    cfg.checkpoint = dict(data.get("checkpoint") or {})
    return cfg


# 与 LIBERO 节点保持同名导出（bench_tool 里按这个别名引用）。
BenchmarkNodeConfig = RobotwinConfig
