from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from config import derive_episode_seed, load_config, resolve_and_validate_plan
from result import BenchmarkResult, EpisodeRecord


ROOT = Path(__file__).parents[1]


class _Adapter:
    num_tasks = 2

    @staticmethod
    def num_init_states(task_id: int) -> int:
        return 3 if task_id == 0 else 2


def test_dataflow_preserves_dora_input_output_contract() -> None:
    dataflow = yaml.safe_load((ROOT / "dataflow.example.yaml").read_text(encoding="utf-8"))
    benchmark = next(node for node in dataflow["nodes"] if node["id"] == "robotwin_benchmark")

    assert benchmark["inputs"] == {"action": "policy/action"}
    assert set(benchmark["outputs"]) == {
        "proprio_state",
        "image/head",
        "image/left_wrist",
        "image/right_wrist",
        "policy_command",
        "benchmark_status",
        "benchmark_result",
    }


def test_config_plan_validation_and_seed_are_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "suite: robotwin\n"
        "task_ids: [0, 1]\n"
        "init_state_ids: [0, 1]\n"
        "num_runs: 2\n"
        f"robotwin_root: {tmp_path}\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    resolve_and_validate_plan(config, _Adapter())

    assert len(config.episode_plan()) == 8
    assert derive_episode_seed(0, 1, 1, 1) == derive_episode_seed(0, 1, 1, 1)
    assert derive_episode_seed(0, 1, 1, 1) != derive_episode_seed(0, 1, 1, 2)


def test_config_rejects_bad_control_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"robotwin_root: {tmp_path}\ncontrol_mode: nope\n", encoding="utf-8"
    )
    try:
        load_config(config_path)
    except ValueError:
        return
    raise AssertionError("control_mode 非法值必须被拒")


def test_config_rejects_missing_robotwin_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("robotwin_root: /nonexistent/robotwin\n", encoding="utf-8")
    try:
        load_config(config_path)
    except ValueError:
        return
    raise AssertionError("robotwin_root 不存在必须被拒")


def test_result_schema_retains_planned_denominator() -> None:
    result = BenchmarkResult(
        suite="robotwin",
        control_mode="qpos",
        max_steps=1800,
        total_episodes=2,
    )
    result.episodes.append(
        EpisodeRecord(
            episode_id="ep-1",
            suite="robotwin",
            task_id=0,
            init_state_id=0,
            run_index=0,
            task_description="test",
            success=True,
            termination="success",
            num_steps=10,
            return_value=1.0,
        )
    )

    payload = result.to_dict()
    assert payload["schema_version"] == "benchmark_execution_result_v1"
    assert payload["success_rate"] == 0.5
    assert json.loads(json.dumps(payload))["episodes"][0]["episode_id"] == "ep-1"


def test_config_reads_checkpoint_declaration(tmp_path: Path) -> None:
    """checkpoint 口径声明必须从 YAML 读进 cfg（否则启动自检的交叉核对静默跳过）。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "suite: robotwin\n"
        f"robotwin_root: {tmp_path}\n"
        "checkpoint:\n"
        "  input_features: [3, 224, 224]\n"
        "  resize_images_to: [224, 224]\n"
        "  state_dim: 16\n"
        "  state_joints: [left_ee_x, left_gripper]\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.checkpoint["input_features"] == [3, 224, 224]
    assert cfg.checkpoint["resize_images_to"] == [224, 224]
    assert cfg.checkpoint["state_dim"] == 16
    assert cfg.checkpoint["state_joints"] == ["left_ee_x", "left_gripper"]


def test_scene_seed_includes_run_index() -> None:
    """num_runs>1 必须产生差异（手册完成标准）：run_index 偏移种子域，
    run_index=0 与官方公式逐位相同。"""
    from adapters.robotwin_adapter import RobotwinAdapter

    assert RobotwinAdapter.scene_seed(0, 5, 0) == 100000 * 1 + 5
    assert RobotwinAdapter.scene_seed(0, 5, 1) == 100000 * 2 + 5
    assert RobotwinAdapter.scene_seed(0, 5, 0) != RobotwinAdapter.scene_seed(0, 5, 1)
    assert RobotwinAdapter.scene_seed(0, 5, 2) != RobotwinAdapter.scene_seed(0, 5, 1)


def test_runner_drops_excess_actions() -> None:
    """1:1 锁步：非等待态到达的多余 action 必须被丢弃（不调 adapter.step）。"""
    from types import SimpleNamespace

    import numpy as np
    from forge_msgs import JointCommand

    from _core.adapter import StepOutcome
    from _core.runner import BenchmarkRunner

    class FakeNode:
        def __init__(self) -> None:
            self.sent: list[tuple[str, object]] = []

        def send_output(self, output_id: str, value: object) -> None:
            self.sent.append((output_id, value))

    class FakeAdapter:
        def __init__(self) -> None:
            self.step_calls = 0

        def state_joint_names(self) -> list[str]:
            return []

        def to_state(self, obs: dict) -> None:
            return None

        def to_images(self, obs: dict) -> dict:
            return {}

        def step(self, action):
            self.step_calls += 1
            return StepOutcome(obs={}, reward=0.0, done=False, success=False)

    cfg = SimpleNamespace(
        benchmark="robotwin",
        suite="robotwin",
        max_steps=10,
        seed=0,
        control_mode="qpos",
        layout_mode="per_episode",
        policy_id="p",
        record_dir=None,
        result_dir=None,
        send_instruction=False,
        episode_plan=lambda: [(0, 0, 0)],
    )
    node = FakeNode()
    adapter = FakeAdapter()
    runner = BenchmarkRunner(node, cfg, adapter)
    action = JointCommand.from_np(
        np.zeros(14, dtype="float32"), [f"j{i}" for i in range(14)], "position"
    )

    runner.on_action(action)  # 初始非等态（批次未开）→ 丢弃
    assert adapter.step_calls == 0
    runner.awaiting_action = True
    runner.on_action(action)  # 等态 → 消费一步（消费后重新等下一动作）
    assert adapter.step_calls == 1
    runner.awaiting_action = False  # episode 边界：_record_episode 置 False
    runner.on_action(action)  # 边界处到达的多余 action → 丢弃
    assert adapter.step_calls == 1


# ---- 整改新增：官方评测协议回归（报告问题 1/2/3/4/6/7a） ----
#
# 测试环境没有 sapien / RoboTwin，adapter 用 object.__new__ 绕过 __init__
# （chdir/任务发现/采样线程），env 用假件；episode_instruction 内部的
# `from envs.utils.create_actor import UnStableError` 用注入的假模块满足。


class _UnStableError(Exception):
    """envs.utils.create_actor.UnStableError 的假件（与注入模块同名同义）。"""


@pytest.fixture
def fake_unstable_error():
    saved = {n: sys.modules.get(n) for n in ("envs", "envs.utils", "envs.utils.create_actor")}
    mods = {}
    for name in ("envs", "envs.utils", "envs.utils.create_actor"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mods[name] = mod
        sys.modules[name] = mod
    mods["envs.utils.create_actor"].UnStableError = _UnStableError
    yield _UnStableError
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


class _FakeEnv:
    """记录 setup/close 调用序列的假 env；按 seed 模拟不稳定与 expert 预筛失败。"""

    def __init__(self, unstable=(), expert_fail=()):
        self.unstable = set(unstable)
        self.expert_fail = set(expert_fail)
        self.setup_log: list[int] = []
        self.close_calls = 0
        self.plan_success = True
        self._cur_seed: int | None = None

    def setup_demo(self, now_ep_num, seed, is_test, **args):
        self._cur_seed = seed
        self.setup_log.append(seed)
        if seed in self.unstable:
            raise _UnStableError(f"unstable {seed}")

    def play_once(self):
        return {"info": {}}

    def check_success(self):
        return self._cur_seed not in self.expert_fail

    def close_env(self):
        self.close_calls += 1


def _bare_adapter(**overrides):
    from adapters.robotwin_adapter import RobotwinAdapter

    adapter = object.__new__(RobotwinAdapter)  # 绕过 __init__（chdir/发现/采样线程）
    cfg = SimpleNamespace(
        seed=0,
        instruction_type="full_description",
        expert_check=False,
        test_num=100,
        max_steps=1800,
        control_mode="qpos",
        task_config="demo_clean",
        robotwin_root=str(ROOT),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    adapter.cfg = cfg
    adapter.root = Path(cfg.robotwin_root)
    adapter.task_names = ["adjust_bottle", "beat_block_hammer"]
    adapter._seed_cursor = {}
    adapter._scene_seed_ledger = []
    adapter._standing = None
    adapter._last_scene_seed = None
    adapter._last_seed_retries = 0
    adapter._task_args_cache = {"adjust_bottle": {}, "beat_block_hammer": {}}
    adapter._task_language_cache = {}
    adapter._load_language = lambda name: f"desc-{name}"
    return adapter


def test_seed_queue_never_repeats_and_closes_per_episode(fake_unstable_error) -> None:
    """报告问题 1/7a：顺序种子队列接受过的 seed 永不重复；每集之间 close_env。"""
    adapter = _bare_adapter()
    env = _FakeEnv(unstable={100000, 100002, 100003})
    adapter._env = env
    adapter._env_task = "adjust_bottle"

    accepted: list[int] = []
    for init_state_id in range(6):
        adapter.episode_instruction(0, init_state_id, 0)
        assert adapter._standing is not None
        accepted.append(adapter._standing[3])
        adapter._standing = None  # 模拟 reset_episode 消费就位场景

    assert accepted == sorted(set(accepted))  # 接受过的 seed 永不重复
    assert len(accepted) == 6
    assert accepted[0] == 100001  # 100000 不稳定被跳过，游标已推进
    assert env.close_calls == 5  # 第 2..6 集开头各关一次旧场景（官方每集 close）


def test_expert_check_prefilters_and_redoes_setup(fake_unstable_error) -> None:
    """报告问题 2：expert 预筛不通过的 seed 跳过；通过后重新 setup 同一 seed
    （官方同款两次 setup：预筛一次 + 正式一次）。"""
    adapter = _bare_adapter(expert_check=True)
    env = _FakeEnv(expert_fail={100000})
    adapter._env = env
    adapter._env_task = "adjust_bottle"

    instruction = adapter.episode_instruction(0, 0, 0)
    assert adapter._standing[3] == 100001
    # 100000：expert setup 一次（预筛失败，无正式 setup）；
    # 100001：expert setup + 正式 eval setup 各一次。
    assert env.setup_log == [100000, 100001, 100001]
    assert env.close_calls == 2  # 两次 play_once 后各关一次
    assert instruction == "desc-adjust_bottle"  # 空 episode_info → 回退静态指令


def test_build_instruction_seen_matches_official_generation() -> None:
    """报告问题 4：seen 指令 = play_once episode_info 经官方生成器填充占位符。"""
    robotwin_root = Path(os.environ.get("ROBOTWIN_ROOT", "~/RoboTwin")).expanduser()
    if not (robotwin_root / "description").is_dir():
        pytest.skip("RoboTwin 仓库不在本机，跳过 seen 指令生成测试")
    adapter = _bare_adapter(instruction_type="seen")
    adapter.root = robotwin_root
    instruction = adapter._build_instruction(
        "beat_block_hammer", {"info": {"{A}": "020_hammer/base0", "{a}": "left"}}
    )
    assert instruction and "{" not in instruction and "hammer" in instruction
    # 生成失败/空 episode_info → 回退 full_description
    assert adapter._build_instruction("beat_block_hammer", {}) == "desc-beat_block_hammer"


def test_aggregate_injects_seed_ledger_and_resources() -> None:
    """报告问题 3/6：失败集 scene seed 经批次级台账进结果 JSON（三态契约不破），
    资源快照随批次注入。"""
    adapter = _bare_adapter()
    adapter._scene_seed_ledger = [
        {
            "task_id": 0,
            "init_state_id": 0,
            "run_index": 0,
            "scene_seed": 100001,
            "seed_retries": 1,
            "instruction": "desc",
        }
    ]
    adapter.cfg.episode_plan = lambda: [(0, 0, 0), (0, 1, 0)]
    adapter._sampler = SimpleNamespace(stop=lambda: {"cpu_percent": 12.0})

    primary, extra = adapter.aggregate([{"success": True}, {}])
    assert primary == {"name": "success_rate", "value": 0.5, "max": 1.0}
    assert extra["resources"]["cpu_percent"] == 12.0
    assert extra["scene_seed_ledger"][0]["scene_seed"] == 100001
    assert extra["scene_seed_ledger"][0]["seed_retries"] == 1


def test_config_reads_expert_check_and_instruction_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "suite: robotwin\n"
        f"robotwin_root: {tmp_path}\n"
        "expert_check: false\n"
        "instruction_type: unseen\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.expert_check is False
    assert cfg.instruction_type == "unseen"


def test_config_rejects_bad_instruction_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"robotwin_root: {tmp_path}\ninstruction_type: nope\n", encoding="utf-8"
    )
    try:
        load_config(config_path)
    except ValueError:
        return
    raise AssertionError("instruction_type 非法值必须被拒")
