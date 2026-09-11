from __future__ import annotations

import json
from pathlib import Path

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
