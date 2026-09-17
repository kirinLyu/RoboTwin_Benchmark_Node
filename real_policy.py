#!/usr/bin/env python3
"""真 policy 节点（本地冒烟件）：XPolicyLab X_VLA，ee 16D。

dora 通道契约与 stub_policy.py 完全一致——每收到一帧 proprio_state 回一个动作，
1:1 锁步，benchmark 节点无需任何改动。推理栈用 XPolicyLab policy/X_VLA 的 Model
类（官方 eval_policy_xpolicylab.py 同款），这里只加 obs dict 组装与 16D 动作扁平化。

X_VLA 输入口径（官方 eval 的 convert_state / convert_vision 同构）：
- 图像只用一个：head 相机 → vision["cam_head"]（X_VLA 模型不看腕部相机）；
- state：left_ee_pose(7, xyz+wxyz 四元数) + left_ee_joint_state(1 夹爪) +
  right_ee_pose(7) + right_ee_joint_state(1)——由 proprio_state 通道的 16D 拼出；
- 指令：policy_command 的 set_instruction。
X_VLA 输出口径：每步 dict {left_ee_pose(7), left_ee_joint_state(1),
right_ee_pose(7), right_ee_joint_state(1)}，本节点拍平成 16D JointCommand
（顺序与 benchmark 侧 take_action(action_type="ee") 的切分一致）。
一次 get_action() 返回一个 action chunk，本节点逐 proprio_state 弹一步，
chunk 耗尽再推理一次（官方 eval 同语义）。

配置（env REAL_POLICY_CONFIG 指向的 yaml，不入库）：
  policy_id: robotwin_policy       # 必须与 bench.yaml 一致，否则忽略命令
  robotwin_root: /path/to/RoboTwin # 提供 XPolicyLab 包
  model_path: /path/to/.../ckpt-30000
  processor_path: null             # 缺省用 model_path；没有 preprocessor_config.json
                                   # 时给 X-VLA-Pt 基座模型目录
  steps: 10                        # 去噪步数
  device: cuda

g05 后端（backend: g05，见 _run_g05）：经 policy_backend.G05WsBackend 连
XPolicyLab websocket policy server（官方 eval_policy_xpolicylab.py 同款
client 与 obs/action 口径），joint 14D [左臂6, 左爪1, 右臂6, 右爪1]，三路
相机齐帧才推理。缺省 backend=g05；X-VLA 通过 backend: xvla 显式启用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dora import Node
from forge_msgs import Image as ForgeImage
from forge_msgs import JointCommand, JointState, PolicyCommand

from policy_backend import (
    CAM_CHANNEL_TO_XPL,
    G05_ACTION_DIM,
    G05WsBackend,
    build_xpl_obs,
    pack_g05_action,
    qpos_state_to_dict,
    resolve_backend,
)

OBS_IDS = {"proprio_state", "image/head", "image/left_wrist", "image/right_wrist"}

# 与 benchmark 侧 adapters/robotwin_adapter.py 的 EE_STATE_NAMES 逐字一致——
# 名字是标签、位置才是契约，两边对不上时按名查表会静默取空。
EE_STATE_NAMES = [
    *(f"left_ee_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    "left_gripper",
    *(f"right_ee_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    "right_gripper",
]
ACTION_DIM = 16


def load_cfg() -> dict[str, Any]:
    path = os.environ.get("REAL_POLICY_CONFIG")
    if not path:
        raise RuntimeError("REAL_POLICY_CONFIG 环境变量未设置")
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg["robotwin_root"] = str(Path(cfg.get("robotwin_root") or os.environ["ROBOTWIN_ROOT"]).expanduser().resolve())
    if os.environ.get("G05_SERVER_URL"):
        cfg["policy_server_url"] = os.environ["G05_SERVER_URL"]
    return cfg


def setup_sys_path(robotwin_root: str) -> None:
    root = Path(robotwin_root).resolve()
    xpl = root / "XPolicyLab"
    xvla = xpl / "policy" / "X_VLA"
    for p in (str(root), str(xpl), str(xpl / "integrations"), str(xvla), str(xvla / "xvla")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_model(cfg: dict[str, Any]):
    from XPolicyLab.policy.X_VLA.model import Model

    model_cfg = {
        "policy_name": "X_VLA",
        "action_type": "ee",
        "model_path": cfg["model_path"],
        "processor_path": cfg.get("processor_path") or None,
        "steps": int(cfg.get("steps", 10)),
        "device": cfg.get("device", "cuda"),
        "domain_id": int(cfg.get("domain_id", 0)),
        "prompt": None,
    }
    print(f"[real_policy] 加载 X_VLA：model_path={cfg['model_path']}", flush=True)
    return Model(model_cfg)


def state_to_dict(values: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """16D 扁平 proprio → XPolicyLab state dict（按名查表，位置是契约）。"""
    by_name = dict(zip(names, np.asarray(values, dtype=np.float64).reshape(-1)))

    def ee(side: str) -> np.ndarray:
        return np.asarray(
            [by_name[f"{side}_ee_{axis}"] for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")],
            dtype=np.float32,
        )

    return {
        "left_ee_pose": ee("left"),
        "left_ee_joint_state": np.asarray([by_name["left_gripper"]], dtype=np.float32),
        "right_ee_pose": ee("right"),
        "right_ee_joint_state": np.asarray([by_name["right_gripper"]], dtype=np.float32),
    }


def action_to_flat(action: dict[str, Any]) -> np.ndarray:
    """XPolicyLab 动作 dict → 16D：[left_ee(7), left_grip(1), right_ee(7), right_grip(1)]。"""
    return np.concatenate(
        [
            np.asarray(action["left_ee_pose"], dtype=np.float32).reshape(-1),
            np.asarray(action["left_ee_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(action["right_ee_pose"], dtype=np.float32).reshape(-1),
            np.asarray(action["right_ee_joint_state"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def _run_g05(cfg: dict[str, Any]) -> int:
    """Source-complete G05 lifecycle shared by source and packaged execution."""
    from g05_node import run_g05
    return run_g05(cfg)


def main() -> int:
    cfg = load_cfg()
    if resolve_backend(cfg) == "g05":
        return _run_g05(cfg)
    setup_sys_path(cfg["robotwin_root"])
    model = build_model(cfg)

    node = Node()
    n_obs = 0
    n_act = 0
    chunk: list[np.ndarray] = []
    obs_cache: dict[str, Any] = {}
    instruction = str(cfg.get("instruction", ""))
    running = False
    last_command = ""
    # benchmark 的 _publish_obs 先发 proprio_state 再发图像（且 reset_scene 会清
    # 缓存），所以 proprio 到达时 head 图可能还没到——用逐通道帧计数判定
    # （head_count >= proprio_count 才是当前帧图像），置位后等 image/head 到达
    # 再推理。避免「新 proprio + 上一步旧 head 图」的混帧输入。
    pending_infer = False

    def send_next() -> None:
        nonlocal chunk, n_act
        action = chunk.pop(0)
        if action.size != ACTION_DIM or not np.all(np.isfinite(action)):
            raise RuntimeError(f"动作维度/数值异常：shape={action.shape}")
        node.send_output(
            "action",
            JointCommand.from_np(action, EE_STATE_NAMES, "position").to_arrow(),
        )
        n_act += 1

    def infer_and_queue() -> None:
        nonlocal chunk
        if "proprio_state" not in obs_cache or "image/head" not in obs_cache:
            raise RuntimeError("proprio_state 或 image/head 观测缺失")
        state_msg = JointState.from_arrow(obs_cache["proprio_state"])
        values = np.asarray(state_msg.position, dtype=np.float64).reshape(-1)
        names = list(state_msg.name)
        if len(values) != ACTION_DIM or len(names) != ACTION_DIM:
            raise RuntimeError(
                f"proprio 维度不对：{len(values)}D（期望 {ACTION_DIM}D）——"
                f"benchmark 侧 control_mode 是否设为 ee？"
            )
        head_rgb = ForgeImage.from_arrow(obs_cache["image/head"]).to_numpy()
        xpl_obs = {
            # 官方 eval 的 convert_vision 口径：vision 字典 + cam_head；X_VLA
            # 的 extract_image 从这里取图（模型只看 head 相机）。
            "vision": {
                "cam_head": {"color": head_rgb, "shape": tuple(head_rgb.shape[:2])}
            },
            "state": state_to_dict(values, names),
            "instruction": instruction,
            "env_idx": 0,
        }
        # 官方 eval 每步 update_obs 一次、chunk 耗尽才 get_action（循环同构：
        # 见 eval_policy_xpolicylab.py 的 update_obs/get_action 节奏），这里照做。
        model.update_obs(xpl_obs)
        if not chunk:
            action_list = model.get_action()
            if not action_list:
                raise RuntimeError("Policy returned an empty action chunk.")
            chunk[:] = [action_to_flat(a) for a in action_list]
            print(f"[real_policy] 推理完成：chunk={len(chunk)} 步", flush=True)

    # 帧对齐计数：每帧恰 1 条 proprio + 1 张 head（逐通道 FIFO）。runner 先发
    # proprio 再发图像，所以 proprio 到达时缓存里的 head 可能是**上一帧**的——
    # head_count < proprio_count 说明本帧 head 还在路上，置 pending 等 head 到达
    # 再推理，杜绝「新 proprio + 旧 head 图」的混帧推理（跨通道乱序下 head 先
    # 到则 head_count >= proprio_count，直接用当前帧图像，无死锁）。
    proprio_count = 0
    head_count = 0

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "policy_command":
                cmd = PolicyCommand.from_arrow(event["value"])
                if cmd.policy_id != cfg["policy_id"]:
                    print(
                        f"[real_policy] 忽略 policy_id 不匹配的命令："
                        f"{cmd.policy_id} != {cfg['policy_id']}",
                        flush=True,
                    )
                    continue
                if cmd.command == "start":
                    running = True
                elif cmd.command == "set_instruction":
                    instruction = str(cmd.inputs().get("instruction", instruction))
                elif cmd.command == "reset_scene":
                    # 清空动作队列与观测缓存，保持 running（手册 §3.5）。
                    chunk.clear()
                    obs_cache.clear()
                    pending_infer = False
                elif cmd.command == "stop":
                    running = False
                last_command = cmd.command
                print(
                    f"[real_policy] command={cmd.command} running={running} "
                    f"instruction={instruction!r}",
                    flush=True,
                )
            elif event["id"] in OBS_IDS:
                obs_cache[event["id"]] = event["value"]
                if event["id"] == "proprio_state":
                    n_obs += 1
                    proprio_count += 1
                    if head_count >= proprio_count:
                        # 每步 update_obs（官方节奏）；chunk 耗尽才 get_action
                        infer_and_queue()
                        send_next()
                    else:
                        # 本帧 head 图还没到（首帧或跨通道乱序），等 image/head 触发
                        pending_infer = True
                elif event["id"] == "image/head":
                    head_count += 1
                    if pending_infer and head_count >= proprio_count:
                        pending_infer = False
                        infer_and_queue()
                        send_next()
            elif event["id"] == "tick":
                pass
        elif event["type"] == "STOP":
            break

    print(
        f"[real_policy] obs={n_obs} act={n_act} (1:1 锁步: {n_obs == n_act}) "
        f"last_command={last_command}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
