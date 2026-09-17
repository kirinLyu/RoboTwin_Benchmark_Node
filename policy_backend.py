#!/usr/bin/env python3
"""G05 policy 后端（XPolicyLab websocket 富协议 env client）+ obs/action 口径转换。

本模块只做纯函数与薄封装：import 时**不**引入 dora/forge_msgs/XPolicyLab，
`client_server.ws.WsModelClient` 在 ``connect()`` 里懒加载。复用官方
``eval_policy_xpolicylab.py`` 的同一个 client 类与同构口径，判据 A 才与
官方侧同协议同行为。

口径对齐依据（官方脚本逐字段同构）：
- ``build_xpl_obs``      ≈ ``robotwin_obs_to_xpolicylab``
- ``qpos_state_to_dict`` ≈ ``convert_state`` 的 joint 路径（14D → 四键）
- ``pack_g05_action``    ≈ ``xpolicylab_action_to_robotwin`` 的 qpos 路径
  （[左臂6, 左爪1, 右臂6, 右爪1]）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

BACKENDS = ("xvla", "g05")

# 与官方 eval_policy_xpolicylab.py 的 CAMERA_NAME_MAP 逐字一致。
CAM_CHANNEL_TO_XPL = {
    "image/head": "cam_head",
    "image/left_wrist": "cam_left_wrist",
    "image/right_wrist": "cam_right_wrist",
}

# G05 单步动作的四键（policy/G05/model.py _format_action_chunk 口径），
# 打包顺序 [左臂6, 左爪1, 右臂6, 右爪1]，与官方 xpolicylab_action_to_robotwin 同序。
G05_ACTION_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
G05_ACTION_DIM = 14

# 官方 action_array/action_gripper 同款键别名（缺键时回退）。
_ACTION_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "left_arm_joint_state": (
        "left_arm_joint_state",
        "left_arm_joint",
        "left_joint_state",
        "arm_joint_state",
        "joint_state",
    ),
    "left_ee_joint_state": (
        "left_ee_joint_state",
        "left_gripper",
        "left_gripper_pos",
        "ee_joint_state",
    ),
    "right_arm_joint_state": ("right_arm_joint_state", "right_arm_joint", "right_joint_state"),
    "right_ee_joint_state": ("right_ee_joint_state", "right_gripper", "right_gripper_pos"),
}


def resolve_backend(cfg: dict[str, Any]) -> str:
    """缺省 G05，与已交付包一致；X-VLA 必须显式选取。"""
    backend = str(cfg.get("backend", "g05")).strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"未知 policy 后端：{backend!r}（可选 {BACKENDS}）")
    return backend


def qpos_state_to_dict(
    values: np.ndarray, names: list[str] | None = None
) -> dict[str, np.ndarray]:
    """14D 扁平 proprio → G05 joint state 四键。

    位置是契约、名字是标签：adapter qpos 口径的 proprio 恒为
    [左臂6, 左夹爪1, 右臂6, 右夹爪1]（``to_state()`` 读 ``joint_action.vector``
    同序）。``names`` 仅做数量校验，不参与切分。
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size != G05_ACTION_DIM:
        raise ValueError(f"G05 qpos state 需要 {G05_ACTION_DIM}D，收到 {v.size}D")
    if names is not None and len(names) != G05_ACTION_DIM:
        raise ValueError(f"proprio 名字数 {len(names)} 与 {G05_ACTION_DIM}D 不符")
    return {
        "left_arm_joint_state": v[0:6],
        "left_ee_joint_state": v[6:7],
        "right_arm_joint_state": v[7:13],
        "right_ee_joint_state": v[13:14],
    }


def build_xpl_obs(
    cameras: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    instruction: str,
    frequency: int = 30,
    env_idx: int = 0,
) -> dict[str, Any]:
    """官方 ``robotwin_obs_to_xpolicylab`` 同构 obs（cameras 以 dora 通道 id 为键）。"""
    vision: dict[str, Any] = {}
    for channel, xpl_name in CAM_CHANNEL_TO_XPL.items():
        if channel not in cameras:
            raise KeyError(f"G05 需要三路相机，缺 {channel}")
        color = np.asarray(cameras[channel], dtype=np.uint8)
        vision[xpl_name] = {"color": color, "shape": tuple(color.shape[:2])}
    text = str(instruction or "")
    return {
        "data_format_version": "v1.0",
        "instruction": text,
        "instructions": [text] if text else [],
        "env_idx": int(env_idx),
        "vision": vision,
        "state": dict(state),
        "additional_info": {"frequency": int(frequency)},
    }


def pack_g05_action(step: dict[str, Any]) -> np.ndarray:
    """G05 单步动作 dict → 14D [左臂6, 左爪1, 右臂6, 右爪1]。缺键 raise。"""

    def part(key: str) -> np.ndarray:
        for candidate in _ACTION_KEY_ALIASES[key]:
            if candidate in step:
                value = step[candidate]
                if isinstance(value, (int, float)):
                    return np.asarray([value], dtype=np.float32)
                return np.asarray(value, dtype=np.float32).reshape(-1)
        raise KeyError(f"G05 动作缺 {key}（试过 {_ACTION_KEY_ALIASES[key]}）")

    flat = np.concatenate([part(k) for k in G05_ACTION_KEYS]).astype(np.float32)
    if flat.size != G05_ACTION_DIM:
        raise ValueError(f"G05 动作拼出 {flat.size}D，期望 {G05_ACTION_DIM}D")
    return flat


class G05WsBackend:
    """XPolicyLab websocket 富协议 env client 薄封装。

    官方 eval 同款调用节奏：``update_obs`` 每帧一次、``get_action`` 仅
    chunk 耗尽时、``reset`` 每集一次。client 构造懒加载（仅 g05 后端需要
    XPolicyLab 的 client_server）。
    """

    def __init__(self, cfg: dict[str, Any], robotwin_root: str):
        host = cfg.get("policy_server_host", "localhost")
        port = cfg.get("policy_server_port")
        url = cfg.get("policy_server_url")
        if not url and port is None:
            raise ValueError(
                "G05 ws 后端需要 policy_server_url 或 policy_server_host/port"
            )
        self._url = url or f"ws://{host}:{int(port)}"
        self._robotwin_root = str(Path(robotwin_root).expanduser().resolve())
        self._kwargs = {
            "evaluation_id": str(cfg.get("evaluation_id", "robotwin-eval")),
            "trial_id": str(cfg.get("trial_id", "robotwin-trial")),
            "action_case_id": str(cfg.get("action_case_id", "robotwin-case")),
            "request_timeout_s": cfg.get("request_timeout_s", 300),  # 首次推理冷启动
            "max_connect_seconds": cfg.get("max_connect_seconds"),
            "ws_ping_interval_s": cfg.get("ws_ping_interval_s"),
            "ws_ping_timeout_s": cfg.get("ws_ping_timeout_s"),
        }
        self._client: Any = None

    def connect(self) -> None:
        if os.environ.get("ROBOTWIN_ENV_SITE"):
            from host_runtime import configure_host_paths
            configure_host_paths("ROBOTWIN_ENV_SITE")
        xpl = Path(self._robotwin_root) / "XPolicyLab"
        for p in (str(xpl), str(xpl / "integrations")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from client_server.ws import WsModelClient  # 官方 eval 同一个类

        self._client = WsModelClient(
            url=self._url,
            **{k: v for k, v in self._kwargs.items() if v is not None},
        )
        print(f"[g05_backend] 连接政策服务器 {self._url}", flush=True)

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._client.call(func_name="update_obs", obs=obs)

    def get_action(self) -> list[dict[str, Any]]:
        result = self._client.call(func_name="get_action")
        # 官方 normalize_action_chunk 同口径：{"actions": [...]} 包装 / None。
        if result is None:
            raise RuntimeError("G05 政策返回空动作 chunk")
        if isinstance(result, dict) and "actions" in result:
            result = result["actions"]
        actions = list(result)
        if not actions:
            raise RuntimeError("G05 政策返回空动作 chunk")
        return actions

    def reset(self) -> None:
        self._client.call(func_name="reset")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
