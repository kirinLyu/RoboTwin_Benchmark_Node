"""G05's Dora lifecycle; model preprocessing remains in official XPolicyLab."""
from __future__ import annotations
import os
from typing import Any
import numpy as np
from dora import Node
from forge_msgs import Image, JointCommand, JointState, PolicyCommand
from policy_backend import (CAM_CHANNEL_TO_XPL, G05_ACTION_DIM, G05WsBackend,
                            build_xpl_obs, pack_g05_action, qpos_state_to_dict)

OBS_IDS = {"proprio_state", *CAM_CHANNEL_TO_XPL}


class G05PolicyNode:
    def __init__(self, cfg: dict[str, Any], backend: Any):
        self.cfg, self.backend = cfg, backend
        self.running = False
        self.instruction: str | None = None
        self.reset_pending = True
        self.observations: dict[str, Any] = {}
        self.chunk: list[np.ndarray] = []
        self.action_count = 0

    def _command(self, value: Any) -> None:
        cmd = PolicyCommand.from_arrow(value)
        if cmd.policy_id != self.cfg["policy_id"]:
            return
        if cmd.command == "start":
            self.running = True
        elif cmd.command == "set_instruction":
            self.instruction = str((cmd.inputs() or {}).get("instruction", ""))
            self.reset_pending = True
        elif cmd.command == "reset_scene":
            self.chunk.clear()
            self.backend.reset()
            self.reset_pending = False
            # Do not discard a new observation that beat reset_scene across
            # channels. Previous frame data was cleared when its action was sent.
        elif cmd.command == "stop":
            self.running = False
            self.instruction = None
            self.reset_pending = True
            self.chunk.clear()
            self.observations.clear()
            self.backend.reset()

    def _maybe_act(self, node: Any) -> None:
        if not self.running or self.instruction is None or self.reset_pending or not OBS_IDS <= self.observations.keys():
            return
        state_msg = JointState.from_arrow(self.observations["proprio_state"])
        values = np.asarray(state_msg.position, dtype=np.float64).reshape(-1)
        names = list(state_msg.name)
        if len(values) != G05_ACTION_DIM or len(names) != G05_ACTION_DIM:
            raise RuntimeError("G05 requires a 14D proprio_state; set benchmark control_mode=qpos")
        cameras = {ch: Image.from_arrow(self.observations[ch]).to_numpy() for ch in CAM_CHANNEL_TO_XPL}
        obs = build_xpl_obs(cameras, qpos_state_to_dict(values, names), self.instruction,
                            frequency=int(self.cfg.get("frequency", 30)))
        # Official cadence: update_obs every frame; get_action only when empty.
        self.backend.update_obs(obs)
        if not self.chunk:
            self.chunk[:] = [pack_g05_action(step) for step in self.backend.get_action()]
            if not self.chunk:
                raise RuntimeError("G05 returned an empty action chunk")
        action = self.chunk.pop(0)
        if action.size != G05_ACTION_DIM or not np.all(np.isfinite(action)):
            raise RuntimeError(f"G05 returned an invalid 14D action: {action.shape}")
        node.send_output("action", JointCommand.from_np(action, names, "position").to_arrow())
        self.observations.clear()
        self.action_count += 1

    def process_event(self, node: Any, event: Any) -> bool:
        if event["type"] == "STOP":
            return False
        if event["type"] == "ERROR":
            raise RuntimeError(f"Dora policy error: {event.get('error', event)}")
        if event["type"] == "INPUT":
            if event["id"] == "policy_command":
                self._command(event["value"])
            elif event["id"] in OBS_IDS:
                self.observations[event["id"]] = event["value"]
            self._maybe_act(node)
        return True


def run_g05(cfg: dict[str, Any]) -> int:
    from policy_tool import PolicyTool
    backend = G05WsBackend(cfg, cfg["robotwin_root"])
    tool = None
    try:
        backend.connect()
        node = Node()
        policy = G05PolicyNode(cfg, backend)
        if os.environ.get("POLICY_TOOL_ENABLED") == "1":
            tool = PolicyTool(node, lambda: {"policy_id": cfg["policy_id"], "backend": "g05",
                                           "ready": True, "running": policy.running, "action_dim": 14})
        for event in node:
            if tool:
                tool.tick()
                if event["type"] == "INPUT" and event["id"] == "tool_in":
                    tool.process(event["value"])
                    continue
            if not policy.process_event(node, event):
                break
    finally:
        try:
            if tool:
                tool.close()
        finally:
            backend.close()
    return 0
