#!/usr/bin/env python3
"""假 policy：验时序闭环，不需要 GPU 与策略权重。

契约：
- 每收到一帧观测（以 proprio_state 计数）回一个 16D 零动作 —— 1:1 锁步；
- policy_command 照单全收（start / set_instruction / reset_scene / stop）；
- 退出时打印 obs/act 计数，两者必须相等；
- success_rate 恒为 0 是预期行为（零动作不可能完成任何任务）。

零动作对 RoboTwin 是合法动作（保持当前关节位形），episode 会跑到 step_lim
为止——所以用它冒烟慢任务时单集可能上千步，建议 task_ids 只开 1~2 个。
"""

from __future__ import annotations

import os

import numpy as np

from dora import Node
from forge_msgs import JointCommand

# 动作维度随 embodiment 变，不是常数：aloha-agilex = 每臂 6 关节 + 1 夹爪 = 14D。
# benchmark 侧按 embodiment 推导并校验，这里用环境变量对齐（默认 aloha-agilex）。
ACTION_DIM = int(os.environ.get("ROBOTWIN_ACTION_DIM", "14"))
# 零动作 = 保持位形。stub 阶段只是为了把时序跑通，数值无所谓。
ZERO_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
# JointCommand.from_np(values, order, field) 三参签名（已对 forge_msgs 1.2.0 源码
# 核实：第三个参数是 field="position"，不是 mode）。benchmark 侧 on_action 只读
# .position，不按名查表，所以这里占位名即可；与 LIBERO 测试写法逐字一致。
JOINT_NAMES = [f"joint{i}" for i in range(ACTION_DIM)]

OBS_IDS = {"proprio_state", "image/head", "image/left_wrist", "image/right_wrist"}


def main() -> int:
    node = Node()
    n_obs = 0
    n_act = 0
    last_instruction = ""

    def reply_actions() -> None:
        nonlocal n_act
        while n_act < n_obs:
            node.send_output("action", JointCommand.from_np(ZERO_ACTION, JOINT_NAMES, "position").to_arrow())
            n_act += 1

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "policy_command":
                from forge_msgs import PolicyCommand

                cmd = PolicyCommand.from_arrow(event["value"])
                if cmd.command == "set_instruction":
                    # PolicyCommand.inputs 是**方法**不是属性（forge_msgs 1.2.0
                    # control.py:91），忘了括号就是 'function' object has no
                    # attribute 'get'。
                    last_instruction = str(cmd.inputs().get("instruction", ""))
                print(
                    f"[stub_policy] command={cmd.command} "
                    f"instruction={last_instruction!r}",
                    flush=True,
                )
            elif event["id"] in OBS_IDS:
                if event["id"] == "proprio_state":
                    n_obs += 1
                    reply_actions()
            elif event["id"] == "tick":
                pass
        elif event["type"] == "STOP":
            break

    print(f"[stub_policy] obs={n_obs} act={n_act} (1:1 锁步: {n_obs == n_act})", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
