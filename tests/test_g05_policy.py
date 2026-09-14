from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from policy_backend import (
    CAM_CHANNEL_TO_XPL,
    G05_ACTION_DIM,
    G05WsBackend,
    build_xpl_obs,
    pack_g05_action,
    qpos_state_to_dict,
    resolve_backend,
)


ROOT = Path(__file__).parents[1]

# G05 qpos 口径：aloha-agilex 14D = [左臂6, 左夹爪1, 右臂6, 右夹爪1]。
QPOS_14D = np.arange(14, dtype=np.float32)
QPOS_NAMES = [f"fl_joint{i}" for i in range(1, 8)] + [f"fr_joint{i}" for i in range(1, 8)]


def _fake_cameras() -> dict[str, np.ndarray]:
    return {
        "image/head": np.zeros((240, 320, 3), dtype=np.uint8),
        "image/left_wrist": np.ones((240, 320, 3), dtype=np.uint8),
        "image/right_wrist": np.full((240, 320, 3), 2, dtype=np.uint8),
    }


class TestResolveBackend:
    def test_default_is_xvla(self) -> None:
        assert resolve_backend({}) == "xvla"
        assert resolve_backend({"backend": "xvla"}) == "xvla"

    def test_g05_case_insensitive(self) -> None:
        assert resolve_backend({"backend": "g05"}) == "g05"
        assert resolve_backend({"backend": "G05"}) == "g05"

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知 policy 后端"):
            resolve_backend({"backend": "mamba"})


class TestQposStateToDict:
    def test_14d_splits_into_four_keys(self) -> None:
        state = qpos_state_to_dict(QPOS_14D, QPOS_NAMES)
        assert list(state) == [
            "left_arm_joint_state",
            "left_ee_joint_state",
            "right_arm_joint_state",
            "right_ee_joint_state",
        ]
        np.testing.assert_array_equal(state["left_arm_joint_state"], np.arange(6))
        np.testing.assert_array_equal(state["left_ee_joint_state"], [6.0])
        np.testing.assert_array_equal(state["right_arm_joint_state"], np.arange(7, 13))
        np.testing.assert_array_equal(state["right_ee_joint_state"], [13.0])

    def test_wrong_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="14D"):
            qpos_state_to_dict(np.zeros(16, dtype=np.float32))
        with pytest.raises(ValueError, match="名字数"):
            qpos_state_to_dict(QPOS_14D, QPOS_NAMES[:13])


class TestBuildXplObs:
    def test_matches_official_schema(self) -> None:
        state = qpos_state_to_dict(QPOS_14D, QPOS_NAMES)
        obs = build_xpl_obs(
            _fake_cameras(), state, "Grab the bottle.", frequency=30, env_idx=7
        )
        assert obs["data_format_version"] == "v1.0"
        assert obs["instruction"] == "Grab the bottle."
        assert obs["instructions"] == ["Grab the bottle."]
        assert obs["env_idx"] == 7
        assert set(obs["vision"]) == set(CAM_CHANNEL_TO_XPL.values())
        head = obs["vision"]["cam_head"]
        assert head["color"].dtype == np.uint8
        assert head["color"].shape == (240, 320, 3)
        assert head["shape"] == (240, 320)
        assert set(obs["state"]) == set(state)
        assert obs["additional_info"] == {"frequency": 30}

    def test_missing_camera_raises(self) -> None:
        cameras = _fake_cameras()
        del cameras["image/left_wrist"]
        with pytest.raises(KeyError, match="image/left_wrist"):
            build_xpl_obs(cameras, qpos_state_to_dict(QPOS_14D), "x")


class TestPackG05Action:
    def test_canonical_order(self) -> None:
        step = {
            "left_arm_joint_state": np.arange(6),
            "left_ee_joint_state": 6.0,
            "right_arm_joint_state": np.arange(7, 13),
            "right_ee_joint_state": 13.0,
        }
        flat = pack_g05_action(step)
        assert flat.size == G05_ACTION_DIM
        np.testing.assert_array_equal(flat, QPOS_14D)

    def test_alias_fallback(self) -> None:
        step = {
            "left_arm_joint": np.arange(6),
            "left_gripper": 6.0,
            "right_arm_joint": np.arange(7, 13),
            "right_gripper": 13.0,
        }
        np.testing.assert_array_equal(pack_g05_action(step), QPOS_14D)

    def test_missing_key_raises(self) -> None:
        step = {"left_arm_joint_state": np.zeros(6)}
        with pytest.raises(KeyError, match="left_ee_joint_state"):
            pack_g05_action(step)

    def test_wrong_dim_raises(self) -> None:
        step = {
            "left_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": np.zeros(2),  # 夹爪应为 1D
            "right_arm_joint_state": np.zeros(6),
            "right_ee_joint_state": 0.0,
        }
        with pytest.raises(ValueError, match="15D"):
            pack_g05_action(step)


class _FakeClient:
    """鸭子注入 G05WsBackend._client：只记录调用序列，不碰网络。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.get_action_result: object = None

    def call(self, func_name: str | None = None, obs: object = None) -> object:
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            return self.get_action_result
        return None

    def close(self) -> None:
        self.calls.append(("close", None))


class TestG05WsBackendRouting:
    def test_calls_routed_like_official_eval(self) -> None:
        fake = _FakeClient()
        fake.get_action_result = [
            {
                "left_arm_joint_state": np.zeros(6),
                "left_ee_joint_state": 0.0,
                "right_arm_joint_state": np.zeros(6),
                "right_ee_joint_state": 0.0,
            }
        ]
        backend = G05WsBackend({"policy_server_url": "ws://127.0.0.1:9"}, "unused")
        backend._client = fake  # 绕过 connect() 的真实连接

        obs = {"instruction": "hello"}
        backend.update_obs(obs)
        actions = backend.get_action()
        backend.reset()
        backend.close()

        assert [name for name, _ in fake.calls] == [
            "update_obs",
            "get_action",
            "reset",
            "close",
        ]
        assert fake.calls[0][1] == obs
        assert len(actions) == 1

    def test_get_action_unwraps_and_rejects_empty(self) -> None:
        backend = G05WsBackend({"policy_server_url": "ws://127.0.0.1:9"}, "unused")
        fake = _FakeClient()

        fake.get_action_result = {"actions": [1, 2]}
        backend._client = fake
        assert backend.get_action() == [1, 2]

        fake.get_action_result = None
        with pytest.raises(RuntimeError, match="空动作 chunk"):
            backend.get_action()
        fake.get_action_result = []
        with pytest.raises(RuntimeError, match="空动作 chunk"):
            backend.get_action()


def test_backend_module_imports_without_dora() -> None:
    """policy_backend 裸 import 不触 dora/forge_msgs/XPolicyLab。

    用子进程验证：全量套件里 _core/tests 的 conftest 已把 forge_msgs 装进
    sys.modules，进程内断言顺序脆弱，子进程才干净。
    """
    code = (
        "import sys; import policy_backend; "
        "bad = [m for m in ('dora', 'forge_msgs', 'client_server', 'XPolicyLab')"
        " if m in sys.modules]; "
        "print('BAD=' + ','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "BAD=", out.stdout


class TestG05WsBackendAgainstFakeServer:
    """真 WsModelClient 打假 ws server：协议层集成测试。

    XPolicyLab checkout 不在本机（~/RoboTwin/XPolicyLab）则跳过——与
    test_contracts.py 的 seen 生成测试同一 guard 口径。
    """

    ACTIONS = [
        {
            "left_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": 0.0,
            "right_arm_joint_state": np.zeros(6),
            "right_ee_joint_state": 0.0,
        },
        {
            "left_arm_joint_state": np.ones(6),
            "left_ee_joint_state": 1.0,
            "right_arm_joint_state": np.ones(6),
            "right_ee_joint_state": 1.0,
        },
    ]

    def test_end_to_end(self) -> None:
        robotwin_root = Path(os.path.expanduser("~/RoboTwin"))
        xpl = robotwin_root / "XPolicyLab"
        if not (xpl / "client_server" / "ws" / "model_client.py").exists():
            pytest.skip("RoboTwin/XPolicyLab 仓库不在本机，跳过 ws 集成测试")
        for p in (str(xpl), str(xpl / "integrations")):
            if p not in sys.path:
                sys.path.insert(0, p)

        # 与 G05WsBackend.connect() 同一机制懒加载 XPolicyLab 协议栈。
        from client_server.ws.protocol.codec import decode_envelope, encode_frame
        from client_server.ws.protocol.messages import MessageType
        from client_server.ws.protocol.schemas import Frame
        import websockets

        calls: list[tuple[str, object]] = []
        ports: list[int] = []
        done = asyncio.Event()  # set() 线程安全（内部 call_soon_threadsafe）

        def reply(frame: Frame, msg_type: MessageType, payload: dict) -> Frame:
            return Frame(
                message_type=msg_type,
                request_id=frame.request_id,
                evaluation_id=frame.evaluation_id,
                action_case_id=frame.action_case_id,
                trial_id=frame.trial_id,
                step=frame.step,
                payload=payload,
            )

        async def handler(ws) -> None:
            try:
                async for raw in ws:
                    frame = decode_envelope(bytes(raw))
                    out = None
                    if frame.message_type == MessageType.HELLO:
                        out = reply(
                            frame,
                            MessageType.HELLO_ACK,
                            {"server_instance_id": "fake-g05-server"},
                        )
                    elif frame.message_type == MessageType.CALL:
                        func = frame.payload.get("func_name")
                        calls.append((func, frame.payload.get("obs")))
                        result = self.ACTIONS if func == "get_action" else None
                        out = reply(frame, MessageType.CALL_RESULT, {"result": result})
                    elif frame.message_type == MessageType.RESET:
                        calls.append(("reset", None))
                        out = reply(frame, MessageType.RESET_RESULT, {"result": None})
                    elif frame.message_type == MessageType.HEARTBEAT:
                        out = reply(frame, MessageType.HEARTBEAT_ACK, {})
                    elif frame.message_type == MessageType.CLOSE:
                        break
                    if out is not None:
                        await ws.send(encode_frame(out))
            except Exception:
                pass

        async def server_main() -> None:
            async with websockets.serve(
                handler, "127.0.0.1", 0, max_size=None, compression=None
            ) as server:
                ports.append(server.sockets[0].getsockname()[1])
                await done.wait()

        thread = threading.Thread(
            target=lambda: asyncio.run(server_main()), daemon=True
        )
        thread.start()
        try:
            deadline = 5.0
            while not ports:
                if not thread.is_alive():
                    pytest.fail("fake ws server 线程提前退出")
                if deadline < 0:
                    pytest.fail("fake ws server 未在 5s 内就绪")
                deadline -= 0.01
                time.sleep(0.01)

            backend = G05WsBackend(
                {
                    "policy_server_url": f"ws://127.0.0.1:{ports[0]}",
                    "request_timeout_s": 10,
                    # 收紧连接重试：默认 max_connect_seconds=900，服务器异常会烧 15 分钟。
                    "max_connect_attempts": 3,
                    "connect_retry_delay_s": 0.1,
                    "max_connect_seconds": 30,
                },
                str(robotwin_root),
            )
            backend.connect()

            # 观测经编码往返核对：三路相机 + 14D joint state（含 numpy 数组）。
            state = qpos_state_to_dict(QPOS_14D, QPOS_NAMES)
            color = _fake_cameras()["image/head"]
            obs = build_xpl_obs(_fake_cameras(), state, "Grab the bottle.")
            backend.update_obs(obs)
            actions = backend.get_action()
            backend.reset()
            backend.close()

            assert [name for name, _ in calls] == ["update_obs", "get_action", "reset"]
            sent = calls[0][1]
            np.testing.assert_allclose(sent["vision"]["cam_head"]["color"], color)
            np.testing.assert_allclose(
                sent["state"]["left_arm_joint_state"], np.arange(6)
            )
            # 动作经 msgpack 编码往返：值逐位一致，顺序 [左臂6, 左爪1, 右臂6, 右爪1]。
            assert len(actions) == 2
            expected = [
                np.concatenate([np.zeros(6), [0.0], np.zeros(6), [0.0]]),
                np.concatenate([np.ones(6), [1.0], np.ones(6), [1.0]]),
            ]
            for got, want in zip(actions, expected):
                np.testing.assert_array_equal(pack_g05_action(got), want)
        finally:
            done.set()
            thread.join(timeout=5.0)
