#!/usr/bin/env python3
"""全 suite 跑分驱动（50 task × 100 init-state × 1 run，X_VLA ee 口径）。

串行跑：每个 task 一次 dora run（每次一个 dataflow/一个 GPU 进程，崩了只丢该
task 的进度）。可续跑：结果 JSON 已存在且 episodes 数齐整的 task 跳过。

路径全部可覆盖（同事换机器只动环境变量）：
    ROBOTWIN_NODE_ROOT   节点根目录（缺省为 forge_runtime 的节点目录）
    ROBOTWIN_ROOT        RoboTwin 仓库根目录（写进生成的 bench 配置）
    ROBOTWIN_MODEL_PATH  X_VLA ckpt-30000 解包目录（写进生成的 policy 配置）
    BENCH_PY             节点解释器绝对路径（缺省为本机 .venv-robotwin 的
                         python-real；uv venv 需用 copy 出的真二进制，见 README §5 坑 10）
    DORA_BIN             dora CLI 所在目录（缺省 ~/.cargo/bin）

多实例并行（每实例独占一对 GPU）：
    tools/run_full_suite.py 0 12 --instance 1   # GPU 0/1
    tools/run_full_suite.py 13 25 --instance 2  # GPU 2/3
    ...
实例 i 用 GPU 2(i-1)/2(i-1)+1，config/dataflow 落盘为 *_i{i}.*，节点进程
argv 带 --instance 标记（节点不解析 argv，仅供本驱动 pkill 按实例定位孤儿）。

用法：
    run_full_suite.py                # 全部 task（单实例）
    run_full_suite.py 5 12           # 只跑 task 5..12
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

NODE = Path(
    os.environ.get("ROBOTWIN_NODE_ROOT")
    or "/data7/home/shenxin/forge_runtime/packages/nodes/benchmarks/robotwin"
)
ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT") or "/data7/home/shenxin/RoboTwin"
MODEL_PATH = os.environ.get("ROBOTWIN_MODEL_PATH") or (
    "/data7/home/shenxin/robotwin_deps/ckpt/"
    "RoboTwin-cotrain-aloha_agilex-ee-0/ckpt-30000"
)
BENCH_PY = (
    os.environ.get("BENCH_PY")
    or "/data7/home/shenxin/RoboTwin/.venv-robotwin/bin/python-real"
)
LOGS = NODE / "full_suite_logs"
LOGS.mkdir(parents=True, exist_ok=True)
PROGRESS = NODE / "full_suite_progress.txt"
NUM_TASKS = 50
INIT_STATES = "0-99"
PER_TASK_TIMEOUT_S = 5 * 3600  # 100 集 × (场景 ~25s + 步进 ~90s) ≈ 3.2h 最坏
INSTANCE = 1  # 并行实例号（main 里从 --instance N 解析）
SUITE_CFG = NODE / "config.fullsuite.yaml"
POLICY_CFG = NODE / "config.fullsuite_policy.yaml"
DATFLOW = NODE / "dataflow_fullsuite.local.yaml"

CHECKPOINT_BLOCK = """checkpoint:
  input_features: [3, 224, 224]
  resize_images_to: [224, 224]
  state_dim: 16
  state_joints: [left_ee_x, left_ee_y, left_ee_z, left_ee_qw, left_ee_qx, left_ee_qy, left_ee_qz, left_gripper, right_ee_x, right_ee_y, right_ee_z, right_ee_qw, right_ee_qx, right_ee_qy, right_ee_qz, right_gripper]
"""

# 真 policy 两节点拓扑内联（仓库只保留 dataflow.example.yaml 一个模板，
# 驱动自包含，不读本地件）。<BENCH_PY> / BENCHMARK_CONFIG / REAL_POLICY_CONFIG
# / GPU 编号由 write_configs 替换。
DATAFLOW_TEMPLATE = """nodes:
  - id: robotwin_benchmark
    path: <BENCH_PY>
    args: -X faulthandler -u main.py
    env:
      BENCHMARK_CONFIG: ./config.real.yaml
      CUDA_VISIBLE_DEVICES: "0"
      PYTHONUNBUFFERED: "1"
    inputs:
      action: real_policy/action
    outputs:
      - proprio_state
      - image/head
      - image/left_wrist
      - image/right_wrist
      - policy_command
      - benchmark_status
      - benchmark_result

  - id: real_policy
    path: <BENCH_PY>
    args: -X faulthandler -u real_policy.py
    env:
      REAL_POLICY_CONFIG: ./config.real_policy.yaml
      CUDA_VISIBLE_DEVICES: "1"
      PYTHONUNBUFFERED: "1"
    inputs:
      policy_command:
        source: robotwin_benchmark/policy_command
        queue_size: 32
      proprio_state: robotwin_benchmark/proprio_state
      image/head: robotwin_benchmark/image/head
      image/left_wrist: robotwin_benchmark/image/left_wrist
      image/right_wrist: robotwin_benchmark/image/right_wrist
    outputs:
      - action
"""


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def result_json(task_id: int) -> Path | None:
    d = NODE / "results" / "full_suite" / f"task_{task_id}"
    jsons = sorted(d.glob("robotwin_*.json")) if d.is_dir() else []
    return jsons[-1] if jsons else None


def task_done(task_id: int) -> bool:
    p = result_json(task_id)
    if p is None:
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return (
        data.get("status") == "succeeded"
        and len(data.get("episodes", [])) == data.get("total_episodes")
    )


def write_configs(task_id: int) -> None:
    SUITE_CFG.write_text(
        "suite: robotwin\n"
        f"task_ids: \"{task_id}\"\n"
        f"init_state_ids: \"{INIT_STATES}\"\n"
        "num_runs: 1\n"
        "max_steps: 1800\n"
        "seed: 0\n"
        f"robotwin_root: {ROBOTWIN_ROOT}\n"
        "task_config: demo_clean\n"
        "control_mode: ee\n"
        "test_num: 100\n"
        "policy_id: robotwin_policy\n"
        "send_instruction: true\n"
        f"result_dir: {NODE}/results/full_suite/task_{task_id}\n"
        "record_dir: null\n"
        + CHECKPOINT_BLOCK,
        encoding="utf-8",
    )
    POLICY_CFG.write_text(
        "policy_id: robotwin_policy\n"
        f"robotwin_root: {ROBOTWIN_ROOT}\n"
        f"model_path: {MODEL_PATH}\n"
        "processor_path: null\n"
        "steps: 10\n"
        "device: cuda\n"
        'instruction: ""\n',
        encoding="utf-8",
    )
    DATFLOW.write_text(
        DATAFLOW_TEMPLATE.replace("<BENCH_PY>", BENCH_PY)
        .replace(
            "BENCHMARK_CONFIG: ./config.real.yaml",
            f"BENCHMARK_CONFIG: {SUITE_CFG}",
        )
        .replace(
            "REAL_POLICY_CONFIG: ./config.real_policy.yaml",
            f"REAL_POLICY_CONFIG: {POLICY_CFG}",
        )
        .replace(
            'CUDA_VISIBLE_DEVICES: "0"',
            f'CUDA_VISIBLE_DEVICES: "{2 * (INSTANCE - 1)}"',
        )
        .replace(
            'CUDA_VISIBLE_DEVICES: "1"',
            f'CUDA_VISIBLE_DEVICES: "{2 * (INSTANCE - 1) + 1}"',
        )
        .replace(
            "-X faulthandler -u main.py",
            f"-X faulthandler -u main.py --instance {INSTANCE}",
        )
        .replace(
            "-X faulthandler -u real_policy.py",
            f"-X faulthandler -u real_policy.py --instance {INSTANCE}",
        ),
        encoding="utf-8",
    )


def cleanup_orphans() -> None:
    """清掉本实例上一个 task 遗留的节点进程（批次提前失败时节点事件循环会悬空，
    dora 退出后节点变孤儿还占着 GPU 内存——不清理跑几个 task 就 OOM）。
    多实例并行时按 --instance 标记只清自己的，别误杀别的实例。"""
    # 注意：模式不能以 "-" 开头（pkill 会把它解析成自己的选项，-u 会被当成
    # user 过滤器导致一个进程都匹配不到）；用 [.] 防 pkill 匹配到自身。
    for pattern in (
        f"main[.]py --instance {INSTANCE}",
        f"real_policy[.]py --instance {INSTANCE}",
        "stub_policy[.]py",
    ):
        proc = subprocess.run(
            ["pkill", "-f", pattern], capture_output=True
        )
        _ = proc


def run_task(task_id: int) -> bool:
    cleanup_orphans()
    write_configs(task_id)
    log(f"task {task_id}: dora run 开始（超时 {PER_TASK_TIMEOUT_S}s）")
    t0 = time.time()
    with open(LOGS / f"task_{task_id}.log", "w", encoding="utf-8") as out:
        proc = subprocess.run(
            ["dora", "run", str(DATFLOW)],
            cwd=NODE,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=PER_TASK_TIMEOUT_S,
        )
    elapsed = time.time() - t0
    p = result_json(task_id)
    if proc.returncode != 0:
        log(f"task {task_id}: dora run 退出码 {proc.returncode}，耗时 {elapsed:.0f}s ✗")
        return False
    if p is None:
        log(f"task {task_id}: 无结果 JSON，耗时 {elapsed:.0f}s ✗")
        return False
    data = json.loads(p.read_text(encoding="utf-8"))
    log(
        f"task {task_id}: {data.get('status')} "
        f"success_rate={data.get('success_rate')} "
        f"({data.get('successes')}/{data.get('total_episodes')}) "
        f"num_steps={data.get('num_steps')} elapsed={data.get('elapsed_s'):.0f}s ✓"
    )
    return True


def main() -> int:
    global INSTANCE, SUITE_CFG, POLICY_CFG, DATFLOW
    argv = list(sys.argv[1:])
    inst = 1
    if "--instance" in argv:
        i = argv.index("--instance")
        inst = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    INSTANCE = inst
    SUITE_CFG = NODE / f"config.fullsuite_i{INSTANCE}.yaml"
    POLICY_CFG = NODE / f"config.fullsuite_policy_i{INSTANCE}.yaml"
    DATFLOW = NODE / f"dataflow_fullsuite_i{INSTANCE}.local.yaml"
    lo, hi = 0, NUM_TASKS - 1
    if len(argv) > 0:
        lo = int(argv[0])
        hi = int(argv[1]) if len(argv) > 1 else lo
    dora_bin = os.environ.get("DORA_BIN")
    os.environ["PATH"] = (
        dora_bin if dora_bin else str(Path.home() / ".cargo/bin")
    ) + ":" + os.environ.get("PATH", "")
    log(f"全 suite 跑分启动（实例 {INSTANCE}，GPU {2 * (INSTANCE - 1)}/{2 * (INSTANCE - 1) + 1}）：task {lo}..{hi}")
    for task_id in range(lo, hi + 1):
        if task_done(task_id):
            log(f"task {task_id}: 已有完整结果，跳过")
            continue
        try:
            run_task(task_id)
        except subprocess.TimeoutExpired:
            log(f"task {task_id}: 超过 {PER_TASK_TIMEOUT_S}s 未跑完，跳过 ✗")
        except Exception as exc:  # noqa: BLE001
            log(f"task {task_id}: 异常 {type(exc).__name__}: {exc} ✗")
    log(f"全 suite 跑分结束（实例 {INSTANCE}）：task {lo}..{hi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
