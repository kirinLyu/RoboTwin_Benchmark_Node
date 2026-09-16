# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-file specification for the RoboTwin benchmark node.

移植自官方 libero_benchmark 配方（Forgelab-Robotics/benchmark-libero）。

冻结边界（与 libero 的一处刻意偏差，理由必须记牢）：
  libero 把 torch-cpu 也冻进包（libero/robosuite 深度静态引用，拆不干净）；
  本节点 torch 只有 adapter 里一处函数级 import，且 RoboTwin envs 用
  torch.cuda —— cpu 版进包会遮蔽评测环境的 CUDA 版。所以 torch/sapien/cv2/
  psutil/pynvml 一律 excludes，运行时由 pyi_rth_robotwin_site.py 把宿主
  site-packages 接进 sys.path（ROBOTWIN_ENV_SITE 或 $ROBOTWIN_ROOT/.venv-robotwin）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


_node_root = Path(os.environ["ROBOTWIN_NODE_ROOT"]).resolve()
_entry = _node_root / "scripts" / "entrypoint.py"
_hooks = _node_root / "scripts" / "hooks"


def _without_tests(name: str) -> bool:
    return (
        ".tests" not in name
        and not name.endswith(".test")
        and not name.endswith("_test")
        and not name.endswith("conftest")
    )


hiddenimports = [
    "_core.adapter",
    "_core.config",
    "_core.node",
    "_core.result",
    "_core.runner",
    "_core.session",
    "_core.video",
    "adapters.robotwin_adapter",
    "bench_tool",
    "bench_tool.descriptor",
    "bench_tool.endpoints",
    "bench_tool.invocations",
    "bench_tool.lease",
    "bench_tool.node",
    "bench_tool.result_shape",
    "websockets",
    "msgpack",
    "msgpack_numpy",
    "numpy.core._multiarray_tests",
]
for package in ("dora", "forge_msgs", "forge_tool", "yaml", "pydantic"):
    hiddenimports.extend(
        module for module in collect_submodules(package) if _without_tests(module)
    )

datas = []
for package in ("dora", "forge_msgs", "pydantic", "yaml"):
    datas.extend(collect_data_files(package))

binaries = []
for package in ("dora", "forge_msgs", "forge_tool", "pyarrow"):
    binaries.extend(collect_dynamic_libs(package))

# NumPy 来自宿主 benchmark 环境，BLAS/LAPACK/Fortran 库在其解释器旁（照 libero
# 配方按 base_prefix 前缀收集，绝不按 checkout 路径写死）。
_runtime_lib_dir = Path(getattr(sys, "base_prefix", sys.prefix)) / "lib"
for _pattern in (
    "libblas.so*",
    "liblapack.so*",
    "libopenblas*.so*",
    "libgfortran.so*",
    "libquadmath.so*",
    "libgcc_s.so*",
):
    for _library in _runtime_lib_dir.glob(_pattern):
        binaries.append((str(_library), "."))

a = Analysis(
    [str(_entry)],
    pathex=[str(_node_root), str(_node_root.parent)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(_hooks)],
    runtime_hooks=[str(_hooks / "pyi_rth_robotwin_site.py")],
    hooksconfig={},
    excludes=[
        "pytest",
        "tests",
        "pyarrow.tests",
        "pyarrow.conftest",
        # 评测环境提供（见文件头冻结边界说明）：
        "torch",
        "torchgen",
        "torchvision",
        "torchaudio",
        "nvidia",
        "triton",
        "cv2",
        "sapien",
        "psutil",
        "pynvml",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="robotwin_benchmark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
