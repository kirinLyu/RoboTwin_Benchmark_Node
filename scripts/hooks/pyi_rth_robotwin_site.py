"""把评测环境的 site-packages 接进冻结进程的 sys.path（torch/sapien/cv2 等）。

冻结边界：二进制只带节点自身 + dora/forge_msgs/forge_tool/numpy/pyarrow 等轻依赖；
torch（CUDA）/sapien/cv2/psutil/pynvml 及 RoboTwin 仓库由评测环境提供（与
libero 官方配方"重资产留运行时"同构，差别是 torch 连 cpu 版都不进包——RoboTwin
envs 用 torch.cuda，cpu 版进包会遮蔽 CUDA 版）。

注入路径（按优先级）：
  1. ROBOTWIN_ENV_SITE —— 显式指定 site-packages 绝对路径；
  2. ROBOTWIN_ROOT 旁的同名 venv：$ROBOTWIN_ROOT/.venv-robotwin/lib/python<xy>/site-packages
追加到 sys.path **末尾**（包内模块始终优先）。
"""

import os
import sys
from pathlib import Path


def _site_candidates():
    explicit = os.environ.get("ROBOTWIN_ENV_SITE")
    if explicit:
        yield Path(explicit)
    root = os.environ.get("ROBOTWIN_ROOT")
    if root:
        yield (
            Path(root)
            / ".venv-robotwin"
            / "lib"
            / f"python{sys.version_info[0]}.{sys.version_info[1]}"
            / "site-packages"
        )


for _cand in _site_candidates():
    if _cand.is_dir():
        _p = str(_cand.resolve())
        if _p not in sys.path:
            sys.path.append(_p)
