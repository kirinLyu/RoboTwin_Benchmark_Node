#!/usr/bin/env bash
# robotwin_benchmark 单文件二进制构建配方（照 libero_benchmark 官方配方移植）。
#
# 冻结进二进制：节点代码 + dora-rs/forge-msgs/forge-tool/numpy/pyarrow/pydantic/pyyaml
#               + websockets/msgpack/msgpack-numpy（纯 python 依赖）。
# 留在评测环境（运行时由 adapters/robotwin_adapter.py 注入 sys.path）：
#               torch(CUDA)/sapien/cv2/psutil/pynvml + RoboTwin 仓库（ROBOTWIN_ROOT）。
# 为什么 torch 连 cpu 版都不进包：RoboTwin envs 用 torch.cuda，cpu 版会遮蔽
#               CUDA 版（与 libero 配方的唯一刻意偏差，spec 文件头有说明）。
#
# 依赖基线：dora-rs==0.4.1、forge-msgs==1.2.0、forge-tool==1.0.0、pyarrow==23.0.1、
#           pydantic==2.12.5、pyyaml==6.0.3（RoboTwin_benchmark_Node/pyproject.toml）。
# 构建环境：任意 py3.12 宿主环境（含 sapien/torch 供静态分析解析），
#           例：/data7/home/shenxin/RoboTwin/.venv-robotwin/bin/python。
#
# 产物：dist/robotwin_benchmark（单文件）→ dist/robotwin_benchmark-<ver>-linux-x86_64.tar.gz
# 命名：candidate 交付名 = <name>-<version>-linux-x86_64-<tarball sha256>.tar.gz
#       （sha256 后缀约定来自 robodojo 参考包）。
#
# 用法：PYTHON_BIN=<base-env-python> bash scripts/build_pyinstaller.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DIST_DIR=$NODE_ROOT/dist
WORK_DIR=$NODE_ROOT/build/pyinstaller
PYTHON_BIN=${PYTHON_BIN:-python3.12}
VERSION=$(grep '^version' "$NODE_ROOT/pyproject.toml" | sed 's/.*= "\(.*\)"/\1/')

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: base python not found: $PYTHON_BIN; set PYTHON_BIN to a py3.12 env with sapien/torch" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib

for name in ("dora", "forge_msgs", "forge_tool", "pyarrow", "numpy", "yaml", "pydantic"):
    importlib.import_module(name)
    print(f"build dependency available: {name}")
PY

BUILD_ENV=$(mktemp -d /tmp/robotwin_benchmark_pyinstaller.XXXXXX)
cleanup() {
    rm -rf "$BUILD_ENV"
}
trap cleanup EXIT

echo "==> creating isolated PyInstaller environment"
"$PYTHON_BIN" -m venv --system-site-packages "$BUILD_ENV"
"$BUILD_ENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    pyinstaller==6.19.0 \
    dora-rs==0.4.1 \
    forge-msgs==1.2.0 \
    forge-tool==1.0.0 \
    pyarrow==23.0.1 \
    pydantic==2.12.5 \
    pyyaml==6.0.3 \
    websockets \
    msgpack \
    msgpack-numpy

rm -rf "$DIST_DIR" "$WORK_DIR"
mkdir -p "$DIST_DIR" "$WORK_DIR"

echo "==> building one-file executable"
ROBOTWIN_NODE_ROOT="$NODE_ROOT" \
PYTHONPATH="$NODE_ROOT:$NODE_ROOT/.." \
    "$BUILD_ENV/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --distpath "$DIST_DIR" \
    --workpath "$WORK_DIR" \
    "$SCRIPT_DIR/robotwin_benchmark.spec"

DIST_FILE="$DIST_DIR/robotwin_benchmark"
if [[ ! -x "$DIST_FILE" ]]; then
    echo "ERROR: executable not found: $DIST_FILE" >&2
    exit 1
fi

echo "==> staging tarball (single entrypoint: robotwin_benchmark)"
STAGE_DIR=$(mktemp -d /tmp/robotwin_benchmark_stage.XXXXXX)
cp "$DIST_FILE" "$STAGE_DIR/robotwin_benchmark"
TARBALL="$DIST_DIR/robotwin_benchmark-${VERSION}-linux-x86_64.tar.gz"
tar -czf "$TARBALL" -C "$STAGE_DIR" robotwin_benchmark
rm -rf "$STAGE_DIR"

# 单 entrypoint 断言：包里必须恰好一个根条目，名字就是 entrypoint。
# 发布前最后一道闸 —— 包结构不对会直接阻断发布。
ROOT_ENTRIES=$(tar -tzf "$TARBALL")
if [[ "$ROOT_ENTRIES" != "robotwin_benchmark" ]]; then
    echo "ERROR: tarball must contain exactly one root entry 'robotwin_benchmark', got: $ROOT_ENTRIES" >&2
    exit 1
fi

echo "==> build successful: $TARBALL"
ls -lh "$TARBALL"
sha256sum "$TARBALL"
