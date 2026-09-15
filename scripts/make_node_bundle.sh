#!/usr/bin/env bash
# 源码形态 Node 制品打包（方案 v2，资产形态 A 的两件产出）：
#   1) robotwin_benchmark-<version>-linux-x86_64.tar.gz
#      —— entrypoint + _core/ + src/（向维护者申请豁免的形态）
#   2) robotwin_benchmark-<version>-linux-x86_64.pyz
#      —— zipapp 单文件备选（字面满足"根目录仅一个与 entrypoint 同名的普通文件"）
#
# 布局约束来自 main.py 的导入结构：
#   - _core 从 main.py 的**父目录**导入（导入期间临时挂 sys.path）
#   - adapters/config/bench_tool 从 main.py 自身目录导入
# 故 tar 形态：根目录 = entrypoint + _core/ + src/；zipapp 形态：全部平铺进归档。
#
# 用法：bash scripts/make_node_bundle.sh [--version X.Y.Z]
# 产物：build/out/ 下两个文件 + sha256/size 输出。
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=${1:-$(grep '^version' pyproject.toml | sed 's/.*= "\(.*\)"/\1/')}
case "${1:-}" in --version) VERSION="$2";; esac
NAME="robotwin_benchmark-${VERSION}-linux-x86_64"
OUT="build/out"
WORK="build/bundle/${NAME}"
STAGE="build/bundle/zipapp_stage"

rm -rf "$WORK" "$STAGE" && mkdir -p "$WORK/src" "$WORK/_core" "$STAGE" "$OUT"

# ---- 运行时子集（28 文件 manifest 的非测试部分；tests/ 不进制品）----
cp main.py config.py result.py "$WORK/src/"
cp -r adapters "$WORK/src/"
mkdir -p "$WORK/src/bench_tool"
for f in __init__.py descriptor.py endpoints.py invocations.py lease.py node.py result_shape.py; do
  cp "bench_tool/$f" "$WORK/src/bench_tool/"
done
for f in __init__.py adapter.py config.py node.py result.py runner.py session.py video.py; do
  cp "_core/$f" "$WORK/_core/"
done

# ---- entrypoint：exec 解释器运行源码（解释器可用 ROBOTWIN_BENCH_PYTHON 覆盖）----
cat > "$WORK/robotwin_benchmark" <<'EOF'
#!/usr/bin/env bash
# robotwin_benchmark 入口：运行依赖（dora-rs/forge-msgs/sapien/torch…）由评测环境提供。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROBOTWIN_BENCH_PYTHON:-python3}" -u "$ROOT/src/main.py" "$@"
EOF
chmod +x "$WORK/robotwin_benchmark"
find "$WORK" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ---- 1) 源码形态 tar（根目录 = robotwin_benchmark + _core/ + src/）----
tar -czf "$OUT/$NAME.tar.gz" -C "$WORK" .

# ---- 2) zipapp 单文件备选（平铺，main:main 入口）----
cp -r "$WORK/src/." "$STAGE/"
cp -r "$WORK/_core" "$STAGE/"
python3 -m zipapp "$STAGE" -m "main:main" -p "/usr/bin/env python3" -o "$OUT/$NAME.pyz"
chmod +x "$OUT/$NAME.pyz"

echo "== $NAME.tar.gz 内容"
tar -tzf "$OUT/$NAME.tar.gz"
echo "== 校验"
sha256sum "$OUT/$NAME.tar.gz" "$OUT/$NAME.pyz"
stat -c '%n %s bytes' "$OUT/$NAME.tar.gz" "$OUT/$NAME.pyz"
