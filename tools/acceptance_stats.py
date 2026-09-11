#!/usr/bin/env python3
"""判据 A 统计判定：官方 runner vs Forge 的成绩对照。

输入：
  official_<task>.log   官方 eval 输出（Success!/Fail! + 每集 current seed）
  Forge 结果 JSON        results/acceptance/robotwin_*.json（逐集 episode_seed/success）

用法：
  acceptance_stats.py --acc-dir <官方日志目录> --forge-dir <Forge results/acceptance 目录>
缺省两个目录都取当前目录。也可用环境变量 ACC_DIR / FORGE_DIR。

输出（每 task + 汇总）：
  n/success/成功率、Wilson 95% CI、Fisher 双侧精确检验 p 值；
  另做 seed 对齐比较（两边都跑过的同一批 scene seed 上再比一次）。
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

ACC = Path(os.environ.get("ACC_DIR") or ".")
FORGE_DIR = Path(os.environ.get("FORGE_DIR") or ".")
TASKS = ["adjust_bottle", "beat_block_hammer"]

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Fisher 双侧精确检验 p 值（scipy 缺省时的纯 Python 实现）。"""
    if fisher_exact is not None:
        return float(fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue)
    n = a + b + c + d
    r1, c1 = a + b, a + c
    obs = math.comb(r1, a) * math.comb(n - r1, c1 - a) / math.comb(n, c1)
    total = 0.0
    for x in range(max(0, c1 - (n - r1)), min(c1, r1) + 1):
        p = math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
        if p <= obs * (1 + 1e-12):
            total += p
    return min(1.0, total)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def parse_official(log_path: Path) -> dict[int, bool]:
    """从官方 eval 日志提取：seed -> success。"""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # 每集以 Success!/Fail! 结束，紧随其后一行打印 current seed: N
    episodes: dict[int, bool] = {}
    pattern = re.compile(
        r"\x1b\[9[12]m(Success!|Fail!)\x1b\[0m.*?current seed: \x1b\[90m(\d+)\x1b\[0m",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        ok = m.group(1) == "Success!"
        seed = int(m.group(2))
        episodes[seed] = ok
    return episodes


def parse_forge_episodes(result_json: Path, task_id: int) -> list[tuple[int | None, bool]]:
    """Forge 结果 JSON：该 task 全部 50 集的 (scene_seed, success)。
    主检验按集数计（分母=50），scene_seed 可能因 UnStableError 重试与相邻集碰撞。"""
    data = json.loads(result_json.read_text(encoding="utf-8"))
    out: list[tuple[int | None, bool]] = []
    for ep in data.get("episodes", []):
        if int(ep.get("task_id", -1)) != task_id:
            continue
        seed = ep.get("metrics") or {}
        s = seed.get("scene_seed")
        out.append((int(s) if s is not None else None, bool(ep.get("success"))))
    return out


def parse_forge(result_json: Path, task_id: int) -> dict[int, bool]:
    """Forge 结果 JSON：scene seed -> success（只取指定 task_id 的集）。
    两个 task 用同一批场景种子（100000+i），必须按 task 过滤，否则互相覆盖。
    episode_seed 是核心派发的记录用身份种子；场景种子在 metrics.scene_seed
    （UnStableError 重试后为最终实际使用的种子，等价官方"计入"口径）。
    失败集无 metrics（无 scene_seed），不在本字典内。"""
    return {s: ok for s, ok in parse_forge_episodes(result_json, task_id) if s is not None}


def compare(official: dict[int, bool], forge_eps: list, forge: dict[int, bool], label: str) -> None:
    def row(k: int, n: int) -> tuple[str, str]:
        return f"{k} / {n}", f"{100 * k / n:.1f}%"

    ok, on = sum(official.values()), len(official)
    fk, fn = sum(1 for _, s in forge_eps if s), len(forge_eps)
    print(f"\n=== {label} ===")
    print(f"官方 runner : {row(ok, on)}  Wilson 95% CI {[f'{100*c:.2f}%' for c in wilson_ci(ok, on)]}")
    print(f"Forge       : {row(fk, fn)}  Wilson 95% CI {[f'{100*c:.2f}%' for c in wilson_ci(fk, fn)]}")
    p = fisher_two_sided(ok, on - ok, fk, fn - fk)
    print(f"Fisher 双侧精确检验 p = {p:.4f}")
    failed = sorted(
        (s for s, suc in forge_eps if not suc), key=lambda x: x if x is not None else -1
    )
    print(f"Forge 失败集 scene_seed：{failed}")

    # seed 对齐：只在官方计入的种子上比较
    common = sorted(set(official) & set(forge))
    ok2 = sum(official[s] for s in common)
    fk2 = sum(forge[s] for s in common)
    print(
        f"seed 对齐（{len(common)} 个共同种子）：官方 {ok2}/{len(common)} "
        f"vs Forge {fk2}/{len(common)}，"
        f"Fisher p = {fisher_two_sided(ok2, len(common) - ok2, fk2, len(common) - fk2):.4f}"
    )
    if len(common) != on or len(common) != fn:
        print(f"  （官方独有种子 {sorted(set(official) - set(forge))[:10]}；"
              f"Forge 独有种子 {sorted(set(forge) - set(official))[:10]}）")


def main() -> int:
    argv = list(sys.argv[1:])
    global ACC, FORGE_DIR
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag in ("--acc-dir", "--acc"):
            ACC = Path(argv.pop(0))
        elif flag in ("--forge-dir", "--forge"):
            FORGE_DIR = Path(argv.pop(0))
        else:
            print(f"未知参数：{flag}", file=sys.stderr)
            return 2
    for tid, task in enumerate(TASKS):
        log = ACC / f"official_{task}.log"
        if not log.exists():
            print(f"官方日志缺失：{log}", file=sys.stderr)
            continue
        official = parse_official(log)
        jsons = sorted(FORGE_DIR.glob("robotwin_*.json"))
        if not jsons:
            print(f"Forge 结果缺失：{FORGE_DIR}", file=sys.stderr)
            continue
        forge_eps = parse_forge_episodes(jsons[-1], tid)
        forge = {s: ok for s, ok in forge_eps if s is not None}
        compare(official, forge_eps, forge, task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
