# PAOS Ubuntu 20.04 integration, 2026-09-17

Published candidate scope: `robotwin_benchmark` 1.1.4, `robotwin_policy` 1.1.1, and `robotwin-benchmark` 0.1.0. Resource registration is handled separately by the owner.

## Validation

RoboTwin source tests: **123 passed**. Real SAPIEN/cuRobo simulation plus a hold-action protocol server completed two 400-step Sessions (800 observation/action requests), with matching scene seed/instruction, separate result ledgers and clean stop.

**Real G05 model acceptance is pending.** The exact gated g05-robotwin20 checkpoint has not been obtained. A separate matching G05 model-server environment, its weights and auxiliary assets, and an official-versus-Forge evaluation are still required. The Skill owns its websocket client only; it does not manage the external server. The hold-action test must not be reported as G05 accuracy.

Both frozen Nodes were built on Ubuntu 20.04 x86_64 with Python 3.12. Their scanned ELF dependencies require at most GLIBC 2.28 (target OS: 2.31). The Skill passed repeat installation, inspection, ready/Query, Session, result retrieval and stop checks in an isolated PAOS test environment. Production Registry installation remains a post-registration step.

## Runtime and sources

Read [HOST_RUNTIME.md](../skills/robotwin-benchmark/HOST_RUNTIME.md), [SKILL.md](../skills/robotwin-benchmark/SKILL.md) and [skill.yaml](../skills/robotwin-benchmark/skill.yaml). These are thin Nodes; model and simulator assets are external. The complete reviewed Skill source is tracked, including its dataflow and immutable Node locks.

The student's original repository history is retained. Benchmark and policy were independently built from the source hashes in their receipts before this publication commit. Adding Skill source and release documentation did not rebuild the frozen Node bytes. The Python project's package version and the two independently versioned binary artifacts are separate identifiers.

## Rebuild commands

On Ubuntu 20.04, with uv and Python 3.12:

```bash
python3.12 scripts/build_release.py --kind benchmark --version 1.1.4
python3.12 scripts/build_release.py --kind policy --version 1.1.1
python3.12 scripts/package_skill.py skills/robotwin-benchmark --output-dir dist
```

The Skill packager verifies every source file against archive-manifest.json before packaging and produces the reviewed archive byte-for-byte. If changing a Skill, update its version/locks/member manifest and validate the changed bundle before publication. A fresh Node build needs new digest verification and relevant runtime acceptance; binary reproducibility is not assumed.

Hashes, sizes, build source hashes and dependency receipts are in [the publication manifest](../releases/paos-ubuntu20-20260917.json). No model weights or simulator datasets are redistributed here.
