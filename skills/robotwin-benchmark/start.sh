#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ROBOTWIN_PYTHON:?Set ROBOTWIN_PYTHON to the Python 3.12 host interpreter}"
exec "${ROBOTWIN_PYTHON}" "$bundle_root/scripts/preflight.py" robotwin
