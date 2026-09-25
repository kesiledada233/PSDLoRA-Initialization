#!/usr/bin/env python3
"""Capture exact software and hardware information without importing training code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = [
    "torch", "transformers", "datasets", "accelerate", "peft", "numpy",
    "scipy", "pandas", "pyarrow", "matplotlib", "seaborn", "PyYAML",
]


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def capture() -> dict:
    torch_info = {}
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            torch_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # environment capture must still succeed
        torch_info = {"import_error": repr(exc)}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
        "packages": package_versions(),
        "torch": torch_info,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = capture()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
