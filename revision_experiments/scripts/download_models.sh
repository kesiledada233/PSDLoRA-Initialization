#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/b205/FDA_INIT"

# hf download resumes partial files automatically. Re-run this script after an interruption.
hf download Qwen/Qwen2.5-7B \
  --revision e25af2efae60472008fbeaf5fb7c4274a87f78d4 \
  --local-dir "$ROOT/pretrained_models/Qwen2.5-7B"

hf download prometheus-eval/prometheus-7b-v2.0 \
  --revision 66ffb1fc20beebfb60a3964a957d9011723116c5 \
  --local-dir "$ROOT/pretrained_models/prometheus-7b-v2.0"
