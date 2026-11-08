#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
prompt="${1:?Usage: bash scripts/quickstart.sh 'your prompt' [Wan CLI options]}"
shift

python Wan2.1/run.py "$@" --mode calibrate --prompt "$prompt" \
  --outdir outputs/quickstart/calibration

python Wan2.1/run.py "$@" --mode infer --prompt "$prompt" \
  --skip-table outputs/quickstart/calibration/tacache_skip_timestep.json \
  --outdir outputs/quickstart/inference
