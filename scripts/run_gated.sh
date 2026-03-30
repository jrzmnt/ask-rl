#!/usr/bin/env bash
set -e

source .venv/bin/activate

echo "==> Running gated rollout"
python src/awu/experiments/run_gated.py
