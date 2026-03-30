#!/usr/bin/env bash
set -e

source .venv/bin/activate

echo "==> Running SLM-only rollout"
python src/awu/experiments/run_slm.py
