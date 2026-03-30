#!/usr/bin/env bash
set -e

source .venv/bin/activate

echo "==> Training PPO agent"
python src/awu/experiments/run_rl.py
