#!/usr/bin/env bash
set -e

echo "==> FULL PIPELINE START"

bash scripts/setup.sh
bash scripts/run_rl.sh
bash scripts/run_slm.sh
bash scripts/run_gated.sh

echo "==> FULL PIPELINE DONE"
