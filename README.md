# When to ASK: Uncertainty-Gated Language Assistance for Reinforcement Learning

[![IJCNN 2026](https://img.shields.io/badge/IJCNN-2026-blue)](https://2026.ijcnn.org)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-ppo--FrozenLake--v1--8x8-yellow)](https://huggingface.co/NathanGavenski/ppo-FrozenLake-v1-8x8)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://arxiv.org)

Official implementation of **ASK** (Adaptive Safety through Knowledge), an extrinsic method that improves out-of-distribution (OOD) generalization in reinforcement learning by selectively querying a Language Model (LM) based on uncertainty estimates, without retraining the RL policy.

> ASK uses Monte Carlo Dropout to measure epistemic and aleatoric uncertainty at each step. When uncertainty exceeds a threshold τ, it queries a LM for an action recommendation. In in-domain scenarios, ASK preserves PPO baseline performance. Under downward generalization (trained on 8×8, tested on 4×4–7×7), 32B/72B models achieve up to **0.95 reward**, where both PPO and LM alone fail completely.

---

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync
source .venv/bin/activate
```

---

## Setup: Generate Evaluation Maps

The FrozenLake evaluation and test maps are not included in the repository and must be generated before running any experiment. Each map size uses 100 fixed contexts for evaluation and 100 for testing (the paper uses 300 total, equally split into train/eval/test).

```bash
python scripts/generate_maps.py
```

This creates `tmp/frozenlake{4..8}/eval/` and `tmp/frozenlake{4..8}/test/` with 100 `.npy` maps each.

You can also generate them manually:

```python
from awu.envs.frozen_lake import FrozenLake

for size in [4, 5, 6, 7, 8]:
    env = FrozenLake(id="FrozenLake-v1", size=size)
    env.create_structures(100, eval=True)   # -> tmp/frozenlake{size}/eval/
    env.create_structures(100, eval=False)  # -> tmp/frozenlake{size}/test/
```

---

## Running Experiments

### Full pipeline

```bash
bash scripts/run_all.sh
```

Runs setup, PPO training, SLM-only rollout, and gated rollout in sequence.

### Individual steps

```bash
# 1. Train the PPO agent
bash scripts/run_rl.sh

# 2. Run SLM-only rollout
bash scripts/run_slm.sh

# 3. Run uncertainty-gated rollout (ASK: PPO + SLM)
bash scripts/run_gated.sh
```

Results are saved under `runs/`.

### Evaluation

Evaluation scripts load the pre-trained PPO model from HuggingFace ([NathanGavenski/ppo-FrozenLake-v1-8x8](https://huggingface.co/NathanGavenski/ppo-FrozenLake-v1-8x8)) and run it over the fixed evaluation maps.

```bash
python eval_ppo.py        # PPO-only
python eval_ppo_slm.py    # ASK: PPO + SLM gated
python eval_slm.py        # SLM-only
```

---

## Configuration

| File | Description |
|---|---|
| `configs/rl/ppo.yaml` | PPO training config (environment regime, timesteps) |
| `configs/slm/small.yaml` | SLM config — Qwen2.5-1.5B-Instruct |
| `configs/slm/medium.yaml` | SLM config — larger Qwen variant |
| `configs/sweeps/` | Sweep configs for threshold and model search |

The `regime` field in `configs/rl/ppo.yaml` controls the experiment type:

```yaml
experiment:
  regime: rl_only   # rl_only | slm_only | gated
```

**Key hyperparameters from the paper:**
- PPO training: 2×10⁷ timesteps (StableBaselines3 defaults)
- MC Dropout: N=100 forward passes, dropout rate 0.2
- LMs: Qwen2.5 family (0.5B–72B), off-the-shelf from HuggingFace, no fine-tuning
- Evaluation: 100 episodes per configuration

---

## Project Structure

```
├── configs/          # YAML experiment configs
├── eval_ppo.py       # Evaluate PPO-only
├── eval_ppo_slm.py   # Evaluate ASK (PPO + SLM gated)
├── eval_slm.py       # Evaluate SLM-only
├── prompts/          # SLM prompt templates
├── scripts/
│   ├── generate_maps.py   # Generate FrozenLake maps (run once)
│   ├── run_all.sh         # Full pipeline
│   ├── run_rl.sh          # Train PPO
│   ├── run_slm.sh         # SLM-only rollout
│   ├── run_gated.sh       # ASK gated rollout
│   └── setup.sh           # Install dependencies
└── src/awu/
    ├── envs/              # FrozenLake and Labyrinth environments
    ├── experiments/       # Training and rollout entry points
    ├── slm/               # SLM loading, prompting, and parsing
    ├── uncertainty/       # MC Dropout uncertainty estimation
    └── utils/             # Callbacks, seeding, I/O utilities
```

---

## Acknowledgments

This work was partially supported by UK Research and Innovation [grant number EP/S023356/1], in the UKRI Centre for Doctoral Training in Safe and Trusted Artificial Intelligence (www.safeandtrustedai.org).

---

## Citation

```bibtex
@inproceedings{monteiro2026ask,
  title     = {When to {ASK}: Uncertainty-Gated Language Assistance for Reinforcement Learning},
  author    = {Monteiro, Juarez and Gavenski, Nathan and Zuin, Gianlucca and Veloso, Adriano},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
}
```
