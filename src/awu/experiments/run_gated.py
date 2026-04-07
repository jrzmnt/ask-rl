"""
Run gated rollouts using a trained PPO policy and an entropy-based gate.
Evaluation is performed over a fixed number of episodes.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

from stable_baselines3 import PPO

from awu.envs.frozen_lake import FrozenLake
from awu.uncertainty.entropy import policy_entropy
from awu.experiments.controls import EntropyGate


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(base: str = "runs") -> Path:
    base = Path(base)
    base.mkdir(exist_ok=True)

    existing = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    )
    idx = len(existing) + 1
    run_dir = base / f"exp_{idx:03d}"
    run_dir.mkdir()
    return run_dir


# -------------------------------------------------
# Main experiment
# -------------------------------------------------

def main():
    cfg = load_config("configs/rl/ppo.yaml")

    # ---------- setup ----------
    run_dir = make_run_dir()
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    device = cfg.get("device", "auto")

    # ---------- environment ----------
    env_regime = cfg["env"]["regime"].split(",")[0]
    raw_env_cfg = cfg["env"]["regimes"][env_regime]

    env_cfg = {
        k: v for k, v in raw_env_cfg.items()
        if k != "max_steps"
    }

    env = FrozenLake(**env_cfg)


    # ---------- model ----------
    model = PPO.load(
        f"runs/ppo/{env_regime}/model.zip",  # trained model path
        env=env,
        device=device,
    )

    # ---------- gating ----------
    gate = EntropyGate(threshold=cfg["gating"]["threshold"])
    agent_regime = cfg["experiment"]["regime"]

    num_episodes = cfg["experiment"]["episodes_eval"]

    # ---------- rollout ----------
    metrics = []
    global_step = 0

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            entropy = policy_entropy(model, obs)

            if agent_regime == "rl_only":
                action, _ = model.predict(obs, deterministic=True)
                source = "rl"

            elif agent_regime == "slm_only":
                raise NotImplementedError(
                    "SLM-only regime not implemented yet."
                )

            elif agent_regime == "gated":
                if gate.should_query(entropy):
                    # placeholder: fallback to RL for now
                    action, _ = model.predict(obs, deterministic=True)
                    source = "slm"
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    source = "rl"

            else:
                raise ValueError(f"Unknown regime: {agent_regime}")

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            metrics.append(
                {
                    "episode": episode_idx,
                    "step": global_step,
                    "entropy": float(entropy),
                    "reward": float(reward),
                    "source": source,
                }
            )

            global_step += 1

    # ---------- save metrics ----------
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "step",
                "entropy",
                "reward",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)

    # ---------- summary ----------
    summary = {
        "episodes": num_episodes,
        "total_steps": global_step,
        "mean_entropy": float(np.mean([m["entropy"] for m in metrics])),
        "slm_activation_rate": float(
            np.mean([m["source"] == "slm" for m in metrics])
        ),
        "total_reward": float(np.sum([m["reward"] for m in metrics])),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GATED ROLLOUT SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
