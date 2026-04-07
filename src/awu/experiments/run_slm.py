from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import yaml
import pandas as pd
import math

from awu.envs.frozen_lake import FrozenLake
from awu.slm.model import load_slm
from awu.slm.prompt import load_prompt, PromptTemplate
from awu.slm.parse import parse_action
from awu.utils.io import save_json, save_yaml, save_csv
from awu.utils.seed import set_seed
from awu.utils.timing import Timer


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def next_experiment_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("exp_")
    ]

    if not existing:
        return base_dir / "exp_001"

    indices = []
    for p in existing:
        try:
            indices.append(int(p.name.split("_")[1]))
        except Exception:
            pass

    next_idx = max(indices) + 1 if indices else 1
    return base_dir / f"exp_{next_idx:03d}"


# -------------------------------------------------
# Main experiment
# -------------------------------------------------

def main():
    cfg_path = "configs/slm/small.yaml"
    cfg = load_config(cfg_path)

    run_name = cfg.get("name", "slm_run")

    # ---------- seed ----------
    set_seed(cfg.get("seed"))

    # ---------- environment ----------
    raw_env_cfg = cfg["env"]
    env_cfg = {k: v for k, v in raw_env_cfg.items() if k != "split"}
    env = FrozenLake(**env_cfg)

    # ---------- SLM ----------
    slm = load_slm(cfg["slm"])
    prompt: PromptTemplate = load_prompt(cfg["prompt"])

    # ---------- HARD ASSERT ----------
    # We expect ASCII rendering from now on
    if prompt.maze_shape is None:
        raise ValueError(
            "Prompt is expected to use ASCII maze rendering, "
            "but 'maze_shape' was not provided in config."
        )

    # ---------- logging buffers ----------
    episodes: List[int] = []
    steps: List[int] = []
    actions: List[int] = []

    rewards: List[float] = []
    entropies: List[float] = []  # placeholder
    latencies: List[float] = []
    costs: List[float] = []
    invalid_actions: List[int] = []

    # ---------- execution params ----------
    max_steps = cfg.get("max_steps", 50)
    use_cache = cfg.get("use_cache", False)
    debug_prompt = cfg.get("debug_prompt", False)

    slm_cache: Dict[tuple, Any] = {}

    # ---------- episodes ----------
    for episode in range(cfg["n_episodes"]):
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            step_count += 1
            if step_count > max_steps:
                break

            # ---------- prompt rendering ----------
            rendered_prompt = prompt.render(obs)

            if debug_prompt and step_count == 1:
                print("\n=== PROMPT SAMPLE (ASCII EXPECTED) ===")
                print(rendered_prompt)
                print("====================================\n")

            obs_key = tuple(obs.tolist())

            # ---------- SLM inference ----------
            if use_cache and obs_key in slm_cache:
                out = slm_cache[obs_key]
                latency = 0.0
            else:
                with Timer() as timer:
                    out = slm.generate(
                        rendered_prompt,
                        decoding=cfg["decoding"],
                    )
                if use_cache:
                    slm_cache[obs_key] = out
                latency = timer.elapsed

            action = parse_action(
                out.text,
                strategy=cfg["parse"]["strategy"],
            )

            # ---------- log common ----------
            episodes.append(episode)
            steps.append(step_count)
            latencies.append(latency)
            costs.append(out.cost)

            # ---------- invalid action ----------
            if action is None:
                actions.append(-1)
                rewards.append(0.0)
                entropies.append(math.nan)
                invalid_actions.append(1)
                continue

            # ---------- valid action ----------
            obs, reward, done, truncated, info = env.step(action)

            actions.append(int(action))
            rewards.append(float(reward))
            entropies.append(math.nan)
            invalid_actions.append(0)

            if truncated:
                break

    env.close()

    # ---------- save ----------
    base_dir = Path("runs") / "slm" / run_name
    run_dir = next_experiment_dir(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "episode": episodes,
        "step": steps,
        "action": actions,
        "reward": rewards,
        "entropy": entropies,
        "latency": latencies,
        "cost": costs,
        "invalid_action": invalid_actions,
    })

    save_csv(df.to_dict(orient="records"), run_dir / "metrics.csv")

    summary = {
        "reward_mean": float(df["reward"].mean()),
        "latency_mean": float(df["latency"].mean()),
        "cost_mean": float(df["cost"].mean()),
        "invalid_action_rate": float(df["invalid_action"].mean()),
        "num_steps": int(len(df)),
        "num_episodes": int(df["episode"].nunique()),
    }

    save_json(summary, run_dir / "summary.json")
    save_yaml(cfg, run_dir / "config.yaml")

    print(f"SLM run finished. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
