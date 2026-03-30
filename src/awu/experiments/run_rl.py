from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, Any
import yaml

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from awu.envs.labyrinth_env import LabyrinthEnv
from awu.envs.frozen_lake import FrozenLake
from awu.utils.ppo import DropoutActorCriticPolicy
from awu.utils.callbacks import EvalCallbackWithEvalMode, EvalCallbackFrozenlake


# -------------------------------------------------
# Utilities
# -------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_env(env_regime: str, env_config: Dict[str, Any]) -> gym.Env:
    if env_regime == "minigrid":
        return FlatObsWrapper(gym.make(**env_config))
    elif "frozenlake" in env_regime:
        return FrozenLake(**env_config)
    else:
        return LabyrinthEnv(**env_config)


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# -------------------------------------------------
# Main training
# -------------------------------------------------


def main():
    cfg = load_config("configs/rl/ppo.yaml")
    set_seed(cfg.get("experiment", {}).get("seed", None))

    # ---------- environment ----------
    env_regime = cfg["env"]["regime"]
    if "," in env_regime:
        regimes = env_regime.split(",")
        for env_regime in regimes:
            raw_env_cfg = cfg["env"]["regimes"][env_regime]

            print()
            print("Training with:", env_regime)
            print("Using settings:", raw_env_cfg)
            print()

            # Remove keys not supported by LabyrinthEnv
            env_cfg = {
                k: v for k, v in raw_env_cfg.items()
                if k != "max_steps"
            }

            env = load_env(env_regime, env_cfg)
            env = Monitor(env)

            # Evaluation environment
            eval_env = load_env(env_regime, env_cfg)
            eval_env = Monitor(eval_env)

            # ---------- PPO ----------
            model = PPO(
                policy=DropoutActorCriticPolicy,
                env=env,
                policy_kwargs={"dropout_rate": 0.2},
                verbose=1,
                device=cfg.get("device", "auto"),
            )

            total_timesteps = cfg["training"]["total_timesteps"]

            # ---------- save intermediate weights ----------
            eval_callback_class = EvalCallbackWithEvalMode
            if "frozenlake" in env_regime:
                eval_callback_class = EvalCallbackFrozenlake

            eval_callback = eval_callback_class(
                eval_env,
                best_model_save_path=f"./logs/{env_regime}/best_model/",
                log_path=f"./logs/{env_regime}/results/",
                eval_freq=10000,
                deterministic=True,
                render=False,
                n_eval_episodes=100,
                verbose=1
            )

            # ---------- train ----------
            model.learn(total_timesteps=total_timesteps, callback=eval_callback)

            # ---------- save ----------
            model_dir = Path("runs/ppo") / env_regime
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "model.zip"
            model.save(model_path)

            print(f"Model saved to {model_path}")

            env.close()
            eval_env.close()


if __name__ == "__main__":
    main()
