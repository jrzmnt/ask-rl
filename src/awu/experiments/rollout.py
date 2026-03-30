"""
Rollout utilities for instrumented policy execution.

This module executes a trained PPO policy step-by-step,
collecting logits, entropy, rewards, and termination signals.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from stable_baselines3 import PPO

from awu.experiments.metrics import policy_entropy_from_logits


def rollout_episode(
    model: PPO,
    env,
    deterministic: bool = False,
    max_steps: int | None = None,
) -> Dict[str, List]:
    """
    Run a single rollout episode with a trained PPO model.

    Args:
        model: trained PPO model
        env: Gym environment
        deterministic: whether to use deterministic actions
        max_steps: optional cap on steps

    Returns:
        Dictionary containing trajectory data
    """
    obs, info = env.reset()

    trajectory = {
        "observations": [],
        "actions": [],
        "logits": [],
        "entropy": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
    }

    step = 0
    done = False

    while not done:
        device = model.device
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(device)

        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            action = distribution.get_actions(deterministic=deterministic)
            logits = distribution.distribution.logits.squeeze(0)

        entropy = policy_entropy_from_logits(logits)
        action_int = int(action.item())
        obs, reward, terminated, truncated, info = env.step(action_int)

        trajectory["observations"].append(obs)
        trajectory["actions"].append(action.item())
        trajectory["logits"].append(logits.cpu().numpy())
        trajectory["entropy"].append(entropy)
        trajectory["rewards"].append(float(reward))
        trajectory["terminated"].append(bool(terminated))
        trajectory["truncated"].append(bool(truncated))

        done = terminated or truncated
        step += 1

        if max_steps is not None and step >= max_steps:
            break

    return trajectory


# if __name__ == "__main__":
#     from gymnasium.wrappers import TimeLimit
#     from stable_baselines3 import PPO

#     from awu.envs.labyrinth_gym_adapter import LabyrinthGymAdapter

#     env = LabyrinthGymAdapter(shape=(5, 5))
#     env = TimeLimit(env, max_episode_steps=50)

#     model = PPO("MlpPolicy", env, verbose=0)
#     model.learn(total_timesteps=5_000)

#     traj = rollout_episode(model, env)

#     print("Steps:", len(traj["entropy"]))
#     print("Entropy (first 5):", traj["entropy"][:5])
#     print("Entropy (mean):", np.mean(traj["entropy"]))
