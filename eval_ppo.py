from __future__ import annotations

import glob
import random
from typing import Dict, Any
import yaml
import os

import gymnasium as gym
from huggingface_sb3 import load_from_hub
import minigrid
from minigrid.wrappers import FlatObsWrapper
import numpy as np
from stable_baselines3 import PPO
from tabulate import tabulate
from tqdm import tqdm
import torch

from awu.envs.labyrinth_env import LabyrinthEnv
from awu.envs.frozen_lake import FrozenLake


# -------------------------------------------------
# Utilities
# -------------------------------------------------
ACTIONS_TO_STR = {x: y for x, y in enumerate(["LEFT", "DOWN", "RIGHT", "UP"])}
STR_TO_ACTIONS = {y: x for x, y in enumerate(["LEFT", "DOWN", "RIGHT", "UP"])}
MAP_INT_TO_CHAR = {value: key for key, value in {'S': 0, 'F': 1, 'H': 2, 'G': 3, 'A': 4}.items()}


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
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def compute_uncertainties(
    model: PPO,
    obs: np.ndarray,
    n_samples: int = 100,
    use_log2: bool = True
) -> dict:
    """
    Compute total, aleatoric, and epistemic uncertainty
    """
    # Enable dropout
    model.policy.mlp_extractor.train()

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)

    # Collect probability distributions
    probs_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            # Get distribution
            distribution = model.policy.get_distribution(obs_tensor)

            # Get action probabilities (for discrete actions)
            if hasattr(distribution.distribution, 'probs'):
                probs = distribution.distribution.probs.squeeze()
            else:
                # For continuous, you'd need to discretize or use different approach
                raise NotImplementedError("This is for discrete actions only")

            probs_list.append(probs)

    # Stack all probability distributions [n_samples, n_actions]
    probs = torch.stack(probs_list)

    # Choose logarithm
    log_fn = torch.log2 if use_log2 else torch.log

    # Mean probability distribution
    mean_probs = probs.mean(dim=0)

    # Total uncertainty: H(E[p])
    total_uncertainty = -(mean_probs * log_fn(mean_probs + 1e-10)).sum()

    entropies = -(probs * log_fn(probs + 1e-10)).sum(dim=1)
    aleatoric_uncertainty = entropies.mean()

    # Epistemic uncertainty: E[KL(p || E[p])]
    # KL(p || q) = sum(p * log(p/q))
    kl_divs = (probs * log_fn((probs + 1e-10) / (mean_probs + 1e-10))).sum(dim=1)
    epistemic_uncertainty = kl_divs.mean()

    # Disable dropout
    model.policy.mlp_extractor.eval()

    return (
        total_uncertainty.item(),
        aleatoric_uncertainty.item(),
        epistemic_uncertainty.item(),
        probs.mean(dim=0)
    )


def print_map(obs, size):
    int_to_char = {value: key for key, value in {
        'S': 0, 'F': 1, 'H': 2, 'G': 3, 'A': 4}.items()}

    map = obs[: size * size]
    agent = obs[size * size:]

    map[np.argmax(agent)] = 4
    map = torch.tensor(map).view((size, size)).numpy()
    map = np.vectorize(int_to_char.get)(map)
    map = ['  '.join(x) for x in map]

    print("Map:")
    for row in map:
        print(row)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------


def evaluate_model(
    model: PPO, env: gym.Env, n_episodes: int = 10,
    deterministic: bool = True, render: bool = False,
    load_local: bool = False, path: str = None,
    verbose: bool = False
):
    """
    Evaluate the model over multiple episodes.

    Returns:
        Dict with mean/std of episode rewards and lengths
    """
    episode_rewards = []
    episode_lengths = []

    if load_local:
        structures = glob.glob(os.path.join(path, "*.npy"))
        if len(structures) < n_episodes:
            env.create_structures(n_episodes, eval="eval" in path)
            structures = glob.glob(os.path.join(path, "*.npy"))

    for episode, structure in enumerate(structures):
        if load_local:
            obs, info = env.load(np.load(structure).tolist())
        else:
            obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        new_obs = None

        while not done:

            if new_obs is not None:
                obs = new_obs

            action, _ = model.predict(obs, deterministic=deterministic)

            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        result = "Goal" if reward == 1.0 else "Truncated"
        death = True if terminated and not truncated and reward == 0 else False
        if death:
            result = "Death"
            total, aleatoric, epistemic, probs = compute_uncertainties(
                model, obs)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if verbose:
            if death:
                print(
                    f"Episode {episode + 1}/{n_episodes}: "
                    f"Reward = {episode_reward:.2f}, Length = {episode_length}, "
                    f"Result = {result}"
                    f"\nUncertainty: Total = {total:.2f}"
                    f", Aleatoric = {aleatoric:.2f}"
                    f", Epistemic = {epistemic:.2f}"
                )
                probs = [round(x, 2) for x in probs.tolist()]
                probs = [f"{y}: {x}" for x, y in zip(
                    probs, ["LEFT", "DOWN", "RIGHT", "UP"])]
                print(f"Probs: {probs}")
                print(f"Action: {ACTIONS_TO_STR[int(action)]}")
                print_map(obs, env.size)
                print()

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

    return results


# -------------------------------------------------
# Main evaluation
# -------------------------------------------------


def main():
    cfg = load_config("configs/rl/ppo.yaml")
    set_seed(cfg.get("experiment", {}).get("seed", None))

    # ---------- environment ----------
    env_regime = cfg["env"]["regime"]
    if "," in env_regime:
        regimes = tqdm(env_regime.split(","))
    else:
        regimes = [env_regime]

    result_table = {}
    for env_regime in regimes:
        raw_env_cfg = cfg["env"]["regimes"][env_regime]

        env_cfg = {
            k: v for k, v in raw_env_cfg.items()
            if k != "max_steps"
        }

        env = load_env(env_regime, env_cfg)

        # ---------- load model ----------
        # You can choose which model to load:

        # Option 1: Load the final model
        # model_path = Path("runs/ppo") / env_regime / "model.zip"

        # Option 2: Load the best model from evaluation callback
        # model_path = Path("./logs/best_model/best_model.zip")

        # Option 3: Load from huggingface
        model_path = load_from_hub(
            "NathanGavenski/ppo-FrozenLake-v1",
            filename=f"frozenlake{raw_env_cfg['size']}.zip",
        )

        model = PPO.load(model_path)
        model.policy.mlp_extractor.eval()

        # ---------- evaluate ----------
        if len(regimes) == 1:
            print("\n" + "=" * 50)
            print("EVALUATING MODEL")
            print("=" * 50)

        results = evaluate_model(
            model=model,
            env=env,
            n_episodes=100,
            deterministic=True,
            render=False,
            load_local=True if "frozenlake" in env_regime else False,
            path=f"./tmp/{env_regime.replace('-', '')}/eval",
            verbose=len(regimes) == 1
        )

        result_table[env_regime] = [
            f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            f"{results['mean_length']:.2f} ± {results['std_length']:.2f}"
        ]

        if len(regimes) == 1:
            print("=" * 50)
            print(f"RESULTS FOR {env_regime}")
            print("=" * 50)
            print(
                f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(
                f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")

        env.close()

    if len(regimes) > 1:
        table = []
        for key, value in result_table.items():
            table.append([key, *value])
        print(tabulate(table, headers=["Name", "Reward", "Length"]))


if __name__ == "__main__":
    main()
