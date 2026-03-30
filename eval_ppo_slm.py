from __future__ import annotations

import glob
import gc
import random
import time
import csv
import os
import re
from typing import Dict, Any, List

import gymnasium as gym
import numpy as np
import torch
import optuna

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

from awu.envs.frozen_lake import FrozenLake
from awu.slm.model import load_slm

import json
# =================================================
# PROMPT TEMPLATE
# =================================================
# PROMPT = """
# You help a robot reach a goal without falling into holes.

# POSITION:
# Agent: row {agent_row}, col {agent_col}
# Goal: row {goal_row}, col {goal_col}

# IMMEDIATE NEIGHBORS (1 step):
# - UP (row {agent_row-1}, col {agent_col}): {up_tile}
# - DOWN (row {agent_row+1}, col {agent_col}): {down_tile}
# - LEFT (row {agent_row}, col {agent_col-1}): {left_tile}
# - RIGHT (row {agent_row}, col {agent_col+1}): {right_tile}

# LOOK AHEAD (2 steps in each direction):
# - UP → UP: {up_up_tile}
# - DOWN → DOWN: {down_down_tile}
# - LEFT → LEFT: {left_left_tile}
# - RIGHT → RIGHT: {right_right_tile}

# Key: SAFE=okay, HOLE=death, GOAL=target, EDGE=boundary

# The robot's autopilot suggests: {action}

# THINK THROUGH THIS:

# Step 1 - Check immediate safety:
# Is {action} a HOLE? Look at the {action} tile above.
# Answer: {action} is [SAFE/HOLE/EDGE/GOAL]

# Step 2 - Check for traps:
# If we go {action}, what's 2 steps ahead in that direction?
# Answer: {action} → {action} leads to [SAFE/HOLE/EDGE/GOAL]

# Step 3 - List all safe immediate moves:
# Check each direction. Which are NOT holes or edges?
# Answer: Safe moves are [list all safe directions]

# Step 4 - Pick best safe move toward goal:
# Goal is at row {goal_row}, col {goal_col}.
# Current position: row {agent_row}, col {agent_col}.
# Need to {"go DOWN" if goal_row > agent_row else "go UP" if goal_row < agent_row else "stay in row"}
# Need to {"go RIGHT" if goal_col > agent_col else "go LEFT" if goal_col < agent_col else "stay in column"}

# From safe moves, which one moves closer to goal?
# Answer: Best move is [direction]

# FINAL: [UP/DOWN/LEFT/RIGHT]
# """

PROMPT = """
You are a robot navigation policy.

Your task is to choose exactly ONE action.

VALID ACTIONS:
UP
DOWN
LEFT
RIGHT

RULES:
- Do NOT explain.
- Do NOT add text.
- Do NOT add markdown.
- Do NOT add JSON keys other than "action".
- If the suggested action is unsafe, choose a safe alternative.
- If multiple safe actions exist, choose the one that moves closer to the goal.

STATE:
Agent position: row {agent_row}, col {agent_col}
Goal position: row {goal_row}, col {goal_col}

IMMEDIATE NEIGHBORS:
UP: {up_tile}
DOWN: {down_tile}
LEFT: {left_tile}
RIGHT: {right_tile}

LOOK AHEAD:
UP->UP: {up_up_tile}
DOWN->DOWN: {down_down_tile}
LEFT->LEFT: {left_left_tile}
RIGHT->RIGHT: {right_right_tile}

Autopilot suggestion: {action}

OUTPUT FORMAT (MANDATORY):
{{"action":"UP"}} OR {{"action":"DOWN"}} OR {{"action":"LEFT"}} OR {{"action":"RIGHT"}}
"""


# =================================================
# Constants
# =================================================
ACTIONS_TO_STR = {x: y for x, y in enumerate(["LEFT", "DOWN", "RIGHT", "UP"])}
STR_TO_ACTIONS = {y: x for x, y in enumerate(["LEFT", "DOWN", "RIGHT", "UP"])}
MAP_INT_TO_CHAR = {value: key for key, value in {"S": 0, "F": 1, "H": 2, "G": 3, "A": 4}.items()}

QWEN_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    
]

FIXED_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 10,
}

CSV_FILE = "experiments.csv"
EXPERIMENTS_DIR = "experiment_logs"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


# =================================================
# Reproducibility
# =================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =================================================
# Helpers
# =================================================
def short_model_name(model_name: str) -> str:
    name = model_name.lower()
    if "0.5b" in name:
        return "qwen_0.5b"
    if "1.5b" in name:
        return "qwen_1.5b"
    if "3b" in name:
        return "qwen_3b"
    if "7b" in name:
        return "qwen_7b"
    if "14b" in name:
        return "qwen_14b"
    if "32b" in name:
        return "qwen_32b"
    if "72b" in name:
        return "qwen_72b"
    return "qwen_unknown"


# =================================================
# Uncertainty
# =================================================
def compute_uncertainties(model: PPO, obs: np.ndarray, n_samples: int = 100):
    model.policy.mlp_extractor.train()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)

    probs_cpu = []
    with torch.no_grad():
        for _ in range(n_samples):
            dist = model.policy.get_distribution(obs_tensor)
            probs_cpu.append(dist.distribution.probs.squeeze(0).cpu())

    probs = torch.stack(probs_cpu)
    mean_probs = probs.mean(dim=0)

    log = torch.log2
    total = -(mean_probs * log(mean_probs + 1e-10)).sum()
    aleatoric = -(probs * log(probs + 1e-10)).sum(dim=1).mean()
    epistemic = (probs * log((probs + 1e-10) / (mean_probs + 1e-10))).sum(dim=1).mean()

    model.policy.mlp_extractor.eval()

    return float(total), float(aleatoric), float(epistemic), mean_probs


# =================================================
# Prompt
# =================================================
def build_prompt(obs, env, action, probs, uncertainty):
    size = env.size

    grid = obs[: size * size].copy()
    agent = obs[size * size :]
    grid[np.argmax(agent)] = 4

    grid = grid.reshape(size, size)

    agent_pos = tuple(map(int, np.argwhere(grid == 4)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    holes = list(map(tuple, np.argwhere(grid == 2)))

    confidence = round(float(torch.softmax(probs, dim=0).max()) * 100, 2)
    total, aleatoric, epistemic = uncertainty

    prompt = PROMPT.format(
        size=size,
        agent=agent_pos,
        goal=goal_pos,
        holes=holes,
        action=ACTIONS_TO_STR[action],
        confidence=confidence,
        total=round(total, 4),
        aleatoric=round(aleatoric, 4),
        epistemic=round(epistemic, 4),
    )

    return prompt

def prompt(obs, size, prediction, uncertainty):
    """
    Generate prompt for SLM using the new format (optimized version).
    """
    prompt_ = PROMPT

    # Map information
    map_ = obs[: size * size]
    agent = obs[size * size :]
    map_[np.argmax(agent)] = 4
    map_ = torch.tensor(map_).view((size, size)).numpy()
    map_ = np.vectorize(MAP_INT_TO_CHAR.get)(map_)
    
    agent_pos = tuple([x.item() for x in np.where(map_ == "A")])
    goal_pos = tuple([x.item() for x in np.where(map_ == "G")])
    
    agent_row, agent_col = agent_pos
    goal_row, goal_col = goal_pos
    
    # Helper function to get tile at position
    def get_tile(row, col):
        """Get tile type at given position."""
        if row < 0 or row >= size or col < 0 or col >= size:
            return "EDGE"
        tile = map_[row, col]
        tile_map = {
            "F": "SAFE",
            "H": "HOLE", 
            "G": "GOAL",
            "S": "SAFE",
            "A": "SAFE"  # Agent position is safe
        }
        return tile_map.get(tile, "SAFE")
    
    # Direction deltas: (row_delta, col_delta)
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    # Get immediate neighbors and look ahead for each direction
    tiles = {}
    for direction, (dr, dc) in directions.items():
        # 1 step
        tiles[f"{direction}_tile"] = get_tile(agent_row + dr, agent_col + dc)
        # 2 steps
        tiles[f"{direction}_{direction}_tile"] = get_tile(agent_row + 2*dr, agent_col + 2*dc)
    
    # Replace all position and tile information
    replacements = {
        "{agent_row}": str(agent_row),
        "{agent_col}": str(agent_col),
        "{goal_row}": str(goal_row),
        "{goal_col}": str(goal_col),
        "{action}": ACTIONS_TO_STR[prediction.get("action")],
        **{f"{{{k}}}": v for k, v in tiles.items()}
    }
    
    for key, value in replacements.items():
        prompt_ = prompt_.replace(key, value)
    
    return prompt_

# =================================================
# SLM Policy
# =================================================
class SLMPolicy:
    def __init__(self, slm):
        self.slm = slm

    def get_action(self, output: str):
        try:
            texto = output.strip()

            # Corrige casos como '{{"action":"RIGHT"}}'
            if texto.startswith("{{") and texto.endswith("}}"):
                texto = texto[1:-1]

            # Extrai o primeiro objeto JSON válido
            match = re.search(r'\{.*?\}', texto)
            if not match:
                return None

            texto = match.group(0)

            # Fecha chave caso esteja incompleto: {"action":"RIGHT"
            if texto.count("{") > texto.count("}"):
                texto += "}"

            dados = json.loads(texto)
            acao = dados.get("action")

            acoes_validas = {"UP", "DOWN", "LEFT", "RIGHT"}

            return acao if acao in acoes_validas else None

        except (json.JSONDecodeError, TypeError):
            return None




    def predict(self, prompt: str):
        with torch.inference_mode():
            output = self.slm.generate(prompt, FIXED_DECODING)

        text = output.text if hasattr(output, "text") else str(output)
        action_str = self.get_action(text)

        action = STR_TO_ACTIONS.get(action_str, None)

        # print("=== SLM Prompt ===")
        # print(prompt)
        # print("=== SLM Output ===")
        # print(text)
        # print(f"=== Parsed Action: {action_str} -> {action} ===\n")
        # print("RAW TEXT REPR:", repr(text))
        # print("TYPE:", type(text))

        return action


# =================================================
# Logging
# =================================================
def save_rows_csv(rows: List[Dict[str, Any]]):
    exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_experiment_txt(
    episode_logs: list[dict],
    env_size: int,
    model_name: str,
    threshold: float,
    trial: int,
):
    model_tag = short_model_name(model_name)
    threshold_tag = f"{threshold:.2f}".replace(".", "_")

    filename = (
        f"frozenlake_{env_size}_"
        f"{model_tag}_"
        f"threshold_{threshold_tag}_"
        f"trial_{trial}.txt"
    )

    path = os.path.join(EXPERIMENTS_DIR, filename)

    rewards = [r["reward"] for r in episode_logs]
    slm_called = [r["slm_called_pct"] for r in episode_logs]
    slm_valid = [r["slm_valid_pct"] for r in episode_logs]
    slm_ovs = [r["slm_overwrite_pct"] for r in episode_logs]

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_slm_called = float(np.mean(slm_called))
    mean_slm_valid = float(np.mean(slm_valid))
    mean_slm_ov = float(np.mean(slm_ovs))

    with open(path, "w") as f:
        # Header
        f.write(
            "Reward\tLength\tResult\t"
            "SLM Called (%)\tSLM Valid (%)\tSLM Overwrite (%)\tEpisode Time (s)\n"
        )

        # Episode-level logs
        for row in episode_logs:
            f.write(
                f"{row['reward']:.2f}\t"
                f"{row['length']}\t"
                f"{row['result']}\t"
                f"{row['slm_called_pct']:.2f}\t"
                f"{row['slm_valid_pct']:.2f}\t"
                f"{row['slm_overwrite_pct']:.2f}\t"
                f"{row['episode_time_sec']:.4f}\n"
            )

        # Summary block
        f.write("\n----------------------------------------\n")
        f.write("SUMMARY\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Mean Reward: {mean_reward:.4f}\n")
        f.write(f"Std Reward: {std_reward:.4f}\n")
        f.write(f"Mean SLM Called (%): {mean_slm_called:.4f}\n")
        f.write(f"Mean SLM Valid (%): {mean_slm_valid:.4f}\n")
        f.write(f"Mean SLM Overwrite (%): {mean_slm_ov:.4f}\n")
        f.write("----------------------------------------\n")


# =================================================
# Evaluation
# =================================================
def evaluate_model(
    model: PPO,
    slm: SLMPolicy,
    env: gym.Env,
    threshold: float,
    n_episodes: int = 100,
    path: str = "./tmp/frozenlake/eval",
):
    logs = []

    structures = glob.glob(os.path.join(path, "*.npy"))
    if len(structures) < n_episodes:
        env.create_structures(n_episodes, eval="eval" in path)
        structures = glob.glob(os.path.join(path, "*.npy"))

    structures = structures[:n_episodes]

    for ep, structure in enumerate(structures):
        obs, info = env.load(np.load(structure).tolist())
        start_ep = time.time()

        done = False
        reward_sum = 0
        steps = 0

        slm_called = 0
        slm_valid = 0
        slm_overwrites = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            total, alea, epis, probs = compute_uncertainties(model, obs)

            if total >= threshold:
                slm_called += 1

            #     prompt = build_prompt(
            #     obs,
            #     env,              # passa o ENV inteiro
            #     int(action),
            #     probs,
            #     (total, alea, epis),
            # )

                slm_prompt = prompt(
                    obs=obs,
                    size=env.size,
                    prediction={"action": int(action.item()), "confidence": probs},
                    uncertainty={"total": total, "aleatoric": alea, "epistemic": epis},
                )

                slm_action = slm.predict(slm_prompt)

                if slm_action is not None:
                    slm_valid += 1

                    #print("SLM OVERRIDE: PPO action", action, "-> SLM action", slm_action, "--->", slm_action == action)
                    if slm_action != action:
                        action = slm_action
                        slm_overwrites += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward_sum += reward
            steps += 1

        ep_time = time.time() - start_ep
        result = "Goal" if reward_sum == 1.0 else "Death" if terminated else "Truncated"

        logs.append({
            "episode": ep + 1,
            "reward": reward_sum,
            "length": steps,
            "result": result,
            "slm_called_pct": (slm_called / steps) * 100,
            "slm_valid_pct": (slm_valid / steps) * 100,
            "slm_overwrite_pct": (slm_overwrites / steps) * 100,
            "episode_time_sec": ep_time,
        })

    mean_reward = float(np.mean([l["reward"] for l in logs]))
    return mean_reward, logs



# =================================================
# Optuna Objective
# =================================================
def objective(trial, model_name: str, env_size: int):
    threshold = trial.suggest_float("threshold", 0.1, 1.2)

    env = FrozenLake(size=env_size)

    model = PPO.load(
        load_from_hub(
            "NathanGavenski/ppo-FrozenLake-v1",
            f"frozenlake{env_size}.zip",
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # slm = SLMPolicy(load_slm(model_name))

    slm_cfg = {
    "provider": "hf",
    "model": model_name,
    }

    slm = SLMPolicy(load_slm(slm_cfg))


    mean_reward, episode_logs = evaluate_model(
        model=model,
        slm=slm,
        env=env,
        threshold=threshold,
        path=f"./tmp/frozenlake{env_size}/eval",
    )

    if mean_reward >= 0.999:
        trial.study.stop()

    for row in episode_logs:
        row.update({
            "model": model_name,
            "env": f"FrozenLake-{env_size}",
            "threshold": threshold,
            "trial": trial.number,
        })

    save_rows_csv(episode_logs)
    save_experiment_txt(
        episode_logs=episode_logs,
        env_size=env_size,
        model_name=model_name,
        threshold=threshold,
        trial=trial.number,
    )

    del model, slm, env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_reward


# =================================================
# Main
# =================================================
def main():
    set_seed(42)

    for model_name in QWEN_MODELS:
        for env_size in [6, 7, 8]:
            study_name = f"{short_model_name(model_name)}_frozenlake_{env_size}"
            study = optuna.create_study(
                direction="maximize",
                storage="sqlite:///optuna.db",
                study_name=study_name,
                load_if_exists=True,
            )
            study.optimize(
                lambda t: objective(t, model_name, env_size),
                n_trials=10,
            )

            print(f"\nBEST for {model_name} | FrozenLake-{env_size}")
            print(study.best_params)
            print("Reward:", study.best_value)


if __name__ == "__main__":
    main()
