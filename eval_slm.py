from __future__ import annotations

import glob
import os
import random
import json
import torch
from pathlib import Path
import re
from typing import Dict, Any, List, Tuple

import yaml
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from awu.envs.frozen_lake import FrozenLake
from awu.slm.model import load_slm
from awu.slm.prompt import load_prompt
from awu.slm.parse import parse_action


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


# -------------------------------------------------
# Observation decoding
# -------------------------------------------------
def decode_obs(obs: np.ndarray, size: int) -> Tuple[List[List[int]], Tuple[int, int]]:
    obs = np.asarray(obs)

    map_flat = obs[: size * size]
    agent_flat = obs[size * size:]

    idx = int(np.argmax(agent_flat))
    agent_pos = (idx // size, idx % size)

    grid = map_flat.reshape(size, size).astype(int).tolist()
    return grid, agent_pos


def find_goal(grid: List[List[int]]) -> Tuple[int, int]:
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 3:
                return i, j
    raise ValueError("Goal not found")


def grid_to_text(grid: List[List[int]]) -> str:
    SYMBOLS = {0: "S", 1: "F", 2: "H", 3: "G"}
    return "\n".join(" ".join(SYMBOLS[c] for c in row) for row in grid)

def render_grid_with_agent(grid: List[List[int]], agent_pos: Tuple[int, int]) -> str:
    SYMBOLS = {0: "S", 1: "F", 2: "H", 3: "G"}

    lines = []
    for i, row in enumerate(grid):
        rendered_row = []
        for j, cell in enumerate(row):
            symbol = SYMBOLS[cell]
            if (i, j) == agent_pos:
                symbol = f"[{symbol}]"
            rendered_row.append(symbol)
        lines.append(" ".join(rendered_row))

    return "\n".join(lines)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_symbolic_state(
    grid: List[List[int]],
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    visited: set[Tuple[int, int]],
    last_action: str | None,
) -> Dict[str, Any]:
    size = len(grid)
    r, c = agent_pos

    def tile_at(pos):
        rr, cc = pos
        if rr < 0 or rr >= size or cc < 0 or cc >= size:
            return "OUT_OF_BOUNDS"
        cell = grid[rr][cc]
        return {0: "S", 1: "F", 2: "H", 3: "G"}[cell]

    adjacent = {
        "UP": tile_at((r - 1, c)),
        "DOWN": tile_at((r + 1, c)),
        "LEFT": tile_at((r, c - 1)),
        "RIGHT": tile_at((r, c + 1)),
    }

    current_distance = manhattan(agent_pos, goal_pos)

    action_effects = {}
    for action, (dr, dc) in {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }.items():
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= size or nc < 0 or nc >= size:
            action_effects[action] = None
        else:
            action_effects[action] = manhattan((nr, nc), goal_pos)

    return {
        "agent_position": {"row": r, "col": c},
        "goal_position": {"row": goal_pos[0], "col": goal_pos[1]},
        "current_distance": current_distance,
        "adjacent_tiles": adjacent,
        "action_effects": action_effects,
        "visited": [{"row": vr, "col": vc} for vr, vc in sorted(visited)],
        "last_action": last_action,
    }


def extract_action(text: str) -> str | None:
    if not text:
        return None

    text = text.lower()

    match = re.search(r"\b(up|down|left|right)\b", text)
    if match:
        return match.group(1)

    return None


# -------------------------------------------------
# SLM Policy
# -------------------------------------------------
class SLMPolicy:
    def __init__(self, slm, prompt_template, size: int, decoding: Dict[str, Any]):
        self.slm = slm
        self.prompt_template = prompt_template
        self.size = size
        self.decoding = decoding
        self.visited = set()
        self.last_action = None
        self.total_steps = 0
        self.invalid_actions = 0

    def predict(self, obs: np.ndarray):
        self.total_steps += 1

        # Decode observation
        grid, agent_pos = decode_obs(obs, self.size)
        goal_pos = find_goal(grid)

        # ----- DEBUG VISUAL (somente para humano) -----
        # print("\nMAP (human-readable):")
        # print(render_grid_with_agent(grid, agent_pos))
        # print(f"Goal position: {goal_pos}")

        # Update visited
        self.visited.add(agent_pos)

        # Build symbolic state
        state = build_symbolic_state(
            grid=grid,
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            visited=self.visited,
            last_action=self.last_action,
        )

        # --- HARD CONSTRAINT FILTERING (deterministic) ---
        valid_actions = []

        for action, tile in state["adjacent_tiles"].items():
            if tile in ("H", "OUT_OF_BOUNDS"):
                continue

            next_dist = state["action_effects"][action]
            if next_dist is None:
                continue

            # forbid actions that increase distance
            if next_dist > state["current_distance"]:
                continue

            # forbid reversing last action if alternatives exist
            if self.last_action:
                reverse = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }
                if action == reverse[self.last_action]:
                    continue

            valid_actions.append(action)

        state["valid_actions"] = valid_actions
        observation = json.dumps(state, indent=2)

        # Render prompt
        prompt = self.prompt_template.render(observation=observation)

        # ---- LLM call ----
        with torch.no_grad():
            output = self.slm.generate(prompt, self.decoding)

        text = output.text if hasattr(output, "text") else str(output)
        text = text.replace("```", "").strip()

        # print("STATE TO LLM:")
        # print(observation)
        # print("RAW LLM OUTPUT:", repr(text))
        # print(30*"-")

        # Parse action
        action_str = extract_action(text)

        if action_str is None:
            self.invalid_actions += 1
            return None, None

        action_str = action_str.upper()

        # Enforce that execution matches exactly the LLM proposal
        if action_str not in valid_actions:
            self.invalid_actions += 1
            return None, None

        action = parse_action(action_str.lower())

        # Store last action
        self.last_action = action_str

        return action, None


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate_slm(
    policy: SLMPolicy,
    env: gym.Env,
    n_episodes: int,
    path: str,
):
    episode_rewards = []
    episode_lengths = []

    structures = glob.glob(os.path.join(path, "*.npy"))
    if len(structures) < n_episodes:
        env.create_structures(n_episodes, eval="eval" in path)
        structures = glob.glob(os.path.join(path, "*.npy"))

    for structure in tqdm(structures[:n_episodes], desc="Evaluating SLM"):
        obs, _ = env.load(np.load(structure).tolist())
        policy.visited = set()
        policy.last_action = None

        done = False
        reward_sum = 0
        steps = 0

        while not done:
            action, _ = policy.predict(obs)

            if action is None:
                break  # episódio termina por ação inválida

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward_sum += reward
            steps += 1

        episode_rewards.append(reward_sum)
        episode_lengths.append(steps)

    invalid_rate = (
        policy.invalid_actions / policy.total_steps
        if policy.total_steps > 0 else 0.0
    )

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "invalid_action_rate": invalid_rate,
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = load_config("configs/slm/small.yaml")
    decoding_cfg = load_config("configs/slm/decoding_action_only.yaml")

    set_seed(cfg.get("experiment", {}).get("seed"))
    
    SIZE = 5
    N_EPISODES = 10

    env = FrozenLake(
        id="FrozenLake-v1",
        size=SIZE,
        render_mode=None,
    )

    slm = load_slm(cfg["slm"])
    slm.model.eval()
    
    prompt = load_prompt({"file": "prompts/action_only.txt"})

    policy = SLMPolicy(
        slm=slm,
        prompt_template=prompt,
        size=SIZE,
        decoding=decoding_cfg,
    )

    results = evaluate_slm(
        policy=policy,
        env=env,
        n_episodes=N_EPISODES,
        path=f"./tmp/frozenlake{SIZE}/test",
    )

    # ---------- Save ----------
    output_dir = Path("runs/slm/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"frozenlake{SIZE}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "env": "FrozenLake-v1",
                "size": SIZE,
                "model": cfg["slm"].get("model", "unknown"),
                "episodes": N_EPISODES,
                "results": results,
            },
            f,
            indent=2,
        )

    print("\nRESULTS")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Invalid Action Rate: {results['invalid_action_rate']:.2%}")
    print(f"Saved to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
