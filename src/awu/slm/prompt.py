from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np


# -------------------------------------------------
# Prompt template
# -------------------------------------------------

@dataclass
class PromptTemplate:
    template: str
    mode: str
    maze_shape: Tuple[int, int] | None = None

    def render(self, observation: Any) -> str:
        obs_str = self._format_observation(observation)
        return self.template.replace("{observation}", obs_str)

    # -------------------------------------------------
    # Observation formatting
    # -------------------------------------------------

    def _format_observation(self, observation: Any) -> str:
        """
        Convert raw observation vector into an ASCII maze representation.
        """
        if self.maze_shape is None:
            return str(observation)

        obs = np.asarray(observation).astype(int)

        h, w = self.maze_shape
        grid_size = h * w

        if len(obs) < grid_size:
            return str(observation)

        # Assume grid is the FIRST part of the observation
        grid_flat = obs[:grid_size]
        grid = grid_flat.reshape(h, w)

        # Heuristics based on Labyrinth conventions
        unique_vals = set(grid_flat.tolist())

        agent_val = max(unique_vals)
        wall_val = 1 if 1 in unique_vals else None
        free_val = 0 if 0 in unique_vals else None

        goal_candidates = [
            v for v in unique_vals
            if v not in {agent_val, wall_val, free_val}
        ]
        goal_val = goal_candidates[0] if goal_candidates else None

        lines = []
        for row in grid:
            chars = []
            for cell in row:
                if cell == agent_val:
                    chars.append("A")
                elif goal_val is not None and cell == goal_val:
                    chars.append("G")
                elif wall_val is not None and cell == wall_val:
                    chars.append("#")
                else:
                    chars.append(".")
            lines.append("".join(chars))

        return "\n".join(lines)


# -------------------------------------------------
# Loader
# -------------------------------------------------

def load_prompt(cfg: Dict[str, Any]) -> PromptTemplate:
    prompt_path = Path(cfg["file"])

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    if "{observation}" not in template:
        raise ValueError(
            "Prompt template must contain the '{observation}' placeholder."
        )

    maze_shape = None
    if "maze_shape" in cfg:
        maze_shape = tuple(cfg["maze_shape"])

    return PromptTemplate(
        template=template,
        mode=cfg.get("mode", "default"),
        maze_shape=maze_shape,
    )
