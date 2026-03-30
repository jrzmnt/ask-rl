import glob

import gymnasium as gym
import numpy as np
from labyrinth.labyrinth import Labyrinth
from labyrinth.file_utils import convert_from_file


class LabyrinthEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for Nathan Gavenski's Labyrinth environment.

    Design decisions:
    - Vector observations only (no images, no rendering)
    - Single consistent configuration
    - Episode length capped at the Gym wrapper level
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        shape: tuple[int] = (5, 5),
        occlusion: bool = True,
        key_and_door: bool = False,
        icy_floor: bool = False,
        labyrinth_dir: str = None
    ) -> None:
        super().__init__()

        # Underlying Labyrinth environment (do NOT pass max_steps here)
        self.env = Labyrinth(
            shape=shape,
            occlusion=occlusion,
            key_and_door=key_and_door,
            icy_floor=icy_floor,
        )

        # Inspect initial observation to define spaces
        obs, _ = self.env.reset()

        # Enforce vector observations
        assert len(obs.shape) == 1, (
            f"Expected vector observation, got shape {obs.shape}. "
            "Ensure Labyrinth is not in image/render mode."
        )

        self.labyrinth_paths = None
        if labyrinth_dir is not None:
            self.labyrinth_paths = sorted(glob.glob(f"{labyrinth_dir}/*.labyrinth"))
            self.current_idx = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        self.action_space = self.env.action_space

    def _load_labyrinth(self) -> None:
        path = self.labyrinth_paths[self.current_idx]
        self.current_idx += 1
        if self.current_idx >= len(self.labyrinth_paths):
            self.current_idx = 0

        structure, variables = convert_from_file(path)
        self.env.load(structure, variables)

    def reset(self, seed=None, options=None):
        if self.labyrinth_paths is not None:
            self._load_labyrinth()
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        # Ensure discrete scalar action
        action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # self.current_step += 1
        # if self.current_step >= self.max_steps:
        #     truncated = True

        return (
            obs.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
