from typing import Union, List
import os

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np


class FrozenLake(gym.Env):

    char_to_int = {b'S': 0, b'F': 1, b'H': 2, b'G': 3}

    def __init__(
        self,
        id: str = "FrozenLake-v1",
        size: int = 4,
        render_mode: Union[None, str] = None
    ):
        super().__init__()
        self.env = None
        self.id = id.split("-")[0].lower()
        self.size = size
        self.render_mode = render_mode
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=3,
            shape=obs.shape,
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def create_obs(self, obs: int):
        map_encoded = np.vectorize(self.char_to_int.get)(self.get_desc())
        agent_onehot = np.zeros(map_encoded.size)
        agent_onehot[obs] = 1

        return np.concatenate([map_encoded.flatten(), agent_onehot]).astype(np.float32)

    def create_env(self):
        return gym.make(
            "FrozenLake-v1",
            success_rate=1. / 1.,
            desc=generate_random_map(size=self.size),
            render_mode=self.render_mode
        )

    def load(self, desc: List[str]):
        self.close()
        self.env = gym.make(
            "FrozenLake-v1",
            success_rate=1. / 1.,
            desc=desc,
            render_mode=self.render_mode
        )
        obs, info = self.env.reset()
        return self.create_obs(obs), info

    def get_desc(self):
        return self.env.env.env.env.desc

    def reset(self, seed=None):
        self.close()
        self.env = self.create_env()
        obs, info = self.env.reset()
        return self.create_obs(obs), info

    def step(self, action):
        action = int(action)
        obs, reward, terminated, truncated, info = self.env.step(action)

        return (
            self.create_obs(obs),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if hasattr(self.env, "render") and self.render_mode is not None:
            self.env.render()

    def close(self):
        if self.env:
            if hasattr(self.env, "close"):
                self.env.close()

    def create_structures(self, number: int, eval: bool = True):
        inner_folder = "eval" if eval else "test"
        os.makedirs(f"./tmp/{self.id}{self.size}/{inner_folder}", exist_ok=True)
        for idx in range(number):
            structure = np.array(generate_random_map(size=self.size))
            np.save(f"./tmp/{self.id}{self.size}/{inner_folder}/{idx}.npy", structure)
