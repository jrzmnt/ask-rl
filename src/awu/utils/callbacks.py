import glob
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv, sync_envs_normalization
)


class EvalCallbackWithEvalMode(EvalCallback):
    def _on_step(self) -> bool:
        # Put policy in eval mode for evaluation
        self.model.policy.set_training_mode(False)

        # Run the original evaluation
        result = super()._on_step()

        # Put policy back in train mode
        self.model.policy.set_training_mode(True)

        return result


class EvalCallbackFrozenlake(EvalCallback):
    def evaluate_policy(
        self, model: PPO, env: DummyVecEnv, n_eval_episodes: int,
        render: bool, deterministic: bool, return_episode_rewards: bool,
        warn: bool, callback: EventCallback,
    ):
        episode_rewards = []
        episode_lengths = []

        env = env.envs[0].env
        size = env.size
        structures = glob.glob(os.path.join(f"./tmp/frozenlake{size}/eval", "*.npy"))
        if len(structures) < n_eval_episodes:
            env.create_structures(n_eval_episodes)
            structures = glob.glob(os.path.join(f"./tmp/frozenlake{size}/eval", "*.npy"))

        for structure in structures:
            obs, info = env.load(np.load(structure).tolist())
            done = False
            current_rewards = 0
            current_lengths = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                current_rewards += reward
                current_lengths += 1

                if render:
                    env.render()

            episode_rewards.append(current_rewards)
            episode_lengths.append(current_lengths)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if return_episode_rewards:
            return episode_rewards, episode_lengths
        return mean_reward, std_reward

    def _on_step(self) -> bool:
        self.model.policy.set_training_mode(False)

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            episode_rewards, episode_lengths = self.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward >= self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        self.model.policy.set_training_mode(True)

        return True
