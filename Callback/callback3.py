import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
# モデルのインポート
from stable_baselines3 import A2C, SAC, PPO, TD3
# ポリシーのインポート
from stable_baselines3.ppo.policies import MlpPolicy
# 評価関数のインポート
from stable_baselines3.common.evaluation import evaluate_policy
# Videoのインポート
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
# Basecallbackのインポート
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

from tqdm.auto import tqdm


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


model = TD3("MlpPolicy", "Pendulum-v1", verbose=0)
# Using a context manager garanties that the tqdm progress bar closes correctly
with ProgressBarManager(2000) as callback:
    model.learn(2000, callback=callback)
