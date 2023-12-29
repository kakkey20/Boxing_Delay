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


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._plot = None

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if self._plot is None: # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else: # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02, 
                                    self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True,True,True)
            self._plot[-1].canvas.draw()
        
# Create log dir
log_dir = "."
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = make_vec_env('MountainCarContinuous-v0', n_envs=1, monitor_dir=log_dir)

plotting_callback = PlottingCallback()
        
model = PPO('MlpPolicy', env, verbose=0)
model.learn(10000, callback=plotting_callback)
