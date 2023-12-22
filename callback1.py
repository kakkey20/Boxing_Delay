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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)

        return True
    
# Create log dir
log_dir = "."
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = make_vec_env("CartPole-v1", n_envs=1, monitor_dir=log_dir)
# it is equivalent to:
# env = gym.make('CartPole-v1')
# env = Monitor(env, log_dir)
# env = DummyVecEnv([lambda: env])

# Create Callback
callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=1)

model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=5000, callback=callback)

