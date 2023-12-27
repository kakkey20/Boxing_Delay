import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm

env = gym.make("ALE/Boxing-v5", render_mode='human')
# env = gym.make("ALE/Boxing-v5")
# model = PPO(MlpPolicy, env, verbose=1)
# model = DQN(MlpPolicy, env, verbose=1)

def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    count = 0
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = vec_env.step(action)

            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward


model = A2C(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=100)

import os
# saveディレクトリ
save_dir = "."
os.makedirs(save_dir, exist_ok=True)

# モデルの保存
model.save(f"{save_dir}/Boxing")

# # モデルを一旦デリート（モデルの保存を一旦保存し、再度ロードするために）
del model

# # モデルのロード
loaded_model = A2C.load(f"{save_dir}/Boxing")

loaded_model.set_env(env)

mean_reward = evaluate(loaded_model, num_episodes=1, deterministic=True)
print(mean_reward)

