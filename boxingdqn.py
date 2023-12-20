import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("CartPole-v1")
# env = gym.make("Boxing-v4")
# env = gym.make("ALE/Boxing-v5")
env = gym.make("ALE/Boxing-v5", render_mode='human')
model = PPO(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=10_000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# mean_reward: 351.68 +/- 107.47
