import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm

# env = gym.make("ALE/Boxing-v5")
# model = PPO(MlpPolicy, env, verbose=1)
# model = DQN(MlpPolicy, env, verbose=1)

class DelayWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, delay_flame):
        super().__init__(env)
        self.actions = []
        # 遅延作成
        self.delay_flame = delay_flame
        for _ in range(self.delay_flame):
            # self.actions.append(np.array([0]))
            self.actions.append(np.int64(0))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        self.actions.insert(0, action)
        action = self.actions.pop(-1)
        # print("action=", action)
        # print("type(action)=", type(action))

        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
env = DelayWrapper(gym.make("ALE/Boxing-v5", render_mode='human'), delay_flame=10)


model = A2C(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)
env = gym.make("ALE/Boxing-v5", render_mode='human')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

