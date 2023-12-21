import gymnasium as gym
from stable_baselines3 import A2C, SAC, PPO, TD3

import os
from stable_baselines3.common.vec_env import DummyVecEnv

# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = PPO("MlpPolicy", "Pendulum-v1", verbose=0).learn(8_000)
# The model will be saved under PPO_tutorial.zip
model.save(f"{save_dir}/PPO_tutorial")

# sample an observation from the environment
obs = model.env.observation_space.sample()

# Check prediction before saving
print("pre saved", model.predict(obs, deterministic=True))

del model  # delete trained model to demonstrate loading

loaded_model = PPO.load(f"{save_dir}/PPO_tutorial")
# Check that the prediction is the same after loading (for the same observation)
print("loaded", loaded_model.predict(obs, deterministic=True))


# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = A2C("MlpPolicy", "Pendulum-v1", verbose=0, gamma=0.9, n_steps=20).learn(8000)
# The model will be saved under A2C_tutorial.zip
model.save(f"{save_dir}/A2C_tutorial")

del model  # delete trained model to demonstrate loading

# load the model, and when loading set verbose to 1
loaded_model = A2C.load(f"{save_dir}/A2C_tutorial", verbose=1)

# show the save hyperparameters
print(f"loaded: gamma={loaded_model.gamma}, n_steps={loaded_model.n_steps}")

# as the environment is not serializable, we need to set a new instance of the environment
loaded_model.set_env(DummyVecEnv([lambda: gym.make("Pendulum-v1")]))
# and continue training
loaded_model.learn(8_000)

class CustomWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    

class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

from gymnasium.envs.classic_control.pendulum import PendulumEnv

# Here we create the environment directly because gym.make() already wrap the environment in a TimeLimit wrapper otherwise
env = PendulumEnv()
# Wrap the environment
env = TimeLimitWrapper(env, max_steps=100)

obs, _ = env.reset()
done = False
n_steps = 0
while not done:
    # Take random actions
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(random_action)
    done = terminated or truncated
    n_steps += 1

print(n_steps, info)

import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        return self.env.reset(**kwargs)
    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float,bool, bool, dict) observation, reward, final state? truncated?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info
    
original_env = gym.make("Pendulum-v1")

print(original_env.action_space.low)
for _ in range(10):
    print(original_env.action_space.sample())

env = NormalizeActionWrapper(gym.make("Pendulum-v1"))

print(env.action_space.low)

for _ in range(10):
    print(env.action_space.sample())

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

env = Monitor(gym.make("Pendulum-v1"))
env = DummyVecEnv([lambda: env])

model = A2C("MlpPolicy", env, verbose=1).learn(int(1000))

normalized_env = Monitor(gym.make("Pendulum-v1"))
# Note that we can use multiple wrappers
normalized_env = NormalizeActionWrapper(normalized_env)
normalized_env = DummyVecEnv([lambda: normalized_env])

model_2 = A2C("MlpPolicy", normalized_env, verbose=1).learn(int(1000))

from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
normalized_vec_env = VecNormalize(env)


obs = normalized_vec_env.reset()
for _ in range(10):
    action = [normalized_vec_env.action_space.sample()]
    obs, reward, _, _ = normalized_vec_env.step(action)
    print(obs, reward)

class MyMonitorWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        # === YOUR CODE HERE ===#
        # Initialize the variables that will be used
        # to store the episode length and episode reward

        # ====================== #

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs = self.env.reset(**kwargs)
        # === YOUR CODE HERE ===#
        # Reset the variables

        # ====================== #
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict)
            observation, reward, is the episode over?, is the episode truncated?, additional information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        # === YOUR CODE HERE ===#
        # Update the current episode reward and episode length

        # ====================== #

        if terminated or truncated:
            # === YOUR CODE HERE ===#
            # Store the episode length and episode reward in the info dict
            pass

            # ====================== #
        return obs, reward, terminated, truncated, info
    
env = gym.make("LunarLander-v2")

from gym.wrappers import TimeLimit


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.0]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super().__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        self._current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionally: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))
    

save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = PPO("MlpPolicy", "Pendulum-v1", verbose=0).learn(8000)
model.save(save_dir + "/PPO_tutorial")

import zipfile

archive = zipfile.ZipFile("/tmp/gym/PPO_tutorial.zip", "r")
for f in archive.filelist:
    print(f.filename)
