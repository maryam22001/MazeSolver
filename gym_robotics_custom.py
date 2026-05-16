import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper

class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(RoboGymObservationWrapper, self).__init__(env)
        
        obs_shape = env.observation_space.spaces['observation'].shape[0]
        goal_shape = env.observation_space.spaces['desired_goal'].shape[0]
        achieved_shape = env.observation_space.spaces['achieved_goal'].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_shape + goal_shape + achieved_shape,), dtype=np.float32)

    def process_observation(self, observation):
        return np.concatenate([observation['observation'], observation['achieved_goal'], observation['desired_goal']])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.process_observation(obs), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        return self.process_observation(observation), reward, done, truncated, info
