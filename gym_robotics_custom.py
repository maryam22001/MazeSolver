import gymnasium_robotics
import numpy as np
from gymnasium import ObservationWrapper


class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(RoboGymObservationWrapper, self).__init__(env)
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.process_observation(observation)
        return observation, info

    def step(self,action):
        observation,reward,done,turncated,info=self.env.step(action)
        observation = self.process_observation(observation)
        return observation,reward,done,turncated,info
    def process_observation(self, observation):
        # observation is a dict with keys like 'observation', 'achieved_goal', 'desired_goal'
        # Return the flat observation array (position + velocity of the point agent)
        obs_map = observation['observation']
        obs_achieveed_goal=observation['achieved_goal']
        obs_desired_goal=observation['desired_goal']
        obs_concatenated=np.concatenate((obs_map,obs_achieveed_goal,obs_desired_goal))
        return obs_concatenated