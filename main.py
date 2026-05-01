import gymnasium as gym
import gymnasium_robotics
import torch
from gym_robotics_custom import RoboGymObservationWrapper
from model import Actor, Critic

env = gym.make('PointMaze_UMaze-v3', render_mode="human")
env = RoboGymObservationWrapper(env)

actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], 256, env.action_space)

obs, info = env.reset()
for _ in range(1000):
    state = torch.FloatTensor(obs).unsqueeze(0)
    action = actor.sample(state).detach().cpu().numpy()[0]
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
env.close()
