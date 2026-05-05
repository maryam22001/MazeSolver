
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from buffer import ReplayMemory

# --- Setup environment ---
env = gym.make('PointMaze_UMaze-v3')
env = RoboGymObservationWrapper(env)

num_inputs = env.observation_space.shape[0]
action_space = env.action_space

print(f"Observation space: {num_inputs}")
print(f"Action space: {action_space.shape[0]}")

# --- Create agent ---
agent = Agent(
    num_inputs=num_inputs,
    action_space=action_space,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    policy='Gaussian',
    target_update_interval=1,
    hidden_size=256,
    lr=3e-4,
    exploration_scaling_factor=1.0
)

# --- Create buffer ---
memory = ReplayMemory(
    max_size=10000,
    input_shape=(num_inputs,),
    n_actions=action_space.shape[0]
)

# --- Run a few steps ---
obs, info = env.reset()
total_steps = 0
episodes = 0

print("\nRunning 3 test episodes...")
for episode in range(3):
    obs, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(obs)               # agent picks action
        next_obs, reward, done, truncated, info = env.step(action)
        memory.store_transition(obs, action, reward, next_obs, done or truncated)

        obs = next_obs
        episode_reward += reward
        total_steps += 1
        done = done or truncated

    episodes += 1
    print(f"  Episode {episode+1} | Steps: {total_steps} | Reward: {episode_reward:.2f} | Buffer size: {memory.mem_cntr}")

# --- Test save/load ---
print("\nTesting save...")
agent.save_model()

print("\nTesting load...")
agent.load_checkpoint(evaluate=False)

print("\n✅ All agent tests passed!")
env.close()