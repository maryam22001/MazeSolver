import gymnasium as gym
import numpy as np

from gym_robotics_custom import RoboGymObservationWrapper
from model import *
from agent import Agent
from buffer import ReplayMemory


# =========================
# MAZES (Curriculum Levels)
# =========================

STRAIGHT_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

MEDIUM_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

HARD_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1]
]

maze_levels = [STRAIGHT_MAZE, MEDIUM_MAZE, HARD_MAZE]


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    # Hyperparameters
    episodes = 1000
    batch_size = 64
    updates_per_step = 4

    gamma = 0.99
    tau = 0.005
    alpha = 0.12
    hidden_size = 512
    lr = 1e-4
    max_episode_steps = 100
    replay_buffer_size = 1_000_000
    exploration_scaling_factor = 1.5

    env_name = "PointMaze_UMaze-v3"
    level = 0

    # =========================
    # INIT ENV
    # =========================
    env = gym.make(
        env_name,
        maze_map=maze_levels[level],
        max_episode_steps=max_episode_steps,
        render_mode="human"
    )

    env = RoboGymObservationWrapper(env)

    obs, _ = env.reset()
    obs_size = obs.shape[0]

    # =========================
    # AGENT + MEMORY
    # =========================
    agent = Agent(
        obs_size,
        env.action_space,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        target_update_interval=1,
        hidden_size=hidden_size,
        lr=lr,
        exploration_scaling_factor=exploration_scaling_factor
    )

    memory = ReplayMemory(
        replay_buffer_size,
        input_size=obs_size,
        n_actions=env.action_space.shape[0]
    )

    total_steps = 0

    # =========================
    # TRAIN LOOP
    # =========================
    for episode in range(episodes):

        # -------------------------
        # CURRICULUM LEARNING
        # -------------------------
        if episode > 0 and episode % 100 == 0 and level < len(maze_levels) - 1:
            level += 1
            print(f"\n🔥 Switching to maze level {level}")

            env.close()

            env = gym.make(
                env_name,
                maze_map=maze_levels[level],
                max_episode_steps=max_episode_steps,
                render_mode="human"
            )

            env = RoboGymObservationWrapper(env)

        state, _ = env.reset()

        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_episode_steps:

            # =========================
            # ACTION
            # =========================
            action = agent.select_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # =========================
            # STORE TRANSITION
            # =========================
            memory.store_transition(
                state,
                action,
                reward,
                next_state,
                float(not done)
            )

            state = next_state
            episode_reward += reward

            step += 1
            total_steps += 1

            # =========================
            # TRAIN AGENT
            # =========================
            if memory.can_sample(batch_size):
                for _ in range(updates_per_step):
                    agent.update_parameters(memory, batch_size, total_steps)

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Level: {level}")

    env.close()