# MazeSolver 🤖🧩

## 1. Project Summary

MazeSolver is a Reinforcement Learning (RL) project that trains an agent to navigate a custom-defined maze using the **PointMaze** environment from [Gymnasium-Robotics](https://robotics.farama.org/).

The agent (a point mass) must learn to move from a start position to a goal position inside the maze. The project lays the foundation for training an RL algorithm (e.g., SAC, TD3, DDPG) by:
- Setting up a custom maze environment
- Wrapping the observation space into a flat vector for RL compatibility
- Running a random-action loop to verify the environment is working end-to-end

```
Agent → takes action → gets reward + next state
         ↑                        ↓
    Actor (policy)         Replay Buffer
         ↑                        ↓
    Critic (Q-values) ← sample mini-batch
```

---

## 2. Environment Setup

### Tools Used

| Tool | Purpose |
|------|---------|
| [Miniconda](https://docs.conda.io/en/latest/miniconda.html) | Python environment manager |
| [conda](https://conda.io) | Creating isolated Python environments |
| [pip](https://pip.pypa.io) | Installing Python packages inside the conda env |

### Creating the Conda Environment

```bash
conda create -n MazeSolver python=3.11
conda activate MazeSolver
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install gymnasium==0.29.1
pip install gymnasium-robotics==1.2.4
pip install pybullet==3.2.6
pip install tensorboard==2.15.1
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Packages & Their Purpose

| Package | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | 0.29.1 | Core RL environment interface (OpenAI Gym successor) |
| `gymnasium-robotics` | 1.2.4 | Provides `PointMaze_UMaze-v3` and other robotics envs |
| `pybullet` | 3.2.6 | Physics simulation engine (for future robotics tasks) |
| `torch` / `torchvision` / `torchaudio` | latest (cu118) | PyTorch deep learning framework for training RL agents |
| `tensorboard` | 2.15.1 | Visualizing training metrics and reward curves |
| `numpy` | — | Array operations for observation processing |
| `mujoco` | 2.3.x | Physics backend used by Gymnasium-Robotics |

> **Note:** `gymnasium==0.29.1` must be used — newer versions (≥1.0) break `gymnasium-robotics==1.2.4` due to internal API changes (`gymnasium.wrappers.time_limit` was removed).

---

## 3. What Was Done

### Step 1 — Maze Environment Setup (`main.py`)
- Imported `gymnasium` and registered `gymnasium-robotics` environments using `gym.register_envs(gymnasium_robotics)` (required to make `PointMaze_UMaze-v3` discoverable)
- Defined a custom **straight maze** layout:
  ```python
  STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1]]
  ```
  Where `1` = wall, `0` = open path
- Created the environment with `render_mode="human"` for visual rendering and `max_episode_steps=100`

### Step 2 — Custom Observation Wrapper (`gym_robotics_custom.py`)
The raw observation from `PointMaze_UMaze-v3` is a **dictionary**:
```python
{
  "observation":    [x, y, vx, vy],   # agent position + velocity  (4,)
  "achieved_goal":  [x, y],           # current position           (2,)
  "desired_goal":   [x, y],           # target position            (2,)
}
```
A custom `ObservationWrapper` called `RoboGymObservationWrapper` was built to:
- Flatten and concatenate all three arrays into a single numpy vector of shape `(8,)`
- Override `reset()` and `step()` to automatically process every observation
- Make the observation compatible with standard RL algorithms that expect flat arrays

### Step 3 — Random Action Loop
Verified the full pipeline works by running 1000 random actions:
```python
for i in range(1000):
    action = env.action_space.sample()
    env.step(action)
```
Output confirmed: observation shape `(8,)` ✅

### Step 4 — Version Fix
Resolved a version mismatch between `gymnasium==1.3.0` (auto-installed) and `gymnasium-robotics==1.2.4` by pinning `gymnasium` back to `0.29.1`.

---

## Running the Project

```bash
conda activate MazeSolver
python main.py
```

Expected output:
```
(8,)
```
