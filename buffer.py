import numpy as np


class ReplayMemory():
    def __init__(self, max_size, input_size, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0

        self.state_memory = np.zeros((self.mem_size, input_size))
        self.next_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    # =========================
    # FIXED
    # =========================
    def can_sample(self, batch_size):
        return self.mem_ctr >= batch_size

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward

        # FIX: store as float
        self.terminal_memory[index] = float(done)

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_ctr, self.mem_size)

        if max_mem < batch_size:
            batch_size = max_mem

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.mem_ctr