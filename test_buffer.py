
from buffer import ReplayMemory

buffer_size = 10
loop_size = 20
memory = ReplayMemory(max_size=buffer_size, input_shape=[1], n_actions=1)

for i in range(loop_size):
    memory.store_transition(state=[i], action=[i], reward=i, new_state=[i+1], done=False)

# Test circular overwrite
print("Testing circular buffer overwrite...")
assert memory.state_memory[0][0] == 10, f"Got {memory.state_memory[0][0]}"  # ✅ fixed
print("✅ Circular buffer test passed")

# Test can_sample
assert memory.can_sample(5) == True
print("✅ can_sample test passed")

# Test sample_buffer
states, actions, rewards, new_states, terminals = memory.sample_buffer(5)
assert states.shape == (5, 1)
print("✅ sample_buffer shape test passed")

print("\nAll buffer tests passed!")