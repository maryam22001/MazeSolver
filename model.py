import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module): 
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        return self.linear3(x1), self.linear6(x2)

    def save_checkpoint(self, path='models'):   # ✅ added
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, 'Critic.pth')
        torch.save(self.state_dict(), filepath)
        print(f"  Saved {filepath}")

    def load_checkpoint(self, path='models'):   # ✅ added
        filepath = os.path.join(path, 'Critic.pth')
        self.load_state_dict(torch.load(filepath, map_location='cpu'))
        print(f"  Loaded {filepath}")


class Actor(nn.Module): 
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.mean(x), torch.clamp(self.log_std(x), min=-20, max=2)

    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        return action

    def save_checkpoint(self, path='models'):   # ✅ added
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, 'Actor.pth')
        torch.save(self.state_dict(), filepath)
        print(f"  Saved {filepath}")

    def load_checkpoint(self, path='models'):   # ✅ added
        filepath = os.path.join(path, 'Actor.pth')
        self.load_state_dict(torch.load(filepath, map_location='cpu'))
        print(f"  Loaded {filepath}")