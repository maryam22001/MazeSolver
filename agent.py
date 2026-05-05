import os
import torch
import torch.nn as nn
from torch.optim import Adam
from model import *
def hard_update(target, source):
    """Copy network parameters from source to target."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    """Soft update model parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)        
class Agent(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy, target_update_interval, hidden_size, lr, exploration_scaling_factor):
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy
        self.target_update_interval = target_update_interval
        #self.exploration_scaling_factor = exploration_scaling_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")
        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = Critic(num_inputs, action_space.shape[0], hidden_size).to(self.device)

        hard_update(self.critic_target, self.critic)   

        self.policy = Actor(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device) 
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.actor = Actor(num_inputs, action_space.shape[0], hidden_size, action_space)
        
       # self.memory = ReplayMemory(capacity=1000000)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate is False:
            action = self.policy.sample(state)
        else:
            action = self.policy.sample(state)      
        return action.detach().cpu().numpy()[0]
    def update_parameters(self):
        pass
    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')  
        print("Saving models...")
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
    def load_checkpoint(self, evaluate=False):
        try:
            print("Loading models...")
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print("Models loaded successfully.")
        except:
            if evaluate:
                raise Exception("Unable to load saved models.")
            else:
                print("Unable to load saved models. Starting from scratch.  This is normal if you haven't trained the agent yet.")
        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()        
