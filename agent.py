import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model import *
from buffer import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import datetime
from icm import ICM


# =========================
# Utils
# =========================
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


# =========================
# Agent
# =========================
class Agent(object):

    def __init__(
        self,
        num_inputs,
        action_space,
        gamma,
        tau,
        alpha,
        target_update_interval,
        hidden_size,
        lr,
        exploration_scaling_factor
    ):

        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Running on {self.device}")

        action_dim = action_space.shape[0]

        # =========================
        # Critic
        # =========================
        self.critic = Critic(num_inputs, action_dim, hidden_size).to(self.device)
        self.critic_target = Critic(num_inputs, action_dim, hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        hard_update(self.critic_target, self.critic)

        # =========================
        # Actor (Policy)
        # =========================
        self.policy = Actor(num_inputs, action_dim, hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        # =========================
        # Intrinsic Curiosity Module (ICM)
        # =========================
        self.icm = ICM(num_inputs, action_dim).to(self.device)
        self.icm_optim = Adam(self.icm.parameters(), lr=lr)

    # =========================
    # Action Selection
    # =========================
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    # =========================
    # Update Step
    # =========================
    def update_parameters(self, memory: ReplayMemory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # =========================
        # ICM (Curiosity)
        # =========================
        pred_next, pred_action = self.icm(state_batch, action_batch, next_state_batch)

        forward_loss = F.mse_loss(pred_next, next_state_batch)
        inverse_loss = F.mse_loss(pred_action, action_batch)

        # update ICM
        icm_loss = forward_loss + inverse_loss


        self.icm_optim.zero_grad()
        icm_loss.backward()
        self.icm_optim.step()
        # =========================
        # Reward shaping
        # =========================
        reward_batch = reward_batch + 0.01 * intrinsic_reward

        # =========================
        # Target Q computation
        # =========================
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(next_state_batch)

            qf1_next, qf2_next = self.critic_target(next_state_batch, next_state_action)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_state_log_pi

            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next

        # =========================
        # Critic loss
        # =========================
        qf1, qf2 = self.critic(state_batch, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # =========================
        # Actor loss
        # =========================
        pi, log_pi = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # =========================
        # Soft update target network
        # =========================
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            icm_loss.item(),
            self.alpha
        )

    # =========================
    # Save / Load
    # =========================
    def save_model(self):
        if not os.path.exists("models"):
            os.makedirs("models")

        self.policy.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_checkpoint(self, evaluate=False):
        try:
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
        except:
            print("No saved model found.")

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()