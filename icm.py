import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()

        # Forward model: (s, a) → s'
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

        # Inverse model: (s, s') → a
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, state, action, next_state):

        # forward prediction
        pred_next_state = self.forward_model(
            torch.cat([state, action], dim=1)
        )

        # inverse prediction
        pred_action = self.inverse_model(
            torch.cat([state, next_state], dim=1)
        )

        return pred_next_state, pred_action