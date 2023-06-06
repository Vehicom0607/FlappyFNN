import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.common = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_linear1 = nn.Linear(128, 128)  # Update critic_linear1 dimensions

        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1)  # Update critic dimensions
        )

    def forward(self, state):
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=-1)

        return value, policy_dist
