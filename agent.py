import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import ActorCritic

# Hyperparameters
hidden_size = 128

# Constants
GAMMA = 0.99


class Agent:
    def __init__(self, num_inputs, num_actions, learning_rate=0.0001):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        self.actor_critic = ActorCritic(num_inputs, num_actions, hidden_size)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.values = []
        self.log_probs = []
        self.rewards = []
        self.masks = []

    def act(self, state, eps):
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            _, policy_dist = self.actor_critic(state)
            action = torch.argmax(policy_dist).item()
        else:
            action = random.choice(range(self.num_actions))
        return action

    def update(self):
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        rewards = self.rewards
        masks = self.masks

        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)

        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + GAMMA * G * masks[t]
            returns[t] = G

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        self.ac_optimizer.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        self.ac_optimizer.step()

    def step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)  # Add unsqueeze(0) to create a batch dimension
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)  # Add unsqueeze(0) to create a batch dimension

        value, policy_dist = self.actor_critic(state)
        next_value, _ = self.actor_critic(next_state)

        self.values.append(value)
        self.log_probs.append(torch.log(policy_dist.squeeze(0)[action]))
        self.rewards.append(reward)
        self.masks.append(1 - done)

        if done:
            self.update()
            self.values = []
            self.log_probs = []
            self.rewards = []
            self.masks = []
