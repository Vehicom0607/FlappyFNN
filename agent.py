import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Network
from collections import deque, namedtuple

BUFFER_SIZE = int(1e5) # Replay buffer size
BATCH_SIZE = 64
GAMMA = 0.99 # Discount factor
TAU = 1e-3 # soft update of target params
LR = 0.01 # learning rate
UPDATE_EVERY = 4 # How often to update the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size):
        # State size is how many dimension of each state
        # Action size is dimentions of each action(1 for flappy bird)

        self.state_size = state_size
        self.action_size = action_size

        self.model_local = Network(state_size, action_size).to(device)
        self.model_target = Network(state_size, action_size).to(device)

        self.optim = optim.Adam(self.model_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0

    def step(self, state, action, reward, next_step, done):
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY
        self.t_step = (self.t_step+1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def act(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model_local.eval()
        with torch.no_grad():
            action_values = self.model_local(state)
        self.model_local.train()

        # Epsilon - greedy/random actions
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        criterion = torch.nn.MSELoss()
        self.model_local.train()
        self.model_target.eval()

        predicted_targets = self.model_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.model_target(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (gamma * labels_next * (1-dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)