import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# hyperparameters
hidden_size = 128
learning_rate = 0.001

# constants
GAMMA = 0.99
num_steps = 300
max_episodes = 10000

class ActorCritic(nn.Module):