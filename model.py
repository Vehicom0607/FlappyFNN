import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x

    def save(self, gen):
        torch.save(self.state_dict(), "./models/gen" + str(gen) + ".pt")
