import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Normal

# =================================================================================================================================================================================================================
# Actor (Controller)
class Policy_Net(nn.Module):
    def __init__(self,input_dim,h1_dim,h2_dim,output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self,x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.log_std)
        return mean,std

# =================================================================================================================================================================================================================
# Critic (State Value Function)
class Value_Function(nn.Module):
    def __init__(self, input_dim,l1_dim,l2_dim,l3_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, l1_dim)
        self.fc2 = nn.Linear(l1_dim, l2_dim)
        self.fc3 = nn.Linear(l2_dim,l3_dim)
        self.fc4 = nn.Linear(l3_dim,1)

    def forward(self,x):
        x = f.tanh(self.fc1(x))
        x = f.tanh(self.fc2(x))
        x = f.tanh(self.fc3(x))
        value_function = self.fc4(x)
        return value_function.squeeze(-1)