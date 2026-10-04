import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Normal

# Actor (Controller)
class Actor(nn.Module):
    def __init__(input_dim, hidden_layers, output_dim):
        super().__init__()
        