from dataclasses import dataclass
import torch
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Actor_Critic_Neural_Networks.Actor_Critic_NNs import Policy_Net, Value_Function
from Hoverboard_Environment.Hbd_Env import Hoverboard
from PPO_Implementation.PPO_Rollout_Function import Rollout, Advantage, Batch_Rollout

# =================================================================================================================================================================================================================
# Parallel Workers definition
Hbd_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_2.xml"
N_workers = 3
input_dim = 8

@dataclass
class Batch:
    obs: np.ndarray
    advantage: np.ndarray
    targets: np.ndarray
    log_probs: np.ndarray
    action: np.ndarray
    episode_length: np.ndarray

class Worker:
    def __init__(self,worker_id,env,Policy_Net,Value_Function):
        torch.set_num_threads(1)
        self.env = env(Hbd_path,input_dim)
        output_dim = 2
        self.Actor = Policy_Net(input_dim,64,64,output_dim)
        self.Critic = Value_Function(input_dim,64,64,64)

workers = [Worker(i,Hoverboard,Policy_Net,Value_Function) for i in range(N_workers)]
    
# =================================================================================================================================================================================================================
# Synchronous PPO Architecture Definition