import numpy as np
import torch

# =================================================================================================================================================================================================================
# PPO Rollout Function

def rollout(env,Policy_Net,Value_Function,Devive):
    obs_list = []
    raw_action_list = []
    rewards = []
    obs, info = env.reset()

    with torch.no_grad():
        for t in range(env.max_count):
            