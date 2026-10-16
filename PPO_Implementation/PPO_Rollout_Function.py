import numpy as np
import torch
from torch.distributions import Normal

# =================================================================================================================================================================================================================
# PPO Rollout Function
def rollout(env,Actor,Critic,device):
    obs_list = []
    raw_action_list = []
    rewards = []
    obs, info = env.reset()

    with torch.no_grad():
        for t in range(env.max_count):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean,std = Actor(obs_tensor)
            policy_distribution = Normal(mean,std)
            action = policy_distribution.sample()
            raw_action = torch.tanh(action).squeeze(0).detach().cpu().numpy()
            motor_ctrl = env.ctrl_low + (raw_action + 1)*(env.high - env.low)/2
            obs,reward,terminate,truncate,_ = env.step(motor_ctrl)
        
            obs_list.append(obs)
            rewards.append(reward)
            raw_action_list.append(raw_action)
            if terminate or truncate:
                break
    return obs_list,rewards,raw_action_list
