import numpy as np
import torch
from torch.distributions import Normal

# =================================================================================================================================================================================================================
# PPO Rollout Function
discount_factor = 0.999
def rollout(env,Actor,Critic,device):
    obs_list = []
    raw_action_list = []
    rewards = []
    critic_list = []
    log_prob_list = []
    obs, info = env.reset()

    with torch.no_grad():
        for t in range(env.max_count):
            obs_list.append(obs)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            value_function = Critic(obs_tensor)
            critic_list.append(value_function.squeeze(0).detach().cpu().numpy())
            mean,std = Actor(obs_tensor)
            policy_distribution = Normal(mean,std)
            action = policy_distribution.sample()
            log_prob = policy_distribution.log_prob(action).sum(dim=1)
            log_prob_list.append(log_prob.squeeze(0).detach().cpu().numpy())
            raw_action = torch.tanh(action).squeeze(0).detach().cpu().numpy()
            motor_ctrl = env.ctrl_low + (raw_action + 1)*(env.ctrl_high - env.ctrl_low)/2
            obs,reward,terminate,truncate,_ = env.step(motor_ctrl)

            rewards.append(reward)
            raw_action_list.append(action)
            if terminate or truncate:
                break
    return obs_list,rewards,raw_action_list,len(obs_list)

def Advantage(critic_list,rewards,episode_length):
    advantage_list = []
    with torch.no_grad():
        for t in range(episode_length):
            advantage = rewards(t) + discount_factor*critic_list[t+1] - critic_list[j]
            advantage_list.append(advantage)
    return advantage_list