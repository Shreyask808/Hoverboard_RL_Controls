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
    terminate = False

    with torch.no_grad():
        for t in range(env.max_count):
            obs_list.append(obs)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            value_function = Critic(obs_tensor)
            critic_list.append(value_function.item())
            mean,std = Actor(obs_tensor)
            policy_distribution = Normal(mean,std)
            action = policy_distribution.sample()
            log_prob = policy_distribution.log_prob(action).sum(dim=1)
            log_prob_list.append(log_prob.squeeze(0).detach().cpu().numpy())
            raw_action = torch.tanh(action).squeeze(0).detach().cpu().numpy()
            motor_ctrl = env.ctrl_low + (raw_action + 1)*(env.ctrl_high - env.ctrl_low)/2
            obs,reward,terminate,truncate,_ = env.step(motor_ctrl)

            rewards.append(reward)
            raw_action_list.append(action.squeeze(0).cpu().numpy())
            if terminate or truncate:
                break
        if terminate:
            last_value = 0
        else:
            last_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = Critic(last_tensor).item()

    return obs_list,log_prob_list,raw_action_list,critic_list,rewards,last_value

def Advantage(critic_list,rewards,last_value,discount_factor):
    current_value = np.array(critic_list, dtype=np.float32)
    next_value = np.array(critic_list[1:] + [last_value], dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    advantage_list = []
    targets_list = []

    targets_list = rewards + discount_factor*next_value
    advantage_list = targets_list - current_value

    return advantage_list,targets_list