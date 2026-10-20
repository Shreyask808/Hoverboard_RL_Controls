import numpy as np
import torch
from torch.distributions import Normal

# =================================================================================================================================================================================================================
# PPO Rollout Function
def Rollout(env,Actor,Critic,device):
    obs_list = []
    raw_action_list = []
    rewards = []
    critic_list = []
    log_prob_list = []
    obs, info = env.reset()
    terminate = False

    with torch.no_grad():
        for t in range(int(env.max_count)):
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

def Batch_Rollout(env,Actor,Critic,batchsize,discount_factor,device):
    batch_obs = []
    batch_log_probs = []
    batch_action = []
    batch_advantage = []
    batch_targets = []
    episode_length = []

    for i in range(batchsize):
        obs_list,log_prob_list,raw_action_list,critic_list,rewards,last_value = Rollout(env,Actor,Critic,device)
        episode_length.append(len(rewards))
        advantage_list,targets_list = Advantage(critic_list,rewards,last_value,discount_factor)
        batch_obs.extend(obs_list)
        batch_advantage.append(advantage_list)
        batch_targets.append(targets_list)
        batch_log_probs.extend(log_prob_list)
        batch_action.extend(raw_action_list)

    batch_obs = np.array(batch_obs, dtype=np.float32)
    batch_log_probs = np.array(batch_log_probs, dtype=np.float32)
    batch_action = np.array(batch_action, dtype=np.float32)
    batch_advantage = np.concatenate(batch_advantage)
    batch_targets = np.concatenate(batch_targets)
    episode_length = np.array(episode_length)
    return batch_obs,batch_advantage,batch_targets,batch_log_probs,batch_action,episode_length