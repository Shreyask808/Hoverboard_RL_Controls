import mujoco
import mujoco.viewer
import gymnasium
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Normal

# =================================================================================================================================================================================================================
# Input Hoverboard xml Model
xml_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"

# =================================================================================================================================================================================================================
# Class Definition
## Gymnasium Class
class hoverboardEnv(gymnasium.Env):
    def __init__(self,xml_path):
        self.hbmodel = mujoco.MjModel.from_xml_path(xml_path)
        self.hbdata = mujoco.MjData(self.hbmodel)

        obs_dim = self.hbmodel.nq + self.hbmodel.nv
        ctrl_dim = self.hbmodel.nu
        self.max_T = np.sum(self.hbmodel.actuator_ctrlrange[:,1]**2)
        self.step_count = 0
        self.discount_factor = 0.99
        self.Low = self.hbmodel.actuator_ctrlrange[:,0]
        self.High = self.hbmodel.actuator_ctrlrange[:,1]
        self.max_count = int(120/self.hbmodel.opt.timestep)

        self.hinge_x_qpos_id = self.hbmodel.joint("pend_hinge_x").qposadr[0]
        self.hinge_y_qpos_id = self.hbmodel.joint("pend_hinge_y").qposadr[0]

        self.hinge_x_qvel_id = self.hbmodel.joint("pend_hinge_x").dofadr[0]
        self.hinge_y_qvel_id = self.hbmodel.joint("pend_hinge_y").dofadr[0]

        self.observation_space = gymnasium.spaces.Box(
            low= -np.inf, high= np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = gymnasium.spaces.Box(
            low= self.Low,
            high= self.High,
            shape= (ctrl_dim,), dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.hbmodel,self.hbdata)

        th_ini = self.np_random.uniform(-np.deg2rad(50), np.deg2rad(50))
        gamma_ini = self.np_random.uniform(-np.deg2rad(50), np.deg2rad(50))
        thdot_ini = 0
        gammadot_ini = 0

        self.hbdata.qpos[self.hinge_x_qpos_id] = gamma_ini
        self.hbdata.qpos[self.hinge_y_qpos_id] = th_ini
        self.hbdata.qvel[self.hinge_x_qvel_id] = gammadot_ini
        self.hbdata.qvel[self.hinge_y_qvel_id] = thdot_ini
        self.step_count = 0
        
        mujoco.mj_forward(self.hbmodel,self.hbdata)
        return self._get_obs(), {}

    def step(self,action):
        self.hbdata.ctrl[:] = action
        mujoco.mj_step(self.hbmodel,self.hbdata)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_done()
        self.step_count += 1
        truncated = self.step_count >= self.max_count
        return obs, reward, terminated, truncated,{}
    
    def _get_obs(self):
        return np.concatenate([self.hbdata.qpos, self.hbdata.qvel])

    def _compute_reward(self):
        th = self.hbdata.qpos[self.hinge_y_qpos_id]
        gamma = self.hbdata.qpos[self.hinge_x_qpos_id]
        thdot = self.hbdata.qvel[self.hinge_y_qvel_id]
        gammadot = self.hbdata.qvel[self.hinge_x_qvel_id]
        balance_reward = -(th**2 + gamma**2)
        rate_reward = -0.1*(thdot**2 + gammadot**2)
        action_reward = -0.001*np.sum(self.hbdata.ctrl**2)/self.max_T
        alive_bonus = 1

        reward = alive_bonus + action_reward + rate_reward + balance_reward
        return reward

    def _check_done(self):
        th = self.hbdata.qpos[self.hinge_y_qpos_id]
        gamma = self.hbdata.qpos[self.hinge_x_qpos_id]

        return bool(abs(th) >= np.deg2rad(60) or abs(gamma) >= np.deg2rad(60))


class PolicyNet(nn.Module):
    def __init__(self, in_dim, l1_dim, l2_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, l1_dim)
        self.fc2 = nn.Linear(l1_dim, l2_dim)
        self.fc3 = nn.Linear(l2_dim, out_dim)
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x)) # Continuous Gaussian Distribution 
        std = torch.exp(self.log_std)
        return mean, std

def rollout(env,policy_net,device):
    log_probability = []
    rewards = []
    obs, info = env.reset()

    for t in range(env.max_T):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mean, std = policy_net(obs_tensor)
        distribution = Normal(mean, std)   
        raw_action = distribution.sample()
        current_log_prob = distribution.log_prop(raw_action).sum(dim=1)
        log_probability.append(current_log_prob)

        action = env.Low + (raw_action + 1)*(env.High - env.Low)/2
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

        rewards.append(reward)

    return log_probability, rewards

def compute_trajectory_loss(log_probability,rewards,env):
    returns = []
    steps = len(rewards)
    for i in range(steps):
        reward_to_go = rewards[i]
        for j in range(steps):
            if j > i:
                reward_to_go = reward_to_go + env.discount_factor**(j-i)*rewards[j]

        returns.append[reward_to_go]

    traj_loss = np.dot(log_probability,returns)
    return traj_loss

# =================================================================================================================================================================================================================
# Mujoco Model and Policy net Definition
hoverboard = hoverboardEnv(xml_path)                                                                            # Hoverboard Model Definition 
input_dim = (hoverboard.hbmodel.nq + hoverboard.hbmodel.nv)                                                     # Inputs to the Policy Net (All qpos + qvel)
output_dim = hoverboard.hbmodel.nu                                                                            # Each Motor Torque is a continuous Gaussian Distribution with a mean and standard deviation as the outputs
nn_policy = PolicyNet(input_dim,64,64,output_dim)                                                               # Control Policy
model_parameters = sum(p.numel() for p in nn_policy.parameters())                                               # Number of Parameters in the Model

batchsize = 32                                                                                                  # Number of 
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("")
print(f"1. Input Dimensions to the NN Policy - {input_dim}")
print("")
print(f"2. Output Dimensions to the NN Policy - {output_dim}")
print("")
print(f"3. Number of Parameters in NN Policy - {model_parameters}")