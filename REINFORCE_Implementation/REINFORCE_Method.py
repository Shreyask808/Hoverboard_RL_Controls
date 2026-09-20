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
from torch.distributions import Normal

# =================================================================================================================================================================================================================
# Input Hoverboard xml Model
xml_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_2.xml"

# =================================================================================================================================================================================================================
# Class Definition
## Gymnasium Class
class hoverboardEnv(gymnasium.Env):
    def __init__(self,xml_path):
        self.hbmodel = mujoco.MjModel.from_xml_path(xml_path)
        self.hbdata = mujoco.MjData(self.hbmodel)

        self.h = self.hbmodel.body("pendulum_mass").pos[2]
        self.g = self.hbmodel.opt.gravity[2]
        self.angle_scale = 1
        self.angular_velocity_scale = np.sqrt(self.h/self.g)

        obs_dim = self.hbmodel.nq + self.hbmodel.nv
        ctrl_dim = self.hbmodel.nu

        self.max_T = np.sum(self.hbmodel.actuator_ctrlrange[:,1]**2)
        self.step_count = 0

        self.discount_factor = 0.99

        self.Low = self.hbmodel.actuator_ctrlrange[:,0]
        self.High = self.hbmodel.actuator_ctrlrange[:,1]

        self.max_count = int(60/self.hbmodel.opt.timestep)

        hinge_x_qpos_id = self.hbmodel.joint("pend_hinge_x").qposadr[0]
        hinge_y_qpos_id = self.hbmodel.joint("pend_hinge_y").qposadr[0]
        hinge_x_qvel_id = self.hbmodel.joint("pend_hinge_x").dofadr[0]
        hinge_y_qvel_id = self.hbmodel.joint("pend_hinge_y").dofadr[0]

        chassis_hinge_z_id = self.hbmodel.joint("chassis_hinge_z").qposadr[0]
        chassis_hinge_x_id = self.hbmodel.joint("chassis_hinge_x").qposadr[0]
        chassis_hinge_z_qvel_id = self.hbmodel.joint("chassis_hinge_z").dofadr[0]
        chassis_hinge_x_qvel_id = self.hbmodel.joint("chassis_hinge_x").dofadr[0]

        self.th = self.hbdata.qpos[hinge_y_qpos_id]
        self.gamma = self.hbdata.qpos[hinge_x_qpos_id]
        self.psi = self.hbdata.qpos[chassis_hinge_z_id]
        self.xi = self.hbdata.qpos[chassis_hinge_x_id]

        self.thdot = self.hbdata.qvel[hinge_y_qvel_id]
        self.gammadot = self.hbdata.qvel[hinge_x_qvel_id]
        self.psidot = self.hbdata[chassis_hinge_z_qvel_id]
        self.xidot = self.hbdata[chassis_hinge_x_qvel_id]

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

        th_ini = self.np_random.uniform(-np.deg2rad(10), np.deg2rad(10))
        gamma_ini = self.np_random.uniform(-np.deg2rad(10), np.deg2rad(10))
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
        balance_reward = -(self.th**2 + self.gamma**2)
        rate_reward = -0.1*(self.thdot**2 + self.gammadot**2)
        action_reward = -0.001*np.sum(self.hbdata.ctrl**2)/self.max_T
        alive_bonus = 1

        reward = alive_bonus + action_reward + rate_reward + balance_reward
        return reward

    def _check_done(self):
        return bool(abs(self.th) >= np.deg2rad(45) or abs(self.gamma) >= np.deg2rad(45) or abs(self.xi) >= np.deg2rad(2))

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
        mean = self.fc3(x) # Continuous Gaussian Distribution 
        std = torch.exp(self.log_std)
        return mean, std

def rollout(env,policy_net,device):
    log_probability = []
    rewards = []
    obs, info = env.reset()

    for t in range(env.max_count):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mean, std = policy_net(obs_tensor)
        distribution = Normal(mean, std)   
        raw_action = distribution.sample()
        current_log_prob = distribution.log_prob(raw_action).sum(dim=1).squeeze(0)
        log_probability.append(current_log_prob)
        u = torch.tanh(raw_action)
        raw_action_np = u.squeeze(0).detach().cpu().numpy()

        action = env.Low + (raw_action_np + 1)*(env.High - env.Low)/2
        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)        

        if terminated or truncated:
            break

    return log_probability,rewards, obs

def compute_reward_to_go(env,log_probability,rewards):
    returns = []
    steps = len(rewards)
    G = 0
    for i in reversed(range(steps)):
        G = rewards[i] + env.discount_factor*G
        returns.append(G)

    returns.reverse()
    traj_return = returns[0]
    return returns, traj_return

# =================================================================================================================================================================================================================
# Mujoco Model and Policy net Definition
hoverboard = hoverboardEnv(xml_path)                                                                            # Hoverboard Model Definition 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                           # Device to compute gradients
input_dim = (hoverboard.hbmodel.nq + hoverboard.hbmodel.nv)                                                     # Inputs to the Policy Net (All qpos + qvel)
output_dim = hoverboard.hbmodel.nu                                                                              # Each Motor Torque is a continuous Gaussian Distribution with a mean and standard deviation as the outputs
nn_policy = PolicyNet(input_dim,64,64,output_dim).to(device)                                                              # Control Policy
model_parameters = sum(p.numel() for p in nn_policy.parameters())                                               # Number of Parameters in the Model

angle_scale = 1                                                                                                 # Angle Scale [rad^-1]
g = hoverboard.hbmodel.opt.gravity[2]
anglular_speed_scale = np.sqrt(hoverboard.h/np.abs(g))                                                          # Angular Velocity scale [sec/rad]

batchsize = 32                                                                                                  # Number of Rollouts per gradient step
max_batches = 1000                                                                                              # Maximum number of batches in the Training
log_probability_list = []
reward_to_go_list = []
avg_reward_to_go_list = []
optimizer = optim.Adam(nn_policy.parameters(), lr = 1e-3)

print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("")
print(f"1. Input Dimensions to the NN Policy - {input_dim}")
print("")
print(f"2. Output Dimensions to the NN Policy - {output_dim}")
print("")
print(f"3. Number of Parameters in NN Policy - {model_parameters}")
print("")
print(f"4. Number of Rollouts per Gradient Step - {batchsize}")
print("")
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

for batch in range(max_batches):
    log_probability_list.clear()
    reward_to_go_list.clear()
    avg_reward_to_go_list.clear()

    for iter in range(batchsize):
        log_probability, rewards, obs_step = rollout(hoverboard,nn_policy,device)
        log_probability = torch.stack(log_probability)
        log_probability_list.append(log_probability)
        returns, traj_return = compute_reward_to_go(hoverboard,log_probability,rewards)
        return_tensors = torch.tensor(returns, dtype=torch.float32, device=device)
        reward_to_go_list.append(return_tensors)
        avg_reward_to_go_list.append(traj_return)

    baseline = np.mean(avg_reward_to_go_list)
    all_rewards_tensor = torch.cat(reward_to_go_list)
    all_log_probs_tensor = torch.cat(log_probability_list)
    loss_function = -torch.dot(all_log_probs_tensor, (all_rewards_tensor - baseline))
    batch_reward = loss_function/batchsize
    optimizer.zero_grad()
    batch_reward.backward()
    optimizer.step()
    print(f"{batch+1}. Batch {batch+1} done ...")

torch.save(nn_policy.state_dict(),"/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Vanilla_PG_Implementation/attempt_2_1000x32_baseline.pth")
print("Weights saved successfully")



