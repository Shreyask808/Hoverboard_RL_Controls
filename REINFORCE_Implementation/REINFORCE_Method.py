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
        #self.R = self.hbmodel.geom("right_wheel_geom").size[0]
        self.angle_scale = 1
        self.angular_velocity_scale = np.sqrt(self.h/np.abs(self.g))

        obs_dim = 8
        ctrl_dim = self.hbmodel.nu

        self.max_T = np.sum(self.hbmodel.actuator_ctrlrange[:,1]**2)
        self.step_count = 0

        self.discount_factor = 0.999

        self.Low = self.hbmodel.actuator_ctrlrange[:,0]
        self.High = self.hbmodel.actuator_ctrlrange[:,1]

        self.max_count = int(10/self.hbmodel.opt.timestep)

        self.hinge_x_qpos_id = self.hbmodel.joint("pend_hinge_x").qposadr[0]
        self.hinge_y_qpos_id = self.hbmodel.joint("pend_hinge_y").qposadr[0]
        self.hinge_x_qvel_id = self.hbmodel.joint("pend_hinge_x").dofadr[0]
        self.hinge_y_qvel_id = self.hbmodel.joint("pend_hinge_y").dofadr[0]

        self.chassis_hinge_z_id = self.hbmodel.joint("chassis_hinge_z").qposadr[0]
        self.chassis_hinge_x_id = self.hbmodel.joint("chassis_hinge_x").qposadr[0]
        self.chassis_hinge_z_qvel_id = self.hbmodel.joint("chassis_hinge_z").dofadr[0]
        self.chassis_hinge_x_qvel_id = self.hbmodel.joint("chassis_hinge_x").dofadr[0]

        self.chassis_x_id = self.hbmodel.joint("chassis_x").qposadr[0]
        self.chassis_y_id = self.hbmodel.joint("chassis_y").qposadr[0]
        self.chassis_z_id = self.hbmodel.joint("chassis_z").qposadr[0]

        self.chassis_x_qvel_id = self.hbmodel.joint("chassis_x").dofadr[0]
        self.chassis_y_qvel_id = self.hbmodel.joint("chassis_y").dofadr[0]
        self.chassis_z_qvel_id = self.hbmodel.joint("chassis_z").dofadr[0]

        self.left_hinge_id = self.hbmodel.joint("left_hinge").qposadr[0]
        self.right_hinge_id = self.hbmodel.joint("right_hinge").qposadr[0]
        self.left_hinge_qvel_id = self.hbmodel.joint("left_hinge").dofadr[0]
        self.right_hinge_qvel_id = self.hbmodel.joint("right_hinge").dofadr[0]
        
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

        self.hbdata.qpos[self.hinge_y_qpos_id] = th_ini             # th_ini
        self.hbdata.qpos[self.hinge_x_qpos_id] = gamma_ini          # gamma_ini

        self.hbdata.qvel[self.hinge_y_qvel_id] = thdot_ini
        self.hbdata.qvel[self.hinge_x_qvel_id] = gammadot_ini

        self.step_count = 0

        mujoco.mj_forward(self.hbmodel,self.hbdata)
        return self._get_obs(), {}

    def step(self,action):
        self.hbdata.ctrl[:] = action
        mujoco.mj_step(self.hbmodel,self.hbdata)
        obs = self._get_obs()
        terminated = self._check_done()
        reward = self._compute_reward(terminated)
        self.step_count += 1
        truncated = self.step_count >= self.max_count
        return obs, reward, terminated, truncated,{}
    
    def _get_obs(self):
        th = self.hbdata.qpos[self.hinge_y_qpos_id]
        gamma = self.hbdata.qpos[self.hinge_x_qpos_id]
        psi = self.hbdata.qpos[self.chassis_hinge_z_id]
        xi = self.hbdata.qpos[self.chassis_hinge_x_id]

        thdot = self.hbdata.qvel[self.hinge_y_qvel_id]
        gammadot = self.hbdata.qvel[self.hinge_x_qvel_id]
        psidot = self.hbdata.qvel[self.chassis_hinge_z_qvel_id]
        xidot = self.hbdata.qvel[self.chassis_hinge_x_qvel_id]

        angles = [th*self.angle_scale, gamma*self.angle_scale, psi*self.angle_scale, xi*self.angle_scale]
        rates = [thdot*self.angular_velocity_scale, gammadot*self.angular_velocity_scale, psidot*self.angular_velocity_scale, xidot*self.angular_velocity_scale]
        observation = np.concatenate([angles, rates])
        return observation

    def _compute_reward(self, terminated):
        th = self.hbdata.qpos[self.hinge_y_qpos_id]
        gamma = self.hbdata.qpos[self.hinge_x_qpos_id]
        xi = self.hbdata.qpos[self.chassis_hinge_x_id]

        thdot = self.hbdata.qvel[self.hinge_y_qvel_id]
        gammadot = self.hbdata.qvel[self.hinge_x_qvel_id]
        yawrate = self.hbdata.qvel[self.chassis_hinge_z_qvel_id]
        xidot = self.hbdata.qvel[self.chassis_hinge_x_qvel_id]

        balance_reward = np.exp((-20*(th**2 + gamma**2)*self.angle_scale)) + np.exp((-20*xi**2*self.angle_scale))
        rate_reward = -0.3*(thdot**2 + gammadot**2 + xidot**2)*self.angular_velocity_scale**2
        action_reward = -0.5*np.sum(self.hbdata.ctrl**2)/self.max_T
        yaw_reward = -1*yawrate**2*self.angular_velocity_scale**2
        alive_bonus = 1

        reward = alive_bonus + action_reward + rate_reward + balance_reward + yaw_reward - 10*int(terminated)
        return reward

    def _check_done(self):
        z = self.hbdata.qpos[self.chassis_z_id]
        xi = self.hbdata.qpos[self.chassis_hinge_x_id]
        th = self.hbdata.qpos[self.hinge_y_qpos_id]
        gamma = self.hbdata.qpos[self.hinge_x_qpos_id]
        return bool(abs(xi) >= np.deg2rad(5) or z >= 0.1 or abs(th) >= np.deg2rad(45) or abs(gamma) >= np.deg2rad(45))

class PolicyNet(nn.Module):
    def __init__(self, in_dim, l1_dim, l2_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, l1_dim)
        self.fc2 = nn.Linear(l1_dim, l2_dim)
        self.fc3 = nn.Linear(l2_dim, out_dim)
        self.log_std = nn.Parameter(-1.5*torch.ones(out_dim))

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x) # Continuous Gaussian Distribution 
        std = torch.exp(self.log_std)
        return mean, std

def rollout(env,policy_net,device):
    obs_list = []
    raw_action_list = []
    rewards = []
    obs, info = env.reset()

    with torch.no_grad():
        for t in range(env.max_count):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, std = policy_net(obs_tensor)
            distribution = Normal(mean, std)   
            raw_action = distribution.sample()
            u = torch.tanh(raw_action)
            raw_action_np = u.squeeze(0).detach().cpu().numpy()

            obs_list.append(obs)
            raw_action_list.append(raw_action.squeeze(0))
            
            action = env.Low + (raw_action_np + 1)*(env.High - env.Low)/2
            obs, reward, terminated, truncated, _ = env.step(action)

            rewards.append(reward)        

            if terminated or truncated:
                break

    return obs_list,raw_action_list,rewards, obs, t

def compute_reward_to_go(env,rewards):
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
input_dim = 8                                                                                                   # Inputs to the Policy Net
output_dim = hoverboard.hbmodel.nu                                                                              # Each Motor Torque is a continuous Gaussian Distribution with a mean and standard deviation as the outputs
nn_policy = PolicyNet(input_dim,64,64,output_dim).to(device)                                                    # Control Policy
model_parameters = sum(p.numel() for p in nn_policy.parameters())
batchsize = 512                                                                                                  # Number of Rollouts per gradient step
max_batches = 100                                                                                              # Maximum number of batches in the Training
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

t0 = time.time()

for batch in range(max_batches):
    log_probability_list.clear()
    reward_to_go_list.clear()
    avg_reward_to_go_list.clear()
    episode_length = 0

    for iter in range(batchsize):
        observation_list,action_list,rewards,obs_step, step_cnt = rollout(hoverboard,nn_policy,device)
        observation_batch = torch.tensor(np.array(observation_list), dtype=torch.float32, device=device)
        action_batch = torch.stack(action_list)
        mean, std = nn_policy(observation_batch)
        log_probability = Normal(mean, std).log_prob(action_batch).sum(dim=1)
        log_probability_list.append(log_probability)
        returns, traj_return = compute_reward_to_go(hoverboard,rewards)
        return_tensors = torch.tensor(returns, dtype=torch.float32, device=device)
        reward_to_go_list.append(return_tensors)
        avg_reward_to_go_list.append(traj_return)
        episode_length += step_cnt

    baseline = np.mean(avg_reward_to_go_list)
    all_rewards_tensor = torch.cat(reward_to_go_list)
    all_log_probs_tensor = torch.cat(log_probability_list)
    loss_function = -torch.dot(all_log_probs_tensor, (all_rewards_tensor - baseline))
    batch_reward = loss_function/batchsize
    optimizer.zero_grad()
    batch_reward.backward()
    optimizer.step()
    episode_length = episode_length*hoverboard.hbmodel.opt.timestep/batchsize
    print(f"{batch+1}. Batch {batch+1} done, Avg Episode Length - {episode_length} [sec]...................")

tf = time.time()
wall_clock_time_min = np.floor((tf -t0)/60)
wall_clock_time_sec = (tf - t0)%60
torch.save(nn_policy.state_dict(),"/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/REINFORCE_Implementation/attempt_10_100x512_baseline.pth")
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("")
print("Weights saved successfully")
print("")
print(f"Wall Clock Time - {wall_clock_time_min} [min] {wall_clock_time_sec} [sec]")
print("")
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")




