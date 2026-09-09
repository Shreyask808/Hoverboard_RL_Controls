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

        self.step_count = 0
        self.max_count = int(120/self.hbmodel.opt.timestep)
        self.hinge_x_qpos_id = self.hbmodel.joint("pend_hinge_x").qposadr[0]
        self.hinge_y_qpos_id = self.hbmodel.joint("pend_hinge_y").qposadr[0]

        self.hinge_x_qvel_id = self.hbmodel.joint("pend_hinge_x").dofadr[0]
        self.hinge_y_qvel_id = self.hbmodel.joint("pend_hinge_y").dofadr[0]

        self.observation_space = gymnasium.spaces.Box(
            low= -np.inf, high= np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = gymnasium.spaces.Box(
            low=self.hbmodel.actuator_ctrlrange[:,0],
            high= self.hbmodel.actuator_ctrlrange[:,1],
            shape= (ctrl_dim,), dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.hbmodel,self.hbdata)

        th_ini = np.deg2rad(0)
        gamma_ini = np.deg2rad(0)
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

class PolicyNet(nn.Module):
    def __init__(self, in_dim, l1_dim, l2_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, l1_dim)
        self.fc2 = nn.Linear(l1_dim, l2_dim)
        self.fc3 = nn.Linear(l2_dim, out_dim)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # motor control torques = low + (NN_output + 1)*(high - low)/2

        return x

hoverboard = hoverboardEnv(xml_path) 
print(hoverboard.hbmodel.nq,hoverboard.hbmodel.nu)
input_dim = hoverboard.hbmodel.nq
output_dim = hoverboard.hbmodel.nu
nn_policy = PolicyNet(input_dim,64,64,output_dim)