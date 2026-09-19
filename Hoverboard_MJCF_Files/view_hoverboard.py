import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

xml_path = "/mnt/c/Users/Admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_2.xml"
weights = "/mnt/c/Users/Admin/Documents/Github/Hoverboard_RL_Controls/REINFORCE_Implementation/attempt_2_1000x32_baseline.pth"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

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


time_log = []
theta = []
thetadot = []
gamma =[]
gammadot = []
Ml_log = []
Mr_log = []

input_dim = (model.nq + model.nv)                   
output_dim = model.nu
nn_policy = PolicyNet(input_dim,64,64,output_dim)
nn_policy.load_state_dict(torch.load(weights, map_location="cpu"))
nn_policy.eval()   

duration= 100
left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id

pend_hinge_x_qpos_id = model.joint("pend_hinge_x").qposadr[0]
pend_hinge_y_qpos_id = model.joint("pend_hinge_y").qposadr[0]
pend_hinge_x_qvel_id = model.joint("pend_hinge_x").dofadr[0]
pend_hinge_y_qvel_id = model.joint("pend_hinge_y").dofadr[0]

data.qpos[pend_hinge_y_qpos_id] = np.deg2rad(0)
data.qpos[pend_hinge_x_qpos_id] = np.deg2rad(0)
mujoco.mj_forward(model,data)

low = model.actuator_ctrlrange[:,0]
high = model.actuator_ctrlrange[:,1]

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running() and data.time < duration:
        step_start = time.time()

        states = np.concatenate([data.qpos,data.qvel])
        inputs = torch.tensor(states, dtype=torch.float32).unsqueeze(0) 
        mean, std = nn_policy(inputs)
        u = torch.tanh(mean)
        action = u.squeeze(0).detach().numpy()
        torques = low + (action + 1)*(high - low)/2
        Ml = torques[0]
        Mr = torques[1]
        data.ctrl[left_motor_id] = Ml
        data.ctrl[right_motor_id] = Mr

        mujoco.mj_step(model,data)
        viewer.sync()

        time_log.append(data.time)
        theta.append(data.qpos[pend_hinge_y_qpos_id])
        gamma.append(data.qpos[pend_hinge_x_qpos_id])

        thetadot.append(data.qvel[pend_hinge_y_qvel_id])
        gammadot.append(data.qvel[pend_hinge_x_qvel_id])

        Ml_log.append(Mr)
        Mr_log.append(Mr)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize = (10,12), sharex=True)

ax1.plot(time_log,np.rad2deg(theta),color='red')
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Theta [deg]')
ax1.grid(True, alpha=0.3)

ax2.plot(time_log,np.rad2deg(gamma),color='red')
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Gamma [deg]')
ax2.grid(True, alpha=0.3)

ax3.plot(time_log,np.rad2deg(thetadot),color='red')
ax3.set_xlabel('Time [sec]')
ax3.set_ylabel('Thetadot [deg/sec]')
ax3.grid(True, alpha=0.3)

ax4.plot(time_log,np.rad2deg(gammadot),color='red')
ax4.set_xlabel('Time [sec]')
ax4.set_ylabel('Gammadot [deg/sec]')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, (ax1) = plt.subplots(1,1, figsize=[10,12], sharex=True)
ax1.plot(time_log,Ml_log,color='blue',label='Left Wheel Torque')
ax1.plot(time_log,Mr_log,color='red',label='Right Wheel Torque')
#ax1.axhline(-T_max,color='black',linestyle='--')
#ax1.axhline(T_max,color='black',linestyle='--')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Motor Torque [N.m]')
ax1.legend()
plt.tight_layout()
plt.show()