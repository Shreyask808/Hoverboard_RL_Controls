import mujoco
import mujoco.viewer
import gymnasium
import time
import numpy as np
import matplotlib.pyplot as plt

xml_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"
th_ini = np.deg2rad(10)
gamma_ini = np.deg2rad(10)
thdot_ini = 0
gammadot_ini = 0

## Gymnasium Class
class hoverboardEnv(gymnasium.Env):
    def __init__(self,xml_path):
        self.hbmodel = mujoco.MjModel.from_xml_path(xml_path)
        self.hbdata = mujoco.MjData(self.hbmodel)

        obs_dim = self.hbdata.nq + self.hbdata.nv
        ctrl_dim = self.hbdata.nu

        self.hinge_x_qpos_id = self.hbmodel.joint("pend_hinge_x").qposadr[0]
        self.hinge_y_qpos_id = self.hbmodel.joint("pend_hinge_y").qposadr[0]

        self.hinge_x_qvel_id = self.hbmodel.joint("pend_hinge_x").dofadr[0]
        self.hinge_y_qvel_id = self.hbmodel.joint("pend_hinge_y").dofadr[0]

        self.observation_space = gymnasium.spaces.Box(
            low= -np.inf, high= np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-self.hbmodel.actuator.ctrlrange[:,0],
            high= self.hbmodel.actuator.ctrlrange[:,1],
            shape= (ctrl_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.hbmodel,self.hbdata)

        th_ini = np.deg2rad(10)
        gamma_ini = np.deg2rad(10)
        thdot_ini = 0
        gammadot_ini = 0

        self.hbdata.qpos[self.hinge_x_qpos_id] = gamma_ini
        self.hbdata.qpos[self.hinge_y_qpos_id] = th_ini
        self.hbdata.qvel[self.hinge_x_qvel_id] = gammadot_ini
        self.hbdata.qvel[self.hinge_y_qvel_id] = thdot_ini

        mujoco.mj_forward(self.hbmodel,self.hbdata)
        return self._get_obs(), {}

    def step(self,action):
        
    