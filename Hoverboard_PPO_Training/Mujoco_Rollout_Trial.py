import mujoco
import mujoco.viewer
import gymnasium
import time
import numpy as np
import matplotlib.pyplot as plt

xml_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"

## Gymnasium Class
class hoverboardEnv(gymnasium.Env):
    def __init__(self,xml_path):
        self.hbmodel = mujoco.MjModel.from_xml_path(xml_path)
        self.hbdata = mujoco.MjData(self.hbmodel)
        obs_dim = self.hbdata.nq + self.hbdata.nv
        ctrl_dim = self.hbdata.nu
        self.observation_space = gymnasium.spaces.Box(
            low= -np.inf, high= np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-self.hbmodel.actuator.ctrlrange[:,0],
            high= self.hbmodel.actuator.ctrlrange[:,1],
            shape= (ctrl_dim,), dtype=np.float32
        )

