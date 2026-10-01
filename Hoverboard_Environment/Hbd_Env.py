import numpy as np
import mujoco
import mujoco.viewer
import gymnasium

# =================================================================================================================================================================================================================
# Class Definition
class Hoverboard(gymnasium.Env):
    def __init__(self,XML_path, input_dim):
        super().__init__()
        self.hbmodel = mujoco.Mjmodel.from_xml_path(XML_path)
        self.hbdata = mujoco.MjData(self.hbmodel)
        self.step_count = 0                                                                                                                     # Step counter in Mujoco Step function

        ## Hoverboard Data
        self.timestep = self.hbmodel.opt.timestep                                                                                               # Timestep [sec]
        self.max_count = 10/self.timestep                                                                                                       # Max number of Timesteps [.]
        self.g = self.hbmodel.opt.gravity[2]                                                                                                    # Gravity [m/sec^2]
        self.ctrl_dim = self.hbmodel.nu                                                                                                         # Number of control inputs
        self.ctrl_low = self.hbmodel.actuator_ctrlrange[:,0]                                                                                    # Lower Bound for Control Inputs [N.m]
        self.ctrl_high = self.hbmodel.actuator_ctrlrange[:,1]                                                                                   # High Bound for Control Inputs [N.m]        
        self.observation_space = gymnasium.spaces.Box(low= -np.inf, high= np.inf, shape=(input_dim,), dtype=np.float64)                         # Observation space definition
        self.sction_space = gymnasium.spaces.Box(low= self.ctrl_low, high= self.ctrl_high, shape=(self.ctrl_dim,), dtype=np.float64)            # Action space definition

        ## States indices
        self.th = self.hbmodel.joint["pend_hinge_y"].qposadr[0]
        self.gamma = self.hbmodel.joint["pend_hinge_x"].qposadr[0]
        self.phi_R = self.hbmodel.joint["right_hinge"].qposadr[0]
        self.phi_L = self.hbmodel.joint["left_hinge"].qposadr[0]
        self.psi = self.hbmodel.joint["chassis_hinge_z"].qposadr[0]
        self.xi = self.hbmodel.joint["chassis_hinge_x"].qposadr[0]
        self.z = self.hbmodel.joint["chassis_z"].qposadr[0]

        # Velocity indices
        self.thdot = self.hbmodel.joint["pend_hinge_y"].dofadr[0]
        self.gammadot = self.hbmodel.joint["pend_hinge_x"].dofadr[0]
        self.phidot_R = self.hbmodel.joint["right_hinge"].dofadr[0]
        self.phidot_L = self.hbmodel.joint["left_hinge"].dofadr[0]
        self.psidot = self.hbmodel.joint["chassis_hinge_z"].dofadr[0]
        self.xidot = self.hbmodel.joint["chassis_hinge_x"].dofadr[0]

    def reset(self,seed=None, options=None, qpos_array=None, qvel_array=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.hbmodel, self.hbdata)
        self.step_count = 0
        self.hbdata.qpos[:] = qpos_array
        self.hbdata.qvel[:] = qvel_array
        mujoco.mj_forward(self.hbmodel,self.hbdata)
        return self._get_obs(),{}

    def step(self,action):
        self.hbdata.ctrl[:] = action
        mujoco.mj_step(self.hbmodel,self.hbdata)
        self.step_count +=1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminate,*_ = self._check_done()
        truncate = self.step_count >= self.max_count
        return obs,reward,terminate,truncate, {}

    def _get_obs(self):
        th = self.hbdata.qpos[self.th]
        gamma = self.hbdata.qpos[self.gamma]
        psi = self.hbdata.qpos[self.psi]
        phi_R = self.hbdata.qpos[self.psi_R]
        phi_L = self.hbdata.qpos[self.psi_L]
        xi = self.bdata.qpos[self.xi]

        thdot = self.hbdata.qvel[self.thdot]
        gammadot = self.hbdata.qvel[self.gammadot]
        psidot = self.hbdata.qvel[self.psidot]
        phidot_R = self.hbdata.qvel[self.psidot_R]
        phidot_L = self.hbdata.qvel[self.psidot_L]
        xidot = self.hbdata.qvel[self.xidot]

        angles = [th,gamma,psi,phi_R,phi_L,xi]
        velocity = [thdot,gammadot,psidot,phidot_R,phidot_L,xidot]
        observation = np.concatenate([angles,velocity])
        return observation

    def _check_done(self):
        th = self.hbdata.qpos[self.th]
        gamma = self.hbdata.qpos[self.gamma]
        xi = self.bdata.qpos[self.xi]
        z = self.hbdata.qpos[self.z]
        status_th = bool(abs(th) >= np.deg2rad(30))
        status_gamma = bool(abs(gamma) >= np.deg2rad(30))
        status_xi = bool(abs(xi) >= np.deg2rad(5))
        status_z = bool(abs(z) >= 0.1)
        status = status_th or status_gamma or status_xi or status_z
        return status,status_th,status_gamma,status_xi,status_z