import numpy as np
import mujoco
import gymnasium

# =================================================================================================================================================================================================================
# Class Definition
class Hoverboard(gymnasium.Env):
    def __init__(self,XML_path):
        super().__init__()
        input_dim = 11
        self.hbmodel = mujoco.MjModel.from_xml_path(XML_path)
        self.hbdata = mujoco.MjData(self.hbmodel)
        self.step_count = 0                                                                                                                     # Step counter in Mujoco Step function

        ## Hoverboard Data
        self.timestep = self.hbmodel.opt.timestep                                                                                               # Timestep [sec]
        self.n_sub = 10                                                                                                                         # Frameskips
        self.dt_ctrl = self.n_sub*self.timestep                                                                                                 # Timestep of Control Input [sec]
        self.max_count = int(round(10/self.dt_ctrl))                                                                                            # Max number of Timesteps [.]
        self.g = self.hbmodel.opt.gravity[2]                                                                                                    # Gravity [m/sec^2]
        self.ctrl_dim = self.hbmodel.nu                                                                                                         # Number of control inputs
        self.ctrl_low = self.hbmodel.actuator_ctrlrange[:,0]                                                                                    # Lower Bound for Control Inputs [N.m]
        self.ctrl_high = self.hbmodel.actuator_ctrlrange[:,1]                                                                                   # High Bound for Control Inputs [N.m]        
        self.observation_space = gymnasium.spaces.Box(low= -np.inf, high= np.inf, shape=(input_dim,), dtype=np.float64)                         # Observation space definition
        self.action_space = gymnasium.spaces.Box(low= self.ctrl_low, high= self.ctrl_high, shape=(self.ctrl_dim,), dtype=np.float64)            # Action space definition

        ## States indices
        self.th = self.hbmodel.joint("pend_hinge_y").qposadr[0]
        self.gamma = self.hbmodel.joint("pend_hinge_x").qposadr[0]
        self.phi_R = self.hbmodel.joint("right_hinge").qposadr[0]
        self.phi_L = self.hbmodel.joint("left_hinge").qposadr[0]
        self.psi = self.hbmodel.joint("chassis_hinge_z").qposadr[0]
        self.xi = self.hbmodel.joint("chassis_hinge_x").qposadr[0]
        self.z = self.hbmodel.joint("chassis_z").qposadr[0]

        # Velocity indices
        self.thdot = self.hbmodel.joint("pend_hinge_y").dofadr[0]
        self.gammadot = self.hbmodel.joint("pend_hinge_x").dofadr[0]
        self.phidot_R = self.hbmodel.joint("right_hinge").dofadr[0]
        self.phidot_L = self.hbmodel.joint("left_hinge").dofadr[0]
        self.psidot = self.hbmodel.joint("chassis_hinge_z").dofadr[0]
        self.xidot = self.hbmodel.joint("chassis_hinge_x").dofadr[0]
        self.zdot = self.hbmodel.joint("chassis_z").dofadr[0]

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.hbmodel, self.hbdata)
        self.step_count = 0

        th_ini = self.np_random.uniform(-np.deg2rad(10), np.deg2rad(10))
        gamma_ini = self.np_random.uniform(-np.deg2rad(10), np.deg2rad(10))
        thdot_ini = 0
        gammadot_ini = 0

        self.hbdata.qpos[self.th] = th_ini             # th_ini
        self.hbdata.qpos[self.gamma] = gamma_ini          # gamma_ini

        self.hbdata.qvel[self.thdot] = thdot_ini
        self.hbdata.qvel[self.gammadot] = gammadot_ini
        mujoco.mj_forward(self.hbmodel,self.hbdata)
        return self._get_obs(),{}

    def step(self, action):
        self.hbdata.ctrl[:] = action
        terminate = False
        for _ in range(self.n_sub):
            mujoco.mj_step(self.hbmodel, self.hbdata)
            terminate, *_ = self._check_done()
            if terminate:
                break
        reward = self._compute_reward(terminate)
        self.step_count += 1
        obs = self._get_obs()
        truncate = self.step_count >= self.max_count
        return obs, reward, terminate, truncate, {}

    def _get_obs(self):
        th = self.hbdata.qpos[self.th]
        gamma = self.hbdata.qpos[self.gamma]
        xi = self.hbdata.qpos[self.xi]
        z = self.hbdata.qpos[self.z]

        thdot = self.hbdata.qvel[self.thdot]
        gammadot = self.hbdata.qvel[self.gammadot]
        psidot = self.hbdata.qvel[self.psidot]
        phidot_R = self.hbdata.qvel[self.phidot_R]
        phidot_L = self.hbdata.qvel[self.phidot_L]
        xidot = self.hbdata.qvel[self.xidot]
        zdot = self.hbdata.qvel[self.zdot]

        displacement = [th,gamma,xi,z]
        velocity = [thdot,gammadot,psidot,phidot_R,phidot_L,xidot,zdot]
        observation = np.concatenate([displacement,velocity])
        return observation

    def _check_done(self):
        th = self.hbdata.qpos[self.th]
        gamma = self.hbdata.qpos[self.gamma]
        xi = self.hbdata.qpos[self.xi]
        z = self.hbdata.qpos[self.z]
        status_th = bool(abs(th) >= np.deg2rad(30))
        status_gamma = bool(abs(gamma) >= np.deg2rad(30))
        status_xi = bool(abs(xi) >= np.deg2rad(5))
        status_z = bool(abs(z) >= 0.1)
        status = status_th or status_gamma or status_xi or status_z
        return status,status_th,status_gamma,status_xi,status_z

    def _compute_reward(self, terminated):
        th = self.hbdata.qpos[self.th]
        gamma = self.hbdata.qpos[self.gamma]
        xi = self.hbdata.qpos[self.xi]
        z = self.hbdata.qpos[self.z]

        alive_bonus = 1
        balance_reward = np.exp(-(th/0.2)**2 - (gamma/0.2)**2)
        xi_pen = (xi / np.deg2rad(5))**2
        z_pen  = (z / 0.1)**2
        
        reward = alive_bonus + balance_reward - 0.5*xi_pen - 0.5*z_pen
        if terminated:
            reward -= 20
        return reward
        