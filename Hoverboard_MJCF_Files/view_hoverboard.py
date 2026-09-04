import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

xml_path = "/mnt/c/Users/Admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

time_log = []
theta = []
thetadot = []
gamma =[]
gammadot = []

duration= 100
left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id

pend_hinge_x_qpos_id = model.joint("pend_hinge_x").qposadr[0]
pend_hinge_y_qpos_id = model.joint("pend_hinge_y").qposadr[0]
pend_hinge_x_qvel_id = model.joint("pend_hinge_x").dofadr[0]
pend_hinge_y_qvel_id = model.joint("pend_hinge_y").dofadr[0]

data.qpos[pend_hinge_y_qpos_id] = np.deg2rad(10)
mujoco.mj_forward(model,data)

T_max = model.actuator("left_motor").ctrlrange[1]

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running() and data.time < duration:
        step_start = time.time()

        Ml = -300*data.qpos[pend_hinge_y_qpos_id] - 200*data.qpos[pend_hinge_x_qpos_id] + 0.1*data.qvel[pend_hinge_y_qvel_id] #- 0.1*data.qvel[pend_hinge_x_qvel_id]
        Mr = -300*data.qpos[pend_hinge_y_qpos_id] + 200*data.qpos[pend_hinge_x_qpos_id] + 0.1*data.qvel[pend_hinge_y_qvel_id] #+ 0.1*data.qvel[pend_hinge_x_qvel_id]

        Ml = np.clip(Ml,-T_max,T_max)
        Mr = np.clip(Mr,-T_max,T_max)

        data.ctrl[left_motor_id] = Ml
        data.ctrl[right_motor_id] = Mr

        mujoco.mj_step(model,data)
        viewer.sync()

        time_log.append(data.time)
        theta.append(data.qpos[pend_hinge_y_qpos_id])
        gamma.append(data.qpos[pend_hinge_x_qpos_id])

        thetadot.append(data.qvel[pend_hinge_y_qvel_id])
        gammadot.append(data.qvel[pend_hinge_x_qvel_id])
    
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
