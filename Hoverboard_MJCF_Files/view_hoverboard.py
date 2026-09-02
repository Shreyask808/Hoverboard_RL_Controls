import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = "/mnt/c/Users/Admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

duration= 100
left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id
pend_hinge_x_qpos_id = model.joint("pend_hinge_x").qposadr[0]
pend_hinge_y_qpos_id = model.joint("pend_hinge_y").qposadr[0]
data.qpos[pend_hinge_y_qpos_id] = np.deg2rad(10)
mujoco.mj_forward(model,data)

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running() and data.time < 100:
        step_start = time.time()

        Ml = -150*data.qpos[pend_hinge_y_qpos_id]
        Mr = -150*data.qpos[pend_hinge_y_qpos_id]

        data.ctrl[left_motor_id] = Ml
        data.ctrl[right_motor_id] = Mr

        mujoco.mj_step(model,data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
