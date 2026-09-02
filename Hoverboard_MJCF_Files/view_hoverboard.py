import mujoco
import mujoco.viewer
import numpy as np

xml_path = "/mnt/c/Users/Admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

duration = 100
framerate = 60

pend_hinge_x_qpos_id = model.joint("pend_hinge_x").qposadr[0]
pend_hinge_y_qpos_id = model.joint("pend_hinge_y").qposadr[0]

data.qpos[pend_hinge_y_qpos_id] = np.deg2rad(10)
data.ctrl[:] = [-1,1]
mujoco.mj_forward(model,data)
mujoco.viewer.launch(model,data)