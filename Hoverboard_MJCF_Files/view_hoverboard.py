import mujoco
import mujoco.viewer
import numpy as np

xml_path = "/mnt/c/Users/Admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_1.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

duration = 5
framerate = 60

while data.time < duration:
    mujoco.mj_step(model, data)
