import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt

xml_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_3.xml"

hboard_model = mujoco.MjModel.from_xml_path(xml_path)
hboard_data = mujoco.MjData(hboard_model)

