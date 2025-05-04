import mujoco
import mujoco_viewer
import numpy as np
import cv2
import mediapy as media
import time

# Load model and data
model = mujoco.MjModel.from_xml_path("unitree_go1/scene.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

# # Create and configure a camera
# cam = mujoco.MjvCamera()
# mujoco.mjv_defaultFreeCamera(model, cam)

# cam.azimuth = 90.0
# cam.elevation = -25.0
# cam.distance = 3.0
# cam.lookat[:] = [0.0, 0.0, 0.0]
# cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# Step and render loop
for i in range(500):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break
viewer.close()