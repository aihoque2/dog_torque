import mujoco
import numpy as np
import cv2
import mediapy as media
import time

# Load model and data
model = mujoco.MjModel.from_xml_path("unitree_go1/scene.xml")
data = mujoco.MjData(model)

# Create renderer
renderer = mujoco.Renderer(model)

# Create and configure a camera
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)

cam.azimuth = 90.0
cam.elevation = -25.0
cam.distance = 3.0
cam.lookat[:] = [0.0, 0.0, 0.0]
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# Step and render loop
for _ in range(300):
    mujoco.mj_step(model, data)

    renderer.update_scene(data, cam)  # Pass camera here
    img = renderer.render()

    # Display frame with mediapy
    cv2.imshow("Mujoco!", img)
    time.sleep(0.001)