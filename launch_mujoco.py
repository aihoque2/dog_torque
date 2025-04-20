import mujoco
import mujoco.viewer
import numpy as np
import time

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,  # FREE camera
}

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("unitree_go1/scene_torque.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Apply camera config to the first free camera (id 0)
    viewer.cam.azimuth = DEFAULT_CAMERA_CONFIG["azimuth"]
    viewer.cam.distance = DEFAULT_CAMERA_CONFIG["distance"]
    viewer.cam.elevation = DEFAULT_CAMERA_CONFIG["elevation"]
    viewer.cam.lookat[:] = DEFAULT_CAMERA_CONFIG["lookat"]
    viewer.cam.type = DEFAULT_CAMERA_CONFIG["type"]

    # Run viewer loop
    start = time.time()
    while viewer.is_running() and time.time() - start < 10:
        mujoco.mj_step(model, data)
        viewer.sync()