from his import Go1MujocoEnv
import numpy as np
import gymnasium
import mujoco
from mujoco import viewer


env = Go1MujocoEnv()
action_size = env.action_space.shape
print(action_size)
viewer.launch_from_path("./unitree_go1/scene.xml")
while (True):
    action = np.zeros(action_size)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
        print("reset!")