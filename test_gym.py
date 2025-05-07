from go1_torque_env import Go1Env
import numpy as np
import gymnasium
import mujoco
from mujoco import viewer


env = Go1Env(render_mode="human")
action_size = env.action_space.shape
print(action_size)

for i in range(300000):
    action = np.zeros(action_size)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        env.reset()
        print("reset!")
    print("here's i: ", i)
