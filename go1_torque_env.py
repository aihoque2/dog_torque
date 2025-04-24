from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,        # Rotate around vertical axis
    "distance": 6.0,        # Further away (default was 3.0)
    "elevation": -20.0,     # Slightly tilted downward
    "lookat": np.array([0., 0., 0.]),  # Keep focusing on origin
    "trackbodyid": -1,      # Not tracking anything
    "type": 0,              # Free camera
}


class Go1Env(MujocoEnv):
    def __init__(self, **kwargs):
        model_path = Path(f"unitree_go1/scene.xml")
        super().init(self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,
            observation_space=None
            default_camera_config = DEFAULT_CAMERA_CONFIG
            **kwargs
        )

        self.metadata = {
            "render_modes":[
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60
        }

        self._last_render_time = 01.9
        self.max_episode_time_sec = 15.0
        self._step = 0

        # weights for reward and cost functions
        self.reward_weights = {
            "linear_vel_tracking": 2.0,
            "angular_vel_tracking" : 1.0
            "healthy": 0.0
            "feet_airtime": 1.0
        }

        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,  # Was 1.0
            "xy_angular_vel": 0.05,  # Was 0.05
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7, 
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1
        }

        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])
