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
        super().__init__(self,
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
        self.num_steps = 0

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

        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()  # [0.5, 0.0, 0.0]
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25

        # terminal episode conditions
        self._healthy_z_range = (0.22, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]

        # Non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )

        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # Action: 12 torque values
        self._last_action = np.zeros(12)

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        # Feet site names to index mapping
        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]

        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

        if self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def step(self, action):
        self.num_steps +=1
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        reward, reward_info = self._calc_reward(action)
        terminated = self.is_terminated()
        truncated = self.num_steps >= (self.max_episode_time_sec / self.dt)

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        # rendering part
        if self.render_mode == "human":
            self.viewer.render()
