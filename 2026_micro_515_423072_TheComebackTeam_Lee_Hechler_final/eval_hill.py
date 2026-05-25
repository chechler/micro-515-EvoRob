from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalHillEnv(MujocoEnv, utils.EzPickle):
    """Hill terrain evaluation environment.

    Termination: robot is terminated when it flips upside-down, gets stuck
    (velocity < 1 cm/s for > 10 s), or produces NaN/Inf accelerations.
    Height-based termination is not used since the robot legitimately climbs.

    Training reward:  healthy_reward + x_position² + z_position² - ctrl_cost - cfrc_cost

    The info dict always exposes the four keys required by the neutral
    leaderboard formula: healthy_reward, x_position, ctrl_cost, cfrc_cost.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.55,
        cfrc_cost_weight: float = 5e-4,
        lateral_penalty_weight: float = 0.1,
        lateral_position_penalty_weight: float = 0.1,
        alignment_weight: float = 2.0,
        fall_penalty: float = 50.0,
        velocity_reward_weight: float = 2.0,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        xml_file_path = robot_path if path.isabs(robot_path) else path.join(
            path.dirname(path.realpath(__file__)), robot_path
        )

        utils.EzPickle.__init__(
            self, xml_file_path, frame_skip, default_camera_config,
            ctrl_cost_weight, cfrc_cost_weight, lateral_penalty_weight,
            lateral_position_penalty_weight, alignment_weight, fall_penalty,
            velocity_reward_weight, reset_noise_scale, **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._lateral_penalty_weight = lateral_penalty_weight
        self._lateral_position_penalty_weight = lateral_position_penalty_weight
        self._alignment_weight = alignment_weight
        self._fall_penalty = fall_penalty
        self._velocity_reward_weight = velocity_reward_weight
        self._reset_noise_scale = reset_noise_scale
        self._stuck_count = 0
        self._init_z = 0.0

        MujocoEnv.__init__(
            self, xml_file_path, frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    _K_EXP: float = 0.15  # stronger position reward; exp(x·k)−1 grows faster on hill

    def step(self, action):
        xyz_before = self.data.body(1).xpos[:3].copy()
        self.do_simulation(action, self.frame_skip)
        xyz_after = self.data.body(1).xpos[:3].copy()

        xyz_velocity = (xyz_after - xyz_before) / self.dt
        x_velocity = float(xyz_velocity[0])
        x_position = float(xyz_after[0])
        z_position = float(xyz_after[2])
        z_gain = z_position - self._init_z

        x_exp_reward = float(np.exp(x_position * self._K_EXP) - 1.0)

        R = self.data.body(1).xmat.reshape(3, 3)
        facing_x = float(R[0, 0])  # +1 = facing world +x, -1 = facing world -x
        alignment_reward = float(self._alignment_weight * facing_x)

        healthy_reward = 1.0
        ctrl_cost = float(np.sum(action ** 2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)
        y_position = float(xyz_after[1])
        lateral_penalty = float(self._lateral_penalty_weight * xyz_velocity[1] ** 2)
        lateral_position_penalty = float(self._lateral_position_penalty_weight * y_position ** 2)

        velocity_reward = float(self._velocity_reward_weight * x_velocity)
        terminated = self._is_terminated(xyz_velocity)
        fall_penalty = self._fall_penalty if terminated else 0.0

        reward = (healthy_reward + x_exp_reward + velocity_reward
                  + max(0.0, z_gain) ** 2
                  + alignment_reward
                  - ctrl_cost - cfrc_cost - lateral_penalty - lateral_position_penalty
                  - fall_penalty)

        info = {
            "healthy_reward": -10.0 if terminated else healthy_reward,
            "x_position": x_position,
            "z_position": z_position,
            "z_gain": z_gain,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "x_velocity": x_velocity,
            "y_velocity": float(xyz_velocity[1]),
            "z_velocity": float(xyz_velocity[2]),
            "lateral_penalty": lateral_penalty,
            "lateral_position_penalty": lateral_position_penalty,
            "alignment_reward": alignment_reward,
            "x_exp_reward": x_exp_reward,
            "velocity_reward": velocity_reward,
            "facing_x": facing_x,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    _Z_FALL_THRESHOLD: float = -0.5  # terminate if torso drops 0.5 m below starting height

    def _is_terminated(self, xyz_velocity: np.ndarray) -> bool:
        qacc = self.data.qacc
        if np.any(np.isnan(qacc) | np.isinf(qacc) | (np.abs(qacc) > 1e6)):
            return True
        if self._torso_upside_down():
            return True
        if float(self.data.body(1).xpos[2]) - self._init_z < self._Z_FALL_THRESHOLD:
            return True
        if np.linalg.norm(xyz_velocity) < 1e-2:
            self._stuck_count += 1
            if self._stuck_count > 10 / self.dt:
                return True
        else:
            self._stuck_count = 0
        return False

    def _torso_upside_down(self) -> bool:
        R = self.data.body(1).xmat.reshape(3, 3)
        return float(R[2, 2]) < 0.0

    def _get_obs(self):
        return np.concatenate((self.data.qpos.flat[2:], self.data.qvel.flat.copy()))

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self._stuck_count = 0
        self._init_z = float(self.data.body(1).xpos[2])
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}
