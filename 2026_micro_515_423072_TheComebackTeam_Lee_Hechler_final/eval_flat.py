from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalFlatEnv(MujocoEnv, utils.EzPickle):
    """Flat terrain evaluation environment.

    Termination: robot torso must stay between 0.2 m and 1.0 m above the
    ground (height-based).

    Training reward:  healthy_reward + x_velocity - ctrl_cost - cfrc_cost

    Tune ctrl_cost_weight and cfrc_cost_weight to shape behaviour on the
    flat surface.

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
        lateral_penalty_weight: float = 0.5,
        lateral_position_penalty_weight: float = 0.1,
        z_vel_penalty_weight: float = 0.2,
        alignment_weight: float = 2.0,
        fall_penalty: float = 500.0,
        velocity_reward_weight: float = 1.5,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        xml_file_path = robot_path if path.isabs(robot_path) else path.join(
            path.dirname(path.realpath(__file__)), robot_path
        )

        utils.EzPickle.__init__(
            self, xml_file_path, frame_skip, default_camera_config,
            ctrl_cost_weight, cfrc_cost_weight, lateral_penalty_weight,
            lateral_position_penalty_weight, z_vel_penalty_weight,
            alignment_weight, fall_penalty, velocity_reward_weight,
            reset_noise_scale, **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._lateral_penalty_weight = lateral_penalty_weight
        self._lateral_position_penalty_weight = lateral_position_penalty_weight
        self._z_vel_penalty_weight = z_vel_penalty_weight
        self._alignment_weight = alignment_weight
        self._fall_penalty = fall_penalty
        self._velocity_reward_weight = velocity_reward_weight
        self._reset_noise_scale = reset_noise_scale
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

    _K_EXP: float = 0.1  # reward doubles every ~7 m; exp(30·k)−1 ≈ 19 vs 1e9 at k=ln2

    def step(self, action):
        xy_before = self.data.qpos[:2].copy()
        z_before = float(self.data.qpos[2])
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.qpos[:2].copy()
        z_after = float(self.data.qpos[2])

        x_velocity = (xy_after[0] - xy_before[0]) / self.dt
        y_velocity = (xy_after[1] - xy_before[1]) / self.dt
        z_velocity = (z_after - z_before) / self.dt
        x_after = float(xy_after[0])

        x_exp_reward = float(np.exp(x_after * self._K_EXP) - 1.0)

        R = self.data.body(1).xmat.reshape(3, 3)
        facing_x = float(R[0, 0])  # +1 = facing world +x, -1 = facing world -x
        alignment_reward = float(self._alignment_weight * facing_x)

        healthy_reward = 1.0
        ctrl_cost = float(np.sum(action ** 2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)
        y_position = float(xy_after[1])
        lateral_penalty = float(self._lateral_penalty_weight * y_velocity ** 2)
        lateral_position_penalty = float(self._lateral_position_penalty_weight * y_position ** 2)
        z_vel_penalty = float(self._z_vel_penalty_weight * z_velocity ** 2)

        velocity_reward = float(self._velocity_reward_weight * x_velocity)
        terminated = self._is_terminated()
        fall_penalty = self._fall_penalty if terminated else 0.0

        reward = (healthy_reward + x_exp_reward + velocity_reward
                  + alignment_reward
                  - ctrl_cost - cfrc_cost - lateral_penalty - lateral_position_penalty
                  - z_vel_penalty - fall_penalty)

        info = {
            "healthy_reward": -10.0 if terminated else healthy_reward,
            "x_position": x_after,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "z_velocity": z_velocity,
            "lateral_penalty": lateral_penalty,
            "lateral_position_penalty": lateral_position_penalty,
            "z_vel_penalty": z_vel_penalty,
            "alignment_reward": alignment_reward,
            "x_exp_reward": x_exp_reward,
            "velocity_reward": velocity_reward,
            "facing_x": facing_x,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    # Flat platform bounds from flat_world.xml:
    #   <geom pos="70 0 0" size="80 5 0.1" type="box"/>
    #   x half-extent=80: back edge at 70-80=-10, front edge at 70+80=150
    #   y half-extent=5: sides at ±5
    _PLATFORM_X_BACK:  float = -10.0
    _PLATFORM_Y_ABS:   float =   5.0
    _Z_FALL_THRESHOLD: float =  -0.5  # terminate if torso drops 0.5 m below starting height

    def _is_terminated(self) -> bool:
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        z = float(self.data.qpos[2])
        return (
            not np.isfinite(self.state_vector()).all()
            or z < 0.3
            or z > 0.8
            or z - self._init_z < self._Z_FALL_THRESHOLD
            or x < self._PLATFORM_X_BACK
            or abs(y) > self._PLATFORM_Y_ABS
        )

    def _get_obs(self):
        return np.concatenate((self.data.qpos.flat[2:], self.data.qvel.flat.copy()))

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self._init_z = float(self.data.qpos[2])
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}
