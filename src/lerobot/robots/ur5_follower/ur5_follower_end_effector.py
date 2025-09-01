# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any, List, Optional

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation


from lerobot.errors import DeviceNotConnectedError


from . import UR5Follower
from .config_ur5_follower import UR5FollowerEndEffectorConfig

import lerobot.debug_tools as D

logger = logging.getLogger(__name__)
EE_FRAME = "gripper_tip"


# TODO: 等着吧
class UR5FollowerEndEffector(UR5Follower):
    """
    URFollower robot with end-effector space control.

    This robot inherits from URFollower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = UR5FollowerEndEffectorConfig
    name = "ur5_follower_end_effector"

    def __init__(self, config: UR5FollowerEndEffectorConfig):
        super().__init__(config)
        self.motors_names = (
            ("ee_x", "ee_y", "ee_z", "roll", "pitch", "yaw")
            if not self.with_gripper
            else ("ee_x", "ee_y", "ee_z", "roll", "pitch", "yaw", "gripper")
        )
        self.action_names = (
            ("delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw")
            if not self.with_gripper
            else ("delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper")
        )

        self.action_bound_max = np.array(self.config.end_effector_bounds["max"][:6], dtype=np.float32)
        self.action_bound_min = np.array(self.config.end_effector_bounds["min"][:6], dtype=np.float32)

        self.unwarpping_angle = [0.001, 0.001, 0.001]

    def esure_safe_action(self, action: np.ndarray, delta_effector_bounds: list[float]) -> np.ndarray:
        """
        Clip the action to the delta_effector_bounds.

        Args:
            action: The action to clip.
            delta_effector_bounds: The bounds to clip the action to.

        Returns:
            The clipped action.
        """
        danger_flag = np.abs(action[:6]) > np.array(delta_effector_bounds)
        if danger_flag.any():
            robot_pos = np.clip(action[:6], -np.array(delta_effector_bounds), np.array(delta_effector_bounds))
            danger_action_names = np.array(self.motors_names[:6])[danger_flag].tolist()
            logger.warning(f"danger action: {danger_action_names}")
            if action.shape[0] == 7:
                robot_pos = np.concatenate([robot_pos, action[6:]], axis=0)
            return robot_pos
        else:
            return action

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {action_name: float for action_name in self.action_names}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'delta_x', 'delta_y', 'delta_z' for end-effector control
                   or a numpy array with [delta_x, delta_y, delta_z]

        Returns:
            The joint-space action that was sent to the motors
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        tcp_raw = self.r_inter.getActualTCPPose()  # [x,y,z, rx,ry,rz] (rx,ry,rz 为 rotvec)
        current_pos_lin = np.array(tcp_raw[:3], dtype=np.float32)
        current_rotvec = np.array(tcp_raw[3:6], dtype=np.float32)

        goal_delta_pos = np.array([action[name] for name in self.action_names], dtype=np.float32)

        delta_pos = self.esure_safe_action(goal_delta_pos, self.config.delta_effector_bounds)

        delta_euler = delta_pos[3:6]  # (d_roll, d_pitch, d_yaw) 约定为 'zyx' 顺序的三个分量
        R_current = Rotation.from_rotvec(current_rotvec)
        R_delta = Rotation.from_euler("zxy", delta_euler, degrees=False)
        R_new = R_current * R_delta
        new_rotvec = R_new.as_rotvec()

        # 线性部分直接累加
        goal_pos = np.concatenate([current_pos_lin + delta_pos[:3], new_rotvec], axis=0)

        if self.with_gripper:
            # 直接使用动作里给的 gripper 值（0~1）
            goal_pos = np.append(goal_pos, delta_pos[-1])

        # 仅对位置做边界裁剪（避免欧拉裁剪引入不连续）
        end_beyond_flag = np.logical_or(
            goal_pos[:3] < self.action_bound_min[:3], goal_pos[:3] > self.action_bound_max[:3]
        )
        if end_beyond_flag.any():
            danger_action_names = np.array(self.motors_names[:3])[end_beyond_flag].tolist()
            logger.warning(
                f"End-effector position {danger_action_names} is beyond bounds {self.action_bound_min[:3]} - {self.action_bound_max[:3]}"
            )
            goal_pos[:3] = np.clip(goal_pos[:3], self.action_bound_min[:3], self.action_bound_max[:3])

        self.command_eef_pos(goal_pos, orientation_is_rotvec=True, **self.move_params)

        tcp_after = self.r_inter.getActualTCPPose()
        lin_after = np.array(tcp_after[:3], dtype=np.float32)
        rot_after = np.array(tcp_after[3:6], dtype=np.float32)
        actual_delta_pos_d = dict(zip(self.action_names, goal_delta_pos.tolist(), strict=True))
        return actual_delta_pos_d

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        eef_pos = self.get_eef_pos()
        obs_dict = dict(zip(self.motors_names, eef_pos, strict=True))
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def get_eef_pos(self, return_np=False) -> List[float] | np.ndarray:
        """
        Get the current end-effector position in the robot's base frame.
        Returns a list of [x, y, z, roll, pitch, yaw] in degrees.
        """
        robot_tcp_pose: list[float] = self.r_inter.getActualTCPPose()
        angles = np.array(robot_tcp_pose[3:6])
        robot_tcp_euler = get_stable_euler_from_rotvec(angles)
        robot_tcp_euler = np.unwrap([self.unwarpping_angle, robot_tcp_euler], axis=0)[1].tolist()
        self.unwarpping_angle = robot_tcp_euler
        robot_tcp_euler = [robot_tcp_pose[0], robot_tcp_pose[1], robot_tcp_pose[2]] + robot_tcp_euler

        if self.with_gripper:
            gripper_pos = self.gripper.get_current_position()
            assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
            gripper_pos = gripper_pos / 255.0
            pos = np.append(robot_tcp_euler, gripper_pos)
        else:
            pos = robot_tcp_euler
        if return_np:
            return pos
        else:
            return pos.tolist()

    def command_eef_pos(
        self,
        eef_pos: np.ndarray,
        move_mode: str = "servo",
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        dt: Optional[float] = None,
        lookahead_time: Optional[float] = None,
        gain: Optional[int] = None,
        gripper_speed: Optional[int] = None,
        gripper_force: Optional[int] = None,
        orientation_is_rotvec: bool = True,
    ) -> None:
        """

        Args:
            eef_pos (np.ndarray): _description_
            move_mode (str, optional): _description_. Defaults to "servo".
            velocity (Optional[float], optional): _description_. Defaults to None.
            acceleration (Optional[float], optional): _description_. Defaults to None.
            dt (Optional[float], optional): _description_. Defaults to None.
            lookahead_time (Optional[float], optional): _description_. Defaults to None.
            gain (Optional[int], optional): _description_. Defaults to None.
            gripper_speed (Optional[int], optional): _description_. Defaults to None.
            gripper_force (Optional[int], optional): _description_. Defaults to None.
            orientation_is_rotvec (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
        """

        robot_eef = eef_pos[:6].copy()
        if not orientation_is_rotvec:
            r = Rotation.from_euler("zxy", robot_eef[3:6], degrees=False)
            robot_eef[3:6] = r.as_rotvec().tolist()

        t_start = self.robot.initPeriod()

        # 使用传入参数或默认参数
        velocity = velocity if velocity is not None else self.move_params["velocity"]
        acceleration = acceleration if acceleration is not None else self.move_params["acceleration"]
        dt = dt if dt is not None else self.move_params["dt"]
        lookahead_time = lookahead_time if lookahead_time is not None else self.move_params["lookahead_time"]
        gain = gain if gain is not None else self.move_params["gain"]

        if move_mode == "moveit":
            self.robot.moveL(robot_eef, velocity, acceleration)
        elif move_mode == "servo":
            self.robot.servoL(robot_eef, velocity, acceleration, dt, lookahead_time, gain)
        else:
            raise ValueError(f"Unknown move model: {move_mode}. Use 'servo' or 'moveit'.")
        if self.with_gripper:
            gripper_pos = eef_pos[-1] * 255
            gripper_speed = gripper_speed if gripper_speed is not None else self.move_params["speed"]
            gripper_force = gripper_force if gripper_force is not None else self.move_params["force"]
            self.gripper.move(gripper_pos, gripper_speed, gripper_force)
        self.robot.waitPeriod(t_start)


def normalize_angle_robust(angle):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def get_stable_euler_from_rotvec(rotvec):
    r = Rotation.from_rotvec(rotvec, degrees=False)
    euler_angles_rad = r.as_euler("zyx")
    stable_euler_angles = [normalize_angle_robust(angle) for angle in euler_angles_rad]
    return stable_euler_angles
