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

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


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

        current_pos = self.get_eef_pos()
        goal_delta_pos = np.array([action[name] for name in self.action_names], dtype=np.float32)
        delta_pos = self.esure_safe_action(goal_delta_pos, self.config.delta_effector_bounds)

        goal_pos = np.array(current_pos) + delta_pos
        if self.with_gripper:
            goal_pos[-1] = delta_pos[-1]  # Gripper position is directly set by the action

        # check xyz and degre beyond end bounds
        end_beyond_flag = np.logical_or(
            goal_pos[:6] < self.action_bound_min, goal_pos[:6] > self.action_bound_max
        )
        if end_beyond_flag.any():
            danger_action_names = np.array(self.motors_names[:6])[end_beyond_flag].tolist()
            logger.warning(
                f"End-effector position {danger_action_names} is beyond bounds {self.action_bound_min} - {self.action_bound_max}"
            )
            goal_pos[:6] = np.clip(goal_pos[:6], self.action_bound_min, self.action_bound_max)
        actual_delta_pos = (goal_pos - current_pos).tolist()
        if self.with_gripper:
            actual_delta_pos[-1] = float(goal_pos[-1])
        self.command_eef_pos(goal_pos, **self.move_params)
        print(f"{self} commanded end-effector position: {goal_pos}")
        return dict(zip(self.action_names, actual_delta_pos, strict=True))

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

    def get_eef_pos(self) -> List[float]:
        """
        Get the current end-effector position in the robot's base frame.
        Returns a list of [x, y, z, roll, pitch, yaw] in degrees.
        """
        robot_joints = self.r_inter.getActualTCPPose()
        if self.with_gripper:
            gripper_pos = self.gripper.get_current_position()
            assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
            gripper_pos = gripper_pos / 255.0
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
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
    ) -> None:
        """Command the leader robot to a given state.

        Args:
            eef_pos (np.ndarray): The state to command the leader robot to.
        """

        robot_eef = eef_pos[:6]
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
