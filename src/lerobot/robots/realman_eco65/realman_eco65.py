# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _robotic_arm_available, require_package

from ..config import RobotConfig
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_realman_eco65 import (
    RealmanECO65CartesianDeltaConfig,
    RealmanECO65JointDeltaConfig,
    RealmanECO65RobotConfigBase,
)

if TYPE_CHECKING or _robotic_arm_available:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
else:
    RoboticArm = None
    rm_thread_mode_e = None

logger = logging.getLogger(__name__)

REALMAN_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
REALMAN_POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def _sequence_from_value(value: Any, names: Sequence[str] | None = None) -> list[float] | None:
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        if names is not None and all(name in value for name in names):
            return [float(value[name]) for name in names]
        return [float(item) for item in value.values()]

    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]

    try:
        return [float(item) for item in list(value)]
    except TypeError:
        return None


class RealmanECO65Base(Robot):
    """Shared logic for RealMan ECO65 robot variants."""

    config_class = RobotConfig
    name = "realman_eco65"

    def __init__(self, config: RealmanECO65RobotConfigBase):
        require_package("robotic-arm", extra="realman", import_name="Robotic_Arm")
        super().__init__(config)

        self.config = config
        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(config.cameras)

        self.arm: RoboticArm | None = None
        self._connected = False

    @property
    def _joint_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in REALMAN_JOINT_NAMES}

    @property
    def _pose_features(self) -> dict[str, type]:
        return {f"ee_pose.{name}": float for name in REALMAN_POSE_NAMES}

    @property
    def _camera_features(self) -> dict[str, tuple[int | None, int | None, int]]:
        features: dict[str, tuple[int | None, int | None, int]] = {}
        for cam_name, cam in self.cameras.items():
            cfg = self.config.cameras[cam_name]
            if getattr(cfg, "use_rgb", True):
                features[cam_name] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam_name}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joint_features, **self._pose_features, **self._camera_features}

    @property
    def is_connected(self) -> bool:
        return self._connected and self.arm is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s does not require a calibration step.", self.__class__.__name__)

    def _make_arm(self) -> RoboticArm:
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(self.config.ip_address, self.config.port)

        handle_id = getattr(getattr(handle, "contents", None), "id", None)
        if handle_id is None:
            handle_id = getattr(getattr(arm, "handle", None), "contents", None)
            handle_id = getattr(handle_id, "id", None)

        if handle_id in (None, -1):
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                logger.debug("Ignoring RealMan delete failure after connect error.", exc_info=True)
            raise ConnectionError(
                f"Failed to connect to {self.__class__.__name__} at "
                f"{self.config.ip_address}:{self.config.port} (handle_id={handle_id})."
            )
        return arm

    def _cleanup_partial_connection(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                try:
                    cam.disconnect()
                except Exception:
                    logger.debug("Ignoring camera disconnect failure during cleanup.", exc_info=True)

        if self.arm is not None:
            try:
                if self.config.disable_torque_on_disconnect:
                    self.arm.rm_set_arm_power(0)
            except Exception:
                logger.debug("Ignoring arm power-off failure during cleanup.", exc_info=True)
            try:
                self.arm.rm_delete_robot_arm()
            except Exception:
                logger.debug("Ignoring arm delete failure during cleanup.", exc_info=True)

        self.arm = None
        self._connected = False

    def _current_state(self) -> dict[str, Any]:
        if self.arm is None:
            raise ConnectionError(f"{self} is not connected.")

        response = self.arm.rm_get_current_arm_state()
        if isinstance(response, (tuple, list)) and len(response) >= 2:
            status, payload = response[0], response[1]
            if status != 0:
                raise RuntimeError(f"{self} failed to read arm state (status={status}).")
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, (tuple, list)) and len(payload) >= 2:
                return {"joint": payload[0], "pose": payload[1]}
            raise RuntimeError(f"Unexpected arm state payload type: {type(payload)}")

        if isinstance(response, dict):
            return response

        raise RuntimeError(f"Unexpected arm state response type: {type(response)}")

    def _current_joint_positions(self, state: dict[str, Any] | None = None) -> list[float]:
        state = self._current_state() if state is None else state
        joint_values = None
        for key in ("joint", "joints", "joint_degree", "joint_position"):
            joint_values = _sequence_from_value(state.get(key), names=REALMAN_JOINT_NAMES)
            if joint_values is not None:
                break
        if joint_values is None:
            raise RuntimeError(f"Could not parse joint positions from arm state keys: {list(state.keys())}")
        if len(joint_values) < len(REALMAN_JOINT_NAMES):
            raise RuntimeError(
                f"Expected at least {len(REALMAN_JOINT_NAMES)} joint values, got {len(joint_values)}."
            )
        return joint_values[: len(REALMAN_JOINT_NAMES)]

    def _current_pose(self, state: dict[str, Any] | None = None) -> list[float]:
        state = self._current_state() if state is None else state
        pose_values = None
        for key in ("pose", "tcp_pose", "cartesian_pose", "end_pose"):
            pose_values = _sequence_from_value(state.get(key), names=REALMAN_POSE_NAMES)
            if pose_values is not None:
                break
        if pose_values is None:
            raise RuntimeError(f"Could not parse end-effector pose from arm state keys: {list(state.keys())}")
        if len(pose_values) < len(REALMAN_POSE_NAMES):
            raise RuntimeError(f"Expected at least {len(REALMAN_POSE_NAMES)} pose values, got {len(pose_values)}.")
        return pose_values[: len(REALMAN_POSE_NAMES)]

    def _read_camera_observations(self, obs_dict: dict[str, Any]) -> None:
        for cam_name, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                obs_dict[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs_dict[f"{cam_name}_depth"] = cam.read_latest_depth()

    def _connect_cameras(self) -> None:
        for cam in self.cameras.values():
            cam.connect()

    def _configure_arm(self) -> None:
        if self.arm is None:
            raise ConnectionError(f"{self} is not connected.")

        try:
            self.arm.rm_clear_system_err()
        except Exception:
            logger.debug("Ignoring RealMan clear-system error failure.", exc_info=True)

        status = self.arm.rm_set_arm_power(1)
        if status != 0:
            raise RuntimeError(f"Failed to power on {self} (status={status}).")

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm = self._make_arm()

        try:
            self._configure_arm()
            self._connect_cameras()
            self._connected = True
        except Exception:
            self._cleanup_partial_connection()
            raise

        logger.info(f"{self} connected.")

    @check_if_not_connected
    def configure(self) -> None:
        self._configure_arm()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: dict[str, Any] = {}

        state = self._current_state()
        joint_values = self._current_joint_positions(state)
        pose_values = self._current_pose(state)

        for index, joint_name in enumerate(REALMAN_JOINT_NAMES):
            obs_dict[f"{joint_name}.pos"] = joint_values[index]
        for index, pose_name in enumerate(REALMAN_POSE_NAMES):
            obs_dict[f"ee_pose.{pose_name}"] = pose_values[index]

        self._read_camera_observations(obs_dict)
        return obs_dict

    def _send_joint_goal(self, goal_joints: list[float]) -> None:
        if self.arm is None:
            raise ConnectionError(f"{self} is not connected.")
        status = self.arm.rm_movej(
            goal_joints,
            self.config.move_speed,
            self.config.move_radius,
            0,
            self.config.move_block,
        )
        if status != 0:
            raise RuntimeError(f"{self} failed to execute joint motion (status={status}).")

    def _send_cartesian_offset(self, offset_pose: list[float]) -> None:
        if self.arm is None:
            raise ConnectionError(f"{self} is not connected.")
        status = self.arm.rm_movel_offset(
            offset_pose,
            self.config.move_speed,
            self.config.move_radius,
            0,
            self.config.cartesian_frame_type,
            self.config.move_block,
        )
        if status != 0:
            raise RuntimeError(f"{self} failed to execute Cartesian motion (status={status}).")

    def _clamp_joint_goal(self, present_joints: list[float], goal_joints: list[float]) -> list[float]:
        if self.config.max_relative_target is None:
            return goal_joints

        goal_present = {
            joint_name: (goal_joints[index], present_joints[index]) for index, joint_name in enumerate(REALMAN_JOINT_NAMES)
        }
        safe_goal = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
        return [safe_goal[joint_name] for joint_name in REALMAN_JOINT_NAMES]

    def _clamp_pose_goal(self, present_pose: list[float], goal_pose: list[float]) -> list[float]:
        if self.config.max_relative_target is None:
            return goal_pose

        goal_present = {
            pose_name: (goal_pose[index], present_pose[index]) for index, pose_name in enumerate(REALMAN_POSE_NAMES)
        }
        safe_goal = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
        return [safe_goal[pose_name] for pose_name in REALMAN_POSE_NAMES]

    @check_if_not_connected
    def disconnect(self) -> None:
        self._cleanup_partial_connection()
        logger.info(f"{self} disconnected.")


class RealmanECO65JointDelta(RealmanECO65Base):
    """RealMan ECO65 arm that interprets actions as joint-space deltas."""

    config_class = RealmanECO65JointDeltaConfig
    name = "realman_eco65_joint_delta"

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.delta": float for joint in REALMAN_JOINT_NAMES}

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        state = self._current_state()
        present_joints = self._current_joint_positions(state)
        joint_delta = []
        for joint_name in REALMAN_JOINT_NAMES:
            for key in (f"{joint_name}.delta", f"{joint_name}.pos"):
                if key in action:
                    joint_delta.append(float(action[key]))
                    break
            else:
                raise KeyError(f"Missing action key for {joint_name}. Expected `*.delta`.")

        if all(abs(delta) <= 1e-9 for delta in joint_delta):
            return {f"{joint_name}.delta": 0.0 for joint_name in REALMAN_JOINT_NAMES}

        goal_joints = [present + delta for present, delta in zip(present_joints, joint_delta)]
        goal_joints = self._clamp_joint_goal(present_joints, goal_joints)
        self._send_joint_goal(goal_joints)
        return {
            f"{joint_name}.delta": goal_joints[index] - present_joints[index]
            for index, joint_name in enumerate(REALMAN_JOINT_NAMES)
        }


class RealmanECO65CartesianDelta(RealmanECO65Base):
    """RealMan ECO65 arm that interprets actions as Cartesian end-effector deltas."""

    config_class = RealmanECO65CartesianDeltaConfig
    name = "realman_eco65_cartesian_delta"

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"ee_delta.{name}": float for name in REALMAN_POSE_NAMES}

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        state = self._current_state()
        present_pose = self._current_pose(state)
        pose_delta = []
        for pose_name in REALMAN_POSE_NAMES:
            for key in (f"ee_delta.{pose_name}", f"ee_pose_delta.{pose_name}"):
                if key in action:
                    pose_delta.append(float(action[key]))
                    break
            else:
                raise KeyError(f"Missing action key for ee_delta.{pose_name}.")

        if all(abs(delta) <= 1e-9 for delta in pose_delta):
            return {f"ee_delta.{name}": 0.0 for name in REALMAN_POSE_NAMES}

        goal_pose = [present + delta for present, delta in zip(present_pose, pose_delta)]
        goal_pose = self._clamp_pose_goal(present_pose, goal_pose)
        offset_pose = [goal - present for goal, present in zip(goal_pose, present_pose)]
        self._send_cartesian_offset(offset_pose)
        return {f"ee_delta.{name}": offset_pose[index] for index, name in enumerate(REALMAN_POSE_NAMES)}
