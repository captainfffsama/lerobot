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
import sys
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_joycon import JoyconTeleopConfig, BiJoyconTeleopConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError



class JoyconTeleop(Teleoperator):
    """
    Optimized version of JoyconTeleop for performance with full 6DOF control.
    """

    config_class = JoyconTeleopConfig
    name = "joycon"

    def __init__(self, config: JoyconTeleopConfig):
        super().__init__(config)
        self.config = config
        self.joycon = None
        self.x_step_size = config.x_step_size
        self.y_step_size = config.y_step_size
        self.z_step_size = config.z_step_size
        self.yaw_step_size = config.yaw_step_size
        self.pitch_step_size = config.pitch_step_size
        self.roll_step_size = config.roll_step_size
        self._is_calibrated = False

        self.action_names = (
            ("delta_x", "delta_y", "delta_z", "delta_yaw", "delta_pitch", "delta_roll", "gripper")
            if self.config.use_gripper
            else ("delta_x", "delta_y", "delta_z", "delta_yaw", "delta_pitch", "delta_roll")
        )

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the joycon."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        from .joycon_driver import JoyconRobotics

        self.joycon = JoyconRobotics(
            device=self.config.device,
            gripper_open=self.config.gripper_open,
            gripper_close=self.config.gripper_close,
            gripper_state=self.config.gripper_state,
            horizontal_stick_mode=self.config.horizontal_stick_mode,
            close_y=self.config.close_y,
            limit_dof=self.config.limit_dof,
            glimit=self.config.glimit,
            offset_position_m=self.config.offset_position_m,
            offset_euler_rad=self.config.offset_euler_rad,
            euler_reverse=self.config.euler_reverse,
            direction_reverse=self.config.direction_reverse,
            dof_speed=self.config.dof_speed,
            rotation_filter_alpha_rate=self.config.rotation_filter_alpha_rate,
            common_rad=self.config.common_rad,
            lerobot=self.config.lerobot,
            pitch_down_double=self.config.pitch_down_double,
            without_rest_init=self.config.without_rest_init,
            pure_xz=self.config.pure_xz,
            pure_x=self.config.pure_x,
            pure_y=self.config.pure_y,
            change_down_to_gripper=self.config.change_down_to_gripper,
        )

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} {self.config.device} is not connected.")

        self.joycon.reset_joycon()
        pose, gripper, control_button = self.joycon.get_control(out_format="euler_deg")
        self.previous_x = pose[0]
        self.previous_y = pose[1]
        self.previous_z = pose[2]
        self.previous_roll = pose[3]
        self.previous_pitch = pose[4]
        self.previous_yaw = pose[5]
        logging.info("Joycon calibrated.")
        self._is_calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def feedback_features(self) -> dict:
        return {}

    def configure(self) -> None:
        """Configure the joycon."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the joycon."""
        # Joycon doesn't support feedback
        pass

    @property
    def action_features(self) -> dict:
        return {action_name: float for action_name in self.action_names}

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} {self.config.device} is not connected.")
        # Get movement deltas from the controller
        pose, gripper, control_button = self.joycon.get_control(out_format="euler_deg")
        # TODO: need dead zone control
        delta_x = pose[0] - self.previous_x
        delta_y = pose[1] - self.previous_y
        delta_z = pose[2] - self.previous_z
        delta_roll = pose[3] - self.previous_roll
        delta_pitch = pose[4] - self.previous_pitch
        delta_yaw = pose[5] - self.previous_yaw

        delta_x = 0 if abs(delta_x) < self.config.dead_zone["delta_x"] else delta_x
        delta_y = 0 if abs(delta_y) < self.config.dead_zone["delta_y"] else delta_y
        delta_z = 0 if abs(delta_z) < self.config.dead_zone["delta_z"] else delta_z
        delta_roll = 0 if abs(delta_roll) < self.config.dead_zone["delta_roll"] else float(delta_roll)
        delta_pitch = 0 if abs(delta_pitch) < self.config.dead_zone["delta_pitch"] else float(delta_pitch)
        delta_yaw = 0 if abs(delta_yaw) < self.config.dead_zone["delta_yaw"] else float(delta_yaw)

        self.previous_x = pose[0]
        self.previous_y = pose[1]
        self.previous_z = pose[2]
        self.previous_roll = pose[3]
        self.previous_pitch = pose[4]
        self.previous_yaw = pose[5]

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "delta_yaw": delta_yaw,
            "delta_pitch": delta_pitch,
            "delta_roll": delta_roll,
        }

        # Default gripper action is to stay
        if self.config.use_gripper:
            action_dict["gripper"] = gripper

        return action_dict

    def disconnect(self):
        if self.is_connected:
            self.joycon.disconnect()
        self.joycon = None

    @property
    def is_connected(self) -> bool:
        return self.joycon is not None and self.joycon.running


class BiJoyconTeleop(Teleoperator):
    config_class = BiJoyconTeleopConfig
    name = "bijoycon"

    def __init__(self, config: BiJoyconTeleopConfig):
        super().__init__(config)
        self.left_joycon = JoyconTeleop(config.left_joycon_config)
        self.right_joycon = JoyconTeleop(config.right_joycon_config)
        self.action_names = tuple(
            [f"l_{name}" for name in self.left_joycon.action_names]
            + [f"r_{name}" for name in self.right_joycon.action_names]
        )

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the joycon."""
        if not self.left_joycon.is_connected:
            self.left_joycon.connect(calibrate=False)
        if not self.right_joycon.is_connected:
            self.right_joycon.connect(calibrate=False)
        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        self.left_joycon.calibrate()
        self.right_joycon.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self.left_joycon.is_calibrated and self.right_joycon.is_calibrated

    @property
    def feedback_features(self) -> dict:
        raise NotImplementedError("BiJoyconTeleop does not support feedback.")

    def configure(self) -> None:
        """Configure the joycon."""
        # No additional configuration needed
        raise NotImplementedError("BiJoyconTeleop does not support configuration.")

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the joycon."""
        # Joycon doesn't support feedback
        raise NotImplementedError("BiJoyconTeleop does not support feedback.")

    @property
    def action_features(self) -> dict:
        return {action_name: float for action_name in self.action_names}

    def get_action(self) -> dict[str, Any]:
        l_action = self.left_joycon.get_action()
        r_action = self.right_joycon.get_action()
        action_dict = {f"l_{k}": v for k, v in l_action.items()}
        for k, v in r_action.items():
            action_dict[f"r_{k}"] = v
        return action_dict

    def disconnect(self):
        if self.is_connected:
            self.left_joycon.disconnect()
            self.right_joycon.disconnect()
        self.left_joycon = None
        self.right_joycon = None

    @property
    def is_connected(self) -> bool:
        return self.left_joycon.is_connected and self.right_joycon.is_connected
