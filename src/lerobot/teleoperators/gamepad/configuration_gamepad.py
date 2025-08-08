#!/usr/bin/env python

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

from dataclasses import dataclass,field
import math

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gamepad")
@dataclass
class GamepadTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("gamepadoptim")
@dataclass
class GamepadTeleopOptimConfig(TeleoperatorConfig):
    use_gripper: bool = True
    x_step_size: float = 0.02  # unit:m
    y_step_size: float = 0.02
    z_step_size: float = 0.02

    yaw_step_deg: float = 3  # unit:deg
    pitch_step_deg: float = 3
    roll_step_deg: float = 3

    tele2joy_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "delta_x": "delta_y",
            "delta_y": "delta_x",
            "delta_z": "delta_z",
            "delta_roll": "delta_roll",
            "delta_pitch": "delta_pitch",
            "delta_yaw": "delta_yaw",
        },  # x, y, z, roll, pitch, yaw
    )
    # Mapping from teleoperator actions to joystick buttons

    def __post_init__(self):
        # Ensure step sizes are positive
        if self.x_step_size <= 0 or self.y_step_size <= 0 or self.z_step_size <= 0:
            raise ValueError("Step sizes must be positive values.")
        self.yaw_step_size: float = math.pi / 180 * self.yaw_step_deg  # unit: r
        self.pitch_step_size: float = math.pi / 180 * self.pitch_step_deg
        self.roll_step_size: float = math.pi / 180 * self.roll_step_deg
