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

from dataclasses import dataclass, field
import math

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("joycon")
@dataclass
class JoyconTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    device: str = "right"  # "left" or "right" joycon
    gripper_open: float = 1.0
    gripper_close: float = 0.0
    gripper_state: float = 1.0

    x_step_size: float = 0.02  # unit:m
    y_step_size: float = 0.02
    z_step_size: float = 0.02

    yaw_step_deg: float = 3  # unit:deg
    pitch_step_deg: float = 3
    roll_step_deg: float = 3

    # Joycon specific parameters
    horizontal_stick_mode: str = "y"
    close_y: bool = False
    limit_dof: bool = False
    glimit: list = field(
        default_factory=lambda: [
            # [0.125, -0.4, 0.046, -3.1, -1.5, -1.57],
            # [0.380, 0.4, 0.23, 3.1, 1.5, 1.57],
            [-10000, -10000, -10000, -10000, -10000, -10000],
            [10000, 10000, 10000, 10000, 10000, 10000],
        ]
    )
    offset_position_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    offset_euler_rad: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    euler_reverse: list = field(default_factory=lambda: [1, 1, 1])
    direction_reverse: list = field(default_factory=lambda: [1, 1, 1])
    dof_speed: list = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    rotation_filter_alpha_rate: float = 1.0
    common_rad: bool = True
    lerobot: bool = False
    pitch_down_double: bool = False
    without_rest_init: bool = False
    pure_xz: bool = True
    pure_x: bool = True
    pure_y: bool = True
    change_down_to_gripper: bool = False

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
    dead_zone: dict[str, float] = field(
        default_factory=lambda: {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_roll": 1.0,
            "delta_pitch": 1.0,
            "delta_yaw": 1.0,
        },
    )

    def __post_init__(self):
        # Ensure step sizes are positive
        if self.x_step_size <= 0 or self.y_step_size <= 0 or self.z_step_size <= 0:
            raise ValueError("Step sizes must be positive values.")
        self.yaw_step_size: float = math.pi / 180 * self.yaw_step_deg  # unit: r
        self.pitch_step_size: float = math.pi / 180 * self.pitch_step_deg
        self.roll_step_size: float = math.pi / 180 * self.roll_step_deg


@TeleoperatorConfig.register_subclass("bijoycon")
@dataclass
class BiJoyconTeleopConfig(TeleoperatorConfig):
    left_joycon_config: JoyconTeleopConfig = field(default_factory=JoyconTeleopConfig)
    right_joycon_config: JoyconTeleopConfig = field(default_factory=JoyconTeleopConfig)

    def __post_init__(self):
        self.left_joycon_config.device = "left"
        self.right_joycon_config.device = "right"
