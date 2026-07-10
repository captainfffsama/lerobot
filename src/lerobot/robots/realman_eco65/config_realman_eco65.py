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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

__all__ = [
    "RealmanECO65CartesianDeltaConfig",
    "RealmanECO65JointDeltaConfig",
    "RealmanECO65RobotConfigBase",
]


@dataclass
class RealmanECO65RobotConfigBase:
    """Shared configuration fields for RealMan ECO65 robot drivers."""

    ip_address: str
    port: int = 8080
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    move_speed: int = 20
    move_radius: int = 0
    move_block: int = 1
    # RealMan frame_type: 0 = work coordinate frame, 1 = tool coordinate frame.
    cartesian_frame_type: int = 0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("realman_eco65_joint_delta")
@dataclass
class RealmanECO65JointDeltaConfig(RobotConfig, RealmanECO65RobotConfigBase):
    pass


@RobotConfig.register_subclass("realman_eco65_cartesian_delta")
@dataclass
class RealmanECO65CartesianDeltaConfig(RobotConfig, RealmanECO65RobotConfigBase):
    pass
