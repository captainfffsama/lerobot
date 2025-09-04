# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from ..configs import CameraConfig


@CameraConfig.register_subclass("intelrealsensepointcloud")
@dataclass
class RealsensePointCloudCameraConfig(CameraConfig):
    name: str | None = None
    channels: int | None = None

    serial_number: int | None = None
    device_name: str = "L515"
    sync_mode: int = 0
    num_points: int = 4096
    z_far: float = 0.8
    z_near: float = 0.1
    use_grid_sampling: bool = True
    use_crop: bool = False
    img_size: int = 224
    box_bounds: list = field(
        default_factory=lambda: [
            -0.3, 0.3, -0.4, -0.05, 0.1, 0.85  # x-, x+, y-, y+, z-, z+
        ]
    )
    
    def __post_init__(self):
        # bool is stronger than is None, since it works with empty strings
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                f"One of them must be set: name or serial_number, but {self.name=} and {self.serial_number=} provided."
            )

        self.channels = 3