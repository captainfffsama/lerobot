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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["Gemini335LECameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("gemini_335le")
@dataclass
class Gemini335LECameraConfig(CameraConfig):
    """Configuration for an Orbbec Gemini 335Le camera."""

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
