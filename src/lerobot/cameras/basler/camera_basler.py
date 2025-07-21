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

"""
Provides the BaslerCamera class for capturing frames from cameras using Basler.
"""

import logging
import time
import traceback
from typing import Any, Dict, List

import cv2
import numpy as np
from pypylon import pylon

from lerobot.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_basler import BaslerCameraConfig, ColorMode

logger = logging.getLogger(__name__)


class BaslerCamera(Camera):
    def __init__(self, config: BaslerCameraConfig):
        """
        Initializes the BaslerCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)
        tl_factory = pylon.TlFactory.GetInstance()
        device_info_list = tl_factory.EnumerateDevices()
        if len(device_info_list) < config.camera_idx:
            raise ValueError(
                f"Invalid camera index {config.camera_idx}. "
                f"Only {len(device_info_list)} Basler cameras found."
            )
        self.basler_cam_info = device_info_list[config.camera_idx]
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.basler_cam_info))
        # 创建图像格式转换器
        self.converter = pylon.ImageFormatConverter()

        # 设置转换器的输出像素格式为RGB8packed
        self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.camera_index = config.camera_index
        self.logs = {}
        self.logs["delta_timestamp_s"] = -1.0
        self.height = 400  # 1200
        self.width = 640  # 1920
        self.channels = 3
        self.fps = 20

        # async read
        self.img_handler = ImageHandler(self.camera.Height.Value, self.camera.Width.Value)
        self.camera.RegisterImageEventHandler(
            self.img_handler, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.basler_cam_info})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return self.camera.IsOpen() and self.camera.IsGrabbing()

    def connect(self, warmup: bool = True):
        self.camera.Open()
        self.camera.AcquisitionFrameRateEnable = True
        self.camera.AcquisitionFrameRate = self.fps
        logger.info(f"max rate: {self.camera.AcquisitionFrameRate.GetValue()}")

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        logger.info(self.camera.PixelFormat.GetSymbolics())
        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        tl_factory = pylon.TlFactory.GetInstance()
        device_info_list = tl_factory.EnumerateDevices()

        return device_info_list

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if not grab_result.GrabSucceeded():
            raise RuntimeError(f"{self} read failed (status={grab_result}).")

        processed_frame = self._postprocess_image(grab_result)
        grab_result.Release()

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, grab_result) -> np.ndarray:
        image = self.converter.Convert(grab_result)
        img = image.GetArray()
        img = cv2.resize(img, (self.width, self.height))
        return img

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        frame = cv2.resize(self.img_handler.img_sum, (self.width, self.height))
        return frame

    def disconnect(self):
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self.camera.DeregisterImageEventHandler(self.img_handler)
        self.camera.StopGrabbing()
        if self.camera.IsOpen():
            self.camera.Close()

        logger.info(f"{self} disconnected.")


class ImageHandler(pylon.ImageEventHandler):
    def __init__(self, height, width, convert):
        super().__init__()
        self.img_sum = np.zeros((height, width), dtype=np.uint16)
        self.convert = convert

    def OnImageGrabbed(self, camera, grab_result):
        """we get called on every image
        !! this code is run in a pylon thread context
        always wrap your code in the try .. except to capture
        errors inside the grabbing as this can't be properly reported from
        the background thread to the foreground python code
        """
        try:
            if grab_result.GrabSucceeded():
                # check image contents
                image = self.converter.Convert(grab_result)
                img = image.GetArray()
                self.img_sum += img
            else:
                raise RuntimeError("Grab Failed")
        except Exception as e:
            traceback.print_exc()
