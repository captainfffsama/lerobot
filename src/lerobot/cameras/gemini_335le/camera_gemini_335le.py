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
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pyorbbecsdk_available, require_package

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_gemini_335le import Gemini335LECameraConfig

if TYPE_CHECKING or _pyorbbecsdk_available:
    import pyorbbecsdk as ob
else:
    ob = None

logger = logging.getLogger(__name__)


def _iter_list_like(items: Any):
    if items is None:
        return
    try:
        for item in items:
            yield item
        return
    except TypeError:
        pass

    count = None
    for attr in ("get_count", "count"):
        fn = getattr(items, attr, None)
        if callable(fn):
            count = int(fn())
            break

    if count is None:
        raise TypeError(f"Unsupported list-like object: {type(items)}")

    for index in range(count):
        yield items[index]


class Gemini335LECamera(Camera):
    """Orbbec Gemini 335Le camera wrapper following the LeRobot camera interface."""

    def __init__(self, config: Gemini335LECameraConfig):
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
        super().__init__(config)

        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.pipeline: ob.Pipeline | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self._color_stream_format: Any | None = None
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.capture_width: int | None = self.width
        self.capture_height: int | None = self.height
        if self.width is not None and self.height is not None and self.rotation in (
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ):
            self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number_or_name})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.thread is not None and self.thread.is_alive()

    @staticmethod
    def _device_info_as_dict(device: Any) -> dict[str, Any]:
        device_info = device.get_device_info()
        info: dict[str, Any] = {
            "name": getattr(device_info, "get_name", lambda: "unknown")(),
            "type": "Gemini335LE",
            "id": getattr(device_info, "get_serial_number", lambda: "unknown")(),
        }
        for attr_name in ("get_pid", "get_vid", "get_firmware_version", "get_connection_type"):
            getter = getattr(device_info, attr_name, None)
            if callable(getter):
                value = getter()
                if value is not None:
                    info[attr_name.removeprefix("get_")] = value

        sensor_types = []
        sensor_list = getattr(device, "get_sensor_list", None)
        if callable(sensor_list):
            try:
                for sensor in _iter_list_like(sensor_list()):
                    sensor_type = getattr(sensor, "get_type", lambda: "unknown")()
                    sensor_types.append(str(sensor_type))
            except Exception:
                pass
        if sensor_types:
            info["sensor_types"] = sensor_types
        return info

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")

        found_cameras_info: list[dict[str, Any]] = []
        context = ob.Context()
        devices = context.query_devices()
        for index in range(int(devices.get_count())):
            device = devices.get_device_by_index(index)
            found_cameras_info.append(Gemini335LECamera._device_info_as_dict(device))
        return found_cameras_info

    def _find_device(self) -> Any:
        context = ob.Context()
        devices = context.query_devices()
        matches: list[Any] = []
        target = self.serial_number_or_name.lower()

        for index in range(int(devices.get_count())):
            device = devices.get_device_by_index(index)
            info = device.get_device_info()
            name = getattr(info, "get_name", lambda: "")().lower()
            serial = getattr(info, "get_serial_number", lambda: "")().lower()
            if target in {name, serial}:
                matches.append(device)

        if not matches:
            raise ConnectionError(
                f"Failed to find {self}. Run `lerobot-find-cameras gemini335le` to inspect available devices."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Device identifier '{self.serial_number_or_name}' is ambiguous for {self}. "
                "Please use the camera serial number instead of the friendly name."
            )
        return matches[0]

    def _select_stream_profile(
        self,
        pipeline: Any,
        sensor_type: Any,
        width: int | None,
        height: int | None,
        fps: int | None,
        ob_format: Any,
    ) -> Any:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        try:
            return profile_list.get_video_stream_profile(width or 0, height or 0, ob_format, fps or 0)
        except Exception:
            return profile_list.get_default_video_stream_profile()

    def _postprocess_color(self, frame: Any) -> NDArray[Any]:
        width = int(frame.get_width())
        height = int(frame.get_height())
        raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
        encoded_color = self._color_stream_format in (
            getattr(ob.OBFormat, "MJPG", None),
            getattr(ob.OBFormat, "H264", None),
        )

        if encoded_color:
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to decode compressed color frame for {self}.")
        else:
            image = raw.reshape((height, width, 3))

        if self.color_mode == ColorMode.RGB:
            if encoded_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_mode == ColorMode.BGR:
            if not encoded_color:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif self.color_mode != ColorMode.RGB:
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            image = cv2.rotate(image, self.rotation)

        if self.width is None or self.height is None:
            self.height, self.width = int(image.shape[0]), int(image.shape[1])
            self.capture_height, self.capture_width = self.height, self.width
        return image

    def _postprocess_depth(self, frame: Any) -> NDArray[Any]:
        width = int(frame.get_width())
        height = int(frame.get_height())
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_scale = float(getattr(frame, "get_depth_scale", lambda: 1.0)())
        depth = (raw.astype(np.float32) * depth_scale).astype(np.float32)

        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            depth = cv2.rotate(depth, self.rotation)

        if depth.ndim == 2:
            depth = depth[..., np.newaxis]

        if self.width is None or self.height is None:
            self.height, self.width = int(depth.shape[0]), int(depth.shape[1])
            self.capture_height, self.capture_width = self.height, self.width
        return depth

    def _read_from_hardware(self, timeout_ms: int = 1000) -> Any:
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError(f"{self} failed to read frames from the camera pipeline.")
        return frames

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        device = self._find_device()
        try:
            self.pipeline = ob.Pipeline(device)
            device_info = device.get_device_info()
            connection_type = str(getattr(device_info, "get_connection_type", lambda: "")()).lower()
            self._color_stream_format = ob.OBFormat.MJPG if "ethernet" in connection_type else ob.OBFormat.RGB

            config = ob.Config()
            if self.use_rgb:
                color_profile = self._select_stream_profile(
                    self.pipeline,
                    ob.OBSensorType.COLOR_SENSOR,
                    self.width,
                    self.height,
                    self.fps,
                    self._color_stream_format,
                )
                config.enable_stream(color_profile)
            if self.use_depth:
                depth_profile = self._select_stream_profile(
                    self.pipeline,
                    ob.OBSensorType.DEPTH_SENSOR,
                    self.width,
                    self.height,
                    self.fps,
                    ob.OBFormat.Y16,
                )
                config.enable_stream(depth_profile)
            if self.use_rgb and self.use_depth:
                aggregate_mode = getattr(ob, "OBFrameAggregateOutputMode", None)
                if aggregate_mode is not None and hasattr(config, "set_frame_aggregate_output_mode"):
                    config.set_frame_aggregate_output_mode(aggregate_mode.FULL_FRAME_REQUIRE)
                if hasattr(self.pipeline, "enable_frame_sync"):
                    self.pipeline.enable_frame_sync()

            if self.use_rgb and (self.width is None or self.height is None or self.fps is None):
                self.width = int(color_profile.get_width())
                self.height = int(color_profile.get_height())
                self.fps = int(color_profile.get_fps())
                self.capture_width, self.capture_height = self.width, self.height
                if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                    self.capture_width, self.capture_height = self.height, self.width
            elif self.use_depth and (self.width is None or self.height is None or self.fps is None):
                self.width = int(depth_profile.get_width())
                self.height = int(depth_profile.get_height())
                self.fps = int(depth_profile.get_fps())
                self.capture_width, self.capture_height = self.width, self.height
                if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                    self.capture_width, self.capture_height = self.height, self.width

            self.pipeline.start(config)
        except Exception as exc:
            self.pipeline = None
            raise ConnectionError(f"Failed to open {self}. Run `lerobot-find-cameras gemini335le` to inspect devices.") from exc

        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            warmup_read = self.async_read if self.use_rgb else self.async_read_depth
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                warmup_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.05)

            with self.frame_lock:
                if self.use_rgb and self.latest_color_frame is None:
                    raise ConnectionError(f"{self} failed to capture a color frame during warmup.")
                if self.use_depth and self.latest_depth_frame is None:
                    raise ConnectionError(f"{self} failed to capture a depth frame during warmup.")

        logger.info(f"{self} connected.")

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not stop_event.is_set():
            try:
                frames = self._read_from_hardware(timeout_ms=200)
                capture_time = time.perf_counter()

                color_frame = frames.get_color_frame() if self.use_rgb else None
                depth_frame = frames.get_depth_frame() if self.use_depth else None

                with self.frame_lock:
                    if color_frame is not None:
                        self.latest_color_frame = self._postprocess_color(color_frame)
                    if depth_frame is not None:
                        self.latest_depth_frame = self._postprocess_depth(depth_frame)
                    self.latest_timestamp = capture_time
                    self.new_frame_event.set()
            except Exception as exc:
                if stop_event.is_set():
                    break
                logger.debug(f"{self} read loop error: {exc}")
                time.sleep(0.05)

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _async_read(self, timeout_ms: float, read_depth: bool = False) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        return frame

    def _read_latest(self, max_age_ms: int, read_depth: bool = False) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return frame

    def _read(self, read_depth: bool = False) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()
        return self._async_read(timeout_ms=1000, read_depth=read_depth)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        if self.use_rgb:
            return self._read()
        if self.use_depth:
            return self._read(read_depth=True)
        raise RuntimeError(f"{self} is not configured to read any stream.")

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.use_rgb:
            return self._async_read(timeout_ms=timeout_ms)
        if self.use_depth:
            return self.async_read_depth(timeout_ms=timeout_ms)
        raise RuntimeError(f"{self} is not configured to read any stream.")

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.use_rgb:
            return self._read_latest(max_age_ms=max_age_ms)
        if self.use_depth:
            return self.read_latest_depth(max_age_ms=max_age_ms)
        raise RuntimeError(f"{self} is not configured to read any stream.")

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth - camera was configured with use_depth=False.")
        self.new_frame_event.clear()
        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth - camera was configured with use_depth=False.")
        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth - camera was configured with use_depth=False.")
        return self._read_latest(max_age_ms=max_age_ms, read_depth=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_read_thread()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        logger.info(f"{self} disconnected.")
