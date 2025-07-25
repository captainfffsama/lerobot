#!/usr/bin/env python

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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_opencv.py::test_connect
# ```

from pathlib import Path

import numpy as np
import pytest

from lerobot.cameras.basler import BaslerCamera,BaslerCameraConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.debug_tools import show_img

# NOTE(Steven): more tests + assertions?
TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
DEFAULT_PNG_FILE_PATH = TEST_ARTIFACTS_DIR / "image_160x120.png"
TEST_IMAGE_SIZES = ["128x128", "160x120", "320x180", "480x270"]
TEST_IMAGE_PATHS = [TEST_ARTIFACTS_DIR / f"image_{size}.png" for size in TEST_IMAGE_SIZES]


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config =BaslerCameraConfig(camera_idx=0)

    _ = BaslerCamera(config)


def test_connect():
    config =BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)

    camera.connect(warmup=False)

    assert camera.is_connected


def test_connect_already_connected():
    config =BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)
    camera.connect(warmup=False)

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)



# def test_invalid_width_connect():
#     config = BaslerCameraConfig(
#         camera_idx=0,
#         width=99999,  # Invalid width to trigger error
#         height=480,
#     )
#     camera = BaslerCamera(config)

#     with pytest.raises(RuntimeError):
#         camera.connect(warmup=False)


# @pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_read():
    config = BaslerCameraConfig(camera_idx=1)
    camera = BaslerCamera(config)
    camera.connect(warmup=False)

    img = camera.read()

    assert isinstance(img, np.ndarray)


def test_read_before_connect():
    config = BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.disconnect()


def test_async_read():
    config = BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)
    camera.connect(warmup=False)

    try:
        img = camera.async_read()

        assert isinstance(img, np.ndarray)
    finally:
        if camera.is_connected:
            camera.disconnect()  # To stop/join the thread. Otherwise get warnings when the test ends


# def test_async_read_timeout():
#     config = BaslerCameraConfig(camera_idx=0)
#     camera = BaslerCamera(config)
#     camera.connect(warmup=False)

#     try:
#         with pytest.raises(TimeoutError):
#             camera.async_read(timeout_ms=0)
#     finally:
#         if camera.is_connected:
#             camera.disconnect()


def test_async_read_before_connect():
    config = BaslerCameraConfig(camera_idx=0)
    camera = BaslerCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()

