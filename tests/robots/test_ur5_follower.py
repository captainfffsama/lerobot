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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.ur5_follower import (
    UR5Follower,
    UR5FollowerConfig,
)

from lerobot.cameras.basler import BaslerCameraConfig


@pytest.fixture
def follower():
    camera_config = {
        "0_top":  BaslerCameraConfig(0),
        "1_right": BaslerCameraConfig(1),
    }
    cfg = UR5FollowerConfig(robot_ip="192.168.1.20", cameras=camera_config)
    robot = UR5Follower(cfg)
    yield robot
    if robot.is_connected:
        robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()
    print(obs)


# def test_send_action(follower):
#     follower.connect()

#     action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
#     returned = follower.send_action(action)

#     assert returned == action

#     goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
#     follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
