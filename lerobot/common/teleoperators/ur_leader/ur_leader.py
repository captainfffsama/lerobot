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

import logging
import time
import glob

import torch
import numpy as np

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_ur_leader import URLeaderConfig

logger = logging.getLogger(__name__)


def make_gello_from_config(config: URLeaderConfig):
    from gello.agents.gello_agent import GelloAgent

    gello_port = config.gello_port
    if gello_port is None:
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError("No gello port found, please specify one or plug in gello")
    gello = GelloAgent(port=gello_port, start_joints=config.start_joints)
    return gello


class URLeader(Teleoperator):
    config_class = URLeaderConfig
    name = "ur_leader"

    def __init__(self, config: URLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = make_gello_from_config(config)
        self._start_pos = None
        self._is_connected = False
        self._is_calibrated = False

        self.action_names = ["q0", "q1", "q2", "q3", "q4", "q5"]
        if self.config.have_gripper:
            self.action_names.append("gripper")

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in self.action_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._start_pos = self.bus.act(None)
        # No calibrattion now
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        goal_pos = self.bus.act(None).tolist()
        action = {k:v for k, v in zip(self.action_names, goal_pos)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self._start_pos = None
        self._is_connected = False
        self._is_calibrated = False
        logger.info(f"{self} disconnected.")
