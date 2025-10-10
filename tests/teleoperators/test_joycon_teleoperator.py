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


import time
from lerobot.teleoperators.joycon.configuration_joycon import JoyconTeleopConfig, BiJoyconTeleopConfig
from lerobot.teleoperators.joycon.teleop_joycon import  JoyconTeleop, BiJoyconTeleop
from hid import HIDException


def test_joycon():
    config = JoyconTeleopConfig(device="left")
    teleop = JoyconTeleop(config)

    teleop.connect()

    while True:
        try:
            time.sleep(0.5)

            action = teleop.get_action()
            print(action)
        except RuntimeError:
            print("JoyCon not connected. Please connect a JoyCon and try again.")

    # Test disconnect method
    teleop.disconnect()

def test_bijoycon():
    config = BiJoyconTeleopConfig()
    teleop = BiJoyconTeleop(config)

    teleop.connect()

    while True:
        try:
            time.sleep(0.5)

            action = teleop.get_action()
            print(action)
        except RuntimeError:
            print("JoyCon not connected. Please connect a JoyCon and try again.")

    # Test disconnect method
    teleop.disconnect()
def test_driver():
    from pyjoycon import GyroTrackingJoyCon, get_R_id
    import time
    import numpy as np

    joycon_id = get_R_id()
    joycon = GyroTrackingJoyCon(*joycon_id)
    time.sleep(0.5)
    previous_ro=np.array([0.0, 0.0, 0.0])
    while True:
        try:
            print("device:",joycon._joycon_device.read(10))
            print("joycon gyro: ", np.array(joycon.gyro_in_rad,dtype=np.float64))
            print("joycon gyro ori:",joycon.gyro_in_rad)
            # if np.array_equal(previous_ro,np.array(joycon.gyro_in_rad,dtype=np.float64)):
            #     print("device die")
            #     break
            # previous_ro=np.array(joycon.gyro_in_rad,dtype=np.float64)
            # time.sleep(1)
        except HIDException:
            print("device:",joycon._joycon_device.read(10))
            print("JoyCon not connected. Please connect a JoyCon and try again.")


test_bijoycon()


