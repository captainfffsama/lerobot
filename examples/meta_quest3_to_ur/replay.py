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
import logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.ur5_follower.ur5_follower_end_effector import UR5FollowerEndEffector
from lerobot.robots.ur5_follower.config_ur5_follower import UR5FollowerEndEffectorConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


logger = logging.getLogger(__name__)

EPISODE_IDX = 0
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


def main():
    # Initialize the robot config
    robot_config = UR5FollowerEndEffectorConfig(
        robot_ip="192.168.1.20",  # change to your UR5 IP
        id="ur5_follower_eef",
        with_gripper=True,
        move_mode="servo",
    )

    # Initialize the robot
    robot = UR5FollowerEndEffector(robot_config)

    # For EE control robot we directly send EE actions recorded in the dataset,
    # so we only need an identity processor (no IK step required).
    robot_ee_passthrough = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Fetch the dataset to replay
    dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset v3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
    actions = episode_frames.select_columns("action")

    # Connect to the robot
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting replay loop...")
    log_say(f"Replaying episode {EPISODE_IDX}")
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        # Get recorded EE action from dataset
        ee_action = {
            name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
        }

        # Get robot observation
        robot_obs = robot.get_observation()

        # Dataset EE -> robot EE passthrough (kept for API parity)
        ee_action_proc = robot_ee_passthrough((ee_action, robot_obs))

        # Send action to robot
        _ = robot.send_action(ee_action_proc)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    # Clean up
    robot.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
