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

import logging
import math
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.pipeline import RobotActionProcessorStep
from lerobot.record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

from lerobot.robots.ur5_follower.config_ur5_follower import UR5FollowerEndEffectorConfig
from lerobot.robots.ur5_follower.ur5_follower_end_effector import UR5FollowerEndEffector

from lerobot.teleoperators.meta_quest3.config_meta_quest3 import MetaQuest3Config
from lerobot.teleoperators.meta_quest3.teleop_meta_quest3 import MetaQuest3Teleop


logger = logging.getLogger(__name__)

# Session configuration
NUM_EPISODES = 2
FPS = 20
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 20
TASK_DESCRIPTION = "Pick-and-place with UR5 controlled by Meta Quest 3"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


class MetaQuest3ToURDeltaStep(RobotActionProcessorStep):
    """
    Convert Meta Quest 3 teleop action (right hand pose and inputs)
    to UR5 end-effector delta action dict expected by UR5FollowerEndEffector.

    - Uses right_hand.pos as position relative to calibration and differentiates over time
    - Converts right_hand.rot quaternion to euler (zxy) and differentiates
    - Maps right_hand.trigger to gripper (0..1) if robot has gripper
    - Applies enable gate: when not enabled, output zeros (hold)
    """

    def __init__(self, with_gripper: bool, pos_scale: float = 0.3, rot_scale: float = 0.8):
        super().__init__()
        self.with_gripper = with_gripper
        self.prev_pos: np.ndarray | None = None
        self.prev_euler: np.ndarray | None = None
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale

    def _quat_to_euler_zxy(self, quat_xyzw: np.ndarray) -> np.ndarray:
        r = Rotation.from_quat(quat_xyzw)
        euler = r.as_euler("zxy", degrees=False)
        # normalize to [-pi, pi]
        euler = (euler + np.pi) % (2 * np.pi) - np.pi
        return euler

    def process(self, teleop_action: dict[str, Any], robot_obs: dict[str, Any] | None = None) -> dict[str, Any]:
        enabled = bool(teleop_action.get("enabled", False))

        # Default zeros
        out: Dict[str, Any] = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_roll": 0.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
        }

        # Use right hand as primary control
        pos = np.array(teleop_action.get("right_hand.pos", np.zeros(3)), dtype=np.float32)
        quat = np.array(teleop_action.get("right_hand.rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        euler = self._quat_to_euler_zxy(quat)

        if self.prev_pos is None:
            self.prev_pos = pos.copy()
        if self.prev_euler is None:
            self.prev_euler = euler.copy()

        # Compute deltas (relative to previous sample) and scale
        dpos = (pos - self.prev_pos) * self.pos_scale
        deuler = (euler - self.prev_euler) * self.rot_scale

        # Update state
        self.prev_pos = pos
        self.prev_euler = euler

        if enabled:
            out["delta_x"] = float(dpos[0])
            out["delta_y"] = float(dpos[1])
            out["delta_z"] = float(dpos[2])
            out["delta_yaw"] = float(deuler[0])
            out["delta_pitch"] = float(deuler[1])
            out["delta_roll"] = float(deuler[2])
        else:
            # hold pose when not enabled
            out = {k: 0.0 for k in out.keys()}

        if self.with_gripper:
            trigger = float(teleop_action.get("right_hand.trigger", 0.0))
            # map trigger (0..1) to gripper open/close directly
            out["gripper"] = float(np.clip(trigger, 0.0, 1.0))

        return out


def main():
    # Configure cameras if needed (optional for dataset video)
    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    }

    # Robot (UR5 end-effector control)
    ur_config = UR5FollowerEndEffectorConfig(
        robot_ip="192.168.1.20",  # change to your UR5 IP
        id="ur5_follower_eef",
        with_gripper=True,
        cameras=camera_config,
        move_mode="servo",
        init_pos_thr=0.25,
    )
    robot = UR5FollowerEndEffector(ur_config)

    # Teleoperator (Meta Quest 3)
    teleop_config = MetaQuest3Config(
        id="meta_quest3_teleop",
        ipaddress="192.168.1.50",  # change to your Quest 3 server IP
        port="30001",
    )
    teleop = MetaQuest3Teleop(teleop_config)

    # Build teleop->robot action processor
    teleop_to_delta = RobotProcessorPipeline[dict[str, Any], RobotAction](
        steps=[
            MetaQuest3ToURDeltaStep(with_gripper=ur_config.with_gripper, pos_scale=0.35, rot_scale=0.9),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Optionally, build observation pipeline (pass-through here)
    obs_passthrough = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Create dataset and features
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_to_delta,
                initial_features=create_initial_features(action=teleop.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=obs_passthrough,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect
    robot.connect()
    teleop.connect(calibrate=True)

    listener, events = init_keyboard_listener()
    _init_rerun(session_name="recording_meta_quest3_ur5")

    if not robot.is_connected or not teleop.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_to_delta,
            robot_observation_processor=obs_passthrough,
        )

        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_to_delta,
                robot_observation_processor=obs_passthrough,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1

    log_say("Stop recording")
    teleop.disconnect()
    robot.disconnect()
    listener.stop()
    dataset.push_to_hub()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
