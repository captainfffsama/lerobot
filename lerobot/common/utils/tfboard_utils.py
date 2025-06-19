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
import shutil
from pathlib import Path
from pprint import pformat

import cv2
import numpy as np
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
import torch

from lerobot.configs.train import TrainPipelineConfig


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"dataset:{cfg.dataset.repo_id}",
        f"seed:{cfg.seed}",
    ]
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


def get_safe_tb_name(name: str):
    """TensorBoard doesn't accept certain characters in names."""
    return name.replace(":", "_").replace("/", "_").replace("\\", "_")


class TensorBoardLogger:
    """A helper class to log objects using TensorBoard."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb  # 复用wandb配置，或者可以添加专门的tensorboard配置
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        self.skip_info_key = {"steps", "samples", "episodes", "epochs"}

        # Set up TensorBoard
        self.tb_log_dir = self.log_dir / "tensorboard"
        self.tb_log_dir.mkdir(exist_ok=True, parents=True)

        self._writer = SummaryWriter(log_dir=str(self.tb_log_dir))

        # 记录配置信息
        config_dict = cfg.to_dict()
        self._log_config(config_dict)

        print(colored("Logs will be saved with TensorBoard.", "blue", attrs=["bold"]))
        logging.info(f"TensorBoard logs --> {colored(str(self.tb_log_dir), 'yellow', attrs=['bold'])}")

    def _log_config(self, config_dict: dict, prefix: str = ""):
        """递归地将配置记录到TensorBoard"""
        self._writer.add_text("config", pformat(config_dict), 0)

    def log_policy(self, checkpoint_dir: Path):
        """记录模型checkpoint信息到TensorBoard."""
        if getattr(self.cfg, "disable_artifact", False):
            return

        step_id = checkpoint_dir.name
        # 记录checkpoint路径信息
        self._writer.add_text(
            "model/checkpoint_path",
            str(checkpoint_dir),
            global_step=int(step_id.split("_")[-1]) if step_id.split("_")[-1].isdigit() else 0,
        )

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(f"Mode must be 'train' or 'eval', got {mode}")

        for k, v in d.items():
            if k in self.skip_info_key:
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"{mode}/{k}", v, step)
            elif isinstance(v, str):
                # self._writer.add_text(f"{mode}/{k}", v, step)
                logging.info(f"{k}:v")
            elif isinstance(v, torch.Tensor):
                if k.startswith("losses"):
                    self._writer.add_scalar(f"{mode}/{k}", v.mean().item(), step)
                else:
                    logging.warning(f"{k} shape is:{v.shape}, dtype is:{v.dtype}, type is:{type(v)}")
            else:
                logging.warning(
                    f'TensorBoard logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(f"Mode must be 'train' or 'eval', got {mode}")
        logging.warning(
            f'TensorBoard video logging is deprecated. Please use "wandb.log" instead for key "video/{mode}" with a list of video paths.'
        )

    def close(self):
        """关闭TensorBoard writer"""
        if hasattr(self, "_writer"):
            self._writer.close()

    def __del__(self):
        """析构函数，确保writer被关闭"""
        self.close()
