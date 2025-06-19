# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 52011))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import os
import torch.nn.functional as F
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import rerun as rr

from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4

    # Custom:
    episode_select: list[int] | None = None

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    jpg_save_dir: str | None = None

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def main(cfg: RecordConfig):
    """使用现成的模型和训练集来尝试分析模型输出的统计结果

    Args:
        cfg (RecordConfig): _description_
    Examples:
        python test.py \
        --dataset.repo_id="123" \
        --dataset.root="/data1/workspace/huqiong/train_log/lerobot/smolvla/250612/dataset" \
        --dataset.episode_select="[0,1,2,3,4,5,6,7,8,9]" \
        --dataset.single_task="dfasf" \
        --robot.type="so100_follower" \
        --robot.port="7200" \
        --policy.path="/data1/workspace/huqiong/train_log/lerobot/smolvla/250612/outputs/train/2025-06-13/03-23-33_smolvla/checkpoints/080000/pretrained_model"

    """
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episode_select,
        delta_timestamps={"action": [i / 20 for i in range(50)]},
    )

    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.reset()
    result = []
    for data in tqdm(dataset, total=len(dataset), desc="Processing dataset"):
        obser = {}
        for k, v in data.items():
            if k in (
                "action",
                "task",
            ):
                continue
            if isinstance(v, torch.Tensor):
                if len(v.shape) > 1 or (len(v.shape) == 1 and v.shape[0] > 1):
                    obser[k] = v
            elif isinstance(v, str):
                continue
            else:
                continue
        action_values = predict_action(
            obser,
            policy,
            get_safe_torch_device(policy.config.device),
            policy.config.use_amp,
            task=data["task"],
        )
        pre_actions = [action_values]
        while len(policy._queues["action"]) > 0:
            action = policy._queues["action"].popleft()
            action = action.squeeze(0)
            action = action.to("cpu")
            pre_actions.append(action)
        pre_actions = torch.stack(pre_actions, dim=0)
        policy.reset()
        l1_dis = F.l1_loss(pre_actions, data["action"], reduction="none")
        result.append(l1_dis.numpy())
        
    # data,action chunk,7
    result = np.array(result)
    # action dim analyer
    r = result.reshape(-1, result.shape[-1])
    rows = np.sqrt(r.shape[-1]).astype(int)
    cols=r.shape[-1] // rows + 1
    plot_multi_dimensional_curve_custom_layout(
        r,
        tile="Action L1 Error per Dimension over Episodes",
        sub_tile="Action Dim",
        sub_x_tile="frame Index",
        sub_y_tile="L1 Error",
        rows=rows,
        cols=cols,
        save_path=os.path.join(cfg.jpg_save_dir, "action_dim_l1.jpg"),
    )
    print("L1 error per action dimension:", r.mean(axis=0))
    print("L1 errot per action dimension max:", r.max(axis=0))
    # action chunk analyer
    r = result.transpose((0,2, 1))
    r = r.reshape(-1, r.shape[-1])
    rows = np.sqrt(r.shape[-1]).astype(int)
    cols=r.shape[-1] // rows + 1
    plot_multi_dimensional_curve_custom_layout(r,
        tile="Action L1 Error per Chunk over Episodes",
        sub_tile="Action Chunk",
        sub_x_tile="Episode Index* action dim",
        sub_y_tile="L1 Error",
        rows=rows,
        cols=cols,
        save_path=os.path.join(cfg.jpg_save_dir, "action_chunk_l1.jpg"),
    )
    print("L1 error per action chunk:", r.mean(axis=0))
    print("L1 errot per action chunk max:", r.max(axis=0))

    # only 1 action exec analyer
    r=result[:, 0, :]
    r = r.reshape(-1, r.shape[-1])
    rows = np.sqrt(r.shape[-1]).astype(int)
    cols=r.shape[-1] // rows + 1
    plot_multi_dimensional_curve_custom_layout(
        r,
        tile="Action L1 Error per Action Exec over Episodes",
        sub_tile="Action Exec",
        sub_x_tile="Episode Index",
        sub_y_tile="L1 Error",
        rows=rows,
        cols=cols,
        save_path=os.path.join(cfg.jpg_save_dir, "action_exec_l1.jpg"),
    )
    print("L1 error per action exec:", r.mean(axis=0))
    print("L1 errot per action exec max:", r.max(axis=0))


def plot_multi_dimensional_curve_custom_layout(
    data,
    tile="Action L1 Error per Dimension over Episodes",
    sub_tile="Action Dim",
    sub_x_tile="Episode Index",
    sub_y_tile="L1 Error",
    save_path="curve_plot.jpg",
    figsize=(15, 12),
    rows=3,
    cols=3,
):
    N, dims = data.shape
    time_steps = np.arange(N)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(tile, fontsize=16, fontweight="bold")

    # 展平 axes 数组以便索引
    axes_flat = axes.flatten()

    colors = plt.cm.tab10.colors

    for dim in range(dims):
        ax = axes_flat[dim]

        ax.plot(
            time_steps,
            data[:, dim],
            color=colors[dim % len(colors)],
            linewidth=2,
            marker="o" if N <= 50 else None,
            markersize=3,
        )

        ax.set_title(f"{sub_tile} {dim + 1}", fontsize=12, fontweight="bold")
        ax.set_xlabel(sub_x_tile, fontsize=10)
        ax.set_ylabel(sub_y_tile, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=9)

    # 隐藏多余的子图
    for i in range(dims, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout()
    plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"图片已保存到: {save_path}")


if __name__ == "__main__":
    main()
