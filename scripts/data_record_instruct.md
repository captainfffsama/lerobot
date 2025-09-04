# conda 环境
```bash
conda activate rjlerobot_new
```
# 代码路径
/home/svt/workspace/code/lerobot/scripts

# 采集脚本
/home/svt/workspace/code/lerobot/scripts/record_realsense.sh

```bash

#!/bin/bash 

#设置环境变量
export PYTHONBREAKPOINT=ipdb.set_trace
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 进入工作目录

lerobot-record  \
    --display_data=True\
    --resume=True \ 
    --robot.type=ur5_follower \
    --robot.robot_ip="192.168.1.20" \
    --robot.init_pos="[0.010746735148131847,-1.7625709772109985,1.9510701894760132,-1.802381157875061,-1.6205466985702515,-0.015358272939920425,0.0117647061124444,]" \
    --robot.with_gripper=True \
    --robot.cameras='{"0_top": {"type": "intelrealsense", "serial_number_or_name": "f1420223", "width": 640, "height": 480, "fps": 30}, "1_right": {"type": "intelrealsense", "serial_number_or_name": "f1480368", "width": 640, "height": 480, "fps": 30}}' \
    --robot.max_relative_target=0.3 \
    --robot.init_pos_thr=2 \
    --robot.move_mode=servo \
    --robot.id=rjnj \
    --dataset.repo_id=rj/ur_250819 \
    --dataset.num_episodes=2 \
    --dataset.root=/home/svt/train_data \
    --dataset.single_task="Grasp the red insulator and hang it on the hook." \
    --teleop.type=ur_leader \
    --teleop.id=rjnj

```
# 示教模式
```bash
cd /home/svt/workspace/code/lerobot/scripts
python ./freeDrive.py
```
# 回放脚本
/home/svt/workspace/code/lerobot/scripts/rjnj_replay.sh
