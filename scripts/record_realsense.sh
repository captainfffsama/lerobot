#!/bin/bash 

#设置环境变量
export PYTHONBREAKPOINT=ipdb.set_trace
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 进入工作目录
rm -rf /home/svt/wxl/data

python -m lerobot.record \
    --display_data=False \
    --robot.type=ur5_follower \
    --robot.robot_ip="192.168.1.20" \
    --robot.with_gripper=True \
    --robot.cameras='{"0_top": {"type": "basler", "camera_idx": 0,}, "1_right": {"type": "intelrealsense", "serial_number_or_name": "f1480368", "width": 640, "height": 480, "fps": 20}}' \
    --robot.max_relative_target=0.3 \
    --robot.init_pos_thr=2 \
    --robot.move_mode=servo \
    --robot.id=rjnj \
    --dataset.repo_id=aliberts/record-test \
    --dataset.num_episodes=2 \
    --dataset.root=/home/svt/wxl/data \
    --dataset.single_task="Grab the cube" \
    --teleop.type=ur_leader \
    --teleop.grpc_host="127.0.0.1:50051" \
    --teleop.id=rjnj

    # <- Teleop optional if you want to teleoperate to record or in between episodes with a policy \
    # --teleop.port=/dev/tty.usbmodem58760431551 \

    # <- Policy optional if you want to record with a policy \
    # --policy.path=${HF_USER}/my_policy \
