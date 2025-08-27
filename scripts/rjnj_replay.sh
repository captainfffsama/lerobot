#!/bin/bash

#设置环境变量
export PYTHONBREAKPOINT=ipdb.set_trace
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 进入工作目录

lerobot-replay \
    --robot.type=ur5_follower_end_effector\
    --robot.robot_ip="192.168.1.20" \
    --robot.with_gripper=True \
    --robot.cameras='{"1_right": {"type": "intelrealsense", "serial_number_or_name": "f1480368", "width": 640, "height": 480, "fps": 30}}' \
    --robot.move_mode=servo \
    --robot.max_relative_target=0.3 \
    --robot.init_pos_thr=7 \
    --robot.id=rjnj \
    --dataset.repo_id="haha" \
    --dataset.episode=0 \
    --dataset.root=/home/svt/workspace/code/tmp/
    # --dataset.episode_time_s=600 \
    # --dataset.single_task="Grasp a red insulator and hang it on hook." \
    # --policy.path=/data1/workspace/huqiong/train_log/lerobot/smolvla/250627/2025-06-27/01-22-45_smolvla/checkpoints/200000/pretrained_model
    # --policy.n_action_steps= 20src/lerobot/model

    # <- Teleop optional if you want to teleoperate to record or in between episodes with a policy \
    # --teleop.port=/dev/tty.usbmodem58760431551 \
    # --teleop.type=ur_leader \
    # --teleop.grpc_host="192.168.1.111:50051" \
    # --teleop.id=rjnj

    # <- Policy optional if you want to record with a policy \
    # --policy.path=${HF_USER}/my_policy \
    # --policy.path=/data1/workspace/huqiong/train_log/lerobot/pi0/0701/2025-07-01/12-29-31_pi0/checkpoints/040000/pretrained_model

    # --policy.path=/data1/workspace/huqiong/train_log/lerobot/pi0/0701/2025-07-01/12-29-31_pi0/checkpoints/040000/pretrained_model

    #--robot.init_pos="[0.010746735148131847,-1.7625709772109985,1.9510701894760132,-1.802381157875061,-1.6205466985702515,-0.015358272939920425,0.0117647061124444,]" \


    # --robot.type=ur5_follower \
    # --robot.robot_ip="192.168.1.20" \
    # --robot.init_pos="[0.010746735148131847,-1.7625709772109985,1.9510701894760132,-1.802381157875061,-1.6205466985702515,-0.015358272939920425,0.0117647061124444,]" \
    # --robot.with_gripper=True \
    # --robot.cameras='{"0_top": {"type": "intelrealsense", "serial_number_or_name": "f1420223", "width": 640, "height": 480, "fps": 30}, "1_right": {"type": "intelrealsense", "serial_number_or_name": "f1480368", "width": 640, "height": 480, "fps": 30}}' \
    # --robot.max_relative_target=0.3 \
    # --robot.init_pos_thr=2 \
    # --robot.move_mode=servo \
    # --robot.id=rjnj \