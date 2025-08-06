#!/bin/bash

#设置环境变量
export PYTHONBREAKPOINT=ipdb.set_trace
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export EVAL_FILE=$(realpath $(dirname $(dirname $(dirname $(python -c "import lerobot; print(lerobot.__file__)"))))/analyer_tools/model_replay_analyer.py)
echo "Using eval file: ${EVAL_FILE}"

python ${EVAL_FILE} \
    --dataset.repo_id="eval" \
    --dataset.root="/data1/datasets/ur_grasp_db/ur_grasp_v2_2000" \
    --dataset.episode_select="[0,1,2,3,4,5,6,7,8,9]" \
    --dataset.single_task="Grasp the red insulator and hang it on the hook." \
    --policy.path=/data1/workspace/huqiong/train_log/lerobot/pi0_fast/0721/new/2025-07-22/01-34-27_pi0fast/checkpoints/080000/pretrained_model \
    --jpg_save_dir="/data1/workspace/huqiong/test"