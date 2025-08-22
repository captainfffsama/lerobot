# 使用方法
## model_relay_analyer.py
```bash
python model_relay_analyer.py \
    --dataset.repo_id="eval" \
    --dataset.root="/data1/datasets/ur_grasp_db/ur_grasp_v2_2000" \
    --dataset.episode_select="[0,1,2,3,4,5,6,7,8,9]" \
    --dataset.single_task="Grasp the red insulator and hang it on the hook." \
    --policy.path="/data1/workspace/huqiong/train_log/lerobot/pi0_fast/0721/new/2025-07-22/01-34-27_pi0fast/checkpoints/080000/pretrained_model" \
    --jpg_save_dir="/data1/workspace/huqiong/test"
```