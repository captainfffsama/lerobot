# 使用方法
## model_relay_analyer.py
```bash
python test.py \
    --dataset.repo_id="123" \
    --dataset.root="/data1/workspace/huqiong/train_log/lerobot/smolvla/250612/dataset" \
    --dataset.episode_select="[0,1,2,3,4,5,6,7,8,9]" \
    --dataset.single_task="dfasf" \
    --robot.type="so100_follower" \
    --robot.port="7200" \
    --policy.path="/data1/workspace/huqiong/train_log/lerobot/smolvla/250612/outputs/train/2025-06-13/03-23-33_smolvla/checkpoints/080000/pretrained_model" \
    --jpg_save_dir="/root/lerobot/hq_test"
```