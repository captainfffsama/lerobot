python model_replay_analyer.py \
    --dataset.repo_id="eval" \
    --dataset.root="/home/svt/train_data" \
    --dataset.episode_select="[0,3,5,7,9,20,30,45,60]" \
    --dataset.single_task="Grasp the red insulator and hang it on the hook." \
    --policy.path="/home/svt/tmp/checkpoints/200000/pretrained_model" \
    --jpg_save_dir="/home/svt/tmp/model_analyer_jpg"
