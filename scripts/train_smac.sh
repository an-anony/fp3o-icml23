#!/bin/sh
env="StarCraft2"
map="3s5z"
exp="numbatch1_episodelength400_clip0.2_step5e6"
gamma=0.99
running_max=5
episode_length=400
n_rollout_threads=8
algo="rfp3o_sep"
num_mini_batch=1
cuda="0"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --cuda_index ${cuda} --algorithm_name ${algo} --env_name ${env} --experiment_name ${exp} --map_name ${map} --running_id ${number} --gamma ${gamma} --n_training_threads 32 --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps 5000000 --ppo_epoch 5 --stacked_frames 1 --use_value_active_masks --use_eval --add_center_xy --use_state_agent --clip_param 0.2 --use_recurrent_policy
done
