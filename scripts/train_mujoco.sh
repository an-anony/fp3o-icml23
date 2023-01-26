#!/bin/sh
env="mujoco"
scenario="manyagent_swimmer"
agent_conf="8x2"
exp="numbatch32_obs5_lr1e5_clr3e4_step10e6_clip0.2_entropycoef1e-3"
running_max=5
agent_obsk=5
algo="fp3o_par"
cuda='0'
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=1 python train/train_mujoco.py --cuda_index ${cuda} --algorithm_name ${algo} --env_name ${env} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 1e-5 --critic_lr 3e-4 --std_x_coef 1 --std_y_coef 5e-1 --running_id ${number} --n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 32 --episode_length 1000 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --use_state_agent --entropy_coef 0.001 --clip_param 0.2 --share_policy --partial_share
done
