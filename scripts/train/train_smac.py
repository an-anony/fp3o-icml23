#!/usr/bin/env python

import sys
import os
# sys.path.append("../")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import socket
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.starcraft2.smac_maps import get_map_params
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
"""Train script for SMAC."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='5m_vs_6m',help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)
    parser.add_argument("--cuda_index", type=str, default='0')
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name[0] == "r":
        assert (all_args.use_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name[0] == "f":
        assert (not all_args.use_recurrent_policy), ("check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name[-3:] == 'sha':
        assert (all_args.share_policy and not all_args.partial_share), ("check partial_share and share_policy!")
    elif all_args.algorithm_name[-3:] == 'par':
        assert (all_args.share_policy and all_args.partial_share), ("check partial_share and share_policy!")
    elif all_args.algorithm_name[-3:] == 'sep':
        assert (not all_args.share_policy), ("check partial_share and share_policy!")
    else:
        raise NotImplementedError

    # check
    experiment_split = all_args.experiment_name.split("_")
    for arg_split in experiment_split:
        if arg_split[:8] == 'numbatch': assert (str(all_args.num_mini_batch) == arg_split[8:]), ("check num_mini_batch!")
        elif arg_split[:13] == 'episodelength': assert (str(all_args.episode_length) == arg_split[13:]), ("check episode_length!")
        elif arg_split[:4] == 'clip': assert (str(all_args.clip_param) == arg_split[4:]), ("check clip_param!")
        elif arg_split[:5] == 'epoch': assert (str(all_args.ppo_epoch) == arg_split[5:]), ("check ppo_epoch!")


    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.running_id
    else:
        all_args.seed=np.random.randint(1000,10000)

    print("seed is :",all_args.seed)
    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = all_args.cuda_index
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    print(device)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.map_name / (
                                                all_args.algorithm_name + "_" + all_args.experiment_name) / str(all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = get_map_params(all_args.map_name)["n_agents"]

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    # run experiments
    if all_args.share_policy and all_args.partial_share:
        from runners.partial.smac_runner import SMACRunner as Runner
    elif all_args.share_policy and not all_args.partial_share:
        from runners.shared.smac_runner import SMACRunner as Runner
    elif not all_args.share_policy:
        from runners.separated.smac_runner import SMACRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

    