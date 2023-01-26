    
import time
import os
import numpy as np
from random import shuffle
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule
from algorithms.fp3o_trainer import PPO_Trainer
# from algorithms.hatrpo_trainer import TRPO_Trainer


def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # add 
        self.use_recurrent_policy = self.all_args.use_recurrent_policy
        self.data_chunk_length = self.all_args.data_chunk_length

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from algorithms.fp3o_trainer import FP3O as TrainAlgo
        from algorithms.fp3o_policy import FP3O_Policy as Policy

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        self.buffer = []
        for agent_id in range(self.num_agents):
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        for agent_id in range(self.num_agents):
            next_value = self.trainer.policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                        self.buffer[agent_id].rnn_states_critic[-1],
                                                        self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer.value_normalizer)

    def train(self):
        train_infos = []
        # random update order

        self.trainer.prep_training()
        parallel_trainer = PPO_Trainer(self.all_args, self.trainer)
  
        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        factor_dicrete = np.ones((self.episode_length, self.n_rollout_threads, self.num_agents), dtype=np.float32)

        # shuffle and non-overlap mapping
        agents_pairs = []
        random_agent_ids = [id for id in range(self.num_agents)]
        shuffle(random_agent_ids)
        agents_pairs.append(random_agent_ids.copy())
        mapping_agent_ids = [random_agent_ids[(i + 1) % self.num_agents] for i in range(self.num_agents)]
        agents_pairs.append(mapping_agent_ids)

        for stage, agent_pair in enumerate(agents_pairs):
            old_actions_logprob_pair = []
            new_actions_logprob_pair = []
            buffer_pair = []

            for agent_id in agent_pair:
                self.buffer[agent_id].update_factor(factor / np.expand_dims(factor_dicrete[..., agent_id], axis=-1))
                self.buffer[agent_id].update_factor_j_p(np.expand_dims(factor_dicrete[..., agent_id], axis=-1))
                buffer_pair.append(self.buffer[agent_id])
            
            if stage == 0:
                for agent_id in agent_pair:
                    available_actions = None if self.buffer[agent_id].available_actions is None \
                        else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

                    with torch.no_grad():
                        if self.use_recurrent_policy:
                            input = self.buffer[agent_id].data_chunk(self.data_chunk_length)
                            old_actions_logprob, _ =self.trainer.policy.actor.evaluate_actions(*input)
                            old_actions_logprob = self.buffer[agent_id].reverse_data_chunk(old_actions_logprob, self.data_chunk_length)
                        else:
                            old_actions_logprob, _ =self.trainer.policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                                    self.buffer[agent_id].rnn_states[:-1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                    self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                                    self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                                    available_actions,
                                                                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                    old_actions_logprob_pair.append(torch.sum(old_actions_logprob,dim=-1,keepdim=True))
            
            train_info_pair = parallel_trainer.train(buffer_pair)

            if stage == 0:
                for agent_id in agent_pair:
                    available_actions = None if self.buffer[agent_id].available_actions is None \
                        else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

                    with torch.no_grad():
                        if self.use_recurrent_policy:
                            input = self.buffer[agent_id].data_chunk(self.data_chunk_length)
                            new_actions_logprob, _ =self.trainer.policy.actor.evaluate_actions(*input)
                            new_actions_logprob = self.buffer[agent_id].reverse_data_chunk(new_actions_logprob, self.data_chunk_length)
                        else:
                            new_actions_logprob, _ =self.trainer.policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                                    self.buffer[agent_id].rnn_states[:-1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                    self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                                    self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                                    available_actions,
                                                                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
                    # new_actions_logprob_pair.append(new_actions_logprob)
                    new_actions_logprob_pair.append(torch.sum(new_actions_logprob,dim=-1,keepdim=True))
                
                old_actions_logprob_pair = torch.cat(old_actions_logprob_pair, dim=-1)
                new_actions_logprob_pair = torch.cat(new_actions_logprob_pair, dim=-1)

                factor_index = np.argsort(np.array(agent_pair))
                factor_dicrete = _t2n(torch.exp(new_actions_logprob_pair-old_actions_logprob_pair)[..., factor_index].reshape(self.episode_length,self.n_rollout_threads, self.num_agents))

                factor = factor * _t2n(torch.prod(torch.exp(new_actions_logprob_pair-old_actions_logprob_pair),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            
            train_infos += train_info_pair


        for i in range(self.num_agents):
            self.buffer[i].after_update()
            
        del parallel_trainer

        index = list(np.argsort(np.array(agents_pairs[0])))
        return [train_infos[i] for i in index]

    def save(self):
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent.pt")

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent.pt')
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
