import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check

import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check

class FP3O():
    """
    Trainer class for FP3O to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (FP3O_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # self.method = args.method
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        # self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # add 
        self.partial_share = args.partial_share

        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss


    def data_concat(self, samples):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch = [], [], [], [], [], [], [], [], [], [], [], [], []

        if not self._use_recurrent_policy:
            for sample in samples:
                share_obs, obs, rnn_states, rnn_states_critic, actions, \
                value_preds, return_, masks, active_masks, old_action_log_probs, \
                adv, available_actions, factor = sample

                share_obs_batch.append(share_obs)
                obs_batch.append(obs)
                rnn_states_batch.append(rnn_states)
                rnn_states_critic_batch.append(rnn_states_critic)
                actions_batch.append(actions)
                value_preds_batch.append(value_preds)
                return_batch.append(return_)
                masks_batch.append(masks)
                active_masks_batch.append(active_masks)
                old_action_log_probs_batch.append(old_action_log_probs)
                adv_targ.append(adv)
                available_actions_batch.append(available_actions)
                factor_batch.append(factor)

            share_obs_batch = np.concatenate(share_obs_batch, axis=0)
            obs_batch = np.concatenate(obs_batch, axis=0)
            rnn_states_batch = np.concatenate(rnn_states_batch, axis=0)
            rnn_states_critic_batch = np.concatenate(rnn_states_critic_batch, axis=0)
            actions_batch = np.concatenate(actions_batch, axis=0)
            value_preds_batch = np.concatenate(value_preds_batch, axis=0)
            return_batch = np.concatenate(return_batch, axis=0)
            masks_batch = np.concatenate(masks_batch, axis=0)
            active_masks_batch = np.concatenate(active_masks_batch, axis=0)
            old_action_log_probs_batch = np.concatenate(old_action_log_probs_batch, axis=0)
            adv_targ = np.concatenate(adv_targ, axis=0)
            if available_actions_batch[0] is not None:
                available_actions_batch = np.concatenate(available_actions_batch, axis=0)
            else:
                available_actions_batch = None
            factor_batch = np.concatenate(factor_batch, axis=0)

            return share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, factor_batch
        else:
            for sample in samples:
                share_obs, obs, rnn_states, rnn_states_critic, actions, \
                value_preds, return_, masks, active_masks, old_action_log_probs, \
                adv, available_actions, factor = sample

                N = rnn_states.shape[0]
                T = int(share_obs.shape[0] / N)

                share_obs_batch.append(share_obs.reshape(T, N, *share_obs.shape[1:]))
                obs_batch.append(obs.reshape(T, N, *obs.shape[1:]))
                rnn_states_batch.append(rnn_states)
                rnn_states_critic_batch.append(rnn_states_critic)

                actions_batch.append(actions.reshape(T, N, *actions.shape[1:]))
                value_preds_batch.append(value_preds.reshape(T, N, *value_preds.shape[1:]))
                return_batch.append(return_.reshape(T, N, *return_.shape[1:]))
                masks_batch.append(masks.reshape(T, N, *masks.shape[1:]))
                active_masks_batch.append(active_masks.reshape(T, N, *active_masks.shape[1:]))
                old_action_log_probs_batch.append(old_action_log_probs.reshape(T, N, *old_action_log_probs.shape[1:]))
                adv_targ.append(adv.reshape(T, N, *adv.shape[1:]))
                if available_actions is not None:
                    available_actions_batch.append(available_actions.reshape(T, N, *available_actions.shape[1:]))
                else:
                    available_actions_batch.append(None)
                factor_batch.append(factor.reshape(T, N, *factor.shape[1:]))

            share_obs_batch = np.concatenate(share_obs_batch, axis=1).reshape(-1, *share_obs_batch[0].shape[2:])
            obs_batch = np.concatenate(obs_batch, axis=1).reshape(-1, *obs_batch[0].shape[2:])

            rnn_states_batch = np.concatenate(rnn_states_batch, axis=0)
            rnn_states_critic_batch = np.concatenate(rnn_states_critic_batch, axis=0)
            actions_batch = np.concatenate(actions_batch, axis=1).reshape(-1, *actions_batch[0].shape[2:])
            value_preds_batch = np.concatenate(value_preds_batch, axis=1).reshape(-1, *value_preds_batch[0].shape[2:])
            return_batch = np.concatenate(return_batch, axis=1).reshape(-1, *return_batch[0].shape[2:])
            masks_batch = np.concatenate(masks_batch, axis=1).reshape(-1, *masks_batch[0].shape[2:])
            active_masks_batch = np.concatenate(active_masks_batch, axis=1).reshape(-1, *active_masks_batch[0].shape[2:])
            old_action_log_probs_batch = np.concatenate(old_action_log_probs_batch, axis=1).reshape(-1, *old_action_log_probs_batch[0].shape[2:])
            adv_targ = np.concatenate(adv_targ, axis=1).reshape(-1, *adv_targ[0].shape[2:])
            if available_actions_batch[0] is not None:
                available_actions_batch = np.concatenate(available_actions_batch, axis=1).reshape(-1, *available_actions_batch[0].shape[2:])
            else:
                available_actions_batch = None
            factor_batch = np.concatenate(factor_batch, axis=1).reshape(-1, *factor_batch[0].shape[2:])

            return share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, factor_batch



    def ppo_update(self, samples, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        assert (not self.partial_share), ("partial shared network cannot use ppo_update")

        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch = self.data_concat(samples)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)


        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)


        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        factor_batch = check(factor_batch).to(**self.tpdv)
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)

        surr1 = imp_weights * adv_targ * factor_batch
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ * factor_batch

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights


    def partial_ppo_update(self, samples, agent_heads, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        assert (self.partial_share), ("non-partially-shared network cannot use partial_ppo_update")

        num = len(samples)
        total_policy_loss = None
        total_value_loss = None
        for sample, agent_head in zip(samples, agent_heads):
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, factor_batch = sample



            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
            adv_targ = check(adv_targ).to(**self.tpdv)


            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
            return_batch = check(return_batch).to(**self.tpdv)


            active_masks_batch = check(active_masks_batch).to(**self.tpdv)

            factor_batch = check(factor_batch).to(**self.tpdv)
            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                obs_batch, 
                                                                                rnn_states_batch, 
                                                                                rnn_states_critic_batch, 
                                                                                actions_batch, 
                                                                                masks_batch, 
                                                                                available_actions_batch,
                                                                                active_masks_batch,
                                                                                agent_head=agent_head)
            # actor update
            imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)

            surr1 = imp_weights * adv_targ * factor_batch
            surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ * factor_batch

            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                                dim=-1,
                                                keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

            value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch) * self.value_loss_coef

            total_policy_loss = total_policy_loss + policy_loss if total_policy_loss is not None else policy_loss
            total_value_loss = total_value_loss + value_loss if total_value_loss is not None else value_loss

        total_policy_loss /= num
        total_value_loss /= num

        # training 
         # actor
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            total_policy_loss.backward()

        assert (isinstance(self.policy.actor.act, list)), ("partial param is wrong!")
        actor_parameters = list(self.policy.actor.parameters())
        for i in range(len(self.policy.actor.act)):
            # list(self.policy.actor.act[i].parameters())[0].grad *= num
            actor_parameters += list(self.policy.actor.act[i].parameters())

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_parameters, self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(actor_parameters)

        self.policy.actor_optimizer.step()

        # critic
        self.policy.critic_optimizer.zero_grad()

        total_value_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()



class PPO_Trainer():
    """
    Trainer class for HAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 trainers,):
        self.share_policy = args.share_policy
        if self.share_policy:
            self.trainer = trainers
        else:
            self.trainers = trainers

        # self.method = args.method
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.ppo_epoch_total = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        # self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # add 
        self.partial_share = args.partial_share


    def shared_train(self, buffers):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        advantages_pair = []

        for buffer in buffers:
            if self._use_popart:
                advantages = buffer.returns[:-1] - self.trainer.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            advantages_pair.append(advantages)

        for _ in range(self.ppo_epoch):
            data_generator_pair = []
            sub_to_generator_pair = []
            for buffer, advantages in zip(buffers, advantages_pair):
                sub_to_generator_pair.append(buffer.subject_to_generator(advantages))
                if self._use_recurrent_policy:
                    data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
                # elif self._use_naive_recurrent:
                #     data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
                else:
                    data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

                data_generator_pair.append(data_generator)

            if self.is_satisfy_st(sub_to_generator_pair):
                for samples in zip(*data_generator_pair):
                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.trainer.ppo_update(samples)

                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['ratio'] += imp_weights.mean()
            else:
                print('skip')

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return [train_info.copy() for _ in range(len(buffers))]

    def partially_shared_train(self, buffers, agent_heads):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        advantages_pair = []

        for buffer in buffers:
            if self._use_popart:
                advantages = buffer.returns[:-1] - self.trainer.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            advantages_pair.append(advantages)

        for _ in range(self.ppo_epoch):
            data_generator_pair = []
            sub_to_generator_pair = []
            for buffer, advantages in zip(buffers, advantages_pair):
                sub_to_generator_pair.append(buffer.subject_to_generator(advantages))
                if self._use_recurrent_policy:
                    data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
                # elif self._use_naive_recurrent:
                #     data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
                else:
                    data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

                data_generator_pair.append(data_generator)

            if self.is_satisfy_st(sub_to_generator_pair):
                for samples in zip(*data_generator_pair):
                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.trainer.partial_ppo_update(samples, agent_heads)

                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['ratio'] += imp_weights.mean()
            else:
                print('skip')

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return [train_info.copy() for _ in range(len(buffers))]

    def seperated_train(self, buffers):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        train_info_pair = [train_info.copy() for _ in range(len(buffers))]

        advantages_pair = []

        for trainer, buffer in zip(self.trainers, buffers):
            if self._use_popart:
                advantages = buffer.returns[:-1] - trainer.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            advantages_pair.append(advantages)

        for _ in range(self.ppo_epoch):
            data_generator_pair = []
            sub_to_generator_pair = []
            for buffer, advantages in zip(buffers, advantages_pair):
                sub_to_generator_pair.append(buffer.subject_to_generator(advantages))
                if self._use_recurrent_policy:
                    data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
                # elif self._use_naive_recurrent:
                #     data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
                else:
                    data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

                data_generator_pair.append(data_generator)

            if self.is_satisfy_st(sub_to_generator_pair):
                for samples in zip(*data_generator_pair):
                    for sample, trainer, train_info in zip(samples, self.trainers, train_info_pair):
                        value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = trainer.ppo_update([sample])

                        train_info['value_loss'] += value_loss.item()
                        train_info['policy_loss'] += policy_loss.item()
                        train_info['dist_entropy'] += dist_entropy.item()
                        train_info['actor_grad_norm'] += actor_grad_norm
                        train_info['critic_grad_norm'] += critic_grad_norm
                        train_info['ratio'] += imp_weights.mean()
            else:
                print('skip')

        num_updates = self.ppo_epoch * self.num_mini_batch

        for train_info in train_info_pair:
            for k in train_info.keys():
                train_info[k] /= num_updates

        return train_info_pair

    def train(self, buffers, agent_heads=None):
        if self.share_policy and not self.partial_share:
            train_info = self.shared_train(buffers)
        elif self.share_policy and self.partial_share:
            assert (agent_heads is not None), ("check agent_heads")
            train_info = self.partially_shared_train(buffers, agent_heads)
        elif not self.share_policy:
            train_info = self.seperated_train(buffers)

        return train_info


    def is_satisfy_st(self, sub_to_generator_pair):
        total_st1 = 0
        total_st2 = 0
        
        for samples in zip(*sub_to_generator_pair):
            for sample in samples:
                adv_targ, active_masks, factor, factor_j_p = sample
                
                total_st1 += (factor * factor_j_p * adv_targ * active_masks).sum()
                total_st2 += (factor_j_p * adv_targ * active_masks).sum()
            
        if total_st1 + 1e-5 < total_st2:
            return False
        else:
            return True
