from copy import deepcopy
import numpy as np
import pybulletgym
import gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
import itertools
from spinup.env_wrapper.pomdp_wrapper import POMDPWrapper
import os.path as osp

DEVICE = "cuda"


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = list(next_obs)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        """

        :param batch_size:
        :param max_hist_len: the length of experiences before current experience
        :return:
        """
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_rew = np.zeros([batch_size, 1])
            hist_done = np.zeros([batch_size, 1])
            hist_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_rew = np.zeros([batch_size, max_hist_len])
            hist_done = np.zeros([batch_size, max_hist_len])
            hist_len = max_hist_len * np.ones(batch_size)
            # Extract history experiences before sampled index
            for hist_i in range(max_hist_len):
                hist_obs[:, -1 - hist_i, :] = self.obs_buf[idxs - hist_i - 1, :]
                hist_act[:, -1 - hist_i, :] = self.act_buf[idxs - hist_i - 1, :]
                hist_obs2[:, -1 - hist_i, :] = self.obs2_buf[idxs - hist_i - 1, :]
                hist_act2[:, -1 - hist_i, :] = self.act_buf[idxs - hist_i, :]  # include a_t
                hist_rew[:, -1 - hist_i] = self.rew_buf[idxs - hist_i - 1]
                hist_done[:, -1 - hist_i] = self.done_buf[idxs - hist_i - 1]
            # If there is done in the backward experiences, only consider the experiences after the last done.
            for batch_i in range(batch_size):
                done_idxs_exclude_last_exp = np.where(hist_done[batch_i][:-1] == 1)  # Exclude last experience
                # If exist done
                if done_idxs_exclude_last_exp[0].size != 0:
                    largest_done_id = done_idxs_exclude_last_exp[0][-1]
                    hist_len[batch_i] = max_hist_len - (largest_done_id + 1)

                    # Only keep experiences after the last done
                    obs_keep_part = np.copy(hist_obs[batch_i, largest_done_id + 1:, :])
                    act_keep_part = np.copy(hist_act[batch_i, largest_done_id + 1:, :])
                    obs2_keep_part = np.copy(hist_obs2[batch_i, largest_done_id + 1:, :])
                    act2_keep_part = np.copy(hist_act2[batch_i, largest_done_id + 1:, :])
                    rew_keep_part = np.copy(hist_rew[batch_i, largest_done_id + 1:])
                    done_keep_part = np.copy(hist_done[batch_i, largest_done_id + 1:])

                    # Set to 0 to make sure all experiences are at the beginning
                    hist_obs[batch_i] = np.zeros([max_hist_len, self.obs_dim])
                    hist_act[batch_i] = np.zeros([max_hist_len, self.act_dim])
                    hist_obs2[batch_i] = np.zeros([max_hist_len, self.obs_dim])
                    hist_act2[batch_i] = np.zeros([max_hist_len, self.act_dim])
                    hist_rew[batch_i] = np.zeros([max_hist_len])
                    hist_done[batch_i] = np.zeros([max_hist_len])

                    # Move kept experiences to the start of the segment
                    hist_obs[batch_i, :max_hist_len - (largest_done_id + 1), :] = obs_keep_part
                    hist_act[batch_i, :max_hist_len - (largest_done_id + 1), :] = act_keep_part
                    hist_obs2[batch_i, :max_hist_len - (largest_done_id + 1), :] = obs2_keep_part
                    hist_act2[batch_i, :max_hist_len - (largest_done_id + 1), :] = act2_keep_part
                    hist_rew[batch_i, :max_hist_len - (largest_done_id + 1)] = rew_keep_part
                    hist_done[batch_i, :max_hist_len - (largest_done_id + 1)] = done_keep_part
        #
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_rew=hist_rew,
                     hist_done=hist_done,
                     hist_len=hist_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

#######################################################################################

#######################################################################################


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,), mem_gate=True):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mem_gate = mem_gate
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()

        self.mem_gate_layer = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory
        #    Pre-LSTM
        mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #    Memeory Gate
        if self.mem_gate:
            self.mem_gate_layer += [
                nn.Linear(self.mem_lstm_layer_sizes[-1] + obs_dim + act_dim, self.mem_lstm_layer_sizes[-1]),
                nn.Sigmoid()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_lstm_layer_sizes[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1

        x = torch.cat([hist_obs, hist_act], dim=-1)

        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_lstm_layer_sizes[-1]).unsqueeze(
                                    1).long()).squeeze(1)
        hist_msk = (hist_seg_len != 0).float().view(-1, 1).repeat(1, self.mem_lstm_layer_sizes[-1]).to(DEVICE)
        #   Memory Gate
        if self.mem_gate:
            memory_gate = torch.cat([hist_out * hist_msk, obs, act], dim=-1)
            for layer in self.mem_gate_layer:
                memory_gate = layer(memory_gate)

        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)
        # Post-Combination
        if self.mem_gate:
            x = torch.cat([memory_gate * hist_out * hist_msk, x], dim=-1)
        else:
            x = torch.cat([hist_out * hist_msk, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return torch.squeeze(x, -1)  # Critical to ensure q has right shape.


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,), mem_gate=True):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.mem_gate = mem_gate
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()

        self.mem_gate_layer = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        #    Pre-LSTM
        mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #    Memeory Gate
        if self.mem_gate:
            self.mem_gate_layer += [nn.Linear(self.mem_lstm_layer_sizes[-1] + obs_dim, self.mem_lstm_layer_sizes[-1]),
                                    nn.Sigmoid()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_lstm_layer_sizes[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1

        x = torch.cat([hist_obs, hist_act], dim=-1)

        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_lstm_layer_sizes[-1]).unsqueeze(
                                    1).long()).squeeze(1)
        hist_msk = (hist_seg_len != 0).float().view(-1, 1).repeat(1, self.mem_lstm_layer_sizes[-1]).to(DEVICE)
        #   Memory Gate
        if self.mem_gate:
            memory_gate = torch.cat([hist_out * hist_msk, obs], dim=-1)
            for layer in self.mem_gate_layer:
                memory_gate = layer(memory_gate)

        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)
        # Post-Combination
        if self.mem_gate:
            x = torch.cat([memory_gate * hist_out * hist_msk, x], dim=-1)
        else:
            x = torch.cat([hist_out * hist_msk, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,), critic_mem_gate=True,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,), actor_mem_gate=True):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes, mem_gate=critic_mem_gate)
        self.q2 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes, mem_gate=critic_mem_gate)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes, mem_gate=actor_mem_gate)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(DEVICE)
            hist_act = torch.zeros(1, 1, self.act_dim).to(DEVICE)
            hist_seg_len = torch.zeros(1).to(DEVICE)
        with torch.no_grad():
            return self.pi(obs, hist_obs, hist_act, hist_seg_len).cpu().numpy()


#######################################################################################

#######################################################################################
def lstm_td3(env_name, seed=0,
             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
             polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
             start_steps=10000,
             update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
             noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
             batch_size=100,
             max_hist_len=100,
             partially_observable=False,
             critic_mem_pre_lstm_hid_sizes=(128,),
             critic_mem_lstm_hid_sizes=(128,),
             critic_cur_feature_hid_sizes=(128,),
             critic_post_comb_hid_sizes=(128,), critic_mem_gate=False,
             actor_mem_pre_lstm_hid_sizes=(128,),
             actor_mem_lstm_hid_sizes=(128,),
             actor_cur_feature_hid_sizes=(128,),
             actor_post_comb_hid_sizes=(128,), actor_mem_gate=False,
             logger_kwargs=dict(), save_freq=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # device = torch.device(DEVICE)

    # Wrapper environment if using POMDP
    if partially_observable:
        env, test_env = POMDPWrapper(env_name), POMDPWrapper(env_name)
    else:
        env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = MLPActorCritic(obs_dim, act_dim, act_limit,
                        critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                        critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                        critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                        critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,
                        critic_mem_gate=critic_mem_gate,
                        actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                        actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                        actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                        actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,
                        actor_mem_gate=actor_mem_gate)
    ac_targ = deepcopy(ac)
    ac.to(DEVICE)
    ac_targ.to(DEVICE)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=replay_size)

    # # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data[
            'hist_len']

        q1 = ac.q1(o, a, h_o, h_a, h_len)
        q2 = ac.q2(o, a, h_o, h_a, h_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2, h_o2, h_a2, h_len)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2, h_o2, h_a2, h_len)
            q2_pi_targ = ac_targ.q2(o2, a2, h_o2, h_a2, h_len)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o, h_o, h_a, h_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_len']
        q1_pi = ac.q1(o, ac.pi(o, h_o, h_a, h_len), h_o, h_a, h_len)
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, o_buff, a_buff, o_buff_len, noise_scale):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(DEVICE)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(DEVICE)
        h_l = torch.tensor([o_buff_len]).float().to(DEVICE)
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(DEVICE),
                       h_o, h_a, h_l).reshape(act_dim)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = get_action(o, o_buff, a_buff, o_buff_len, 0)
                o2, r, d, _ = test_env.step(a)

                ep_ret += r
                ep_len += 1
                # Add short history
                if max_hist_len != 0:
                    if o_buff_len == max_hist_len:
                        o_buff[:max_hist_len - 1] = o_buff[1:]
                        a_buff[:max_hist_len - 1] = a_buff[1:]
                        o_buff[max_hist_len - 1] = list(o)
                        a_buff[max_hist_len - 1] = list(a)
                    else:
                        o_buff[o_buff_len + 1 - 1] = list(o)
                        a_buff[o_buff_len + 1 - 1] = list(a)
                        o_buff_len += 1
                o = o2

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, obs_dim])
        a_buff = np.zeros([max_hist_len, act_dim])
        o_buff[0, :] = o
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, obs_dim])
        a_buff = np.zeros([1, act_dim])
        o_buff_len = 0

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    for t in range(total_steps):
        if t % 200 == 0:
            end_time = time.time()
            print("t={}, {}s".format(t, end_time - start_time))
            start_time = end_time
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, o_buff, a_buff, o_buff_len, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Add short history
        if max_hist_len != 0:
            if o_buff_len == max_hist_len:
                o_buff[:max_hist_len - 1] = o_buff[1:]
                a_buff[:max_hist_len - 1] = a_buff[1:]
                o_buff[max_hist_len - 1] = list(o)
                a_buff[max_hist_len - 1] = list(a)
            else:
                o_buff[o_buff_len + 1 - 1] = list(o)
                a_buff[o_buff_len + 1 - 1] = list(a)
                o_buff_len += 1

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch_with_history(batch_size, max_hist_len)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                update(data=batch, timer=j)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_hist_len', type=int, default=10)
    parser.add_argument('--partially_observable', type=bool, default=True)
    parser.add_argument('--critic_mem_pre_lstm_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--critic_mem_lstm_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--critic_cur_feature_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--critic_post_comb_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--critic_mem_gate', type=bool, default=False)
    parser.add_argument('--actor_mem_pre_lstm_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--actor_mem_lstm_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--actor_cur_feature_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--actor_post_comb_hid_sizes', type=tuple, default=(128,))
    parser.add_argument('--actor_mem_gate', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default='lstm_td3')
    parser.add_argument("--data_dir", type=str, default='spinup_data_lstm')
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs
    data_dir = osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))),
        args.data_dir)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    lstm_td3(args.env,
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             max_hist_len=args.max_hist_len,
             partially_observable=args.partially_observable,
             critic_mem_pre_lstm_hid_sizes=args.critic_mem_pre_lstm_hid_sizes,
             critic_mem_lstm_hid_sizes=args.critic_mem_lstm_hid_sizes,
             critic_cur_feature_hid_sizes=args.critic_cur_feature_hid_sizes,
             critic_post_comb_hid_sizes=args.critic_post_comb_hid_sizes,
             critic_mem_gate=args.critic_mem_gate,
             actor_mem_pre_lstm_hid_sizes=args.actor_mem_pre_lstm_hid_sizes,
             actor_mem_lstm_hid_sizes=args.actor_mem_lstm_hid_sizes,
             actor_cur_feature_hid_sizes=args.actor_cur_feature_hid_sizes,
             actor_post_comb_hid_sizes=args.actor_post_comb_hid_sizes,
             actor_mem_gate=args.actor_mem_gate,
             logger_kwargs=logger_kwargs)
