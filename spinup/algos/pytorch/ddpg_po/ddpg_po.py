from copy import deepcopy
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import pybulletgym
import pybullet_envs
import time
import spinup.algos.pytorch.ddpg_po.core as core
from spinup.utils.mpi_logx import EpochLogger
from spinup.env_wrapper.pomdp_wrapper import POMDPWrapper
import os.path as osp
import os


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.thld_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        # Global info
        self.ptr, self.size, self.max_size = 0, 0, size
        # Info used for sampling
        self.sampled_num_buf = np.zeros(size, dtype=np.float32)
        self.pred_q_buf = {i: [] for i in range(size)}
        self.targ_q_buf = {i: [] for i in range(size)}
        self.targ_next_q_buf = {i: [] for i in range(size)}
        self.hist_tuned_indicator_buf = {i: [] for i in range(size)}
        self.sampled_time_buf = {i: [] for i in range(size)}    # indicate when the sample is sampled

    def store(self, obs, act, thld, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.thld_buf[self.ptr] = thld
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def add_sample_hist(self, sample_idxs, pred_q, targ_q, targ_next_q, tuned_indicator, t):
        """Add prediction history."""
        for i in range(len(sample_idxs)):
            self.pred_q_buf[sample_idxs[i]].append(pred_q[i])
            self.targ_q_buf[sample_idxs[i]].append(targ_q[i])
            self.targ_next_q_buf[sample_idxs[i]].append(targ_next_q[i])
            self.hist_tuned_indicator_buf[sample_idxs[i]].append(tuned_indicator[i])
            self.sampled_time_buf[sample_idxs[i]].append(t)

    def sample_batch(self, batch_size=32, device=None, sample_type='genuine_random'):
        # (1) pseudo_random  (2) genuine_random
        if sample_type == 'genuine_random':
            # sample_weights = (self.max_size - self.sampled_num_buf) / self.max_size

            # 1e-6 is causes only the latest experience being sampled in a mini-batch. It's good at the beginning of
            # the learning, but causes very slow improvement in later phase.
            # sample_weights = 1 / (self.sampled_num_buf + 1e-1)
            # import pdb; pdb.set_trace()
            # 0.5 is better than 1e-6, because it give older experiences a chance to be sampled.
            # sample_weights = 1 / (self.sampled_num_buf + 0.5)

            sample_weights = 1 / (self.sampled_num_buf + 0.1)

            # sample_weights = np.exp(-self.sampled_num_buf)  # Very bad performance

            # Pessimistic weights: increase the sampling probability of experiences with lower reward
            # Optimistic weights: increase the sampling probability of experiences with higher reward
            # sample_weights = sample_weights + np.exp(self.rew_buf)
            # sample_weights = np.exp(self.rew_buf)

            population_id = np.arange(self.size)

            batch_size = min(batch_size, self.size)

            # # With replacement
            # idxs = random.choices(population_id, sample_weights[:self.size], k=batch_size)

            # Without replacement
            idxs = np.random.choice(population_id, size=batch_size, replace=False,
                                    p=sample_weights[:self.size] / sample_weights[:self.size].sum())

        elif sample_type == "alternate_random":
            # TODO: alternate between genuine_random and pseudo_random to allow a balance between
            # recent and past experiences.
            pass
        elif sample_type == 'pseudo_random':
            idxs = np.random.randint(0, self.size, size=batch_size)

        # Increase sampled_num by 1 and update sample_weights
        # TODO: only update sample_weights on sampled experiences
        # Crucial note: if sample with replace there may be experiences sample multiple times, so use for loop.
        for i in idxs:
            self.sampled_num_buf[i] += 1

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     thld=self.thld_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     )
        batch_hist = dict(pred_q_hist=[self.pred_q_buf[i] for i in idxs],
                          targ_q_hist=[self.targ_q_buf[i] for i in idxs],
                          targ_next_q_hist=[self.targ_next_q_buf[i] for i in idxs],
                          sampled_time_hist=[self.sampled_time_buf[i] for i in idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}, batch_hist, idxs


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256], threshold_dim=1):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer_sizes = [obs_dim + act_dim+threshold_dim] + hidden_sizes + [1]

        self.layers = nn.ModuleList()
        # Hidden layers
        for h_i in range(len(self.layer_sizes) - 2):
            self.layers += [nn.Linear(self.layer_sizes[h_i], self.layer_sizes[h_i + 1]),
                            nn.ReLU()]
        # Output layer
        self.output_layer = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

    def forward(self, obs, act, threshold):
        x = torch.cat([obs, act, torch.reshape(threshold, (-1, 1))], dim=-1)
        # Hidden layers
        for h_i in range(len(self.layers)):
            x = self.layers[h_i](x)
        # Output layer
        out = self.output_layer(x)
        return torch.squeeze(out, -1)  # Critical to ensure q has right shape.


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, shared_hidden_sizes=[256, 256],
                 act_hidden_size=[64], threshold_hidden_size=[64]):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.shared_layer_sizes = [obs_dim] + shared_hidden_sizes
        self.act_layer_sizes = [shared_hidden_sizes[-1]] + act_hidden_size + [act_dim]
        self.threshold_layer_sizes = [shared_hidden_sizes[-1]] + threshold_hidden_size + [1]

        self.shared_layers = nn.ModuleList()
        # Shared Hidden layers
        for h_i in range(len(self.shared_layer_sizes) - 1):
            self.shared_layers += [nn.Linear(self.shared_layer_sizes[h_i], self.shared_layer_sizes[h_i + 1]),
                                   nn.ReLU()]
        # Act layers
        self.act_layers = nn.ModuleList()
        for h_i in range(len(self.act_layer_sizes) - 2):
            self.act_layers += [nn.Linear(self.act_layer_sizes[h_i], self.act_layer_sizes[h_i + 1]),
                                nn.ReLU()]
        self.act_layers += [nn.Linear(self.act_layer_sizes[-2], self.act_layer_sizes[-1]),
                            nn.Tanh()]
        # Threshold layers
        self.threshold_layers = nn.ModuleList()
        for h_i in range(len(self.threshold_layer_sizes) - 2):
            self.threshold_layers += [nn.Linear(self.threshold_layer_sizes[h_i], self.threshold_layer_sizes[h_i + 1]),
                                      nn.ReLU()]
        self.threshold_layers += [nn.Linear(self.threshold_layer_sizes[-2], self.threshold_layer_sizes[-1]),
                                  nn.Tanh()]

    def forward(self, obs):
        shared_x = obs
        # Shared layers
        for h_i in range(len(self.shared_layers)):
            shared_x = self.shared_layers[h_i](shared_x)
        # Act layers
        act_x = shared_x
        for h_i in range(len(self.act_layers)):
            act_x = self.act_layers[h_i](act_x)
        # Threshold layers
        threshold_x = shared_x
        for h_i in range(len(self.threshold_layers)):
            threshold_x = self.threshold_layers[h_i](threshold_x)
        # Output layer
        return self.act_limit * act_x, threshold_x


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 critic_hidden_sizes=[256, 256],
                 actor_shared_hidden_sizes=[256, 256],
                 actor_act_hidden_size=[64], actor_threshold_hidden_size=[64]):
        super(MLPActorCritic, self).__init__()
        self.q = MLPCritic(obs_dim, act_dim, critic_hidden_sizes)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           actor_shared_hidden_sizes, actor_act_hidden_size, actor_threshold_hidden_size)

    def act(self, obs):
        with torch.no_grad():
            a, thld = self.pi(obs)
            return a.cpu().numpy(), thld.cpu().numpy()


def ddpg(env_name, partially_observable=False,
         pomdp_type = 'remove_velocity',
         flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1,
         actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_name : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        partially_observable:

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

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

    # Wrapper environment if using POMDP
    if partially_observable:
        env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
        test_env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
    else:
        env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = MLPActorCritic(obs_dim, act_dim, act_limit,
                        critic_hidden_sizes=[256, 256],
                        actor_shared_hidden_sizes=[256, 256],
                        actor_act_hidden_size=[64], actor_threshold_hidden_size=[64])
    ac_targ = deepcopy(ac)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    thld_min = 0
    thld_max = 10

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data, batch_hist, t):
        o, a, thld, r, o2, d = data['obs'], data['act'], data['thld'], data['rew'], data['obs2'], data['done']

        # batch_hist['pred_q_hist']
        # batch_hist['targ_q_hist']
        # batch_hist['targ_next_q_hist']
        # batch_hist['sampled_time_hist']

        q = ac.q(o, a, thld)

        # Bellman backup for Q function
        with torch.no_grad():
            pi_targ, thld_targ = ac_targ.pi(o2)
            q_pi_targ = ac_targ.q(o2, pi_targ, thld_targ)

            mean_targ_next_q_hist = []
            tuned_indicator = np.zeros(q_pi_targ.shape)

            threshold = ((thld_targ[:, 0]+act_limit)/(2*act_limit))*(thld_max-thld_min)+thld_min
            # threshold = 1*torch.ones(q_pi_targ.shape).to(device)
            # import pdb; pdb.set_trace()
            for i in range(len(batch_hist['targ_next_q_hist'])):
                tmp_batch_hist = np.asarray(batch_hist['targ_next_q_hist'][i])
                tmp_batch_hist = np.append(tmp_batch_hist, q_pi_targ[i].item())  # add new prediction
                change_rate = tmp_batch_hist[1:] - tmp_batch_hist[:-1]
                if len(tmp_batch_hist)==1:
                    avg_window = tmp_batch_hist[-1]
                else:
                    if change_rate[-1] > thld_targ[i]:
                        avg_window = tmp_batch_hist[-2] + threshold[i]
                        tuned_indicator[i] = 1
                    else:
                        avg_window = tmp_batch_hist[-1]
                mean_targ_next_q_hist.append(avg_window)

            # if t>10000:
            #     import pdb; pdb.set_trace()
            avg_q_pi_targ = torch.as_tensor(mean_targ_next_q_hist, dtype=torch.float32).to(device)
            backup = r + gamma * (1 - d) * avg_q_pi_targ
            # backup = r + gamma * (1 - d) * q_pi_targ
        # import pdb;
        # pdb.set_trace()

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy(),
                         Threshold=threshold.cpu().detach().numpy(), TunedNum=tuned_indicator.sum())

        return loss_q, loss_info, q, backup, avg_q_pi_targ, tuned_indicator  # Crucial log shapped q_pi_targ to history

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, thld = ac.pi(o)
        q_pi = ac.q(o, pi, thld)
        # loss_pi = -q_pi.mean()
        threshold = ((thld[:, 0] + act_limit) / (2 * act_limit)) * (thld_max - thld_min) + thld_min
        thld_weight = 0.001
        loss_pi = -(q_pi-thld_weight*threshold).mean()
        return loss_pi

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, batch_hist, t):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info, q, backup, q_pi_targ, tuned_indicator = compute_loss_q(data, batch_hist, t)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging. (Common choice: 0.995)
        # # TODO: remove later
        # polyak = 0.4
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q.cpu().detach().numpy(), backup.cpu().detach().numpy(), q_pi_targ.cpu().detach().numpy(), tuned_indicator

    def get_action(o, noise_scale):
        a, thld = ac.act(torch.as_tensor(o, dtype=torch.float32).to(device))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit), thld

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a, thld = get_action(o, 0)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a, thld = get_action(o, act_noise)
        else:
            a, thld = env.action_space.sample(), random.uniform(-act_limit, act_limit)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, thld, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                sample_type = 'pseudo_random'  # 'pseudo_random'  genuine_random
                batch, batch_hist, batch_idxs = replay_buffer.sample_batch(batch_size, device=device, sample_type=sample_type)
                q, backup, q_pi_targ, tuned_indicator = update(data=batch, batch_hist=batch_hist, t=t)
                replay_buffer.add_sample_hist(batch_idxs, q, backup, q_pi_targ, tuned_indicator, t)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # # Save model
            # fpath = osp.join(logger.output_dir, 'pyt_save')
            # os.makedirs(fpath, exist_ok=True)
            # context_fname = 'checkpoint-context-' + (
            #     'Step-%d' % t if t is not None else '') + '.pt'
            # context_fname = osp.join(fpath, context_fname)
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     logger.save_state({'env': env}, None)
            #     torch.save({'replay_buffer': replay_buffer}, context_fname)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Threshold', with_min_and_max=True)
            logger.log_tabular('TunedNum', with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


def str2bool(v):
    """Function used in argument parser for converting string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--partially_observable', type=str2bool, nargs='?', const=True, default=False, help="Using POMDP")
    parser.add_argument('--pomdp_type',
                        choices=['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing',
                                 'remove_velocity_and_flickering', 'remove_velocity_and_random_noise',
                                 'remove_velocity_and_random_sensor_missing', 'flickering_and_random_noise',
                                 'random_noise_and_random_sensor_missing', 'random_sensor_missing_and_random_noise'],
                        default='remove_velocity')
    parser.add_argument('--flicker_prob', type=float, default=0.2)
    parser.add_argument('--random_noise_sigma', type=float, default=0.1)
    parser.add_argument('--random_sensor_missing_prob', type=float, default=0.1)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument("--data_dir", type=str, default='spinup_data_lstm')
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs
    # data_dir = osp.join(
    #     osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))),
    #     args.data_dir)
    data_dir = osp.join(
        osp.dirname("D:\spinup_new_data"),
        args.data_dir)

    # import pdb; pdb.set_trace()
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    ddpg(env_name=args.env, partially_observable=args.partially_observable,
         pomdp_type=args.pomdp_type,
         flicker_prob=args.flicker_prob,
         random_noise_sigma=args.random_noise_sigma,
         random_sensor_missing_prob=args.random_sensor_missing_prob,
         actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
