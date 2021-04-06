from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import pybulletgym
import time
import spinup.algos.pytorch.td3_test.core as core
from spinup.utils.logx import EpochLogger
from spinup.env_wrapper.pomdp_wrapper import POMDPWrapper
import os.path as osp

DEVICE = "cuda"  # "cuda" "cpu"


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256]):
        super(MLPCritic, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer_sizes = [obs_dim + act_dim] + hidden_sizes + [1]

        self.layers = nn.ModuleList()
        # Hidden layers
        for h_i in range(len(self.layer_sizes) - 2):
            self.layers += [nn.Linear(self.layer_sizes[h_i], self.layer_sizes[h_i + 1]),
                            nn.Sigmoid()]
        # Output layer
        self.layers += [nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]),
                        nn.Identity()]

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        hid_activation = []
        # Hidden layers
        for h_i in range(len(self.layers) - 2):
            x = self.layers[h_i](x)
            # Store activation
            if h_i % 2 == 1:
                hid_activation.append(x)
        # Output layer
        x = self.layers[-2](x)
        x = self.layers[-1](x)
        return torch.squeeze(x, -1), hid_activation # Critical to ensure q has right shape.


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=[256, 256]):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.layer_sizes = [obs_dim] + hidden_sizes + [act_dim]

        self.layers = nn.ModuleList()
        # Hidden layers
        for h_i in range(len(self.layer_sizes) - 2):
            self.layers += [nn.Linear(self.layer_sizes[h_i], self.layer_sizes[h_i + 1]),
                            nn.Sigmoid()]
        # Output layer
        self.layers += [nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]),
                        nn.Tanh()]

    def forward(self, obs):
        x = obs
        hid_activation = []
        # Hidden layers
        for h_i in range(len(self.layers) - 2):
            x = self.layers[h_i](x)
            # Store activation
            if h_i % 2 == 1:
                hid_activation.append(x)
        # Output layer
        x = self.layers[-2](x)
        x = self.layers[-1](x)
        return self.act_limit * x, hid_activation


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, critic_hidden_sizes=[256, 256], actor_hidden_sizes=[256, 256]):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim, critic_hidden_sizes)
        self.q2 = MLPCritic(obs_dim, act_dim, critic_hidden_sizes)
        self.pi = MLPActor(obs_dim, act_dim, act_limit, actor_hidden_sizes)

    def act(self, obs):
        with torch.no_grad():
            a, _ = self.pi(obs)
            return a.cpu().numpy()


def td3(env_name, partially_observable=False,
        pomdp_type = 'remove_velocity',
        flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1,
        actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_name : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        partially_observable:

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
    critic_sparsity_parameter_rho = 0.1
    critic_sparsity_penalty_beta = 0.1 #0.5
    ac = MLPActorCritic(obs_dim, act_dim, act_limit,
                        critic_hidden_sizes=[256, 256], actor_hidden_sizes=[256, 256])
    ac_targ = deepcopy(ac)
    ac.to(DEVICE)
    ac_targ.to(DEVICE)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1, q1_hid_activation = ac.q1(o, a)
        q2, q2_hid_activation = ac.q2(o, a)
        # q1 = ac.q1(o, a)
        # q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ, _ = ac_targ.q1(o2, a2)
            q2_pi_targ, _ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        q1_avg_hid_activation = torch.cat(q1_hid_activation, dim=1).mean(axis=0)
        q2_avg_hid_activation = torch.cat(q2_hid_activation, dim=1).mean(axis=0)

        avoid_divide_zero = torch.tensor(1e-15).to(DEVICE)
        rho = torch.ones(q1_avg_hid_activation.shape).to(DEVICE) * critic_sparsity_parameter_rho
        q1_sparsity_penalty = torch.sum(
            rho * torch.log(rho / (q1_avg_hid_activation + avoid_divide_zero)) + (1 - rho) * torch.log(
                (1 - rho) / (1 - q1_avg_hid_activation + avoid_divide_zero)))
        q2_sparsity_penalty = torch.sum(
            rho * torch.log(rho / (q2_avg_hid_activation + avoid_divide_zero)) + (1 - rho) * torch.log(
                (1 - rho) / (1 - q2_avg_hid_activation + avoid_divide_zero)))

        # q1_sparsity_penalty = nn.functional.kl_div(q1_avg_hid_activation,
        #         #                                            torch.ones(q1_avg_hid_activation.shape).to(
        #         #                                                DEVICE) * critic_sparsity_parameter_rho,
        #         #                                            reduction='sum')
        #         # q2_sparsity_penalty = nn.functional.kl_div(q2_avg_hid_activation,
        #         #                                            torch.ones(q2_avg_hid_activation.shape).to(
        #         #                                                DEVICE) * critic_sparsity_parameter_rho,
        #         #                                            reduction='sum')
        q1_mse = ((q1 - backup)**2).mean()
        q2_mse = ((q2 - backup)**2).mean()
        loss_q1 = q1_mse + critic_sparsity_penalty_beta * q1_sparsity_penalty
        loss_q2 = q2_mse + critic_sparsity_penalty_beta * q2_sparsity_penalty

        # loss_q1 = ((q1 - backup) ** 2).mean()
        # loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q1HidActivation=torch.cat(q1_hid_activation, dim=1).detach().cpu().numpy(),
                         Q1SparsityPenalty=q1_sparsity_penalty.detach().cpu().numpy(),
                         Q1MSE=q1_mse.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q2HidActivation=torch.cat(q2_hid_activation, dim=1).detach().cpu().numpy(),
                         Q2SparsityPenalty=q2_sparsity_penalty.detach().cpu().numpy(),
                         Q2MSE=q2_mse.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o_ = data['obs']
        a_, _ = ac.pi(o_)
        q1_pi, _ = ac.q1(o_, a_)
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

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32).to(DEVICE))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
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
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

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
            logger.log_tabular('Q1HidActivation', with_min_and_max=True)
            logger.log_tabular('Q1SparsityPenalty', with_min_and_max=True)
            logger.log_tabular('Q1MSE', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Q2HidActivation', with_min_and_max=True)
            logger.log_tabular('Q2SparsityPenalty', with_min_and_max=True)
            logger.log_tabular('Q2MSE', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
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
    parser.add_argument('--exp_name', type=str, default='td3_test')
    parser.add_argument("--data_dir", type=str, default='spinup_data')
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs

    data_dir = osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))),
        args.data_dir)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    td3(env_name=args.env, partially_observable=args.partially_observable,
        pomdp_type=args.pomdp_type,
        flicker_prob=args.flicker_prob,
        random_noise_sigma=args.random_noise_sigma,
        random_sensor_missing_prob=args.random_sensor_missing_prob,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
