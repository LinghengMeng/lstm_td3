from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import pybullet_envs
import time
import spinup.algos.pytorch.td3_sc_pixel.core as core
from spinup.utils.logx import EpochLogger, setup_logger_kwargs
import os.path as osp

class Env():
    """
    Test environment wrapper for CarRacing
    """

    def __init__(self, seed=0, img_stack=4, action_repeat=8, flat_obs=False):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_space = self.env.action_space
        # Gray image
        width = self.env.observation_space.shape[0]
        height = self.env.observation_space.shape[1]
        self.flat_obs = flat_obs
        if self.flat_obs:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(img_stack*width*height,))
        else:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(img_stack, width, height))
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        if self.flat_obs:
            return np.array(self.stack).flatten()
        else:
            return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state" (the original code has die state penalty)
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        if self.flat_obs:
            obs = np.array(self.stack).flatten()
        else:
            obs = np.array(self.stack)
        return obs, total_reward, done or die, die

    def render(self, *arg):
        self.env.render(*arg)

    def close(self):
        self.env.close()

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

# class Env():
#     """
#     Environment wrapper for CarRacing
#     """
#
#     def __init__(self, env_name, random_seed, img_stack, action_repeat):
#         self.env = gym.make(env_name)
#         self.env.seed(random_seed)
#         self.action_space = self.env.action_space
#         # Gray image
#         width = self.env.observation_space.shape[0]
#         height = self.env.observation_space.shape[1]
#         self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(img_stack, width, height))
#         self.reward_threshold = self.env.spec.reward_threshold
#         self.img_stack = img_stack
#         self.action_repeat = action_repeat
#
#     def reset(self):
#         self.counter = 0
#         self.av_r = self.reward_memory()
#
#         self.die = False
#         img_rgb = self.env.reset()
#         #         print(img_rgb)
#         img_gray = self.rgb2gray(img_rgb)
#         self.stack = [np.expand_dims(img_gray, axis=0)] * self.img_stack  # four frames for decision
#         return torch.FloatTensor(self.stack).permute(1, 0, 2, 3)
#
#     def step(self, action):
#         total_reward = 0
#         for i in range(self.action_repeat):
#             img_rgb, reward, die, _ = self.env.step(action)
#             # # don't penalize "die state"
#             # if die:
#             #     reward += 100
#             # # green penalty
#             # if np.mean(img_rgb[:, :, 1]) > 185.0:
#             #     reward -= 0.05
#             total_reward += reward
#             # if no reward recently, end the episode
#             done = True if self.av_r(reward) <= -0.1 else False
#             if done or die:
#                 done = done or die
#                 break
#
#         img_gray = self.rgb2gray(img_rgb)
#         self.stack.pop(0)
#         self.stack.append(np.expand_dims(img_gray, axis=0))
#         assert len(self.stack) == self.img_stack
#         return torch.FloatTensor(self.stack).permute(1, 0, 2, 3), total_reward, done, die
#
#     def render(self, *arg):
#         self.env.render(*arg)
#
#     def close(self):
#         self.env.close()
#
#     @staticmethod
#     def rgb2gray(rgb, norm=True):
#         # rgb image -> gray [0, 1]
#         gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
#         if norm:
#             # normalize
#             gray = gray / 128. - 1.
#         return gray
#
#     @staticmethod
#     def reward_memory():
#         # record reward for last 100 steps
#         count = 0
#         length = 100
#         history = np.zeros(length)
#
#         def memory(reward):
#             nonlocal count
#             history[count] = reward
#             count = (count + 1) % length
#             return np.mean(history)
#
#         return memory


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

    def sample_batch(self, batch_size=32, device=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}


#######################################################################################
#
#######################################################################################
class CNNCritic(nn.Module):
    def __init__(self, act_dim, img_stack,
                 cnn_out_channels=[16, 32, 64, 128, 256, 512], full_hidden_sizes=[256]):
        super(CNNCritic, self).__init__()
        self.cnn_layers = nn.ModuleList()
        self.full_layers = nn.ModuleList()

        # CNN layers
        cnn_layer_sizes = [img_stack] + cnn_out_channels
        # [(W−K+2P)/S]+1
        self.cnn_layers += [nn.Conv2d(img_stack, 8, kernel_size=4, stride=2), # (8, 47, 47)
                            nn.ReLU(),  # activation
                            nn.Conv2d(8, 16, kernel_size=3, stride=2),        # (16, 23, 23)
                            nn.ReLU(),  # activation
                            nn.Conv2d(16, 32, kernel_size=3, stride=2),       # (32, 11, 11)
                            nn.ReLU(),  # activation
                            nn.Conv2d(32, 64, kernel_size=3, stride=2),       # (64, 5, 5)
                            nn.ReLU(),  # activation
                            nn.Conv2d(64, 128, kernel_size=3, stride=1),      # (128, 3, 3)
                            nn.ReLU(),  # activation
                            nn.Conv2d(128, 5, kernel_size=3, stride=1),     # (256, 1, 1)
                            nn.ReLU()]

        # Fully connection layers
        # full_layer_sizes = [cnn_layer_sizes[-1]+act_dim] + full_hidden_sizes + [1]
        full_layer_sizes = [int(5)+act_dim] + full_hidden_sizes + [1]
        for f_i in range(len(full_layer_sizes) - 2):
            self.full_layers += [nn.Linear(full_layer_sizes[f_i], full_layer_sizes[f_i + 1]), nn.ReLU()]
        self.full_layers += [nn.Linear(full_layer_sizes[-2], full_layer_sizes[-1]), nn.Identity()]

    def forward(self, obs, act):
        x = obs
        for c_layer in self.cnn_layers:
            x = c_layer(x)
        x = x.view(-1, 5)
        x = torch.cat((x, act), dim=1)
        for f_layer in self.full_layers:
            x = f_layer(x)
        return torch.squeeze(x, -1)  # Critical to ensure q has right shape.

class CNNActor(nn.Module):
    def __init__(self, act_dim, act_limit, img_stack,
                 cnn_out_channels=[16, 32, 64, 128, 256, 512], full_hidden_sizes=[256]):
        super(CNNActor, self).__init__()
        self.act_limit = act_limit

        self.cnn_layers = nn.ModuleList()
        self.full_layers = nn.ModuleList()

        # CNN layers
        cnn_layer_sizes = [img_stack] + cnn_out_channels
        # [(W−K+2P)/S]+1
        self.cnn_layers += [nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),  # (8, 47, 47)
                            nn.ReLU(),  # activation
                            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (16, 23, 23)
                            nn.ReLU(),  # activation
                            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 11, 11)
                            nn.ReLU(),  # activation
                            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 5, 5)
                            nn.ReLU(),  # activation
                            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (128, 3, 3)
                            nn.ReLU(),  # activation
                            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 1, 1)
                            nn.ReLU()]

        # Fully connection layers
        # full_layer_sizes = [cnn_layer_sizes[-1]] + full_hidden_sizes + [act_dim]
        full_layer_sizes = [int(256)] + full_hidden_sizes + [act_dim]
        for f_i in range(len(full_layer_sizes)-2):
            self.full_layers += [nn.Linear(full_layer_sizes[f_i], full_layer_sizes[f_i+1]), nn.ReLU()]
        # Crucial CarRacing-v0 action space
        self.full_layers += [nn.Linear(full_layer_sizes[-2], full_layer_sizes[-1]), nn.Sigmoid()]

    def forward(self, obs):
        x = obs
        for c_layer in self.cnn_layers:
            x = c_layer(x)
        x = x.view(-1, 256)
        for f_layer in self.full_layers:
            x = f_layer(x)
        # CarRacing-v0 action_space.high[1,1,1] action_space.low[-1,0,0]
        x = torch.cat(((x[:, 0]*2-1).reshape(-1,1), x[:, 1:]), dim=1)
        return x


class CNNActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, img_stack=4):
        super(CNNActorCritic, self).__init__()
        self.q1 = CNNCritic(act_dim, img_stack)
        self.q2 = CNNCritic(act_dim, img_stack)
        self.pi = CNNActor(act_dim, act_limit, img_stack)

    def act(self, obs):
        with torch.no_grad():
            a = self.pi(obs)
            return a.cpu().numpy()

#################################################################################
#
#################################################################################
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
                            nn.ReLU()]
        # Output layer
        self.layers += [nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]),
                        nn.Identity()]

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        # Hidden layers
        for h_i in range(len(self.layers)):
            x = self.layers[h_i](x)
        return torch.squeeze(x, -1)  # Critical to ensure q has right shape.


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
                            nn.ReLU()]
        # Output layer: crucial CarRacing-v0 action space [0.1]
        self.layers += [nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]),
                        nn.Sigmoid()]

    def forward(self, obs):
        x = obs
        # Hidden layers
        for h_i in range(len(self.layers)):
            x = self.layers[h_i](x)
        return x

# Fully concatenated
# class ConcatSkipConnMLPActor(nn.Module):
#     """Concatenated Skip Connection MLP actor"""
#     def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=[256, 256]):
#         super(ConcatSkipConnMLPActor, self).__init__()
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.act_limit = act_limit
#         self.layer_sizes = [obs_dim] + hidden_sizes + [act_dim]
#
#         self.layers = nn.ModuleList()
#         # Hidden layers
#         for h_i in range(len(self.layer_sizes) - 2):
#             self.layers += [nn.Linear(np.sum(self.layer_sizes[:h_i + 1]), self.layer_sizes[h_i + 1]),
#                             nn.ReLU()]
#         # Output layer
#         self.layers += [nn.Linear(np.sum(self.layer_sizes[:-1]), self.layer_sizes[-1]),
#                         nn.Tanh()]
#
#     def forward(self, obs):
#         x = obs
#         x_list = [x]
#         # Hidden layers
#         for h_i in range(len(self.layers)):
#             if isinstance(self.layers[h_i], nn.Linear):
#                 if len(x_list) == 1:
#                     concat_input = x_list[0]
#                 else:
#                     concat_input = torch.cat(x_list, dim=1)
#                 x = self.layers[h_i](concat_input)
#             else:  # All activation layers including nn.ReLU and nn.Tanh
#                 x = self.layers[h_i](x)
#                 x_list.append(x)
#         return self.act_limit * x


# Only last layer concatenated
class ConcatSkipConnMLPActor(nn.Module):
    """Concatenated Skip Connection MLP actor"""
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=[256, 256]):
        super(ConcatSkipConnMLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.layer_sizes = [obs_dim] + hidden_sizes + [act_dim]

        self.layers = nn.ModuleList()
        # Hidden layers
        for h_i in range(len(self.layer_sizes) - 2):
            self.layers += [nn.Linear(self.layer_sizes[h_i], self.layer_sizes[h_i + 1]),
                            nn.ReLU()]
        # Output layer
        self.layers += [nn.Linear(np.sum(self.layer_sizes[:-1]), self.layer_sizes[-1]),
                        nn.Tanh()]

    def forward(self, obs):
        x = obs
        x_list = [x]
        # Hidden layers
        for h_i in range(len(self.layers)):
            if isinstance(self.layers[h_i], nn.Linear):
                if len(x_list) == len(self.layer_sizes)-1:
                    concat_input = torch.cat(x_list, dim=1)
                else:
                    concat_input = x_list[-1]
                x = self.layers[h_i](concat_input)
            else:  # All activation layers including nn.ReLU and nn.Tanh
                x = self.layers[h_i](x)
                x_list.append(x)
        return self.act_limit * x


class AdditionSkipConnMLPActor(nn.Module):
    """Addition skip connection MLP actor"""
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=[256, 256]):
        super(AdditionSkipConnMLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.layer_sizes = [obs_dim] + hidden_sizes + [act_dim]

        self.layers = nn.ModuleList()
        # Hidden layers
        for h_i in range(len(self.layer_sizes) - 2):
            self.layers += [nn.Linear(self.layer_sizes[h_i], self.layer_sizes[h_i + 1]),
                            nn.ReLU()]
        # Output layer
        self.layers += [nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]),
                        nn.Tanh()]

    def forward(self, obs):
        x = obs
        x_list = [x]
        # Hidden layers
        for h_i in range(len(self.layers)):
            if isinstance(self.layers[h_i], nn.Linear):
                if len(x_list) <= 2:
                    added_input = x_list[-1]
                else:
                    added_input = torch.sum(torch.stack(x_list[1:]), dim=0)
                x = self.layers[h_i](added_input)
            else:  # All activation layers including nn.ReLU and nn.Tanh
                x = self.layers[h_i](x)
                x_list.append(x)
        return self.act_limit * x


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 critic_hidden_sizes=[256, 256], actor_hidden_sizes=[256, 256], actor_type='MLP'):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim, critic_hidden_sizes)
        self.q2 = MLPCritic(obs_dim, act_dim, critic_hidden_sizes)
        if actor_type == 'MLP':
            self.pi = MLPActor(obs_dim, act_dim, act_limit, actor_hidden_sizes)
        elif actor_type == 'ConcatSkipConnMLP':
            self.pi = ConcatSkipConnMLPActor(obs_dim, act_dim, act_limit, actor_hidden_sizes)
        elif actor_type == 'AdditionSkipConnMLP':
            self.pi = AdditionSkipConnMLPActor(obs_dim, act_dim, act_limit, actor_hidden_sizes)
        else:
            raise ValueError('Wrong actor_type: {}!'.format(actor_type))

    def act(self, obs):
        with torch.no_grad():
            a = self.pi(obs)
            return a.cpu().numpy()


def td3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
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

    img_stack = 4
    action_repeat = 8
    # env = Env('CarRacing-v0', seed, img_stack, action_repeat)
    flat_obs = False
    env = Env(flat_obs=flat_obs)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]
    # import pdb; pdb.set_trace()
    # Create actor-critic module and target networks
    if flat_obs:
        ac = MLPActorCritic(obs_dim[0], act_dim, act_limit)
    else:
        ac = CNNActorCritic(obs_dim, act_dim, act_limit, img_stack=img_stack)
    ac_targ = deepcopy(ac)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    # q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_size = 100000
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)

            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                         Q2Vals=q2.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
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
        # Log q grad
        q_grad_log = {}
        # for h_i in range(len(ac.q1.layers)):
        #     if h_i % 2 == 0:
        #         q_grad_log['Q1Layer{}WeightGrad'.format(h_i)] = ac.q1.layers[h_i].weight.grad.cpu().detach().numpy()
        #         q_grad_log['Q1Layer{}BiasGrad'.format(h_i)] = ac.q1.layers[h_i].bias.grad.cpu().detach().numpy()
        logger.store(**q_grad_log)
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

            # # Tune grad
            grad_log = {}
            # for h_i in range(len(ac.pi.cnn_layers)):
            # for h_i in range(1):
            #     if h_i % 2 == 0:
            #         grad_log['Layer{}PiWeightGrad'.format(h_i)] = ac.pi.cnn_layers[h_i].weight.grad.cpu().detach().numpy()
            #         grad_log['Layer{}PiBiasGrad'.format(h_i)] = ac.pi.cnn_layers[h_i].bias.grad.cpu().detach().numpy()

                    # # Random set value for grad equal to 0
                    # ac.pi.cnn_layers[h_i].weight.grad[ac.pi.cnn_layers[h_i].weight.grad == 0] = np.random.uniform(-1, 1)*1e-6 # np.random.rand() * 1e-8
                    # ac.pi.cnn_layers[h_i].bias.grad[ac.pi.cnn_layers[h_i].bias.grad == 0] = np.random.uniform(-1, 1)*1e-6 # np.random.rand() * 1e-8

                    # # Random set value for |grad| less than 1e-6
                    # grad_threshold = 1
                    # ac.pi.cnn_layers[h_i].weight.grad[torch.abs(ac.pi.cnn_layers[h_i].weight.grad) < grad_threshold] = np.random.uniform(-1, 1) * grad_threshold
                    # ac.pi.cnn_layers[h_i].bias.grad[torch.abs(ac.pi.cnn_layers[h_i].bias.grad) < grad_threshold] = np.random.uniform(-1, 1) * grad_threshold
                    #
                    # grad_log['AfterTunedLayer{}PiWeightGrad'.format(h_i)] = ac.pi.cnn_layers[h_i].weight.grad.cpu().detach().numpy()
                    # grad_log['AfterTunedLayer{}PiBiasGrad'.format(h_i)] = ac.pi.cnn_layers[h_i].bias.grad.cpu().detach().numpy()
            # for h_i in range(1):
            #     if h_i % 2 == 0:
            #         grad_log['Layer{}PiWeightGrad'.format(h_i)] = ac.pi.layers[h_i].weight.grad.cpu().detach().numpy()
            #         grad_log['Layer{}PiBiasGrad'.format(h_i)] = ac.pi.layers[h_i].bias.grad.cpu().detach().numpy()

                    # # Random set value for grad equal to 0
                    # ac.pi.cnn_layers[h_i].weight.grad[ac.pi.cnn_layers[h_i].weight.grad == 0] = np.random.uniform(-1, 1)*1e-6 # np.random.rand() * 1e-8
                    # ac.pi.cnn_layers[h_i].bias.grad[ac.pi.cnn_layers[h_i].bias.grad == 0] = np.random.uniform(-1, 1)*1e-6 # np.random.rand() * 1e-8

                    # # Random set value for |grad| less than 1e-6
                    # grad_threshold = 1
                    # ac.pi.layers[h_i].weight.grad[
                    #     torch.abs(ac.pi.layers[h_i].weight.grad) < grad_threshold] = np.random.uniform(-1,
                    #                                                                                        1) * grad_threshold
                    # ac.pi.layers[h_i].bias.grad[
                    #     torch.abs(ac.pi.layers[h_i].bias.grad) < grad_threshold] = np.random.uniform(-1,
                    #                                                                                      1) * grad_threshold
                    # #
                    # grad_log['AfterTunedLayer{}PiWeightGrad'.format(h_i)] = ac.pi.layers[
                    #     h_i].weight.grad.cpu().detach().numpy()
                    # grad_log['AfterTunedLayer{}PiBiasGrad'.format(h_i)] = ac.pi.layers[
                    #     h_i].bias.grad.cpu().detach().numpy()

            #
            pi_optimizer.step()

            logger.store(LossPi=loss_pi.item(), **grad_log)

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
        a = ac.act(torch.as_tensor(o.reshape((1,)+o.shape), dtype=torch.float32).to(device))[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(num_test_episodes):
        # test_env = Env('CarRacing-v0', seed, img_stack, action_repeat)
        test_env = Env(flat_obs=flat_obs)
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        test_env.close()

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # # TODO: delete
    steps_per_epoch = 1000
    update_after = 100

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
            print(a)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        # print('reward at {}: {}'.format(t, r))
        env.render()
        ep_ret += r
        ep_len += 1
        # import pdb; pdb.set_trace()
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
                batch = replay_buffer.sample_batch(batch_size, device=device)
                # import pdb;
                # pdb.set_trace()
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # # Test the performance of the deterministic version of the agent.
            # num_test_episodes = 1
            # test_agent(num_test_episodes)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            # for h_i in range(len(ac.q1.layers)):
            #     if h_i % 2 == 0:
            #         logger.log_tabular('Q1Layer{}WeightGrad'.format(h_i), with_min_and_max=True)
            #         logger.log_tabular('Q1Layer{}BiasGrad'.format(h_i), with_min_and_max=True)
            #
            # # for h_i in range(len(ac.pi.cnn_layers)):
            # for h_i in range(1):
            #     if h_i % 2 == 0:
            #         logger.log_tabular('Layer{}PiWeightGrad'.format(h_i), with_min_and_max=True)
            #         logger.log_tabular('Layer{}PiBiasGrad'.format(h_i), with_min_and_max=True)
            # # for h_i in range(len(ac.pi.cnn_layers)):
            # for h_i in range(1):
            #     if h_i % 2 == 0:
            #         logger.log_tabular('AfterTunedLayer{}PiWeightGrad'.format(h_i), with_min_and_max=True)
            #         logger.log_tabular('AfterTunedLayer{}PiBiasGrad'.format(h_i), with_min_and_max=True)
            # for h_i in range(1):
            #     if h_i % 2 == 0:
            #         logger.log_tabular('Layer{}PiWeightGrad'.format(h_i), with_min_and_max=True)
            #         logger.log_tabular('Layer{}PiBiasGrad'.format(h_i), with_min_and_max=True)
            # for h_i in range(len(ac.pi.layers)):
            #     if h_i % 2 == 0:
            #         logger.log_tabular('AfterTunedLayer{}PiWeightGrad'.format(h_i), with_min_and_max=True)
            #         logger.log_tabular('AfterTunedLayer{}PiBiasGrad'.format(h_i), with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--act_hid', type=int, default=256)
    parser.add_argument('--act_l', type=int, default=2)
    parser.add_argument('--cri_hid', type=int, default=256)
    parser.add_argument('--cri_l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--act_noise', type=float, default=0.1)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument("--data_dir", type=str, default='po_spinup_data')
    args = parser.parse_args()

    data_dir = osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))),
        args.data_dir)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    td3(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(act_hidden_sizes=[args.act_hid]*args.act_l,
                       cri_hidden_sizes=[args.cri_hid]*args.cri_l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        act_noise=args.act_noise,
        logger_kwargs=logger_kwargs)
