import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class LSTMActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, lstm_hidden_dim=128, lstm_hidden_num_layers=2,
                 fc_hidden_sizes=(256,)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        # LSTM layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_hidden_num_layers = lstm_hidden_num_layers
        self.lstm = nn.LSTM(obs_dim, self.lstm_hidden_dim, self.lstm_hidden_num_layers, batch_first=True)
        # Fully connected layers
        self.fc_layers = []
        self.fc_hidden_sizes = [self.lstm_hidden_dim] + list(fc_hidden_sizes)
        for j in range(len(self.fc_hidden_sizes) - 1):
            self.fc_layers += [nn.Linear(self.fc_hidden_sizes[j], self.fc_hidden_sizes[j + 1]), nn.ReLU()]
        # Output layer
        self.out_layer = [nn.Linear(self.fc_hidden_sizes[-1], self.act_dim), nn.Tanh()]

    def forward(self, obs, seg_len=None):
        # LSTM layers
        if seg_len is not None:
            obs_packed = pack_padded_sequence(obs, lengths=seg_len, batch_first=True, enforce_sorted=False)
        else:
            obs_packed = pack_padded_sequence(obs, lengths=[obs.size(1) for _ in range(obs.size(0))], batch_first=True,
                                              enforce_sorted=False)
        lstm_output_packed, (lstm_hidden_state, lstm_cell_state) = self.lstm(obs_packed)
        lstm_output_padded, lstm_output_lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # Fully connected layers
        fc_output = lstm_output_padded
        for fc_l in self.fc_layers:
            fc_output = fc_l(fc_output)
        # Return output from network scaled to action space limits.
        output = fc_output
        for out_l in self.out_layer:
            output = out_l(output)
        return self.act_limit * output


class LSTMQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, lstm_hidden_dim=128, lstm_hidden_num_layers=2, fc_hidden_sizes=(256,)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # LSTM layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_hidden_num_layers = lstm_hidden_num_layers
        self.lstm_layer = nn.LSTM(obs_dim+act_dim, self.lstm_hidden_dim, self.lstm_hidden_num_layers, batch_first=True)
        # Fully connected layers
        self.fc_layers = []
        self.fc_hidden_sizes = [self.lstm_hidden_dim] + list(fc_hidden_sizes)
        for j in range(len(self.fc_hidden_sizes) - 1):
            self.fc_layers += [nn.Linear(self.fc_hidden_sizes[j], self.fc_hidden_sizes[j + 1]), nn.ReLU()]
        # Output layer
        self.out_layer = [nn.Linear(self.fc_hidden_sizes[-1], 1), nn.Identity()]
        # self.layers = nn.ModuleList()
        # self.layers += [self.lstm_layer] + self.fc_layers + self.out_layer

    def forward(self, obs, act, seg_len=None):
        # LSTM layers
        cat_input = torch.cat([obs, act], dim=-1)
        if seg_len is not None:
            input_packed = pack_padded_sequence(cat_input, lengths=seg_len,
                                                batch_first=True, enforce_sorted=False)
        else:
            input_packed = pack_padded_sequence(cat_input, lengths=[cat_input.size(1) for _ in range(cat_input.size(0))],
                                                batch_first=True, enforce_sorted=False)

        lstm_output_packed, (lstm_hidden_state, lstm_cell_state) = self.lstm_layer(input_packed)
        lstm_output_padded, lstm_output_lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # Fully connected layers
        fc_output = lstm_output_padded
        for fc_l in self.fc_layers:
            fc_output = fc_l(fc_output)
        output = fc_output
        for out_l in self.out_layer:
            output = out_l(output)
        return output


class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        # build policy and value functions
        self.pi = LSTMActor(obs_dim, act_dim, act_limit, lstm_hidden_dim=128, lstm_hidden_num_layers=2,
                            fc_hidden_sizes=(256,))
        self.q = LSTMQFunction(obs_dim, act_dim, lstm_hidden_dim=128, lstm_hidden_num_layers=2,
                               fc_hidden_sizes=(256,))

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

