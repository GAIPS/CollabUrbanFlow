"""DQN version 2.0

    References:
    ----------
    Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks.
    https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/train.py
"""
from torch import nn
import torch
import torch.nn.functional as F

from approximators.mlp import MLP

class DQN4(nn.Module):
    def __init__(self, n_agents=3, n_input=4, n_hidden=16, n_output=2):
        """Dense version of GAT."""
        super(DQN4, self).__init__()
        self.n_agents = n_agents
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.nets = []
        for n_a in range(n_agents):
            self.nets.append(
                MLP(n_input, n_output, n_hidden)
            )
            self.add_module(f'dqn_{n_a}', self.nets[-1])

        self.hparameters = {
            'hparams.n_agents': n_agents,
            'hparams.n_input': n_input,
            'hparams.n_hidden': n_hidden,
            'hparams.n_output': n_output,
        }

    def state_dict(self):
        state_dict = super(DQN4, self).state_dict()
        state_dict.update(self.get_extra_state())
        return state_dict

    def get_extra_state(self):
        return self.hparameters

    def forward(self, x):
        ''' x [B, n_agents, n_input] '''

        dim = 1 if len(x.shape) == 3 else 0
        xs = torch.tensor_split(x, self.n_agents, dim=dim)
        ys = []
        for n_a, x_a in enumerate(xs):
            net = self.nets[n_a] 
            ys.append(net(x_a))
        ret = torch.cat(ys, dim=dim)
        return ret
