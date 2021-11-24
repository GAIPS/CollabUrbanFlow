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

class GAT(nn.Module):
    def __init__(self, adjacency, n_input=4, n_embeddings=8,
                 n_hidden=16, n_heads=5, n_output=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        self.adjacency = adjacency
        self.n_agents = adjacency.shape[0]
        self.n_input = n_input
        self.n_embeddings = n_embeddings
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.n_output = n_output
        self.embeddings = nn.Linear(n_input, n_embeddings)
        self.add_module('embeddings', self.embeddings)

        self.attentions = []
        for n_h in range(n_heads):
            self.attentions.append(GraphAttentionLayer(adjacency, n_embeddings, n_hidden, n_hidden))
            self.add_module(f'attention_{n_h}', self.attentions[-1])

        self.head = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

        self.hparameters = {
            'hparams.n_agents': self.n_agents,
            'hparams.n_input': n_input,
            'hparams.n_embeddings':n_embeddings,
            'hparams.n_hidden': n_hidden,
            'hparams.n_heads': n_heads, 
            'hparams.n_output': n_output,
        }

    def state_dict(self):
        state_dict = super(GAT, self).state_dict()
        state_dict.update(self.get_extra_state())
        return state_dict

    def get_extra_state(self):
        return self.hparameters

    def forward(self, x):
        ''' x [B, n_agents, n_input] '''
        # 1) Converts features to embeddings.
        x = F.relu(self.embeddings(x))

        # 2) Run n_heads attention mechanisms.
        x = torch.stack([att(x) for att in self.attentions])

        # 3) Average the stacked heads (dim=0)
        x = torch.sum(x, dim=0) * (1 / self.n_heads)

        # 4) Apply a non-linear activation
        x = F.relu(self.head(x))

        # 5) Final prediction.
        x = self.predict(x) 

        # dim = 1 if len(x.shape) == 3 else 0
        # xs = torch.tensor_split(x, self.n_agents, dim=dim)
        # ys = []
        # for n_a, x_a in enumerate(xs):
        #     net = self.predictions[n_a] 
        #     ys.append(net(x_a))
        # ret = torch.cat(ys, dim=dim)
        # return ret
        return x

class GraphAttentionLayer(nn.Module):
    """Graph attention network"""
    def __init__(self, adjacency, n_input, n_hidden, n_output,  **kwargs):
        """ Defines a single head Graph attention network

        Parameters:
        -----------
        * n_input: int
        size of the input layer -- for reinforcement learning
        observation/state size of the environment.
        * n_output: int
        size of the output layer -- for reinforcement learning
        number of discrete actions available in the environment.
        * hidden_size: int
        size of hidden layers.
        """
        super(GraphAttentionLayer, self).__init__()
        self.adj = torch.tensor(adjacency)
        self.n_input = n_input
        self.n_output = n_output

        self.Ws = nn.Parameter(torch.empty(size=(n_input, n_hidden)))
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)

        self.Wt = nn.Parameter(torch.empty(size=(n_input, n_hidden)))
        nn.init.xavier_normal_(self.Wt.data, gain=1.414)

        self.Wc = nn.Parameter(torch.empty(size=(n_input, n_output)))
        nn.init.kaiming_normal_(self.Wc.data, nonlinearity='relu')


    def forward(self, h):
        # h.shape: [B, N, E],
        # adj.shape: [N, N] 
        # Whx.shape: [B, N, H], x in (s, t)
        adj = self.adj
        if len(h.shape) == 3: adj.repeat(h.shape[0], 1, 1)
        Whs = torch.matmul(h, self.Ws)
        Wht = torch.matmul(h, self.Wt)

        # WhtT.shape: [B, H, N]
        WhtT = torch.transpose(Wht, dim0=-2, dim1=-1)

        # e.shape: [B, N, N]
        e = torch.matmul(Whs, WhtT)

        # zij = eij if j --> i
        # zij = epsilon otherwise, -9e15
        zero_vec = -9e15 * torch.ones_like(e)
        z = torch.where(adj > 0, e, zero_vec)
        alpha = F.softmax(z, dim=-1)

        Whc = torch.matmul(h, self.Wc)
        h_prime = torch.matmul(alpha, Whc)

        return h_prime
