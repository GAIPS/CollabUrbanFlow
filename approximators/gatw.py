"""Graph Attention Mechanism

    References:
    ----------
    Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks.
    https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/train.py
"""
from torch import nn
import torch
import torch.nn.functional as F

class GATW(nn.Module):
    def __init__(self,
            in_features=4,
            n_embeddings=8,
            n_hidden=16,
            out_features=2,
            n_heads=1):
        """Dense version of GAT."""
        super(GATW, self).__init__()
        self.n_heads = n_heads

        self.embeddings = nn.Linear(in_features, n_embeddings)

        self.attentions = [GraphAttentionLayer(n_embeddings, n_hidden, n_hidden) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        self.heads = nn.Linear(n_hidden, n_hidden)

        self.prediction = nn.Linear(n_hidden, out_features)

    def forward(self, x, adj):
        # 1) Converts features to embeddings.
        x = self.embeddings(x)

        # TODO: Repeat 2-3-4 for multiple layers.
        # 2) Run n_heads attention mechanisms.
        x = torch.stack([att(x, adj) for att in self.attentions])

        # 3) Average the stacked heads (dim=0)
        x = torch.sum(x, dim=0) * (1 / self.n_heads)

        # 4) Apply a non-linear activation
        x = F.relu(self.heads(x))

        # 5) Predicion layer  
        x = self.prediction(x)

        return x

class GraphAttentionLayer(nn.Module):
    """Graph attention network"""
    def __init__(self, in_features, n_hidden, out_features,  **kwargs):
        """ Defines a single head Graph attention network

        Parameters:
        -----------
        * in_features: int
        size of the input layer -- for reinforcement learning
        observation/state size of the environment.
        * out_features: int
        size of the output layer -- for reinforcement learning
        number of discrete actions available in the environment.
        * hidden_size: int
        size of hidden layers.
        """
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.Ws = nn.Parameter(torch.empty(size=(in_features, n_hidden)))
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)

        self.Wt = nn.Parameter(torch.empty(size=(in_features, n_hidden)))
        nn.init.xavier_uniform_(self.Wt.data, gain=1.414)

        self.Wc = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.Wc.data, gain=1.414)


    def forward(self, h, adj):
        # h.shape: [B, N, E],
        # adj.shape: [N, N] 
        # Whx.shape: [B, N, H]
        Whs = torch.matmul(h, self.Ws)
        Wht = torch.matmul(h, self.Wt)

        # e.shape: [B, N, N]
        WhtT = torch.transpose(Wht, dim0=-2, dim1=-1)
        e = torch.matmul(Whs, WhtT)

        # zij = eij if j --> i
        # zij = epsilon otherwise, -9e15
        zero_vec = -9e15 * torch.ones_like(e)
        z = torch.where(adj > 0, e, zero_vec)
        alpha = F.softmax(z, dim=-1)

        Whc = torch.matmul(h, self.Wc)
        h_prime = torch.matmul(alpha, Whc)

        return h_prime
