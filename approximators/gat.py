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

class GAT(nn.Module):
    def __init__(self, in_features, n_hidden, n_classes, dropout, alpha, n_heads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_features, n_hidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_hidden * n_heads, n_classes, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    """Graph attention network"""
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, **kwargs):
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

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # adj_mtrx [source, in_feature]  [target, in_feature]
    def forward(self, hi, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.matmul(hi, self.W) 
        # if len(hi.shape) == 2:
        #     Wh1 = torch.mm(hi, self.W) 
        #     try:
        #         assert torch.equal(Wh1, Wh)
        #     except AssertionError:
        #         import ipdb; ipdb.set_trace()

        e = self._prepare_attentional_mechanism_input(Wh)
        
        # if len(hi.shape) == 3:
        #     import ipdb; ipdb.set_trace()
        zero_vec = -9e15*torch.ones_like(e)
        try:
            attention = torch.where(adj > 0, e, zero_vec)
        except Exception:
            import ipdb; ipdb.set_trace()
        # attention = F.softmax(attention, dim=1)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        # e1 = Wh1 + Wh2.T

        #e2 = Wh1 + torch.transpose(Wh2, dim0=0,dim1=-1)
        # if len(Wh.shape) == 3:
        e2 = Wh1 + torch.transpose(Wh2, dim0=-2,dim1=-1)
        # else:
        #     e2 = Wh1 + torch.transpose(Wh2, dim0=0,dim1=-1)
        # try:
        #     assert torch.equal(e1, e2)
        # except AssertionError:
        #     import ipdb; ipdb.set_trace()
        return self.leakyrelu(e2)


