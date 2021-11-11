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
    def __init__(self, in_features=4, n_embeddings=8, n_hidden=16,
                out_features=2, n_heads=1, n_layers=1):
        """Dense version of GAT."""
        super(GATW, self).__init__()
        self.n_agents = 3
        self.in_features = in_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        # self.embeddings = nn.Linear(in_features, n_embeddings)

        self.nets = []
        for n_a in range(self.n_agents):
            self.nets.append(
                nn.Sequential(
                    nn.Linear(in_features, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, out_features)
                )
            )
            self.add_module(f'dqn_{n_a}', self.nets[-1])

        self.hparameters = {
            'hparams.in_features': in_features,
            'hparams.n_embeddings':n_embeddings,
            'hparams.n_hidden': n_hidden,
            'hparams.out_features': out_features,
            'hparams.n_heads': n_heads, 
            'hparams.n_layers':n_layers
        }

    def state_dict(self):
        state_dict = super(GATW, self).state_dict()
        state_dict.update(self.get_extra_state())
        return state_dict

    def get_extra_state(self):
        return self.hparameters

    def forward(self, x, adj):
        ''' x [B, n_agents, n_input] '''
        # TODO: DEBUG
        dim = 1 if len(x.shape) == 3 else 0
        xs = torch.tensor_split(x, self.n_agents, dim=dim)
        # if dim == 1: import ipdb; ipdb.set_trace()

        ys = []
        for n_a in range(self.n_agents):
            net = self.nets[n_a] 
            ys.append(net(xs[n_a]))
        
        # stack
        y = torch.cat(ys, dim=dim)
        return y

        # # 1) Converts features to embeddings.
        # x = self.embeddings(x)

        # # Repeat 2-3-4 for multiple layers.
        # for j in range(self.n_layers):

        #     # 2) Run n_heads attention mechanisms.
        #     x = torch.stack([att(x, adj) for att in self.attentions[j]])

        #     # 3) Average the stacked heads (dim=0)
        #     x = torch.sum(x, dim=0) * (1 / self.n_heads)

        #     # 4) Apply a non-linear activation
        #     head = self.heads[j]
        #     x = F.relu(head(x))



# class GATW(nn.Module):
#     def __init__(self, in_features=4, n_embeddings=8, n_hidden=16,
#                 out_features=2, n_heads=1, n_layers=1):
#         """Dense version of GAT."""
#         super(GATW, self).__init__()
#         self.n_agents = 3
#         self.in_features = in_features
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         # self.embeddings = nn.Linear(in_features, n_embeddings)
# 
#         self.attentions = []
#         self.heads = []
#         # for j in range(n_layers):
#         #     # first layer has different size
#         #     n_input = n_hidden if j > 0 else n_embeddings
#         #     self.attentions.append([
#         #         GraphAttentionLayer(n_input, n_hidden, n_hidden) for _ in range(n_heads)
#         #     ])
#         #     for i, attention in enumerate(self.attentions[j]):
#         #         self.add_module(f'attention_{i}{j}', attention)
# 
# 
#         # self.heads.append(nn.Linear(n_hidden, n_hidden))
#         # self.add_module(f'heads_{j}', self.heads[-1])
#         # TODO: ERASEME
# 
#         self.heads, self.prediction = [], []
#         for n_a in range(self.n_agents):
# 
#             self.heads.append(nn.Linear(in_features, n_hidden))
#             self.add_module(f'head_{n_a}', self.heads[-1])
# 
#             self.prediction.append(nn.Linear(n_hidden, out_features))
#             self.add_module(f'pred_{n_a}', self.prediction[-1])
# 
# 
#         self.hparameters = {
#             'hparams.in_features': in_features,
#             'hparams.n_embeddings':n_embeddings,
#             'hparams.n_hidden': n_hidden,
#             'hparams.out_features': out_features,
#             'hparams.n_heads': n_heads, 
#             'hparams.n_layers':n_layers
#         }
# 
#     def state_dict(self):
#         state_dict = super(GATW, self).state_dict()
#         state_dict.update(self.get_extra_state())
#         return state_dict
# 
#     def get_extra_state(self):
#         return self.hparameters
# 
#     def forward(self, x, adj):
#         # # 1) Converts features to embeddings.
#         # x = self.embeddings(x)
# 
#         # # Repeat 2-3-4 for multiple layers.
#         # for j in range(self.n_layers):
# 
#         #     # 2) Run n_heads attention mechanisms.
#         #     x = torch.stack([att(x, adj) for att in self.attentions[j]])
# 
#         #     # 3) Average the stacked heads (dim=0)
#         #     x = torch.sum(x, dim=0) * (1 / self.n_heads)
# 
#         #     # 4) Apply a non-linear activation
#         #     head = self.heads[j]
#         #     x = F.relu(head(x))
# 
# 
#         # TODO: DEBUG
#         dim = 1 if len(x.shape) == 3 else 0
#         xs = torch.tensor_split(x, self.n_agents, dim=dim)
# 
#         ys = []
#         for n_a in range(self.n_agents):
#             head, pred = self.heads[n_a], self.prediction[n_a]
#             x = F.relu(head(xs[n_a]))
#             # 5) Predicion layer  
#             ys.append(pred(x))
# 
#         
#         # stack
#         y = torch.cat(ys, dim=dim)
# 
#         return y

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
        nn.init.xavier_normal_(self.Wt.data, gain=1.414)

        self.Wc = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.kaiming_normal_(self.Wc.data, nonlinearity='relu')


    def forward(self, h, adj):
        # h.shape: [B, N, E],
        # adj.shape: [N, N] 
        # Whx.shape: [B, N, H], x in (s, t)
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
