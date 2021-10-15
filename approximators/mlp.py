import torch
from torch import nn

class MLP(nn.Module):
    """Multi-Layer Perceptron  network used for function approximation.

    >>> MLP(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACML
    MLP(
      (net): Sequential(...)
    )
    """

    def __init__(self, obs_size=4, n_actions=2, hidden_size=32):
        """ Defines a single multi-layer percentron neutral networks.

        Parameters:
        -----------
        * obs_size: int
        size of the input layer -- for reinforcement learning
        observation/state size of the environment.
        * n_actions: int
        size of the output layer -- for reinforcement learning
        number of discrete actions available in the environment.
        * hidden_size: int
        size of hidden layers.
        """
        super().__init__()
        # at beginning of the script
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        ).to(device)

    def forward(self, x):
        return self.net(x.float())



class RNN(nn.Module):
    """Multi-Layer Perceptron  network used for function approximation.

    >>> MLP(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACML
    MLP(
      (net): Sequential(...)
    )
    """

    def __init__(self, obs_size=4, n_actions=2, hidden_size=32):
        """ Defines a single multi-layer percentron neutral networks.

        Parameters:
        -----------
        * obs_size: int
        size of the input layer -- for reinforcement learning
        observation/state size of the environment.
        * n_actions: int
        size of the output layer -- for reinforcement learning
        number of discrete actions available in the environment.
        * hidden_size: int
        size of hidden layers.
        """
        super(RNN, self).__init__()
        # at beginning of the script
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.layer_dim = 1
        self.rnn = nn.RNN(obs_size, hidden_size, self.layer_dim, batch_first=True, nonlinearity='relu').to(self.device)
        self.fc = nn.Linear(hidden_size, n_actions).to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_size, device = self.device).requires_grad_()
        x_ = x.view(-1, 1, self.obs_size).float().to(self.device)
        # One time step
        out, hn = self.rnn(x_, h0)
        out = self.fc(out[:, -1, :])
        return out



