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
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


