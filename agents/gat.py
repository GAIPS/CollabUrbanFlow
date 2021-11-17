"""Deep Reinforcement Learning for Multi-agent system.

    Graph Attention Reinforcement Learning

    Paradigms:
    ---------
    * Parameter Sharing: att, W.
    * Centralized Training: Decentralized Execution.
    * Learning to communicate: Sends message to agents.

    To run a template:
    1) set agent_type = GATW
    >>> python models/train.py
    >>> tensorboard --logdir lightning_logs


    TODO:
    ----
    * Move adjacency matrix to module.

    References:
    -----------
    `Graph attention networks. 2017`
    https://arxiv.org/abs/1710.10903
    Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017.
    https://github.com/Diego999/pyGAT/blob/master/train.py
"""

import argparse
from functools import cached_property
from collections import deque, namedtuple, OrderedDict
from pathlib import Path


import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from environment import Environment
from utils.utils import flatten
from utils.network import get_adjacency_from_env
from agents.experience import Experience, ReplayBuffer, RLDataset
from approximators.gat import GAT

class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = get_environment('arterial')
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...gat.Agent object at ...>
    """

    def __init__(self, env, replay_buffer=None):
        """
        Parameters:
        -----------
        * env: environment.Environment
            Trainning environment
        * replay_buffer: dqn.ReplayBuffer
            replay buffer storing experiences
        """

        self.env = env
        self.replay_buffer = replay_buffer

        self.reset()

    @cached_property
    def adjacency_matrix(self):
        return get_adjacency_from_env(self.env)

    def reset(self):
        """Resets the environment and updates the state."""
        # Assumption: All intersections have the same phase.
        self.state = list(flatten(self.env.reset().values()))

    def act(self, net, epsilon, device):
        """For a given network, decide what action to carry out
            using an epsilon-greedy policy.

        Parameters:
        -----------
        * epsilon: float
            Value to determine likelihood of taking a random action
        * device: current device

        Returns:
        --------
        * actions: dict<str, int>
        contains actions from all agents.
        """
        # actions = {}
        n_agents = len(self.env.tl_ids)
        state = torch.tensor([self.state]).reshape((n_agents, -1)).to(device)

        actions = net(state).argmax(dim=-1).clone().detach().cpu().numpy()
        choice = np.random.choice((0, 1), replace=True, size=n_agents)
        flip = np.random.rand(n_agents) < epsilon
        actions = np.where(flip, choice, actions)
        return dict(zip(self.env.tl_ids, actions))

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        """Carries out a single interaction step between the agent
        and the environment.

        Parameters:
        -----------
        * net: list<dqn.DQN>
        Deep Q-network one per agent.
        * epsilon: float
        Value to determine likelihood of taking a random action
        * device: str
        Current device: 'cpu' or 'tpu'

        Returns:
        --------
        * reward: list<float>
        reward for each agent.
        * done: bool
        if the episode is over
        """
        actions = self.act(net, epsilon, device)

        # do step in the environment -- 10s
        for _ in range(10):
            experience = self.env.step(actions)

        next_state, reward, done, _ = experience
        next_state, reward, actions = \
            list(flatten(next_state.values())), list(reward.values()), list(actions.values())
        if epsilon > 0.0:
            exp = Experience(self.state, actions, reward, done, next_state)

            self.replay_buffer.append(exp)

        self.state = next_state
        if done:
            self.reset()
        return reward, done



class GATLightning(pl.LightningModule):
    """Basic DQN Model.

    >>> GATLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )
    """

    def __init__(self, env, device, replay_size=200, warm_start_steps=0,
                 gamma=0.98, epsilon_init=1.0, epsilon_final=0.01, epsilon_timesteps=3500,
                 sync_rate=10, lr=1e-2, episode_timesteps=3600, batch_size=1000,
                 save_path=None, **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_timesteps = epsilon_timesteps
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_timesteps = episode_timesteps
        self.batch_size = batch_size

        self.env = env
        self.save_path = save_path
        self.num_episodes = 0


        self.n_agents = len(self.env.tl_ids)
        self.n_input = 4
        self.n_embeddings = 16
        self.n_hidden = 32
        self.n_heads = 5
        self.n_output = 2

        # Auxiliary variables
        self._state_view_shape = (-1, self.n_agents, self.n_input)
        self._timestep = 0
        self._reward = 0

        self.reset(device=device)

    @property
    def episode_timestep(self):
        return self.agent.env.timestep

    @property
    def timestep(self):
        return self._timestep + self.episode_timestep

    @property
    def reward(self):
        return self._reward

    @property
    def epsilon(self):
        epsilon = max(self.epsilon_final,
                      self.epsilon_init - \
                      (self.timestep + 1) / self.epsilon_timesteps)
        return epsilon

    @cached_property
    def adjacency_matrix(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(self.agent.adjacency_matrix).to(device)

    def reset(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if self.num_episodes > 0:
            state_dict = torch.load(self.save_path / str(self.timestep) / f'GAT.chkpt')
            self.net.load_state_dict(state_dict, strict=False)
            self.target_net.load_state_dict(state_dict, strict=False)
            self.agent.reset()
        else:
            self.buffer = ReplayBuffer(self.replay_size)
            self.agent = Agent(self.env, self.buffer)
            self.net = GAT(
                self.adjacency_matrix,
                self.n_input,
                self.n_embeddings,
                self.n_hidden,
                self.n_heads,
                self.n_output,
            ).to(device)
            self.target_net = GAT(
                self.adjacency_matrix,
                self.n_input,
                self.n_embeddings,
                self.n_hidden,
                self.n_heads,
                self.n_output,
            ).to(device)
            if self.warm_start_steps > 0: self.populate(device=device, steps=self.warm_start_steps)
        self.episode_reward = 0

    def populate(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), steps=1000):
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=self.epsilon, device=device)
        self.env.reset(); self.agent.reset()

    def forward(self, x):
        """Passes in a state `x` through the network and gets the `q_values` of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = self._debatch(batch)

        q_values = self.net(states)
        state_action_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values, _ = self.target_net(next_states).max(-1)
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch, nb_batch):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, self.epsilon, device)
        self.episode_reward += sum(reward) * 0.001

        # calculates training loss
        opt = self.optimizers(use_pl_optimizer=True)
        loss = self.dqn_mse_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if done:
            self.num_episodes += 1
            # save and reset the network
            # update log.
            self._timestep += self.episode_timesteps
            self._reward = self.episode_reward
            if self.save_path is not None:
                self.save_checkpoint(self.save_path, self.timestep)
                self.reset()
            print('')  # Skip an output line

        log = {
            "steps": torch.tensor(self.episode_timestep).to(device),
            "reward": torch.tensor(self.reward).to(device),
            "epsilon": torch.tensor(np.round(self.epsilon, 4)).to(device),
        }

        self.log('loss', loss.clone().detach().to(device), logger=True, prog_bar=True)
        for k, v in log.items():
            self.log(k, v, logger=True, prog_bar=True)


    def _debatch(self, batch):
        '''Splits and processes batch.

        Parameters:
        -----------
        * batch: list<torch.Tensor> 
        List containing five elements:
        * state, torch.DoubleTensor<B, N * n_input>
        * action, torch.LongTensor<B, N>
        * reward, torch.DoubleTensor<B, N>
        * dones, torch.BoolTensor<B>
        * next_state, torch.DoubleTensor<B, N * n_input>


        Returns:
        --------
        * state: torch.FloatTensor<B, N, n_input>
        * action: torch.LongTensor<B, N>
        * reward: torch.DoubleTensor<B, N>
        * dones: torch.BoolTensor<B>
        * next_state: torch.FloatTensor<B, N * n_input>
        ''' 
        device = self.get_device(batch)
        states, actions, rewards, dones, next_states = batch
        states = self._debatch_state(states)
        next_states = self._debatch_state(next_states)
        return states, *batch[1:-1], next_states

    def _debatch_state(self, states):
        ret = states.view(self._state_view_shape). \
              type(torch.FloatTensor).to(states.device)
        return ret
    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.episode_timesteps)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)
        return dataloader

    def train_dataloader(self):
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch):
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    """ Serialization """
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'GAT.chkpt'
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.net.state_dict(), file_path)

def load_checkpoint(env, chkpt_dir_path, rollout_time=None, network=None, chkpt_num=None):
    if chkpt_num == None:
        chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)

    agent = Agent(env)
    state_dict = torch.load(chkpt_path / f'GAT.chkpt')
    n_agents = state_dict['hparams.n_agents']
    n_input = state_dict['hparams.n_input']
    n_embeddings = state_dict['hparams.n_embeddings']
    n_hidden = state_dict['hparams.n_hidden']
    n_heads = state_dict['hparams.n_heads']
    n_output = state_dict['hparams.n_output']

    net = GAT(agent.adjacency_matrix, n_input=n_input,
               n_embeddings=n_embeddings, n_hidden=n_hidden,
               n_heads=n_heads, n_output=n_output)
    net.load_state_dict(state_dict, strict=False)
    return agent, net
