"""Deep Reinforcement Learning for Multi-agent system.

    Graph Attention Reinforcement Learning

    Paradigms:
    ---------
    * Parameter Sharing: att, W.
    * Centralized Training: Decentralized Execution.
    * Learning to communicate: Sends message to agents. 
    
    To run a template:
    1) set agent_type = GATV
    >>> python models/train.py
    >>> tensorboard --logdir lightning_logs


    TODO:
    ----
    * Move adjacency matrix to module.

    References:
    -----------
    Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero,
    Pietro Lio, and Yoshua Bengio. 2017.
    `Graph attention networks. 2017`
    https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/train.py
"""
import argparse
from pathlib import Path
from collections import OrderedDict

from tqdm.auto import trange
import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils.file_io import parse_train_config, \
    expr_logs_dump, expr_path_create, \
    expr_path_test_target
from plots.train_plots import main as train_plots
from plots.test_plots import main as test_plots
from utils.utils import concat, flatten
from utils.network import get_neighbors
from agents.experience import Experience, ReplayBuffer, RLDataset
from approximators.gatw import GATW

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

def simple_hash(x): return hash(x) % (11 * 255)

def get_adjacency_matrix(edge_list):
    num_nodes = max(edge_list, key=lambda x: max(x))

    data = np.ones(len(edge_list), dtype=int)
    adj = csr_matrix((data, zip(*edge_list)), dtype=int).todense()
    adj = Variable(torch.tensor(adj))
    return adj

# Should this agent be general?
# Or, should function approximation be associated to agnt
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

        edge_list, _ = get_neighbors(env.incoming_roadlinks, env.outgoing_roadlinks)
        self._adjacency_matrix = get_adjacency_matrix(edge_list)
        self.reset()

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

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
        actions = {}
        N = len(self.env.tl_ids)

        state = torch.tensor([self.state]).reshape((N, -1))

        if device not in ["cpu"]:
            state = state.cuda(device)

        q_values = net(state, self.adjacency_matrix)

        # Exploration & Exploitation:
        # XOR operation flips correctly
        # TODO: verify flip
        actions = torch.argmax(q_values, dim=1).numpy()
        flip = np.random.random(N)
        actions = np.logical_xor(actions.astype(bool), flip < epsilon)
        actions = dict(zip(self.env.tl_ids, actions.astype(int).tolist()))
        return actions

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

        new_state, reward, done, _ = experience
        new_state, reward, actions = \
            list(flatten(new_state.values())), list(reward.values()), list(actions.values())
        if epsilon > 0.0:
            exp = Experience(self.state, actions, reward, done, new_state)

            self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class GATWLightning(pl.LightningModule):
    """ Graph Attention Networks

    * For function approximation.
    * target_net: a dephased copy
    * ReplayBuffer: for storing experiences.


    >>> env = get_environment('arterial')
    >>> agent = get_agent('GATW', env, epsilon_init, epsilon_final, epsilon_timesteps)
    >>> train_loop = get_loop('GATV')
    >>> info_dict = train_loop(env, agent, experiment_time,
                               save_agent_interval, chkpt_dir, seed)
    """

    def __init__(self, env, replay_size=200, warm_start_steps=0,
                 gamma=0.98, epsilon_init=1.0, epsilon_final=0.01, epsilon_timesteps=3500,
                 sync_rate=10, lr=1e-2, episode_timesteps=3600, batch_size=1000,
                 save_path=None, **kwargs,
                 ):
        super().__init__(**kwargs)
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
        self.obs_size = 4
        self.embeddings = 8
        self.hidden_size = 16
        self.num_actions = 2
        self.num_intersections = len(self.env.tl_ids)

        # Define GAT's parameters
        n_heads = 1

        self.net = GATW(
            self.obs_size,
            self.embeddings,
            self.hidden_size,
            self.num_actions,
            n_heads
        ) 
        self.target_net = GATW(
            self.obs_size,
            self.embeddings,
            self.hidden_size,
            self.num_actions,
            n_heads
        )

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self._total_timestep = 0
        self.cum_loss = 0
        if self.warm_start_steps > 0: self.populate(self.warm_start_steps)
        self.save_path = save_path

    @property
    def timestep(self):
        return self.agent.env.timestep

    @property
    def total_timestep(self):
        return self._total_timestep + self.timestep

    @property
    def adjacency_matrix(self):
        return self.agent.adjacency_matrix

    def populate(self, steps=1000):
        """Carries out several random steps through the
           environment to initially fill up the replay buffer with
           experiences.

        Parameters:
        -----------
        * steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
        self.agent.reset()

    def forward(self, x):
        """Passes in a state `x` through the network and gets the
           `q_values` of each action as an output.

        Parameters:
        -----------
        * x: environment state

        Returns:
        --------
        * q-values
        """
        adj = self.adjacency_matrix
        # batch_size != num_intersections
        adj = adj.repeat(x.shape[0], 1, 1)
        output = self.net(x, adj)
        return output

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Parameters:
        -----------
        * batch: torch.tensor([B, N * obs_size])
        Current mini batch of replay data

        Returns:
        --------
        * loss: torch.tensor([B, N * obs_size])
        """
        states, actions, rewards, dones, next_states = batch


        x = states.view((-1, self.num_intersections, self.obs_size))
        x = x.type(torch.FloatTensor)
        q_values = self.forward(x)
        state_action_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)


        with torch.no_grad():
            y = states.view((-1, self.num_intersections, self.obs_size))
            y = y.type(torch.FloatTensor)


            adj = self.adjacency_matrix.repeat(y.shape[0], 1, 1)
            next_state_values = self.target_net(y, adj).argmax(-1)

            next_state_values[dones] = 0.0

            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        return loss 

    def training_step(self, batch, nb_batch):
        """Carries out a single step through the environment to update
           the replay buffer. Then calculates loss based on the minibatch
           received.

        Parameters:
        -----------
        * batch:
        Current mini batch of replay data
        * nb_batch:
        Batch number
        """
        device = self.get_device(batch)
        epsilon = max(self.epsilon_final,
                      self.epsilon_init - \
                      (self.total_timestep + 1) / self.epsilon_timesteps)

        
        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += sum(reward) * 0.001

        loss = self.dqn_mse_loss(batch)
        self.cum_loss += loss.clone().detach().numpy()
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self._total_timestep += self.episode_timesteps
            self.cum_loss = 0
            if self.save_path is not None:
                self.save_checkpoint(self.save_path, self.total_timestep)
            print('')  # Skip an output line

        # Soft update of target network
        if self.timestep % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "steps": torch.tensor(self.timestep).to(device),
            "reward": torch.tensor(self.total_reward).to(device),
            "epsilon": torch.tensor(np.round(epsilon, 4)).to(device),
        }

        self.log('loss', loss.clone().detach().to(device), logger=True, prog_bar=True)
        for k, v in log.items():
            self.log(k, v, logger=True, prog_bar=True)

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

        for i, tl_id in enumerate(self.env.tl_ids):
            file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'GATW.chkpt'
            file_path.parent.mkdir(exist_ok=True)
            torch.save(self.net.state_dict(), file_path)

def load_checkpoint(env, chkpt_dir_path, rollout_time, network, chkpt_num=None):
    if chkpt_num == None:
        chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)

    # TODO: dropout, alpha, n_heads ?
    state_dict = torch.load(chkpt_path / f'GATW.chkpt')
    in_features, n_hidden = state_dict['attention_0.W'].shape
    out_features = state_dict['out_att.W'].shape[-1]
    net = GATV(in_features=in_features, n_hidden=n_hidden, n_classes=out_features, n_heads=5)
    net.load_state_dict(state_dict)
    agent = Agent(env)
    return agent, net
