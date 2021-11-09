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
from pathlib import Path
from collections import OrderedDict
from functools import cached_property
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
from utils.utils import concat, flatten
from utils.network import get_adjacency_from_env
from agents.experience import Experience, ReplayBuffer, RLDataset
from approximators.gatw import GATW

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

def simple_hash(x): return hash(x) % (11 * 255)

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

        self.reset()

    @cached_property
    def adjacency_matrix(self):
        return get_adjacency_from_env(self.env)

    def reset(self):
        """Resets the environment and updates the state."""
        # Assumption: All intersections have the same phase.
        self.state = list(flatten(self.env.reset().values()))

    def act(self, net, epsilon, device, adj):
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

        state = torch.tensor([self.state]).reshape((N, -1)).to(device)


        q_values = net(state, adj)

        # Exploration & Exploitation:
        # XOR operation flips correctly
        with torch.no_grad():
            actions = torch.argmax(q_values, dim=1)
            flip = torch.rand((N,)).to(device) < epsilon / 2
            actions = actions.type(torch.bool).bitwise_xor(flip)
            actions = dict(zip(self.env.tl_ids, actions.cpu().numpy().astype(int).tolist()))
        return actions

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu", adj=None):
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
        if adj is None: adj = torch.tensor(self.adjacency_matrix).to(device)
        actions = self.act(net, epsilon, device, adj)

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

    def __init__(self, env, device, replay_size=200, warm_start_steps=0,
                 gamma=0.98, epsilon_init=1.0, epsilon_final=0.01, epsilon_timesteps=3500,
                 sync_rate=10, lr=1e-2, episode_timesteps=3600, batch_size=1000,
                 save_path=None, **kwargs):

        super(GATWLightning, self).__init__(**kwargs)
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
        self.num_intersections = len(self.env.tl_ids)
        self.num_episodes = 0

        # TODO: GATs hyperparams
        self.obs_size = 4
        self.num_embeddings = 8
        self.hidden_size = 16
        self.num_actions = 2
        self.num_heads = 5
        self.num_layers = 1

        # Auxiliary variables
        self._state_view_shape = (-1, self.num_intersections, self.obs_size)
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
            state_dict = torch.load(self.save_path / str(self.timestep) / f'GATW.chkpt')
            self.net.load_state_dict(state_dict, strict=False)
            self.target_net.load_state_dict(state_dict, strict=False)
            self.agent.reset()
        else:
            self.net = GATW(
                self.obs_size,
                self.num_embeddings,
                self.hidden_size,
                self.num_actions,
                self.num_heads,
                self.num_layers
            ).to(device)
            self.target_net = GATW(
                self.obs_size,
                self.num_embeddings,
                self.hidden_size,
                self.num_actions,
                self.num_heads,
                self.num_layers
            ).to(device)

            self.buffer = ReplayBuffer(self.replay_size)
            self.agent = Agent(self.env, self.buffer)
            if self.warm_start_steps > 0: self.populate(device=device, steps=self.warm_start_steps)
        self.episode_reward = 0

    def populate(self, device=None, steps=1000):
        """Carries out several random steps through the
           environment to initially fill up the replay buffer with
           experiences.

        Parameters:
        -----------
        * steps: number of random steps to populate the buffer with
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=self.epsilon, device=device, adj=self.adjacency_matrix)
        self.agent.reset()

    def loss_step(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.


        Parameters:
        -----------
        * batch: list<torch.Tensor> 
        List containing five elements:
        * state: torch.DoubleTensor<B, N * obs_size>
        * action: torch.LongTensor<B, N>
        * reward: torch.DoubleTensor<B, N>
        * dones: torch.BoolTensor<B>
        * next_state: torch.DoubleTensor<B, N * obs_size>

        Returns:
        --------
        * loss: torch.tensor([B])
        """
        states, actions, rewards, dones, next_states, adj = self._debatch(batch)

        q_values = self.net(states, adj)
        state_action_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states, adj).argmax(-1)

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
        adj = self.adjacency_matrix

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, self.epsilon, device, adj)
        self.episode_reward += sum(reward) * 0.001

        loss = self.loss_step(batch)

        # Soft update of target network
        if self.episode_timestep % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict(), strict=False)

        if done:
            self.num_episodes += 1
            # save and reset the network
            # update log.
            self._timestep += self.episode_timesteps
            self._reward += self.episode_reward
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


    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.episode_timesteps)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)
        return dataloader

    def _debatch(self, batch):
        '''Splits and processes batch.

        Parameters:
        -----------
        * batch: list<torch.Tensor> 
        List containing five elements:
        * state, torch.DoubleTensor<B, N * obs_size>
        * action, torch.LongTensor<B, N>
        * reward, torch.DoubleTensor<B, N>
        * dones, torch.BoolTensor<B>
        * next_state, torch.DoubleTensor<B, N * obs_size>


        Returns:
        --------
        * state: torch.FloatTensor<B, N, obs_size>
        * action: torch.LongTensor<B, N>
        * reward: torch.DoubleTensor<B, N>
        * dones: torch.BoolTensor<B>
        * next_state: torch.FloatTensor<B, N * obs_size>
        * adj: torch.FloatTensor<B, N, N>
        ''' 
        device = self.get_device(batch)
        states, actions, rewards, dones, next_states = batch
        states = self._debatch_state(states)
        next_states = self._debatch_state(next_states)
        adj = self.adjacency_matrix.repeat(states.shape[0], 1, 1)
        return states, *batch[1:-1], next_states, adj

    
    def _debatch_state(self, states):
        ret = states.view(self._state_view_shape). \
              type(torch.FloatTensor).to(states.device)
        return ret

    def train_dataloader(self):
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch):
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    """ Serialization """
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'GATW.chkpt'
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.net.state_dict(), file_path)

def load_checkpoint(env, chkpt_dir_path, rollout_time=None, network=None, chkpt_num=None):
    if chkpt_num == None:
        chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)

    state_dict = torch.load(chkpt_path / f'GATW.chkpt')
    in_features = state_dict['hparams.in_features']
    num_embeddings = state_dict['hparams.n_embeddings']
    n_hidden = state_dict['hparams.n_hidden']
    out_features = state_dict['hparams.out_features']
    num_heads = state_dict['hparams.n_heads']
    num_layers = state_dict['hparams.n_layers']

    net = GATW(in_features=in_features, n_embeddings=num_embeddings,
               n_hidden=n_hidden, out_features=out_features,
               n_heads=num_heads, n_layers=num_layers)
    net.load_state_dict(state_dict, strict=False)
    agent = Agent(env)
    return agent, net
