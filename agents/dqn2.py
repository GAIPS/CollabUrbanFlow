"""Deep Reinforcement Learning for Multi-agent system.

    Graph Attention Reinforcement Learning

    Paradigms:
    ---------
    * Parameter Sharing: att, W.
    * Centralized Training: Decentralized Execution.
    * Learning to communicate: Sends message to agents.

    To run a template:
    1) set agent_type = DQN2
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
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils.file_io import parse_train_config, \
    expr_logs_dump, expr_path_create, \
    expr_path_test_target
from utils.utils import concat, flatten
from agents.experience import Experience, ReplayBuffer, RLDataset
from approximators.dqn2 import DQN2

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
        n_agents = len(self.env.tl_ids)
        state = torch.tensor([self.state]).reshape((n_agents, -1)).to(device)

        for n_a, tl_id in enumerate(self.env.tl_ids):
            if np.random.random() < epsilon:
                action = np.random.choice((0, 1))
            else:
                q_values = net(state[n_a, :], n_a)
                action = int(torch.argmax(q_values, dim=-1))
            actions[tl_id] = int(action)
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


class DQN2Lightning(pl.LightningModule):
    """ Graph Attention Networks

    * For function approximation.
    * target_net: a dephased copy
    * ReplayBuffer: for storing experiences.


    >>> env = get_environment('arterial')
    >>> agent = get_agent('DQN2', env, epsilon_init, epsilon_final, epsilon_timesteps)
    >>> train_loop = get_loop('GATV')
    >>> info_dict = train_loop(env, agent, experiment_time,
                               save_agent_interval, chkpt_dir, seed)
    """

    def __init__(self, env, device, replay_size=200, warm_start_steps=0,
                 gamma=0.98, epsilon_init=1.0, epsilon_final=0.01, epsilon_timesteps=3500,
                 sync_rate=10, lr=1e-2, episode_timesteps=3600, batch_size=1000,
                 save_path=None, **kwargs):

        super(DQN2Lightning, self).__init__(**kwargs)
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
        self.n_hidden = 16
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

    
    def reset(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if self.num_episodes > 0:
            state_dict = torch.load(self.save_path / str(self.timestep) / f'DQN2.chkpt')
            self.net.load_state_dict(state_dict, strict=False)
            self.target_net.load_state_dict(state_dict, strict=False)
            self.agent.reset()
        else:
            self.net = DQN2(
                self.n_agents,
                self.n_input,
                self.n_hidden,
                self.n_output,
            ).to(device)
            self.target_net = DQN2(
                self.n_agents,
                self.n_input,
                self.n_hidden,
                self.n_output,
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
            self.agent.play_step(self.net, epsilon=self.epsilon, device=device)
        self.agent.reset()

    def loss_step(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.


        Parameters:
        -----------
        * batch: list<torch.Tensor> 
        List containing five elements:
        * state: torch.DoubleTensor<B, N * n_input>
        * action: torch.LongTensor<B, N>
        * reward: torch.DoubleTensor<B, N>
        * dones: torch.BoolTensor<B>
        * next_state: torch.DoubleTensor<B, N * n_input>

        Returns:
        --------
        * loss: torch.tensor([B])
        """
        states, actions, rewards, dones, next_states = self._debatch(batch)

        losses = []
        for n_a in range(self.n_agents):
            x = states[:, n_a, :].squeeze(1) 
            y = next_states[:, n_a, :].squeeze(1)
            u = actions[:, n_a].unsqueeze(-1)
            v = rewards[:, n_a]

            q_values = self.net(x, n_a)
            state_action_values = q_values.gather(-1, u).squeeze(-1)
            with torch.no_grad():
                next_state_values = self.target_net(y, n_a).argmax(-1)
                next_state_values = next_state_values.detach()


            expected_state_action_values = next_state_values * self.gamma + v

            loss = nn.MSELoss()(state_action_values, expected_state_action_values)
            losses.append(loss)

        return losses

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

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, self.epsilon, device)
        self.episode_reward += sum(reward) * 0.001


        # calculates training loss
        optimizers = self.optimizers(use_pl_optimizer=True)

        losses = self.loss_step(batch)
        loss = 0
        for i, opt in enumerate(optimizers):
            opt.zero_grad()
            losses[i].backward()
            opt.step()
            loss += losses[i].clone().detach()

        # Soft update of target network
        if self.episode_timestep % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict(), strict=False)

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

        self.log('loss', loss.to(device), logger=True, prog_bar=True)
        for k, v in log.items():
            self.log(k, v, logger=True, prog_bar=True)


    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizers = []
        for n_a in range(self.n_agents):
            optimizers.append(
                optim.Adam(self.net.individuals[n_a].parameters(), lr=self.lr)
            )
        return optimizers

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

    def train_dataloader(self):
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch):
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    """ Serialization """
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'DQN2.chkpt'
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.net.state_dict(), file_path)

def load_checkpoint(env, chkpt_dir_path, rollout_time=None, network=None, chkpt_num=None):
    if chkpt_num == None:
        chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)

    state_dict = torch.load(chkpt_path / f'DQN2.chkpt')
    n_agents = state_dict['hparams.n_agents']
    n_input = state_dict['hparams.n_input']
    n_hidden = state_dict['hparams.n_hidden']
    n_output = state_dict['hparams.n_output']

    net = DQN2(n_agents=n_agents, n_input=n_input,
               n_hidden=n_hidden, n_output=n_output)
    net.load_state_dict(state_dict, strict=False)
    agent = Agent(env)
    return agent, net
