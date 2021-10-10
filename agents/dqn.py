"""Deep Reinforcement Learning for Multi-agent system.

* Deep Q-networks

* Using Deep Q-networks for independent learners.

* TODO:
    Creates a MAS class that controls agents.
    Separate buffers.

To run the template, just run:
`python dqn.py`

`tensorboard --logdir lightning_logs`

References:
-----------
* Deep Q-network (DQN)

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""
import argparse
# from collections import deque, namedtuple, OrderedDict
# from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from tqdm.auto import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
# from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.file_io import parse_train_config, \
    expr_logs_dump, expr_path_create, \
    expr_path_test_target
from plots.train_plots import main as train_plots
from plots.test_plots import main as test_plots
from utils.utils import concat
from agents.experience import Experience, ReplayBuffer, RLDataset
from approximators.mlp import MLP

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

def simple_hash(x): return hash(x) % (11 * 255)


def flatten(items, ignore_types=(str, bytes)):
    """

    Usage:
    -----
    > items = [1, 2, [3, 4, [5, 6], 7], 8]

    > # Produces 1 2 3 4 5 6 7 8
    > for x in flatten(items):
    >         print(x)

    Ref:
    ----

    David Beazley. `Python Cookbook.'
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


def get_experience(batch, obs_size, agent_index):
    states, actions, rewards, dones, next_states = batch
    return (
               states.split(obs_size, dim=1)[agent_index],
               actions.split(1, dim=-1)[agent_index],
               rewards.split(1, dim=-1)[agent_index],
               next_states.split(obs_size, dim=1)[agent_index]
           ), dones



class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = get_environment('arterial')
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...dqn.Agent object at ...>
    """

    def __init__(self, env, replay_buffer=None):
        """
        Parameters:
        -----------
        * env: environment.EnvironmentGymWrapper
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
        for i, tl_id in enumerate(self.env.tl_ids):
            if np.random.random() < epsilon:
                # TODO: Actions are change or keep for all intersections.
                action = np.random.choice((0, 1))
            else:
                state = torch.tensor([self.state]).split(4, dim=1)[i]

                if device not in ["cpu"]:
                    state = state.cuda(device)

                q_values = net[i](state)
                _, action = torch.max(q_values, dim=1)
                action = int(action.item())
            actions[tl_id] = action
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


class DQNLightning(pl.LightningModule):
    """Basic DQN Model.

    * Multi-layer Perceptron: for function approximation.
    * target_net: a dephased copy
    * ReplayBuffer: for storing experiences.

    >>> DQNLightning(env="intersection")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): MLP(
        (net): Sequential(...)
      )
      (target_net): MLP(
        (net): Sequential(...)
      )
    )
    """

    def __init__(self, network='intersection', replay_size=200, warm_start_steps=0,
                 gamma=0.98, epsilon_init=1.0, epsilon_final=0.01, epsilon_timesteps=3500,
                 sync_rate=10, lr=1e-2, episode_timesteps=3600, batch_size=1000,
                 save_path=None, **kwargs,
                 ):
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

        from environment import get_environment
        self.env = get_environment(network, episode_timesteps=episode_timesteps)
        self.obs_size = 4
        self.hidden_size = 32
        self.num_actions = 2
        self.num_intersections = len(self.env.tl_ids)

        self.net = [MLP(
            self.obs_size,
            self.num_actions,
            self.hidden_size
        ) for _ in range(self.num_intersections)]
        self.target_net = [MLP(
            self.obs_size,
            self.num_actions,
            self.hidden_size
        ) for _ in range(self.num_intersections)]

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self._total_timestep = 0
        if self.warm_start_steps > 0: self.populate(self.warm_start_steps)
        self.save_path = save_path

    @property
    def timestep(self):
        return self.agent.env.timestep

    @property
    def total_timestep(self):
        return self._total_timestep + self.timestep

    def populate(self, net, steps=1000):
        """Carries out several random steps through the
           environment to initially fill up the replay buffer with
           experiences.

        Parameters:
        -----------
        * steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(net, epsilon=1.0)

    def forward(self, i, x):
        """Passes in a state `x` through the network and gets the
           `q_values` of each action as an output.

        Parameters:
        -----------
        * x: environment state

        Returns:
        --------
        * q-values
        """
        output = self.net[i](x)
        return output

    def dqn_mse_loss(self, batch, agent_index):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Parameters:
        -----------
        * batch: torch.tensor([B, N * obs_size])
        Current mini batch of replay data

        Returns:
        --------
        * loss: torch.tensor([B, N * obs_size])
        """
        # split and iterate over sections! -- beware assumption
        # gets experience from the agent
        batch_data, dones = get_experience(batch, self.obs_size, agent_index)
        net = self.net[agent_index]
        target_net = self.target_net[agent_index]
        s_, a_, r_, s1_ = batch_data
        state_action_values = self.net[agent_index](s_).gather(1, a_).squeeze(-1)

        with torch.no_grad():
            next_state_values = target_net(s1_).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + r_.squeeze(-1)

        return nn.MSELoss()(state_action_values, expected_state_action_values)

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

        # calculates training loss
        optimizers = self.optimizers(use_pl_optimizer=True)

        # MAS -- Q_n(s_n, u_n), for n=1,...,|N|
        # N Independent Learners
        for i, opt in enumerate(optimizers):
            loss = self.dqn_mse_loss(batch, i)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self._total_timestep += self.episode_timesteps
            if self.save_path is not None:
                self.save_checkpoint(self.save_path, self.total_timestep)
            print('')  # Skip an output line

        # Soft update of target network
        if self.timestep % self.sync_rate == 0:
            # TODO: ModuleList
            # self.target_net.load_state_dict(self.net.state_dict())
            for i in range(self.num_intersections):
                self.target_net[i].load_state_dict(self.net[i].state_dict())

        log = {
            "steps": torch.tensor(self.timestep).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.mean(torch.tensor(reward).clone().detach()).to(device),

            "exploration_rate": torch.tensor(np.round(epsilon, 4)).to(device),
            "train_loss": loss.clone().detach().to(device)
        }
        status_bar = {
            "episode_reward": torch.tensor(self.episode_reward).to(device),
            "status_steps": torch.tensor(self.timestep).to(device),
            "epsilon": torch.tensor(epsilon).to(device)
        }

        for k, v in log.items():
            self.log(k, v, logger=True, prog_bar=False)
        for k, v in status_bar.items():
            self.log(k, v, logger=False, prog_bar=True)

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizers = [optim.Adam(self.net[i].parameters(), lr=self.lr) for i in range(self.num_intersections)]
        return optimizers

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

    # Serializes the object's copy -- sets get_wave to null.
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):

        for i, tl_id in enumerate(self.env.tl_ids):
            file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{tl_id}.chkpt'
            file_path.parent.mkdir(exist_ok=True)
            torch.save(self.net[i].state_dict(), file_path)

    # deserializes object -- except for get_wave.
    @classmethod
    def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
        class_name = cls.__name__.lower()
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{class_name}.chkpt'
        return cls.load_from_checkpoint(checkpoint_path=file_path.as_posix())

"""Deep Reinforcement Learning for Multi-agent system.

* Deep Q-networks

* Using Deep Q-networks for independent learners.

* TODO:
    Creates a MAS class that controls agents.
    Separate buffers.

To run the template, just run:
`python dqn.py`

`tensorboard --logdir lightning_logs`

References:
-----------
* Deep Q-network (DQN)

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""
import argparse

import numpy as np
import torch

from environment import get_environment
from approximators.mlp import MLP
import pytorch_lightning as pl

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'


class DQN_MODEL:

    def __init__(self, epsilon_init, epsilon_final, epsilon_timesteps, network, save_agent_interval, experiment_time, save_path, seed):

        parser = self.add_model_specific_args()
        args = parser.parse_args()
        args.epsilon_init = epsilon_init
        args.epsilon_final = epsilon_final
        args.epsilon_timesteps = epsilon_timesteps
        args.network = network
        args.episode_timesteps = save_agent_interval
        args.save_path = save_path
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = DQNLightning(**vars(args))


        # FIXME: RLDataLoader apperantly has an irregular
        # number of epochs.
        # * 2456 are the first 5 episodes
        # * 197 epochs for each episode thereafter
        # for 80 eps:
        # num_epochs = 2456 + 197 * 75
        self.num_episodes = experiment_time / save_agent_interval
        self.num_epochs = 2456 + 197 * (self.num_episodes-5)
        # self.trainer = pl.Trainer(
        #     max_epochs=self.num_epochs,
        #     log_every_n_steps=500,
        #     val_check_interval=100)


        # 10000 of populate. Since each trainer step runs 10 env timesteps, we need to divide max_steps by 10.
        assert experiment_time > save_agent_interval
        max_trainer_steps = (experiment_time-10000)/10

        # self.trainer = pl.Trainer(
        #     max_steps=max_trainer_steps,
        #     log_every_n_steps=500,
        #     val_check_interval=100)



    @staticmethod
    def load_checkpoint(env, chkpt_dir_path, rollout_time, network, chkpt_num=None):

        
        if chkpt_num == None:
            chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
        chkpt_path = chkpt_dir_path / str(chkpt_num)
        print("Loading checkpoint: ", chkpt_path)

        nets = []
        for tl_id in env.tl_ids:
            dqn = MLP()
            dqn.load_state_dict(torch.load(chkpt_path / f'{tl_id}.chkpt'))
            nets.append(dqn)

        agent = Agent(env)
        return agent, nets

    @staticmethod
    def add_model_specific_args():  # pragma: no-cover
        parent_parser = argparse.ArgumentParser(add_help=False)
        parser = parent_parser.add_argument_group("DQNLightning")
        parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
        parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
        parser.add_argument("--network", type=str, default="intersection", help="roadnet name")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--sync_rate", type=int, default=500,
                            help="how many frames do we update the target network")
        parser.add_argument("--replay_size", type=int, default=50000, help="capacity of the replay buffer")
        parser.add_argument("--warm_start_steps", type=int, default=1000,
                            help="how many samples do we use to fill our buffer at the start of training",
                            )
        parser.add_argument("--epsilon_timesteps", type=int, default=3500,
                            help="what frame should epsilon stop decaying")
        parser.add_argument("--epsilon_init", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--epsilon_final", type=float, default=0.01, help="final value of epsilon")
        parser.add_argument("--episode_timesteps", type=int, default=3600, help="max length of an episode")

        parser.add_argument("--save_path", type=str, default=None, help="Directory to save experiments.")
        return parent_parser

# def main(args, train_config_path=TRAIN_CONFIG_PATH, seed=0):
#     # Setup config parser path.
#     print(f'Loading train parameters from: {train_config_path}')
#
#     train_args = parse_train_config(train_config_path)
#     args.network = train_args['network']
#     experiment_time = train_args['experiment_time']
#     args.episode_timesteps = train_args['experiment_save_agent_interval']
#     # num_epochs = experiment_time // 10
#     num_steps = train_args['experiment_time']
#
#     # Epsilon
#     args.epsilon_init = train_args['epsilon_init']
#     args.epsilon_final = train_args['epsilon_final']
#     args.epsilon_timesteps = train_args['epsilon_schedule_timesteps']
#
#     if args.save_path is None:
#         expr_path = expr_path_create(args.network)
#         args.save_path = expr_path / 'checkpoints'
#         Path(args.save_path).mkdir(exist_ok=True)
#
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     model = DQNLightning(**vars(args))
#
#     # FIXME: RLDataLoader apperantly has an irregular
#     # number of epochs.
#     # * 2456 are the first 5 episodes
#     # * 197 epochs for each episode thereafter
#     # num_epochs = 2456 + 197 * 75
#     num_epochs = 2456
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         log_every_n_steps=500,
#         val_check_interval=100)
#     trainer.fit(model)
#     # TODO: Move this routine to env or create a delegate chain.
#     expr_logs_dump(args.save_path, 'train_log.json', model.agent.env.info_dict)
#
#     # 2) Create train plots.
#     load_path = Path('data/emissions/arterial_20211007235041/checkpoints/')
#     train_plots(load_path.parent)
#
#     # Load checkpoint for testing
#     rollout_timesteps = 21600
#     loaded_nets = []
#     chkpt_num = max(int(folder.name) for folder in load_path.iterdir())
#     chkpt_path = load_path / str(chkpt_num)
#     env = get_environment('arterial', episode_timesteps=rollout_timesteps)
#
#     net = []
#     for tl_id in env.tl_ids:
#         dqn = DQN()
#         dqn.load_state_dict(torch.load(chkpt_path / f'{tl_id}.chkpt'))
#         net.append(dqn)
#
#     num_rollouts = 2
#     rollout_timesteps = args.episode_timesteps
#     rollout_list = []
#     for num_rollout in trange(num_rollouts, position=0):
#         agent = Agent(env)
#         target_path = expr_path_test_target(load_path.parent, network=args.network)
#
#         # TODO: Get device
#         # TODO: Move emissions to a separate module.
#         # TODO: Refactor emissions -- separate Log?
#         emissions = []
#         for timestep in trange(rollout_timesteps, position=1):
#             agent.play_step(net, epsilon=0.0)
#             update_emissions(env.engine, emissions)
#         info_dict = agent.env.info_dict
#         info_dict['id'] = chkpt_num
#         rollout_list.append(info_dict)
#
#         expr_logs_dump(target_path, 'emission_log.json', emissions)
#
#         env = get_environment('arterial', episode_timesteps=rollout_timesteps)
#     # expr_logs_dump(args.save_path, 'train_log.json', model.agent.env.info_dict)
#
#     res = concat(rollout_list)
#     filename = f'rollouts_test.json'
#     target_path = batch_path / filename
#     with target_path.open('w') as fj:
#         json.dump(res, fj)
#
#     test_plots(load_path)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(add_help=False)
#     parser = DQNLightning.add_model_specific_args(parser)
#     args = parser.parse_args()
#
#     main(args)
