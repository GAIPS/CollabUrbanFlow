# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deep Reinforcement Learning: Deep Q-network (DQN)

The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.

To run the template, just run:
`python reinforce_learn_Qnet.py`

After ~1500 steps, you will see the total_reward hitting the max score of 475+.
Open up TensorBoard to see the metrics:

`tensorboard --logdir default`

References
----------

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""

import argparse
from collections import deque, namedtuple, OrderedDict
from collections import Iterable
# from typing import Iterator, List, Tuple

# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from environment import get_environment
from utils.file_io import parse_train_config, \
                expr_logs_dump, expr_path_create

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'


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

def zip2sections(batch, obs_size):
    states, actions, rewards, dones, next_states = batch
    return zip(
        states.split(obs_size, dim=1),
        actions.split(1, dim=-1),
        rewards.split(1, dim=-1),
        next_states.split(obs_size, dim=1)
    ), dones

class DQN(nn.Module):
    """Simple MLP network.

    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )
    """

    def __init__(self, obs_size=4, n_actions=2, hidden_size=32, n_inter=1):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
            ) for _ in range(n_inter)
        ])

    def forward(self, i, x):
        return self.net[i](x.float())


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
        "Experience", 
        field_names=["state", "action", "reward", "done", "new_state"]
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    >>> ReplayBuffer(5)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.ReplayBuffer object at ...>
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size) 
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>
    """

    def __init__(self, buffer, sample_size):
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = gym.make("CartPole-v1")
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.Agent object at ...>
    """

    def __init__(self, env, replay_buffer):
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self):
        """Resets the environment and updates the state."""
        # Assumption: All intersections have the same phase.
        self.state = list(flatten(self.env.reset().values()))
        

    def get_action(self, net, i, epsilon, device):
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: list<DQN>
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            # TODO: wrapper?
            # action = self.env.action_space.sample()
            action = np.random.choice((0, 1))
        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(i, state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        actions = {}
        for i, tl_id in enumerate(self.env.tl_ids):
            actions[tl_id] = self.get_action(net, i, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(actions)
        new_state, reward, actions = \
                list(flatten(new_state.values())), list(reward.values()), list(actions.values())
        exp = Experience(self.state, actions, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(pl.LightningModule):
    """Basic DQN Model.

    >>> DQNLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )
    """

    def __init__(self, network='intersection', replay_size=200, warm_start_steps=0,
        gamma=0.99, epsilon_init=1.0, epsilon_final=0.01, epsilon_timesteps=3500,
        sync_rate=10, lr=1e-2, episode_timesteps=3600, batch_size=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.epsilon_init = epsilon_init 
        self.epsilon_final= epsilon_final
        self.epsilon_timesteps= epsilon_timesteps
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_timesteps = episode_timesteps
        self.batch_size = batch_size

        # Instanciate environment.
        # self.env = gym.make(env)
        # obs_size = self.env.observation_space.shape[0]
        # n_actions = self.env.action_space.n
        self.env = get_environment(network, episode_timesteps=episode_timesteps)
        self.obs_size = 4
        self.num_actions = 2
        self.num_intersections = len(self.env.tl_ids)

        self.net = DQN(self.obs_size, self.num_actions)
        self.target_net = DQN(self.obs_size, self.num_actions)

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self._total_timestep = 0
        if self.warm_start_steps > 0 : self.populate(self.warm_start_steps)


    @property
    def timestep(self):
        return self.agent.env.timestep

    @property
    def total_timestep(self):
        return self._total_timestep + self.timestep

    def populate(self, steps=1000):
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, i, x):
        """Passes in a state `x` through the network and gets the `q_values` of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net[i](x)
        return output

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        # split and iterate over sections! -- beware assumption
        # states, actions, rewards, dones, next_states = batch
        ni = 0
        gen, dones = zip2sections(batch, self.obs_size)
        loss = torch.zeros(self.num_intersections)
        for s_, a_, r_, s1_ in gen:
            state_action_values = self.net(ni, s_).gather(1, a_).squeeze(-1)

            with torch.no_grad():
                next_state_values = self.target_net(ni, s1_).max(1)[0]
                next_state_values[dones] = 0.0
                next_state_values = next_state_values.detach()

            expected_state_action_values = next_state_values * self.gamma + r_.squeeze(-1)

            loss[ni] = nn.MSELoss()(state_action_values, expected_state_action_values)
            ni += 1
        return loss

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
        epsilon = max(self.epsilon_final,
                      self.epsilon_init - \
                      (self.total_timestep + 1) / self.epsilon_timesteps)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += sum(reward) * 0.001

        # calculates training loss
        loss = self.dqn_mse_loss(batch)
    

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self._total_timestep += self.episode_timesteps
            print('') # Skip an output line

        # Soft update of target network
        if self.timestep % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "steps": torch.tensor(self.timestep).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "step_loss": torch.mean(loss.clone().detach()).to(device),
            "epsilon": torch.tensor(epsilon).to(device)
        }
        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

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
    # Serializes the object's copy -- sets get_wave to null.
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        class_name = type(self).__name__.lower()
        file_path = Path(chkpt_dir_path) / chkpt_num / f'{class_name}.chkpt'  
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, mode='wb') as f:
            dill.dump(self, f)

    # deserializes object -- except for get_wave.
    @classmethod
    def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
        class_name = cls.__name__.lower()
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{class_name}.chkpt'  
        return cls.load_from_checkpoint(checkpoint_path=file_path.as_posix())

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("DQNLightning")
        parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
        parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
        parser.add_argument("--network", type=str, default="intersection", help="roadnet name")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
        parser.add_argument("--replay_size", type=int, default=50000, help="capacity of the replay buffer")
        parser.add_argument( "--warm_start_steps", type=int, default=10,
            help="how many samples do we use to fill our buffer at the start of training",
        )
        parser.add_argument("--epsilon_timesteps", type=int, default=3500, help="what frame should epsilon stop decaying")
        parser.add_argument("--epsilon_init", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--epsilon_final", type=float, default=0.01, help="final value of epsilon")
        parser.add_argument("--episode_timesteps", type=int, default=3600, help="max length of an episode")
        return parent_parser

def main(args, train_config_path=TRAIN_CONFIG_PATH, seed=0):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    train_args = parse_train_config(train_config_path)
    args.network = train_args['network']
    experiment_time = train_args['experiment_time']
    args.episode_timesteps = train_args['experiment_save_agent_interval']
    # num_epochs = experiment_time // args.episode_timesteps

    # Epsilon 
    args.epsilon_init = train_args['epsilon_init']
    args.epsilon_final = train_args['epsilon_final']
    args.epsilon_timesteps = train_args['epsilon_schedule_timesteps']

    expr_path = expr_path_create(args.network)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DQNLightning(**vars(args))

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="checkpoint",
    #     dirpath=expr_path,
    #     filename=f"sample-{args.network}",
    #     save_top_k=3,
    #     mode="min",
    # )

    # trainer = pl.Trainer(gpus=1, accelerator="dp", val_check_interval=100)
    num_epochs = 7000
    trainer = pl.Trainer(
            max_steps=-1,
            max_epochs=num_epochs,
            # checkpoint_callbacks=[checkpoint_callback],
            val_check_interval=100)
    trainer.fit(model)
    # TODO: Move this routine to env or create a delegate chain.
    expr_logs_dump(expr_path, 'train_log.json', model.agent.env.info_dict)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser = DQNLightning.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
