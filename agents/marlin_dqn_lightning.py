import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim, nn

from agents import DQN_Lightning
from agents.DQN_Lightning import flatten
from agents.experience import Experience
from approximators.mlp import MLP
from utils.utils import tuple_to_int


class Agent_MARLIN(DQN_Lightning.Agent):
    """Base Agent class handling the interaction with the environment.

    >>> env = get_environment('arterial')
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...dqn.Agent object at ...>
    """

    def __init__(self, env, single_obs_size, single_action_size, edges, replay_buffer=None):
        super().__init__(env, replay_buffer=replay_buffer)
        self.single_obs_size = single_obs_size
        self.single_action_size = single_action_size
        self.edges = edges

    def get_policy_estimate(self, id1, id2, state1, state2):
        # Make actual policy estimate with network
        return torch.tensor([1.0 for _ in range(self.single_action_size)])

        # visited_sum = sum(self._policy_estimate[id1][id2][state1][state2].values())
        # if visited_sum == 0:
        #     return 0
        # else:
        #     return self._policy_estimate[id1][id2][state1][state2][action2] \
        #            / visited_sum

    def get_q_value_sum(self, nets, id1, id2, action1, state1, state2):
        _sum = 0
        concat_state = torch.cat((state1, state2), 1)
        q_values = nets[id1][id2](concat_state)
        q_slice = torch.tensor([0, 1]) if action1 == 0 else torch.tensor([2, 3])
        q_sliced = torch.index_select(q_values[0], 0, q_slice)
        policy_estimate = self.get_policy_estimate(id1, id2, state1, state2)

        return torch.matmul(q_sliced, policy_estimate)

    def choose_greedy_action(self, nets, id1, state):
        _max = (float("-inf"), -1)
        for action1 in range(self.single_action_size):
            _sum = 0
            for id2 in self.edges[id1]:
                _sum += self.get_q_value_sum(nets, id1, id2, action1, state[id1], state[id2])
            if _sum > _max[0]:
                _max = (_sum, action1)
        return _max[1]

    def act(self, nets, epsilon, device):
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
        split_state = dict(zip(self.env.tl_ids, torch.tensor([self.state]).split(self.single_obs_size, dim=1)))
        actions = {}
        for i, tl_id in enumerate(self.env.tl_ids):
            action = self.choose_greedy_action(nets, tl_id, split_state)
            if np.random.random() < epsilon:
                action = np.random.choice([a for a in range(self.single_action_size) if a != action])
            actions[tl_id] = action

        return actions

    @torch.no_grad()
    def play_step(self, nets, epsilon=0.0, device="cpu"):
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
        actions = self.act(nets, epsilon, device)

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


class DQNLightning_MARLIN(DQN_Lightning.DQNLightning):
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

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.single_obs_size = self.obs_size
        self.obs_size = self.single_obs_size * 2

        self.single_num_actions = self.num_actions
        self.num_actions = self.single_num_actions ** 2

        with open('data/networks/' + kwargs["network"] + "/edges.json") as f:
            self.edges = json.load(f)

        self.net = defaultdict(lambda: defaultdict(lambda: 0))
        self.target_net = defaultdict(lambda: defaultdict(lambda: 0))

        # Net for every (intersection1, intersection2) pair
        for id1 in self.env.tl_ids:
            for id2 in self.edges[id1]:
                self.net[id1][id2] = MLP(self.obs_size,
                                         self.num_actions,
                                         self.hidden_size
                                         )

                self.target_net[id1][id2] = MLP(self.obs_size,
                                                self.num_actions,
                                                self.hidden_size
                                                )

        self.agent = Agent_MARLIN(self.env, self.single_obs_size, self.single_num_actions, self.edges, self.buffer)

    def forward(self, i, j, x):
        """Passes in a state `x` through the network and gets the
           `q_values` of each action as an output.

        Parameters:
        -----------
        * x: environment state

        Returns:
        --------
        * q-values
        """
        output = self.net[i][j](x)
        return output

    def get_experience(self, batch, agent1, agent2):
        agent1_index = self.env.tl_ids.index(agent1)
        agent2_index = self.env.tl_ids.index(agent2)

        states, actions, rewards, dones, next_states = batch
        state_split = states.split(self.single_obs_size, dim=1)
        concat_state = torch.cat((state_split[agent1_index], state_split[agent2_index]), 1)

        action_split = actions.split(1, dim=-1)
        concat_action = tuple_to_int((action_split[agent1_index], action_split[agent2_index]),
                                     length=self.single_num_actions)

        reward_split = [rewards.squeeze() for rewards in rewards.split(1, dim=-1)]
        mean_reward = torch.mean(torch.stack((reward_split[agent1_index], reward_split[agent2_index]), dim=-1), dim=-1)

        next_state_split = next_states.split(self.single_obs_size, dim=1)
        concat_next_state = torch.cat((next_state_split[agent1_index], next_state_split[agent2_index]), 1)

        return (
                   concat_state,
                   concat_action,
                   mean_reward,
                   concat_next_state
               ), dones

    def dqn_mse_loss(self, batch, agent1, agent2):
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
        batch_data, dones = self.get_experience(batch, agent1, agent2)
        target_net = self.target_net[agent1][agent2]
        s_, a_, r_, s1_ = batch_data

        state_action_values = self.net[agent1][agent2](s_).gather(1, a_).squeeze(-1)

        with torch.no_grad():
            next_state_values = target_net(s1_).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + r_.squeeze(-1)

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def loss_step(self, batch, optimizers):
        losses = []
        i = 0
        for id1 in self.env.tl_ids:
            for id2 in self.edges[id1]:
                opt = optimizers[i]
                loss = self.dqn_mse_loss(batch, id1, id2)
                losses.append(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
                i += 1

        return torch.mean(torch.stack(losses))

    def update_target_nerwork(self):
        # Soft update of target network
        if self.timestep % self.sync_rate == 0:
            # TODO: ModuleList
            for id1 in self.env.tl_ids:
                for id2 in self.edges[id1]:
                    self.target_net[id1][id2].load_state_dict(self.net[id1][id2].state_dict())

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizers = []
        for id1 in self.env.tl_ids:
            for id2 in self.edges[id1]:
                optimizers.append(optim.Adam(self.net[id1][id2].parameters(), lr=self.lr))

        return optimizers

    # Serializes the object's copy -- sets get_wave to null.

    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        for id1 in self.env.tl_ids:
            for id2 in self.edges[id1]:
                file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{id1}_{id2}.chkpt'
                file_path.parent.mkdir(exist_ok=True)
                torch.save(self.net[id1][id2].state_dict(), file_path)
