import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from itertools import product

import pandas as pd
from tqdm.auto import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from agents import DQNLightning
from agents.dqn import Agent, flatten
from utils.file_io import parse_train_config, \
    expr_logs_dump, expr_path_create, \
    expr_path_test_target
from plots.train_plots import main as train_plots
from plots.test_plots import main as test_plots
from utils.utils import concat, tuple_to_int
from agents.experience import Experience, ReplayBuffer, RLDataset
from approximators.mlp import MLP, RNN

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'


def get_experience(batch, obs_size):
    states, actions, rewards, dones, next_states = batch
    return (
               states.split(obs_size, dim=1),
               actions.split(1, dim=-1),
               rewards.split(1, dim=-1),
               next_states.split(obs_size, dim=1)
           ), dones




class CGAgent:

    def __init__(self, name, possible_actions):
        self.name = name
        self.possible_actions = possible_actions
        self.payout_functions = []
        self.dependant_agents = []

class ActionTable:
    def __init__(self, agents, data=None):
        self.agents = agents
        self.agent_names = [agent.name for agent in self.agents]

        if len(agents) != 0:
            if data is None:
                size = tuple([len(agent.possible_actions) for agent in self.agents])
                data = np.zeros(size)
            self.table = pd.Series(data.flat,
                                   index=pd.MultiIndex.from_product([agent.possible_actions for agent in self.agents],
                                                                    names=self.agent_names))

    def get_value(self, actions):
        if len(self.agents) == 0:
            return self.action

        res = self.table
        indexer = []
        for agent in self.agent_names:
            indexer.append(actions[agent])
        return res.loc[tuple(indexer)]

    def set_action(self, action):
        self.action = action

    def set_value(self, actions, value):
        res = self.table
        indexer = []
        for agent in self.agent_names:
            indexer.append(actions[agent])

        res.loc[tuple(indexer)] = value


class VEAgent(Agent):
    """Base Agent class handling the interaction with the environment.

    >>> env = get_environment('arterial')
    >>> buffer = ReplayBuffer(10)
    >>> VEAgent(env, buffer)  # doctest: +ELLIPSIS
    <...dqn.Agent object at ...>
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.unique_edges = []
        self.CGAgents = {}
        self.num_actions = len(list(self.env.phases.values())[0])

        for tid in self.env.tl_ids:
            self.CGAgents[tid] = CGAgent(tid, list(range(self.num_actions)))

        for link in self.unique_edges:
            self.CGAgents[link[0]].dependant_agents.append(link[1])
            self.CGAgents[link[1]].dependant_agents.append(link[0])

        with open('data/networks/' + self.env.network + "/edges.json") as f:
            self.edges = json.load(f)

        self.unique_edges = [tuple(sorted((agent1, agent2))) for agent1 in self.edges for agent2 in self.edges[agent1]]
        self.unique_edges = list(set([i for i in self.unique_edges]))

        for link in self.unique_edges:
            self.CGAgents[link[0]].dependant_agents.append(link[1])
            self.CGAgents[link[1]].dependant_agents.append(link[0])
    @staticmethod
    def maximizeAgent(agent, action_dict):
        _max = ("-1", float("-inf"))
        # Figure out the max and maxArg of current agent actions
        for agent_action in agent.possible_actions:
            _sum = 0
            actions = dict({agent.name: agent_action}, **action_dict)
            # Maximizing the sum of every payout function
            for function in agent.payout_functions:
                _sum += function.get_value(actions)
            if _sum >= _max[1]:
                _max = (agent_action, _sum)
        return _max

    def variable_elimination(self, node_net, edge_net, debug=False, epsilon=0):
        agents = self.CGAgents
        state = torch.tensor([self.state]).split(4, dim=1)
        #Get Q_values
        for (tid, agent) in agents.items():
            qTable = node_net(state[self.env.tl_ids.index(tid)]).numpy()[0]
            payoutFunction = ActionTable([agent], qTable)
            agent.payout_functions = [payoutFunction]

        for (agent1, agent2) in self.unique_edges:
            index1 = self.env.tl_ids.index(agent1)
            index2 = self.env.tl_ids.index(agent2)
            concat_state = torch.cat((state[index1],state[index2]), dim=-1)
            qTable = edge_net(concat_state).numpy()[0]
            agent1 = agents[agent1]
            agent2 = agents[agent2]
            payoutFunction = ActionTable([agent1, agent2], qTable)
            agent1.payout_functions.append(payoutFunction)
            agent2.payout_functions.append(payoutFunction)

        elimination_agents = list(agents.values())
        # First Pass
        for agent in elimination_agents:

            # For every agent that depends on current agent
            dependant_agents = [agents[agent_name] for agent_name in agent.dependant_agents]

            # If last agent to eliminate, create best response.
            if len(dependant_agents) == 0:
                agent.best_response = ActionTable([])
                _max = self.maximizeAgent(agent, {})
                if np.random.random() < epsilon:
                    action = np.random.choice(
                        [n for n in agent.possible_actions if n != _max[0]])  # Select random suboptimal action
                    # action = random.choice(agent.possible_actions)  # Select random action
                    agent.best_response.set_action(action)
                else:
                    agent.best_response.set_action(_max[0])
                continue

            action_product = list(product(*[dependant_agent.possible_actions for dependant_agent in dependant_agents]))

            new_function = ActionTable(dependant_agents)
            agent.best_response = ActionTable(dependant_agents)

            # For every action pair of dependant agents
            for joint_action in action_product:
                action_dict = {agent.dependant_agents[i]: joint_action[i] for i in range(len(agent.dependant_agents))}

                _max = self.maximizeAgent(agent, action_dict)

                # Save new payout and best response
                if np.random.random() < epsilon:
                    action = np.random.choice(
                        [n for n in agent.possible_actions if n != _max[0]])  # Select random suboptimal action
                    # action = random.choice(agent.possible_actions)  # Select random action
                    agent.best_response.set_value(action_dict, action)
                    _sum = 0
                    for function in agent.payout_functions:  # Get value for new action
                        _sum += function.get_value(
                            dict({agent.name: np.random.choice([n for n in agent.possible_actions if n != _max[0]])},
                                 **action_dict))

                    new_function.set_value(action_dict, _sum)

                else:
                    agent.best_response.set_value(action_dict, _max[0])
                    new_function.set_value(action_dict, _max[1])

            # Delete all payout functions that involve the parent agent from all the dependant agents
            # And add the new payout functions to dependants
            for dependant_agent in agent.dependant_agents:
                # Remove all functions that have agent_name in the dependants
                agents[dependant_agent].payout_functions = [function for function in
                                                            agents[dependant_agent].payout_functions if
                                                            agent.name not in function.agent_names]
                if agent.name in agents[dependant_agent].dependant_agents:
                    agents[dependant_agent].dependant_agents.remove(agent.name)

                agents[dependant_agent].payout_functions.append(new_function)

                # Add all dependants (except himself) to the agent's list if they are not already in
                agents[dependant_agent].dependant_agents.extend([agent_name for agent_name in agent.dependant_agents if
                                                                 agent_name != dependant_agent and agent_name not in
                                                                 agents[
                                                                     dependant_agent].dependant_agents])

        # Second Pass, Reverse Order
        actions = {}
        for agent in list(elimination_agents)[::-1]:
            actions[agent.name] = int(agent.best_response.get_value(actions))

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
        node_net = nets[0]
        edge_net = nets[1]

        actions = self.act(node_net, edge_net, epsilon, device)

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

    def act(self, node_net, edge_net, epsilon, device):
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

        actions = self.variable_elimination(node_net, edge_net, epsilon=epsilon)
        return actions


class VELightning(DQNLightning):
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

    def __init__(self, env, **kwargs: Any):
        warm_steps = kwargs["warm_start_steps"]
        kwargs["warm_start_steps"] = 0
        super().__init__(env, **kwargs)
        self.node_net = self.init_net(self.obs_size, self.num_actions, self.hidden_size)
        self.node_target_net = self.init_net(self.obs_size, self.num_actions, self.hidden_size)
        self.edge_net = self.init_net(self.obs_size * 2, self.num_actions ** 2, self.hidden_size)
        self.edge_target_net = self.init_net(self.obs_size * 2, self.num_actions ** 2, self.hidden_size)
        self.agent = VEAgent(self.env, self.buffer)


        with open('data/networks/' + env.network + "/edges.json") as f:
            self.edges = json.load(f)

        self.unique_edges = [tuple(sorted((agent1, agent2))) for agent1 in self.edges for agent2 in self.edges[agent1]]
        self.unique_edges = list(set([i for i in self.unique_edges]))


        self.agent.unique_edges = self.unique_edges
        if warm_steps > 0: self.populate(warm_steps)

    def init_net(self, obs_size, num_actions, hidden_size):
        return RNN(
            obs_size,
            num_actions,
            hidden_size)

    def populate(self, steps=1000):
        """Carries out several random steps through the
           environment to initially fill up the replay buffer with
           experiences.

        Parameters:
        -----------
        * steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step((self.node_net, self.edge_net), epsilon=1.0)

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
        reward, done = self.agent.play_step((self.node_net, self.edge_net), epsilon, device)
        self.episode_reward += sum(reward) * 0.001

        # calculates training loss
        optimizers = self.optimizers(use_pl_optimizer=True)

        # MAS -- Q_n(s_n, u_n), for n=1,...,|N|
        # N Independent Learners
        loss = self.loss_step(batch, optimizers)

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
            self.node_target_net.load_state_dict(self.node_net.state_dict())
            self.edge_target_net.load_state_dict(self.edge_net.state_dict())

        log = {
            "steps": torch.tensor(self.timestep).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.mean(torch.tensor(reward).clone().detach()).to(device),

            "exploration_rate": torch.tensor(np.round(epsilon, 4)).to(device),
        }
        status_bar = {
            "episode_reward": torch.tensor(self.episode_reward).to(device),
            "status_steps": torch.tensor(self.timestep).to(device),
            "epsilon": torch.tensor(epsilon).to(device),
            "train_loss": loss.clone().detach().to(device)

        }

        for k, v in log.items():
            self.log(k, v, logger=True, prog_bar=False)
        for k, v in status_bar.items():
            self.log(k, v, logger=False, prog_bar=True)

    def get_q_total(self, state, target=False):
        node_net = self.node_target_net if target else self.node_net
        edge_net = self.edge_target_net if target else self.edge_net

        a = [list(range(self.num_actions)) for _ in range(self.num_intersections)]
        batch_size = state[0].size(0)
        q_total_size = (self.num_actions,) * self.num_intersections + (batch_size,)

        q_total = torch.zeros(q_total_size)
        for action_pair in list(product(*a)):
            for i, action in enumerate(action_pair):
                q_total[action_pair].add_(node_net(state[i]).T[action])
            for agent1, agent2 in self.unique_edges:
                index1 = self.env.tl_ids.index(agent1)
                index2 = self.env.tl_ids.index(agent2)
                concat_state = torch.cat((state[index1], state[index2]), dim=-1)
                q_values = edge_net(concat_state)
                joined_action = tuple_to_int((action_pair[index1], action_pair[index2]), length=self.num_actions)
                q_total[action_pair].add_(q_values.T[joined_action])

                #Permute Q-values from [2,2,2,1000] to [1000,2,2,2]
        return q_total.permute((self.num_intersections,) + tuple(range(self.num_intersections)))

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
        # split and iterate over sections! -- beware assumption
        # gets experience from the agent

        batch_data, dones = get_experience(batch, self.obs_size)
        s_, a_, r_, s1_ = batch_data

        q_total = self.get_q_total(s_)
        state_action_values = torch.stack([q_total[j] for j in tuple(((i,) + tuple(torch.cat(a_, dim=-1)[i].tolist())) for i in range(batch[0].size(0)))])

        with torch.no_grad():
            next_state_values = self.get_q_total(s1_, target=True).view(batch[0].size(0),self.num_actions**self.num_intersections).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + torch.sum(torch.cat(r_, dim=-1), dim=1)

        return nn.MSELoss()(state_action_values, expected_state_action_values)


    def loss_step(self, batch, optimizers):

        loss = self.dqn_mse_loss(batch)
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        loss.backward()
        optimizers[0].step()
        optimizers[1].step()

        return loss

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizers = []
        optimizers.append(optim.Adam(self.node_net.parameters(), lr=self.lr))
        optimizers.append(optim.Adam(self.edge_net.parameters(), lr=self.lr))
        return optimizers

    """ Serialization """

    # Serializes the object's copy -- sets get_wave to null.
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'node.chkpt'
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.node_net.state_dict(), file_path)
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'edge.chkpt'
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.edge_net.state_dict(), file_path)


# TODO: Fix load_checkpoint
def load_checkpoint(env, chkpt_dir_path, rollout_time, network, chkpt_num=None):
    if chkpt_num == None:
        chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)
    single_obs_size = 4
    single_action_size = 2

    nets = []
    dqn = RNN(single_obs_size, single_action_size)
    dqn.load_state_dict(torch.load(chkpt_path / f'node.chkpt'))
    nets.append(dqn)

    dqn = RNN(single_obs_size * 2, single_action_size ** 2)
    dqn.load_state_dict(torch.load(chkpt_path / f'edge.chkpt'))
    nets.append(dqn)

    agent = VEAgent(env)
    return agent, nets
