import json
from collections import defaultdict

import torch

from agents import dqn
from agents.marlin_DQN_Lightning import DQNLightning_MARLIN, Agent_MARLIN
from approximators.mlp import MLP
from environment import get_environment


class MARLIN_DQN_MODEL(dqn.DQN_MODEL):

    def init_lightning_model(self, args):
        return DQNLightning_MARLIN(**vars(args))

    @staticmethod
    def get_rollout_agent(network, rollout_time, single_obs_size, single_action_size, edges):
        env = get_environment(network, episode_timesteps=rollout_time)
        env.emit = True
        return Agent_MARLIN(env, single_obs_size, single_action_size, edges)

    @staticmethod
    def load_checkpoint(chkpt_dir_path, rollout_time, network, chkpt_num=None):
        single_obs_size = 4
        single_action_size = 2

        if chkpt_num == None:
            chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
        chkpt_path = chkpt_dir_path / str(chkpt_num)
        print("Loading checkpoint: ", chkpt_path)

        with open('data/networks/' + network + "/edges.json") as f:
            edges = json.load(f)

        nets = defaultdict(lambda: defaultdict(lambda: 0))
        agent = MARLIN_DQN_MODEL.get_rollout_agent(network, rollout_time, single_obs_size, single_action_size, edges)

        for id1 in agent.env.tl_ids:
            for id2 in edges[id1]:
                dqn = MLP(obs_size=single_obs_size * 2, n_actions=single_action_size ** 2)
                dqn.load_state_dict(torch.load(chkpt_path / f'{id1}_{id2}.chkpt'))
                nets[id1][id2] = dqn

        return agent, nets
