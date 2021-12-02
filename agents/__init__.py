import torch
from pytorch_lightning import seed_everything
import numpy as np

from agents.actor_critic import ACAT
from agents.marlin import MARLIN
from agents import dqn, gat

def load_agent(env, agent_type, chkpt_dir_path, chkpt_num, rollout_time, network):
    if agent_type == 'ACAT': return ACAT.load_checkpoint(chkpt_dir_path, chkpt_num), None
    if agent_type == 'MARLIN': return MARLIN.load_checkpoint(chkpt_dir_path, chkpt_num), None
    if agent_type == 'DQN': return dqn.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    if agent_type == 'GAT': return gat.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    raise ValueError(f'{agent_type} not defined.')

# def get_agent(agent_type, env, epsilon_init, epsilon_final, epsilon_timesteps):
def get_agent(env, train_args):

    agent_type = train_args.agent_type
    if agent_type == 'ACAT':
        return ACAT(env.phases, train_args.epsilon_init,
                    train_args.epsilon_final, train_args.epsilon_timesteps)
    if agent_type == 'MARLIN':
        return MARLIN(env.phases, train_args.epsilon_init,
                      train_args.epsilon_final, train_args.epsilon_timesteps, env.network)

    if agent_type in ('DQN', 'GAT'):
        # TODO: Hparameters becomes model_args
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hparams = { 
            'batch_size': 1000,
            'lr': 1e-3, 
            'gamma': 0.8,
            'sync_rate': 18000,
            'replay_size':10000,
            'warm_start_steps': 1000,
        }
        if agent_type == 'DQN':
            model = dqn.DQNLightning
        elif agent_type == 'GAT':
            model = gat.GATLightning
        return model(env, device, train_args, **hparams).to(device)
    raise ValueError(f'{agent_type} not defined.')
