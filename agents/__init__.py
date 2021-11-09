import torch
from pytorch_lightning import seed_everything
import numpy as np

from agents.actor_critic import ACAT
from agents.marlin import MARLIN
from agents import dqn, dqn2, gatv, gatw

def load_agent(env, agent_type, chkpt_dir_path, chkpt_num, rollout_time, network):
    if agent_type == 'ACAT': return ACAT.load_checkpoint(chkpt_dir_path, chkpt_num), None
    if agent_type == 'MARLIN': return MARLIN.load_checkpoint(chkpt_dir_path, chkpt_num), None
    if agent_type == 'DQN': return dqn.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    if agent_type == 'DQN2': return dqn2.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    if agent_type == 'GATV': return gatv.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    if agent_type == 'GATW': return gatw.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    raise ValueError(f'{agent_type} not defined.')

def get_agent(agent_type, env, epsilon_init, epsilon_final, epsilon_timesteps):

    if agent_type == 'ACAT':
        return ACAT(env.phases, epsilon_init,
                    epsilon_final, epsilon_timesteps)
    if agent_type == 'MARLIN':
        return MARLIN(env.phases, epsilon_init,
                      epsilon_final, epsilon_timesteps, network)
    if agent_type in ('DQN', 'DQN2', 'GATV', 'GATW'):
        # TODO: Place here the episode_timesteps
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hparams = { 
            'batch_size': 1000,
            'lr': 5e-3, 
            'gamma': 0.98,
            'sync_rate': 500,
            'replay_size':50000,
            'warm_start_steps': 1000,
            'epsilon_init':epsilon_init,
            'epsilon_final':epsilon_final,
            'epsilon_timesteps':epsilon_timesteps,
            'device': device,
        }
        if agent_type == 'GATV':
            model = gatv.GATVLightning
        elif agent_type == 'GATW':
            model = gatw.GATWLightning
        elif agent_type == 'DQN':
            model = dqn.DQNLightning
        elif agent_type == 'DQN2':
            model = dqn2.DQN2Lightning
        return model(env, **hparams).to(device)
    raise ValueError(f'{agent_type} not defined.')
