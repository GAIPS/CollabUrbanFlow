""" 
    Tests A_CAT agent.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
import json

from datetime import datetime

from pathlib import Path
from tqdm.auto import trange
import configparser
import numpy as np
from cityflow import Engine
from agents import load_agent
from models import get_loop

from environment import Environment
from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, \
    expr_path_test_target, parse_test_config, parse_train_config

# prevent randomization
PYTHONHASHSEED=-1

TRAIN_CONFIG_PATH = 'config/train.config'

def main(test_config_path=None):
    # Setup config parser path.
    args = parse_test_config(test_config_path)
    orig_path =  args['orig_path']
    rollout_time =  args['rollout_time']
    chkpt_num =  args['chkpt_num']
    seed =  args['seed']
    agent_type = args['agent_type']
    network = args['network']
    chkpt_dir_path = Path(orig_path) / 'checkpoints' 

    target_path = expr_path_test_target(orig_path)

    # TODO: replace by pathlib
    print(f'Experiment: {str(target_path)}\n')
    
    # Have to update the target path
    config_dir_path = Path(orig_path) / 'config'

    config, flows, roadnet = engine_load_config(config_dir_path)
    config['dir'] = f'{config_dir_path}/'
    with (config_dir_path / 'config.json').open('w') as f: json.dump(config, f)
    eng = engine_create(config_dir_path / 'config.json', seed=seed, thread_num=4)

    np.random.seed(seed)

    env = Environment(roadnet, eng)
    approx = TileCodingApproximator(roadnet, flows)

    # TODO: nets are approximators -- make load_agent produce the two. 
    agent, nets = load_agent(env, agent_type, chkpt_dir_path, chkpt_num, rollout_time, network)

    rollback_loop = get_loop(agent_type, train=False)
    if agent_type == "DQN":
        info_dict = rollback_loop(env, agent, nets, rollout_time, target_path)
    else:
        agent.stop()
        info_dict = rollback_loop(env, agent, approx, rollout_time, target_path)
    
    info_dict['id'] = chkpt_num
    return info_dict

if __name__ == '__main__':
    main(test_config_path='data/emissions/20211010185842.098537/')
