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

    train_args = parse_train_config(TRAIN_CONFIG_PATH)
    agent_type = train_args['agent_type']
    agent = load_agent(agent_type, chkpt_dir_path, chkpt_num, rollout_time, train_args["network"])

    rollback_loop = get_loop(agent, train=False)
    if agent_type == "DQN":
        info_dict = rollback_loop(agent[0], agent[1], target_path, rollout_time)
    else:
        agent.stop()
        info_dict = rollback_loop(env, agent, approx, rollout_time, target_path, chkpt_num)
    
    info_dict['id'] = chkpt_num
    return info_dict

if __name__ == '__main__':
    main(run_path='train.config')
