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
# TODO: Build a factory
from models.train import TRAIN_CONFIG_PATH
from agents import load_agent
from environment import Environment, rollback_loop
from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, \
    expr_path_test_target, parse_test_config, parse_train_config

# prevent randomization
PYTHONHASHSEED=-1



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
    agent = load_agent(agent_type, chkpt_dir_path, chkpt_num)
    agent.stop()

    info_dict = rollback_loop(env, agent, approx, rollout_time, target_path, chkpt_num)
    # emissions = []
    # 
    # gen = env.loop(rollout_time)

    # try:
    #     while True:
    #         experience = next(gen)
    #         if experience is not None:
    #             observations = experience[0]
    #             state = approx.approximate(observations)
    #             actions = ctrl.act(state)

    #             s_prev = state
    #             a_prev = actions
    #             gen.send(actions)
    #         update_emissions(eng, emissions)

    # except StopIteration as e:
    #     result = e.value
    # expr_logs_dump(target_path, 'emission_log.json', emissions)
    
    # env.info_dict['id'] = chkpt_num
    return info_dict

if __name__ == '__main__':
    main(run_path='train.config')
