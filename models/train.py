""" 
    Trains a Reinforcement learning system.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
import os, sys
from collections import defaultdict
from pathlib import Path
#from shutil import copyfile
# FIXME: DEBUG
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
# print(sys.path)


import numpy as np
from environment import Environment
from agents import get_agent
from models import get_loop

from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, \
                        expr_path_create, expr_config_dump, expr_logs_dump, \
                        parse_train_config

# prevent randomization
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

# Main abstracts the training loop and assigns an
# agent to its environment
# TODO: Put logging on a decorator.
def main(train_config_path=TRAIN_CONFIG_PATH):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    train_args = parse_train_config(train_config_path)
    network = train_args['network']
    agent_type = train_args['agent_type']

    experiment_time = int(train_args['experiment_time'])
    save_agent_interval = int(train_args['experiment_save_agent_interval'])
    seed = int(train_args.get('experiment_seed', 0))

    # Epsilon 
    epsilon_init = train_args['epsilon_init']
    epsilon_final = train_args['epsilon_final']
    epsilon_timesteps = train_args['epsilon_schedule_timesteps']

    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 
    env = Environment(network, roadnet, eng, episode_timesteps=save_agent_interval)

    np.random.seed(seed)
    expr_path = expr_path_create(network)
    chkpt_dir = Path(f"{expr_path}/checkpoints/")
    chkpt_dir.mkdir(exist_ok=True)


    expr_config_dump(network, expr_path, config, flows, roadnet)
    approx = TileCodingApproximator(roadnet, flows)
    agent = get_agent(agent_type, env, epsilon_init,
                      epsilon_final, epsilon_timesteps)

    train_loop = get_loop(agent_type)
    if agent_type in ('DQN', 'VE'):
        info_dict = train_loop(env, agent, experiment_time,
                               save_agent_interval, chkpt_dir, seed)
    else:
        info_dict = train_loop(env, agent, approx, experiment_time, save_agent_interval, chkpt_dir, seed)

    # Store train info dict.
    expr_logs_dump(expr_path, 'train_log.json', info_dict)

    return str(expr_path)

if __name__ == '__main__':
    main(train_config_path='config/train.config')
