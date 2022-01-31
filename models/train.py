""" 
    Trains a Reinforcement learning system.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
from collections import defaultdict
from pathlib import Path
ys
import numpy as np
from environment import Environment
from agents import get_agent
from models import get_loop

from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, \
                        expr_path_create, expr_config_dump, expr_logs_dump, \
                        parse_env_parameters, parse_train_parameters, parse_mdp_parameters

# prevent randomization
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

# Main abstracts the training loop and assigns an
# agent to its environment
# TODO: Put logging on a decorator.
def main(train_config_path=TRAIN_CONFIG_PATH):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    opt = parse_train_parameters(train_config_path)
    network = opt.network
    save_agent_interval = int(opt.save_agent_interval)
    seed = opt.experiment_seed

    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 

    #TODO: Make a special config section for the env.
    env_args = parse_env_parameters(train_config_path)
    mdp_args = parse_mdp_parameters(train_config_path)

    env = Environment(network, roadnet, env_args, mdp_args, eng,
        episode_timesteps=save_agent_interval
    )

    np.random.seed(seed)
    expr_path = expr_path_create(network)
    chkpt_dir = Path(f"{expr_path}/checkpoints/")
    chkpt_dir.mkdir(exist_ok=True)

    expr_config_dump(network, expr_path, config, flows, roadnet)
    approx = TileCodingApproximator(roadnet, flows)
    agent = get_agent(env, opt)

    train_loop = get_loop(opt.agent_type)
    if opt.agent_type in ('DQN', 'GAT'):
        info_dict = train_loop(env, agent, opt, chkpt_dir, seed)
    else:
        info_dict = train_loop(env, agent, approx, opt, chkpt_dir, seed)

    # Store train info dict.
    expr_logs_dump(expr_path, 'train_log.json', info_dict)

    return str(expr_path)

if __name__ == '__main__':
    main(train_config_path='config/train.config')
