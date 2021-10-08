""" 
    Trains a Reinforcement learning system.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
import ipdb
import os, sys
# import json, os, sys
from collections import defaultdict
from pathlib import Path
#from shutil import copyfile
# FIXME: DEBUG
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
# print(sys.path)


from tqdm.auto import trange
# import configparser
import numpy as np

from environment import Environment
from agents.actor_critic import ACAT
from agents.marlin import MARLIN

from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, \
                            expr_path_create, expr_config_dump, expr_logs_dump, \
                            parse_train_config

# prevent randomization
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

def get_agent(agent_type, env, epsilon_init, epsilon_final,
                   epsilon_timesteps, network):

    if agent_type == 'ACAT':
        return ACAT(env.phases, epsilon_init, epsilon_final,
                    epsilon_timesteps)
    if agent_type == 'MARLIN':
        return MARLIN(env.phases, epsilon_init, epsilon_final, epsilon_timesteps, network)
    raise ValueError(f'{agent_type} not defined.')


# Main abstracts the training loop and assigns an
# agent to its environment
def main(train_config_path=TRAIN_CONFIG_PATH, seed=0):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    train_args = parse_train_config(train_config_path)
    network = train_args['network']
    agent_type = train_args['agent_type']

    experiment_time = int(train_args['experiment_time'])
    episode_time = int(train_args['experiment_save_agent_interval'])

    # Epsilon 
    epsilon_init = train_args['epsilon_init']
    epsilon_final = train_args['epsilon_final']
    epsilon_timesteps = train_args['epsilon_schedule_timesteps']

    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 

    np.random.seed(seed)
    expr_path = expr_path_create(network)
    chkpt_dir = Path(f"{expr_path}/checkpoints/")


    expr_config_dump(network, expr_path, config, flows, roadnet)
    env = Environment(roadnet, eng)
    approx = TileCodingApproximator(roadnet, flows)
    agent = get_agent(agent_type, env, epsilon_init, epsilon_final,
                      epsilon_timesteps, network)

    info_dict = train_loop(env, agent, approx, experiment_time, episode_time, chkpt_dir)

    # Store train info dict.
    expr_logs_dump(expr_path, 'train_log.json', info_dict)

    return str(expr_path)

def train_loop(env, agent, approx, experiment_time, episode_time, chkpt_dir):
    # 1) Seed everything
    num_episodes = int(experiment_time / episode_time)

    s_prev = None
    a_prev = None

    for eps in trange(num_episodes, position=0):
        gen = env.loop(episode_time)

        try:
            while True:
                experience = next(gen)
                if experience is not None:
                    observations, reward = experience[:2]
                    state = approx.approximate(observations)
                    actions = agent.act(state)

                    if s_prev is None and a_prev is None:
                        s_prev = state
                        a_prev = actions

                    else:
                        agent.update(s_prev, a_prev, reward, state)
                        
                    s_prev = state
                    a_prev = actions
                    gen.send(actions)

        except StopIteration as e:
            result = e.value

            chkpt_num = str(eps * episode_time)
            os.makedirs(chkpt_dir, exist_ok=True)
            agent.save_checkpoint(chkpt_dir, chkpt_num)

            s_prev = None
            a_prev = None
            agent.reset()
    return env.info_dict

def train_torch(env, engine, agent):
    # 1) Seed everything
    pass

if __name__ == '__main__':
    main(train_config_path='config/train.config')
