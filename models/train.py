""" 
    Trains an A_CAT agent.

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

def get_controller(agent_type, env, epsilon_init, epsilon_final, epsilon_timesteps, network):
    if agent_type == 'ACAT': return ACAT(env.phases, epsilon_init, epsilon_final, epsilon_timesteps)
    if agent_type == 'MARLIN': return MARLIN(env.phases, epsilon_init, epsilon_final, epsilon_timesteps, network)
    raise ValueError(f'{agent_type} not defined.')


def main(train_config_path=TRAIN_CONFIG_PATH, seed=0):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    train_args = parse_train_config(train_config_path)
    network = train_args['network']
    agent_type = train_args['agent_type']

    experiment_time = train_args['experiment_time']
    episode_time = train_args['experiment_save_agent_interval']

    # Epsilon 
    epsilon_init = train_args['epsilon_init']
    epsilon_final = train_args['epsilon_final']
    epsilon_timesteps = train_args['epsilon_schedule_timesteps']

    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 

    np.random.seed(seed)

    expr_path = expr_path_create(network)


    expr_config_dump(network, expr_path, config, flows, roadnet)
    env = Environment(roadnet, eng)
    approx = TileCodingApproximator(roadnet, flows)
    ctrl = get_controller(agent_type, env, epsilon_init, epsilon_final, epsilon_timesteps, network)

    info_dict = defaultdict(lambda : [])
    s_prev = None
    a_prev = None

    num_episodes = int(experiment_time / episode_time)
    for eps in trange(num_episodes, position=0):
        gen = env.loop(episode_time)

        try:
            while True:
                observations = next(gen)
                if observations is not None:
                    state = approx.approximate(observations)

                    # Rounded delay state
                    # state = {tid: (*obs[:2], round(obs[2]), round(obs[3])) for tid, obs in observations.items()}

                    actions = ctrl.act(state)

                    if s_prev is None and a_prev is None:
                        s_prev = state
                        a_prev = actions

                    else:
                        r_next = {_id: -sum(_obs[2:]) for _id, _obs in observations.items()}
                        ctrl.update(s_prev, a_prev, r_next, state)
                        
                        sum_speeds = sum(([float(vel) for vel in env.speeds.values()]))
                        num_vehicles = len(env.speeds)
                        info_dict["rewards"].append(r_next)
                        info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
                        info_dict["vehicles"].append(num_vehicles)
                        info_dict["observation_spaces"].append(observations) # No function approximation.
                        info_dict["actions"].append(actions)
                        info_dict["states"].append(state)

                    s_prev = state
                    a_prev = actions
                    gen.send(actions)

        except StopIteration as e:
            result = e.value

            chkpt_dir = Path(f"{expr_path}/checkpoints/")
            chkpt_num = str(eps * episode_time)
            os.makedirs(chkpt_dir, exist_ok=True)
            ctrl.save_checkpoint(chkpt_dir, chkpt_num)

            s_prev = None
            a_prev = None
            ctrl.reset()


    # Store train info dict.
    expr_logs_dump(expr_path, 'train_log.json', info_dict)

    return str(expr_path)

if __name__ == '__main__':
    main(train_config_path='config/train.config')
