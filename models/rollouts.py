""" 
    Tests A_CAT agent.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
from collections import defaultdict
import json

from datetime import datetime

from pathlib import Path
from tqdm.auto import trange
import configparser
import numpy as np
from cityflow import Engine
# TODO: Build a factory
from models.train import TRAIN_CONFIG_PATH
from agents.actor_critic import ACAT
from agents.marlin import MARLIN
from environment import Environment
from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, expr_logs_dump, \
    expr_path_test_target, parse_test_config, parse_train_config

# prevent randomization
PYTHONHASHSEED=-1


def get_controller(agent_type, chkpt_dir_path, chkpt_num):
    if agent_type == 'ACAT': return ACAT.load_checkpoint(chkpt_dir_path, chkpt_num)
    if agent_type == 'MARLIN': return MARLIN.load_checkpoint(chkpt_dir_path, chkpt_num)
    raise ValueError(f'{agent_type} not defined.')



def update_emissions(eng, emissions):
    """Builds sumo like emission file"""
    for veh_id in eng.get_vehicles(include_waiting=False):
        data = eng.get_vehicle_info(veh_id)

        emission_dict = {
            'time': eng.get_current_time(),
            'id': veh_id,
            'lane': data['drivable'],
            'pos': float(data['distance']),
            'route': simple_hash(data['route']),
            'speed': float(data['speed']),
            'type': 'human',
            'x': 0,
            'y': 0
        }
        emissions.append(emission_dict)

def simple_hash(x):
    return hash(x) % (11 * 255)

def make_list():
    return []

def make_info_dict():
    return defaultdict(make_list)

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
    ctrl = get_controller(agent_type, chkpt_dir_path, chkpt_num)
    ctrl.stop()

    s_prev = None
    a_prev = None

    info_dict = make_info_dict()
    emissions = []
    
    gen = env.loop(rollout_time)

    try:
        while True:
            observations = next(gen)
            if observations is not None:
                state = approx.approximate(observations)
                actions = ctrl.act(state)

                if s_prev is None and a_prev is None:
                    s_prev = state
                    a_prev = actions

                else:
                    r_next = {_id: -sum(_obs[2:]) for _id, _obs in observations.items()}

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
            update_emissions(eng, emissions)

    except StopIteration as e:
        result = e.value
    expr_logs_dump(target_path, 'emission_log.json', emissions)
    
    info_dict['id'] = chkpt_num
    return info_dict

if __name__ == '__main__':
    main(run_path='train.config')
