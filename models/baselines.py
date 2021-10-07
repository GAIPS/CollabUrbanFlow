""" 
    Tests A_CAT agent.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
from collections import defaultdict
import json

from shutil import copyfile
from datetime import datetime

from pathlib import Path
from tqdm.auto import trange
import configparser
import numpy as np
from cityflow import Engine

# TODO: Build a factory
from environment import Environment
from controllers import MaxPressure
from utils.file_io import engine_create, engine_load_config, expr_logs_dump, \
                            expr_path_create, expr_config_dump
# prevent randomization
PYTHONHASHSEED=-1


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

def simple_hash(x): return hash(x) % (11 * 255)
def g_list(): return []
def g_dict(): return defaultdict(g_list)
def get_controller(ts_type):
    if ts_type == 'max_pressure': return MaxPressure(5, 90, 5)
    raise ValueError(f'{ts_type} not defined.')

def main(baseline_config_path=None):

    # Read configs.
    baseline_config = configparser.ConfigParser()
    baseline_config.read(baseline_config_path)
    network = baseline_config.get('baseline_args', 'network')
    ts_type = baseline_config.get('baseline_args', 'ts_type')
    # demand_type = baseline_config.get('baseline_args', 'demand_type')
    # demand_mode = baseline_config.get('baseline_args', 'demand_mode')
    seed = int(baseline_config.get('baseline_args', 'seed'))
    rollout_time = int(baseline_config.get('baseline_args', 'rollout-time'))

    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 
    np.random.seed(seed)
    eng.set_random_seed(seed)

    target_path = expr_path_create(network, seed)
    expr_config_dump(network, target_path, config, flows,
                     roadnet, dump_train_config=False)
    # Move temporary to target
    copyfile(baseline_config_path, target_path / 'config' / 'baseline.config')

    # Have to update the target path
    config_dir_path = Path(target_path) / 'config'
    config, flows, roadnet = engine_load_config(config_dir_path)
    config['dir'] = f'{config_dir_path}/'
    config['seed'] = seed
    with (config_dir_path / 'config.json').open('w') as f: json.dump(config, f)

    ctrl = get_controller(ts_type)
    env = Environment(roadnet, eng, feature=ctrl.feature)
    # TODO: Allow for more types of controllers.

    info_dict = g_dict()
    emissions = []
    gen = env.loop(rollout_time)
    actions = {}
    try:
        while True:
            observations = next(gen)
            if observations is not None:

                actions = ctrl.act(observations)

                r_next = {_id: -sum(_obs[2:]) for _id, _obs in observations.items()}

                sum_speeds = sum(([float(vel) for vel in env.speeds.values()]))
                num_vehicles = len(env.speeds)
                info_dict["rewards"].append(r_next)
                info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
                info_dict["vehicles"].append(num_vehicles)
                info_dict["observation_spaces"].append(observations) # No function approximation.
                info_dict["actions"].append(actions)
                info_dict["states"].append(observations)

                gen.send(actions)
            update_emissions(eng, emissions)

    except StopIteration as e:
        result = e.value
    expr_logs_dump(target_path, 'emission_log.json', emissions)
    
    info_dict['id'] = f'{ts_type}-{seed}'
    return (target_path, info_dict)

if __name__ == '__main__':
    main(run_path='train.config', baseline='max_pressure')
