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
from controllers import MaxPressure, Random, Static, Webster
from utils.file_io import engine_create, engine_load_config, expr_logs_dump, \
                            expr_path_create, expr_config_dump, parse_env_parameters, \
                            parse_mdp_parameters
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
def g_dict(): return defaultdict(list)

def get_controller(ts_type, env_args, mdp_args, n_actions, config_folder):
    if ts_type == 'max_pressure':
        return MaxPressure(env_args, mdp_args, n_actions)
    if ts_type == 'random': return Random(env_args, n_actions)
    if ts_type == 'static': return Static(env_args, mdp_args, config_folder)
    if ts_type == 'webster': return Webster(env_args, mdp_args, config_folder)
    raise ValueError(f'{ts_type} not defined.')

def update_info_dict(actions, env, info_dict, observations):
    r_next = {_id: -sum(_obs[2:]) for _id, _obs in observations.items()}
    sum_speeds = sum(([float(vel) for vel in env.speeds.values()]))
    num_vehicles = len(env.speeds)
    info_dict["rewards"].append(r_next)
    info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
    info_dict["vehicles"].append(num_vehicles)
    info_dict["observation_spaces"].append(observations)  # No function approximation.
    info_dict["actions"].append(actions)
    info_dict["states"].append(observations)



def main(baseline_config_path=None):

    # Read configs.
    baseline_config = configparser.ConfigParser()
    baseline_config.read(baseline_config_path)
    network = baseline_config.get('baseline_args', 'network')
    ts_type = baseline_config.get('baseline_args', 'ts_type')

    #TODO: Make a special config section for the env.
    env_args = parse_env_parameters(baseline_config_path)
    mdp_args = parse_mdp_parameters(baseline_config_path)

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

    emit = ts_type != 'static'
    env = Environment(network, roadnet, env_args, mdp_args, eng, emit=emit,
                      episode_timesteps=rollout_time)

    ctrl = get_controller(ts_type, env_args, mdp_args, 
                          env.n_actions, config_dir_path)

    info_dict = g_dict()
    emissions = []
    actions = {}
    # Updates every second -- manipulates engine.
    # Beware of incompatible roadnet definitions.
    # to do -- create a step by step loop.
    if ts_type in ('static', 'webster'):
        env.reset()

        for _ in trange(rollout_time):
            actions = ctrl.act(env.timestep)

            if env.timestep % 5 == 0:
                update_info_dict(actions, env, info_dict, env.observations)
        
            # action_schema == `set` 
            for tl, action in actions.items():
                env.engine.set_tl_phase(tl, 2 * action)
            
            update_emissions(eng, emissions)
            if ts_type  == 'webster':
                ctrl.update(env.vehicles)
            env.engine.next_step()
    else:
        gen = env.loop(rollout_time)

        try:
            while True:
                experience = next(gen)
                if experience is not None:
                    observations = experience[0]
                    actions = ctrl.act(observations)

                    update_info_dict(actions, env, info_dict, observations)

                    gen.send(actions)

        except StopIteration as e:
            result = e.value
            emissions = env.emissions

    expr_logs_dump(target_path, 'emission_log.json', emissions)
    
    info_dict['id'] = f'{ts_type}-{seed}'
    return (target_path, info_dict)



if __name__ == '__main__':
    main(run_path='train.config', baseline='max_pressure')
