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
from agents.actor_critic import ACAT
from environment import Environment
from approximators.tile_coding import TileCodingApproximator

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

def simple_hash(x):
    return hash(x) % (11 * 255)

def make_list():
    return []

def make_info_dict():
    return defaultdict(make_list)

def main(run_path=None):
    # Setup config parser path.
    
    # Load config file with parameters.
    rollouts_config = configparser.ConfigParser()

    # TODO: Change path to checkpoints
    # TODO: Read train
    rollouts_config.read(run_path)
    rollouts_args = rollouts_config['test_args']

    # Setup parser with custom path (load correct train parameters).
    orig_path = rollouts_args['run-path']
    rollout_time = int(rollouts_args['rollout-time'])
    chkpt_num = int(rollouts_args['chkpt-number'])
    seed = int(rollouts_args['seed'])
    checkpoints_dir_path = Path(orig_path) / 'checkpoints' 

    # training data
    train_path = Path(orig_path) / 'config' / 'train.config'

    train_config = configparser.ConfigParser()
    train_config.read(train_path)

    network = train_config.get('train_args', 'network') 
    # TODO: Build agent
    # agent_type = train_config.get('agent_type', 'agent_type')

    target_path = Path(orig_path) / 'eval'
    target_path.mkdir(exist_ok=True)
    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'

    target_path =  target_path / f'{network}_{timestamp}'
    target_path.mkdir(exist_ok=True)
    

    # TODO: replace by pathlib
    print(f'Experiment: {str(target_path)}\n')
    
    # Build engine
    config_file_path = Path(orig_path) / 'config' / 'config.json' 
    flow_file_path = Path(f'data/networks/{network}/flow.json')
    roadnet_file_path = Path(orig_path) / 'config' / 'roadnet.json' 
    with roadnet_file_path.open() as f: roadnet = json.load(f)
    with flow_file_path.open() as f: flows = json.load(f)
    with config_file_path.open() as f: cfg = json.load(f)
    cfg['dir'] = f'{config_file_path.parent}/'
    with config_file_path.open('w') as f: json.dump(cfg, f)
    eng = Engine(config_file_path.as_posix(), thread_num=4)

    np.random.seed(seed)
    eng.set_random_seed(seed)

    env = Environment(roadnet, eng)
    approx = TileCodingApproximator(roadnet, flows)

    # TODO: get_agent('A_CAT')
    # TODO: Load agent.
    acat = ACAT.load_checkpoint(checkpoints_dir_path, chkpt_num)
    acat.stop()

    s_prev = None
    a_prev = None

    info_dict = make_info_dict()
    emissions = []
    
    obs_dict = {}
    state_dict = {}
    action_dict = {}
    reward_dict = {}


    gen = env.loop(rollout_time)

    try:
        while True:
            observations = next(gen)
            if observations is not None:
                state = approx.approximate(observations)
                actions = acat.act(state) 

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


    # TODO: Turn all of this into Path standard
    logs_dir_path = Path(target_path) / 'logs'
    logs_dir_path.mkdir(exist_ok=True)
    

    emission_log_path = logs_dir_path / "emission_log.json"
    with emission_log_path.open('w') as f:
        json.dump(emissions, f)
    info_dict['id'] = chkpt_num
    return info_dict

if __name__ == '__main__':
    main(run_path='train.config')
