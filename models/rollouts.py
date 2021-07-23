from collections import defaultdict
import json

from datetime import datetime

from pathlib import Path
from tqdm import tqdm
import configparser
import numpy as np
from cityflow import Engine

# TODO: Build a factory
from agents.actor_critic import ACAT, WAVE
from tile_coding import TileCodingMapper

NUM_EPISODES=2
EPISODE=2 * 3600

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

def build():
    pass

def load():
    pass


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
    roadnet_file_path = Path(orig_path) / 'config' / 'roadnet.json' 
    with roadnet_file_path.open() as f: roadnet = json.load(f)
    with config_file_path.open() as f: cfg = json.load(f)
    cfg['dir'] = f'{config_file_path.parent}/'
    with config_file_path.open('w') as f: json.dump(cfg, f)
    eng = Engine(config_file_path.as_posix(), thread_num=4)

    np.random.seed(seed)
    eng.set_random_seed(seed)

    # Build agents
    intersections = [item for item in roadnet['intersections'] if not item['virtual']]

    # TODO: prepare intersection
    a_cats = []
    phases_per_edges = {}
    p = 0
    for intersection in intersections:
        lightphases = intersection['trafficLight']['lightphases']
        for linkids in lightphases:
            if any(linkids['availableRoadLinks']):
                linkids = linkids['availableRoadLinks']
                edges = []
                for linkid in linkids:
                    edgeid = intersection['roadLinks'][linkid]['startRoad']
                    if edgeid not in edges: edges.append(edgeid)
                phases_per_edges[p] = edges
                p += 1


        # tl_id = intersection['id']
        # num_phases = len(phases_per_edges)
        # TODO: WAVE is the function approximator.
        # TODO: replace by delay calculator.
        wave = WAVE(eng, phases_per_edges)
        # TODO: TileCoding must receive the capacities 
        mapper = TileCodingMapper(len(phases_per_edges), 1)
        # TODO: get_agent('A_CAT')
        # TODO: Load agent.
        acat = ACAT.load_checkpoint(checkpoints_dir_path, chkpt_num)
        acat.get_wave = wave
        # acat = ACAT(tl_id, num_phases, wave, mapper)
        a_cats.append(acat)


    info_dict = defaultdict(lambda : [])
    emissions = []
    
    for time_step in tqdm(range(rollout_time)):
        obs_dict = {}
        state_dict = {}
        action_dict = {}
        reward_dict = {}
        time_episode = time_step % EPISODE
        for a_cat in a_cats:
            num_phases, tl_id, obser = a_cat.num_phases, a_cat.tl_id, a_cat.get_wave()

            # Must come before compute otherwise will miss next_change.
            phase_id, next_change = a_cat.phase_ctrl
            state_dict[tl_id], action_dict[tl_id], reward_dict[tl_id] = a_cat.compute()
            obs_dict[tl_id] = [int(obs) for obs in obser]

            #print(time_step, phase_id, next_change)
            if time_episode == (next_change - 5): # only visits "odd" phases (yellow)
                eng.set_tl_phase(a_cat.tl_id, (2 * phase_id + 1) % (2 * num_phases))
            elif time_episode == next_change: # Only visits even phases
                eng.set_tl_phase(a_cat.tl_id, (2 * phase_id) % (2 * num_phases))

        sum_speeds = sum(([float(vel) for vel in eng.get_vehicle_speed().values()]))
        num_vehicles = eng.get_vehicle_count()
        info_dict["rewards"].append(reward_dict)
        info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
        info_dict["vehicles"].append(num_vehicles)
        info_dict["observation_spaces"].append(obs_dict)
        info_dict["actions"].append(action_dict)
        info_dict["states"].append(state_dict)

        update_emissions(eng, emissions)
        eng.next_step()

    # Store train info dict.
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
