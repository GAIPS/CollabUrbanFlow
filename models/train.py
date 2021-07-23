from collections import defaultdict
import json
import os
from shutil import copyfile

from datetime import datetime

from pathlib import Path
from tqdm import tqdm
import configparser
import numpy as np
from matplotlib import pyplot as plt
from cityflow import Engine

from agents.actor_critic import ACAT, WAVE
from tile_coding import TileCodingMapper

NUM_EPISODES=2
EPISODE=2 * 3600

# prevent randomization
PYTHONHASHSEED=-1


def build():
    pass

def load():
    pass

def main(train_config_path=None):
    # Setup config parser path.
    if train_config_path is None:
        seed = 0
        network = 'intersection'
    else:
        print(f'Loading train parameters from: {train_config_path}')

        # Load config file with parameters.
        train_config = configparser.ConfigParser()
        train_config.read(train_config_path)
        train_args = train_config['train_args']

        # only seed
        experiment_seed = eval(train_args['experiment_seed'])
        seed = int(experiment_seed) if experiment_seed  is not None else 0
        network = train_args['network']

    # Parse train parameters.
    config_file_path = Path(f'data/networks/{network}/config.json')
    roadnet_file_path = Path(f'data/networks/{network}/roadnet.json')
    flow_file_path = Path(f'data/networks/{network}/flow.json')
    eng = Engine(config_file_path.as_posix(), thread_num=4)

    np.random.seed(seed)
    eng.set_random_seed(seed)
    with roadnet_file_path.open() as f: roadnet = json.load(f)
    with config_file_path.open() as f: config = json.load(f)


    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'
    experiment_path =  f'data/emissions/intersection_{timestamp}'
    # TODO: replace by pathlib
    os.makedirs(experiment_path, exist_ok=True)
    print(f'Experiment: {str(experiment_path)}\n')

    # TODO: save logs
    config['dir'] = f'{experiment_path}/'
    
    save_dir_path = Path(experiment_path) / 'config'
    if not save_dir_path.exists():
        save_dir_path.mkdir()
    copyfile(train_config_path, save_dir_path / 'train.config')
    copyfile(flow_file_path, save_dir_path / 'flow.json')
    copyfile(roadnet_file_path, save_dir_path / 'roadnet.json')
    with (save_dir_path / 'config.json').open('w') as f: json.dump(config, f)
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


        tl_id = intersection['id']
        num_phases = len(phases_per_edges)
        # TODO: WAVE is the function approximator.
        # TODO: replace by delay calculator.
        wave = WAVE(eng, phases_per_edges)
        # TODO: TileCoding must receive the capacities 
        mapper = TileCodingMapper(len(phases_per_edges), 1)
        acat = ACAT(tl_id, num_phases, wave, mapper)
        a_cats.append(acat)


    info_dict = defaultdict(lambda : [])
    emissions = []
    for time_step in tqdm(range(int(EPISODE * NUM_EPISODES))):
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

            # save time_step
            # if time_episode % save_agent_interval == 0:
            #     pass

        sum_speeds = sum(([float(vel) for vel in eng.get_vehicle_speed().values()]))
        num_vehicles = eng.get_vehicle_count()
        info_dict["rewards"].append(reward_dict)
        info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
        info_dict["vehicles"].append(num_vehicles)
        info_dict["observation_spaces"].append(obs_dict)
        info_dict["actions"].append(action_dict)
        info_dict["states"].append(state_dict)

        # update_emissions(eng, emissions)
        eng.next_step()

        # TODO: use path
        chkpt_dir = f"{experiment_path}/checkpoints/"
        os.makedirs(chkpt_dir, exist_ok=True)
        if (eng.get_current_time() + 1) % EPISODE == 0:
            eng.reset()
            chkpt_dir = Path(chkpt_dir)
            chkpt_num = str(int(eng.get_current_time() + 1))
            for a_cat in a_cats:
                os.makedirs(chkpt_dir, exist_ok=True)
                a_cat.save_checkpoint(chkpt_dir, chkpt_num)
                eng.set_tl_phase(a_cat.tl_id, 0)


    # Store train info dict.
    # TODO: Turn all of this into Path standard
    logs_dir_path = Path(experiment_path) / 'logs'
    print(logs_dir_path)
    os.makedirs(logs_dir_path.as_posix(), exist_ok=True)
    train_log_path = logs_dir_path / "train_log.json"
    with train_log_path.open('w') as f:
        json.dump(info_dict, f)

    return str(experiment_path)

if __name__ == '__main__':
    main(train_config_path='train.config')
