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

from actor_critic import ACAT, WAVE
from tile_coding import TileCodingMapper

from ipdb import set_trace

NUM_EPISODES=20
EPISODE=24 * 3600
# TODO: separate parameter parsing with a decorator
def main(train_config_path=None):

    # Setup config parser path.
    if train_config_path is not None:
        print(f'Loading train parameters from: {train_config_path}')

        # Load config file with parameters.
        train_config = configparser.ConfigParser()

        train_config.read(train_config_path)

        train_args = train_config['train_args']

        # only seed
        seed = int(train_args['experiment_seed']) if train_args['experiment_seed'] is not None else None
    else:
        # print('Loading train parameters from: configs/train.config [DEFAULT]')
        seed = 0

    # Parse train parameters.
    #train_args = config_parser.parse_train_params(print_params=True)

    config_file_path = 'network/intersection/config.json'
    eng = Engine(config_file_path, thread_num=4)

    np.random.seed(seed)
    eng.set_random_seed(seed)
    with open('network/intersection/roadnet.json', 'r') as f:
        network = json.load(f)



    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'
    experiment_path =  f'data/intersection_{timestamp}'
    os.makedirs(experiment_path, exist_ok=True)
    print(f'Experiment: {str(experiment_path)}\n')

    intersections = [item for item in network['intersections'] if not item['virtual']]
    # prepare intersection
    a_cats = []
    phases_per_edges = {}
    p = 0
    for i, intersection in enumerate(intersections):
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
        wave = WAVE(eng, phases_per_edges)
        mapper = TileCodingMapper(len(phases_per_edges), 1)
        acat = ACAT(tl_id, num_phases, wave, mapper)
        a_cats.append(acat)


    info_dict = defaultdict(lambda : [])
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

        sum_speeds = sum(([float(vel) for vel in eng.get_vehicle_speed().values()]))
        num_vehicles = eng.get_vehicle_count()
        info_dict["rewards"].append(reward_dict)
        info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
        info_dict["vehicles"].append(num_vehicles)
        info_dict["observation_spaces"].append(obs_dict)
        info_dict["actions"].append(action_dict)
        info_dict["states"].append(state_dict[tl_id])

        eng.next_step()
        # TODO: use path
        chkpt_dir = f"{experiment_path}/checkpoint/"
        os.makedirs(chkpt_dir, exist_ok=True)
        if (eng.get_current_time() + 1) % EPISODE == 0:
            eng.reset()
            for a_cat in a_cats:
                this_chkpt_dir = f"{chkpt_dir}/{a_cat.tl_id}"
                os.makedirs(this_chkpt_dir, exist_ok=True)
                a_cat.save(file_dir=this_chkpt_dir, chkpt_num=int((time_step + 1) / EPISODE) + 1)
                eng.set_tl_phase(a_cat.tl_id, 0)


        # eng.get_current_time()
        # eng.get_lane_vehicle_count()
        # eng.get_lane_waiting_vehicle_count()
        # eng.get_lane_vehicles()
        # eng.get_vehicle_speed()
        # do something 





    chkpt_dir = f"{experiment_path}/checkpoint/"
    os.makedirs(chkpt_dir, exist_ok=True)
    for a_cat in a_cats:
        a_cat.save(file_dir=chkpt_dir)
    # chkpt_dir = f"{experiment_path}/checkpoint/"
    # os.makedirs(chkpt_dir, exist_ok=True)
    # a_cat.save(file_dir=chkpt_dir)
    # with open('data/train_log.json', 'w') as f:
    #     json.dump(info_dict, f)

    # Store train parameters (config file). 
    # config_parser.store_config(experiment_path / 'config')

    save_dir_path = Path(experiment_path) / 'config'
    if not save_dir_path.exists():
        save_dir_path.mkdir()
    copyfile(train_config_path, save_dir_path / 'train.config')
    # Store a copy of the tls_config.json file.
    # tls_config_path = NETWORKS_PATH / train_args.network / 'tls_config.json'
    # copyfile(tls_config_path, experiment_path / 'tls_config.json')

    # Store a copy of the demands.json file.
    # demands_file_path = NETWORKS_PATH / train_args.network / 'demands.json'
    # copyfile(demands_file_path, experiment_path / 'demands.json')

    # Run the experiment.
    #info_dict = exp.run(train_args.experiment_time)

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
    main()
