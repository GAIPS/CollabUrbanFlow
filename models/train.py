""" 
    Trains an A_CAT agent.

    TODO: Make this script specific to A_CAT agent.
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
# FIXME: DEBUG
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
# print(sys.path)

from datetime import datetime

from tqdm import tqdm
import configparser
import numpy as np
from cityflow import Engine

from agents.actor_critic import ACAT, epsilon_decay
from converters import DelayConverter
from approximators.tile_coding import TileCodingApproximator

# prevent randomization
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

def main(train_config_path=TRAIN_CONFIG_PATH, seed=0):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    # Load train config file with parameters.
    train_config = configparser.ConfigParser()
    train_config.read(train_config_path)
    train_args = train_config['train_args']
    network = train_args['network']
    experiment_time = int(train_args['experiment_time'])
    experiment_save_agent_interval = int(train_args['experiment_save_agent_interval'])

    # Parse train parameters.
    config_file_path = Path(f'data/networks/{network}/config.json')
    roadnet_file_path = Path(f'data/networks/{network}/roadnet.json')
    flow_file_path = Path(f'data/networks/{network}/flow.json')
    eng = Engine(config_file_path.as_posix(), thread_num=4)

    np.random.seed(seed)
    eng.set_random_seed(seed)
    with config_file_path.open() as f: config = json.load(f)
    with flow_file_path.open() as f: flows = json.load(f)
    with roadnet_file_path.open() as f: roadnet = json.load(f)

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

    dc = DelayConverter(roadnet, eng)
    approx = TileCodingApproximator(roadnet, flows)
    acat = ACAT(dc.phases)

    info_dict = defaultdict(lambda : [])

    min_green = 5
    max_green = 90
    yellow = 5
    s_prev = None
    a_prev = None
    for time_counter in tqdm(range(experiment_time)):
        step_counter = time_counter % experiment_save_agent_interval

        decision_step = step_counter % 5 == 0 
        if decision_step:
            # State: is composed by the internal state and delay.
            # internal state is affected by environment conditions
            # or by yellew and green rules.
            observations, exclude_actions = dc.convert()
            state = approx.approximate(observations)
            actions = acat.act(state, exclude_actions=exclude_actions)
            dc.update(actions)

            if s_prev is None and a_prev is None:
                s_prev = state
                a_prev = actions

            else:
                # INTERLEAVED COOPERATION
                r_next = {tl_id: -sum(obs[2:]) for tl_id, obs in observations.items()}
                acat.update(s_prev, a_prev, r_next, state)

                def fn(x, u):
                    # First cycle ignore yellow transitions
                    if step_counter <= min_green: return False
                    # Switch to yellow
                    if int(x[0]) != u: return True
                    if int(x[1])  == min_green: return True
                    return False

                def ctrl(x, u):
                    # Switch to yellow
                    if int(x[0]) != u: return int(2 * x[0] + 1)
                    # Switch to green
                    if int(x[1]) == yellow: return int(2 * x[0])

                controller_actions = {
                    tl_id: ctrl(obs, actions[tl_id])
                    for tl_id, obs in observations.items() if fn(obs, actions[tl_id])
                }
                this_observation = observations.get('247123161', {})
                this_action = actions.get('247123161', {})
                this_phase_id = controller_actions.get('247123161', {})
                print(f'{time_counter}:{this_observation} --> {this_action} --> {this_phase_id}') 
                for tl_id, tl_phase_id in controller_actions.items():
                    eng.set_tl_phase(tl_id, tl_phase_id)
                
                
                sum_speeds = sum(([float(vel) for vel in eng.get_vehicle_speed().values()]))
                num_vehicles = eng.get_vehicle_count()
                info_dict["rewards"].append(r_next)
                info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
                info_dict["vehicles"].append(num_vehicles)
                info_dict["observation_spaces"].append(observations) # No function approximation.
                info_dict["actions"].append(actions)
                info_dict["states"].append(state)

                s_prev = state
                a_prev = actions



        if step_counter == experiment_save_agent_interval - 1:
            # TODO: use path
            chkpt_dir = f"{experiment_path}/checkpoints/"
            os.makedirs(chkpt_dir, exist_ok=True)

            eng.reset()
            chkpt_dir = Path(chkpt_dir)
            chkpt_num = str(time_counter)
            os.makedirs(chkpt_dir, exist_ok=True)
            acat.save_checkpoint(chkpt_dir, chkpt_num)
            for tl_id in acat.tl_ids:
                eng.set_tl_phase(tl_id, 0)

            s_prev = None
            a_prev = None
            num_episodes = time_counter // experiment_save_agent_interval + 1
            acat.eps += epsilon_decay(num_episodes)
            dc.reset()
            import ipdb; ipdb.set_trace()
        else:
            eng.next_step()


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
    main(train_config_path='config/train.config')
