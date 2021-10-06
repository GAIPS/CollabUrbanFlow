import configparser
import json
import os
import random
import time
from pathlib import Path
from shutil import copyfile

# TODO: Build a factory
import numpy as np

from environment import Environment
from jobs.rollouts import concat
from models.rollouts import make_info_dict, update_emissions
from plots.test_plots import main as test_plots
from utils.file_io import engine_create, engine_load_config, expr_logs_dump, \
    parse_test_config, parse_train_config, expr_path_create
from jobs.rollouts import NonDaemonicPool

# prevent randomization
PYTHONHASHSEED = -1
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'
TEST_CONFIG_PATH = 'config/test.config'

def delay_roll(args):
    """Delays execution.

        Parameters:
        -----------
        * args: tuple
            Position 0: execution delay of the process.
            Position 1: store the train config file.

        Returns:
        -------
        * fnc : function
            An anonymous function to be executed with a given delay
    """
    time.sleep(args[0])
    return main(experiment_dir=args[1], seed=args[2])

# TODO: yellow is being taken from environment not from roadnet file
def main(experiment_dir=None, seed=0):
    train_args = parse_train_config(TRAIN_CONFIG_PATH)
    network_name = train_args['network']

    if experiment_dir is None:
        target_path = expr_path_create(network_name)
    else:
        target_path = experiment_dir

    # Setup config parser path.
    args = parse_test_config(TEST_CONFIG_PATH)
    rollout_time = args['rollout_time']
    # Have to update the target path
    config_dir_path = Path(target_path) / 'config'

    config, flows, roadnet = engine_load_config(network_name)
    config['dir'] = f'{config_dir_path}/'
    os.mkdir(config_dir_path)
    with (config_dir_path / 'config.json').open('w+') as f:
        json.dump(config, f)
    copyfile(TRAIN_CONFIG_PATH, target_path / 'config/train.config')

    eng = engine_create(Path('data/networks/' + network_name + "/config.json"), seed=seed, thread_num=4)
    env = Environment(roadnet, eng)
    np.random.seed(seed)
    info_dict = make_info_dict()
    emissions = []

    with open('data/networks/' + network_name + '/roadnet.json', 'r') as f:
        network = json.load(f)
    intersections = [item for item in network['intersections'] if not item['virtual']]


    gen = env.loop(rollout_time)
    try:
        while True:
            observations = next(gen)
            if observations is not None:
                actions = {}
                for i, intersection in enumerate(intersections):
                    phases = [inter for inter in intersection['trafficLight']['lightphases'] if
                          len(inter['availableRoadLinks']) > 0]
                    actions[intersection["id"]] = random.choice(range(len(phases)))

                r_next = {_id: -sum(_obs[2:]) for _id, _obs in observations.items()}
                sum_speeds = sum(([float(vel) for vel in env.speeds.values()]))
                num_vehicles = len(env.speeds)

                info_dict["rewards"].append(r_next)
                info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
                info_dict["vehicles"].append(num_vehicles)
                info_dict["observation_spaces"].append(observations)  # No function approximation.
                info_dict["actions"].append(actions)
                info_dict["states"].append(observations)

                gen.send(actions)
            update_emissions(eng, emissions)

    except StopIteration as e:
        pass

    expr_logs_dump(target_path, 'emission_log.json', emissions)

    info_dict['id'] = seed
    return info_dict


if __name__ == '__main__':
    train_args = parse_train_config(TRAIN_CONFIG_PATH)
    network_name = train_args['network']
    target_path = expr_path_create(network_name)

    run_config = configparser.ConfigParser()
    run_config.read('config/run.config')

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                                 'must match the number of runs (`num_runs`) argument.')
    rollouts_cfg_paths = []
    for seed in train_seeds:
        rollouts_cfg_paths.append((target_path / str(seed), seed))
        os.mkdir(target_path / str(seed))


    # rvs: directories' names holding experiment data
    if num_processors > 1:
        pool = NonDaemonicPool(num_processors)
        rvs = pool.map(delay_roll, [(delay, cfg[0], cfg[1])
                                    for (delay, cfg) in zip(range(len(rollouts_cfg_paths)), rollouts_cfg_paths)])
        pool.close()
    else:
        rvs = []
        for cfg,seed in rollouts_cfg_paths:
            rvs.append(delay_roll((0.0, cfg, seed)))

    res = concat(rvs)
    filename = f'rollouts_test.json'
    dump_path = target_path / filename
    with dump_path.open('w') as fj:
        json.dump(res, fj)

    test_plots(experiment_root_folder=target_path)
