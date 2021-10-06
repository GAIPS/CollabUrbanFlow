import json
import os
import random
from pathlib import Path
from shutil import copyfile

# TODO: Build a factory
from environment import Environment
from jobs.rollouts import concat
from models.rollouts import make_info_dict, update_emissions
from plots.test_plots import main as test_plots
from utils.file_io import engine_create, engine_load_config, expr_logs_dump, \
    parse_test_config, parse_train_config, expr_path_create

# prevent randomization
PYTHONHASHSEED = -1
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'
TEST_CONFIG_PATH = 'config/test.config'


# TODO: yellow is being taken from environment not from roadnet file
def main(experiment_dir=None):
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

    eng = engine_create(Path('data/networks/' + network_name + "/config.json"), thread_num=4)
    env = Environment(roadnet, eng)

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

    except StopIteration as e:
        pass

    expr_logs_dump(target_path, 'emission_log.json', emissions)

    info_dict['id'] = 0
    return concat([info_dict])


if __name__ == '__main__':
    train_args = parse_train_config(TRAIN_CONFIG_PATH)
    network_name = train_args['network']
    target_path = expr_path_create(network_name)
    res = main(experiment_dir=target_path)

    filename = f'rollouts_test.json'
    dump_path = target_path / filename
    with dump_path.open('w') as fj:
        json.dump(res, fj)

    test_plots(experiment_root_folder=target_path)