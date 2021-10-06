"""
    Tests A_CAT agent.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
import json
import os
from pathlib import Path
from shutil import copyfile

# TODO: Build a factory
from agents.webster import WEBSTER
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


# TODO: yellow needs to be 6 sec
def main(experiment_dir=None, cycle_time=60, aggregation_period=600):
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

    with open('data/networks/' + network_name + '/roadnet.json', 'r') as f:
        network = json.load(f)

    eng = engine_create(Path('data/networks/' + network_name + "/config.json"), thread_num=4)
    env = Environment(roadnet, eng, step_size=1, yellow=6)

    info_dict = make_info_dict()
    emissions = []
    intersections = [item for item in network['intersections'] if not item['virtual']]
    phasectl = [(-1, 0)] * len(intersections)

    webster = WEBSTER(env, cycle_time=cycle_time, aggregation_period=aggregation_period)

    env._reset()
    for step in range(rollout_time):
        observations = env.observations
        if observations is not None:
            timings = webster.act(env.vehicles)
            timings = {intersection: [timing[0], timing[2]- timing[1]] for intersection,timing in timings.items()} # Ignore yellows
            if (step+1) % aggregation_period == 0: # When plan changes recalculate next shift
                for i, intersection in enumerate(env.phases):
                    phasectl[i] = (0, last_step + int(timings[intersection][0]) + env.yellow)

            #Apply webster timings to keep/change structure
            actions = {}
            for i, intersection in enumerate(env.phases):
                phaseid, next_change = phasectl[i]
                tl = {phaseid: int(timings[intersection][int(phaseid)]) for phaseid in env.phases[intersection]}
                if next_change == step :
                    actions[intersection] = 1
                    phaseid = (phaseid + 1) % len(tl)

                    yellow = env.yellow if step != 0 else 0
                    next_change = step + tl[phaseid] + yellow
                    last_step = step
                    phasectl[i] = (phaseid, next_change)
                else:
                    actions[intersection] = 0

            env._phase_ctl(actions)

            r_next = {_id: -sum(_obs[2:]) for _id, _obs in observations.items()}
            sum_speeds = sum(([float(vel) for vel in env.speeds.values()]))
            num_vehicles = len(env.speeds)
            info_dict["rewards"].append(r_next)
            info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
            info_dict["vehicles"].append(num_vehicles)
            info_dict["observation_spaces"].append(observations)  # No function approximation.
            info_dict["actions"].append(actions)
            info_dict["states"].append(observations)

            eng.next_step()

        update_emissions(eng, emissions)

    expr_logs_dump(target_path, 'emission_log.json', emissions)
    webster.terminate()
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



