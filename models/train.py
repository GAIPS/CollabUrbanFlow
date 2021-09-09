""" 
    Trains an A_CAT agent.

    References:
    -----------
    * Generators
        http://www.dabeaz.com/finalgenerator/FinalGenerator.pdf
"""
import ipdb
import os, sys
# import json, os, sys
from collections import defaultdict
from pathlib import Path
#from shutil import copyfile
# FIXME: DEBUG
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
# print(sys.path)


from tqdm.auto import trange
# import configparser
import numpy as np

from environment import Environment
from agents.actor_critic import ACAT
from approximators.tile_coding import TileCodingApproximator
from utils.file_io import engine_create, engine_load_config, \
                            expr_path_create, expr_config_dump, expr_train_dump, \
                            parse_train_config

# prevent randomization
TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'

def main(train_config_path=TRAIN_CONFIG_PATH, seed=0):
    # Setup config parser path.
    print(f'Loading train parameters from: {train_config_path}')

    # Load train config file with parameters.
    # train_config = configparser.ConfigParser()
    # train_config.read(train_config_path)
    # train_args = train_config['train_args']
    train_args = parse_train_config(train_config_path)
    network = train_args['network']
    experiment_time = train_args['experiment_time']
    episode_time = train_args['experiment_save_agent_interval']

    # Epsilon 
    epsilon_init = train_args['epsilon_init']
    epsilon_final = train_args['epsilon_final']
    epsilon_timesteps = train_args['epsilon_schedule_timesteps']

    # Parse train parameters.
    # config_file_path = Path(f'data/networks/{network}/config.json')
    # roadnet_file_path = Path(f'data/networks/{network}/roadnet.json')
    # flow_file_path = Path(f'data/networks/{network}/flow.json')
    # eng = Engine(config_file_path.as_posix(), thread_num=4)
    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 

    np.random.seed(seed)
    # eng.set_random_seed(seed)
    # with config_file_path.open() as f: config = json.load(f)
    # with flow_file_path.open() as f: flows = json.load(f)
    # with roadnet_file_path.open() as f: roadnet = json.load(f)

    expr_path = expr_path_create(network)
    # timestamp = f'{datetime.now():%Y%m%d%H%M%S}'
    # experiment_path =  f'data/emissions/{network}_{timestamp}'
    # # TODO: replace by pathlib
    # os.makedirs(experiment_path, exist_ok=True)
    # print(f'Experiment: {str(experiment_path)}\n')

    # TODO: save logs
    # config['dir'] = f'{experiment_path}/'
    # 
    # save_dir_path = Path(experiment_path) / 'config'
    # if not save_dir_path.exists():
    #     save_dir_path.mkdir()
    # copyfile(train_config_path, save_dir_path / 'train.config')
    # copyfile(flow_file_path, save_dir_path / 'flow.json')
    # copyfile(roadnet_file_path, save_dir_path / 'roadnet.json')
    # with (save_dir_path / 'config.json').open('w') as f: json.dump(config, f)

    expr_config_dump(network, expr_path, config, flows, roadnet)
    env = Environment(roadnet, eng)
    approx = TileCodingApproximator(roadnet, flows)
    acat = ACAT(env.phases, epsilon_init, epsilon_final, epsilon_timesteps)

    info_dict = defaultdict(lambda : [])
    s_prev = None
    a_prev = None

    num_episodes = int(experiment_time / episode_time)
    for eps in trange(num_episodes, position=0):
        gen = env.loop(episode_time)

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
                        acat.update(s_prev, a_prev, r_next, state)
                        
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

        except StopIteration as e:
            result = e.value
            # End of episode: save iterations
            # # TODO: Put this logic inside actor_critic.checkpoint
            # chkpt_dir = f"{expr_path}/checkpoints/"
            # os.makedirs(chkpt_dir, exist_ok=True)

            chkpt_dir = Path(f"{expr_path}/checkpoints/")
            chkpt_num = str(eps * episode_time)
            os.makedirs(chkpt_dir, exist_ok=True)
            acat.save_checkpoint(chkpt_dir, chkpt_num)

            s_prev = None
            a_prev = None
            acat.reset()


    # Store train info dict.
    # TODO: Turn all of this into Path standard
    # logs_dir_path = Path(expr_path) / 'logs'
    # print(logs_dir_path)
    # os.makedirs(logs_dir_path.as_posix(), exist_ok=True)
    # train_log_path = logs_dir_path / "train_log.json"
    # with train_log_path.open('w') as f:
    #     json.dump(info_dict, f)
    expr_train_dump(expr_path, info_dict)

    return str(expr_path)

if __name__ == '__main__':
    main(train_config_path='config/train.config')
