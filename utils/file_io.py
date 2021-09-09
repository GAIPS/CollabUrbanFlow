""" 
    Functions to parse configs, save and load files.
"""
from datetime import datetime
import json
from pathlib import Path
from shutil import copyfile

import configparser

from cityflow import Engine

def engine_load_config(network):
    """Loads config, roadnet and flows 

    * Loads json files from data/networks

    Params:
    ------
    * network: str
      String representing a network.

    Returns:
    --------
    * config: dict
      Dictionary representation of config file.
    * roadnet: dict
      Dictionary representation of config file.
    """ 

    # Parse train parameters.
    config_file_path = Path(f'data/networks/{network}/config.json')
    roadnet_file_path = Path(f'data/networks/{network}/roadnet.json')
    flow_file_path = Path(f'data/networks/{network}/flow.json')


    with config_file_path.open() as f: config = json.load(f)
    with flow_file_path.open() as f: flows = json.load(f)
    with roadnet_file_path.open() as f: roadnet = json.load(f)

    return config, flows, roadnet

def engine_create(network_or_path, seed=0, thread_num=4):
    """Loads config, roadnet and flows 

    * Loads json files from data/networks

    Params:
    ------
    * network: str
      String representing a network.

    """ 
    if isinstance(network_or_path, str):
        config_file_path = Path(f'data/networks/{network_or_path}/config.json')
    elif isinstance(network_or_path, Path):
        raise NotImplementedError
    eng = Engine(config_file_path.as_posix(), thread_num=4)
    eng.set_random_seed(seed)
    return eng

def expr_path_create(network):
    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'
    expr_path =  Path(f'data/emissions/{network}_{timestamp}')
    Path.mkdir(expr_path, exist_ok=True)
    print(f'Experiment: {str(expr_path)}\n')
    return expr_path

def expr_config_dump(network, expr_path, config, flow, roadnet):
    config['dir'] = f'{expr_path.as_posix()}/'
    
    save_dir_path = Path(expr_path) / 'config'
    save_dir_path.mkdir(exist_ok=True)
    # if not save_dir_path.exists():
    #     save_dir_path.mkdir()
    copyfile('config/train.config', save_dir_path / 'train.config')

    with (save_dir_path / 'config.json').open('w') as f: json.dump(config, f)
    with (save_dir_path / 'flow.json').open('w') as f: json.dump(flow, f)
    with (save_dir_path / 'roadnet.json').open('w') as f: json.dump(roadnet, f)

def expr_train_dump(expr_path, info_dict):
    logs_dir_path = Path(expr_path) / 'logs'
    print(logs_dir_path)
    logs_dir_path.mkdir(exist_ok=True)
    train_log_path = logs_dir_path / "train_log.json"
    with train_log_path.open('w') as f:
        json.dump(info_dict, f)
    return logs_dir_path

def parse_train_config(train_config_path,
        args_list=['network', 'experiment_time', 'experiment_save_agent_interval',
            'epsilon_init', 'epsilon_final', 'epsilon_schedule_timesteps'] ):
    if isinstance(train_config_path, str):
        train_config_path = Path(train_config_path)

    ret = {}

    # Load train config file with parameters.
    train_config = configparser.ConfigParser()
    train_config.read(train_config_path)
    train_args = train_config['train_args']

    ret['network'] = train_args['network']
    ret['experiment_time']= int(train_args['experiment_time'])
    ret['experiment_save_agent_interval']= int(train_args['experiment_save_agent_interval'])

    # Epsilon 
    ret['epsilon_init'] = float(train_args['epsilon_init'])
    ret['epsilon_final'] = float(train_args['epsilon_final'])
    ret['epsilon_schedule_timesteps'] = float(train_args['epsilon_schedule_timesteps'])

    return ret
