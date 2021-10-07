""" 
    Functions to parse configs, save and load files.
"""
from datetime import datetime
import json
from pathlib import Path
from shutil import copyfile

import configparser

from cityflow import Engine

def engine_load_config(network_or_path):
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

    if isinstance(network_or_path, str):
        file_path = Path(f'data/networks/{network_or_path}')

    elif isinstance(network_or_path, Path):
        file_path = network_or_path

    # Parse train parameters.
    config_file_path = file_path / 'config.json'
    roadnet_file_path = file_path / 'roadnet.json'
    flow_file_path = file_path / 'flow.json'


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
        config_file_path = network_or_path
    eng = Engine(config_file_path.as_posix(), thread_num=4)
    eng.set_random_seed(seed)
    return eng

def expr_path_create(network, seed=""):
    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'
    base_path = f'data/emissions/{network}_{timestamp}'
    if seed: base_path += f'-{seed}'
    expr_path =  Path(base_path)
    Path.mkdir(expr_path, exist_ok=True)
    print(f'Experiment: {str(expr_path)}\n')
    return expr_path

def expr_path_test_target(orig_path):
    args = parse_train_config(Path(orig_path) / 'config' / 'train.config', args_list=['network'])
    network = args['network']

    target_path = Path(orig_path) / 'eval'
    target_path.mkdir(exist_ok=True)
    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'

    target_path =  target_path / f'{network}_{timestamp}'
    target_path.mkdir(exist_ok=True)
    return target_path

def expr_config_dump(network, expr_path,
                     config, flow, roadnet, dump_train_config=True):
    config['dir'] = f'{expr_path.as_posix()}/'
    
    save_dir_path = Path(expr_path) / 'config'
    save_dir_path.mkdir(exist_ok=True)
    # if not save_dir_path.exists():
    #     save_dir_path.mkdir()
    if dump_train_config:
        copyfile('config/train.config', save_dir_path / 'train.config')

    with (save_dir_path / 'config.json').open('w') as f: json.dump(config, f)
    with (save_dir_path / 'flow.json').open('w') as f: json.dump(flow, f)
    with (save_dir_path / 'roadnet.json').open('w') as f: json.dump(roadnet, f)

def expr_logs_dump(expr_path, filename, data):
    logs_dir_path = Path(expr_path) / 'logs'
    logs_dir_path.mkdir(exist_ok=True)
    logs_path = logs_dir_path / filename
    with logs_path.open('w') as f:
        json.dump(data, f)
        print(logs_path)
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
    ret['agent_type'] = train_config["agent_type"]["agent_type"]
    # Epsilon 
    ret['epsilon_init'] = float(train_args['epsilon_init'])
    ret['epsilon_final'] = float(train_args['epsilon_final'])
    ret['epsilon_schedule_timesteps'] = float(train_args['epsilon_schedule_timesteps'])

    return ret

def parse_test_config(test_config_path):

    if isinstance(test_config_path, str):
        test_config_path = Path(test_config_path)
    ret = {}

    # Load test config file with parameters.
    test_config = configparser.ConfigParser()
    test_config.read(test_config_path)
    test_args = test_config['test_args']

    ret['orig_path'] = test_args['run-path']
    ret['rollout_time'] = int(test_args['rollout-time'])
    ret['chkpt_num'] = int(test_args['chkpt-number'])
    ret['seed'] = int(test_args['seed'])
    ret['chkpt_dir_path'] = Path(ret['orig_path']) / 'checkpoints' 

    return ret


