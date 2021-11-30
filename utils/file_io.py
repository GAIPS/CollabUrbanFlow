""" 
    Functions to parse configs, save and load files.
"""
from types import SimpleNamespace
from datetime import datetime
import json
from pathlib import Path
from shutil import copyfile

import configparser

from cityflow import Engine

from utils import str2bool

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

def expr_path_test_target(orig_path, network=None):
    if network is None:
        args = parse_train_parameters(Path(orig_path) / 'config' / 'train.config', args_list=['network'])
        network = args.network

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

def parse_train_parameters(train_config_path,
        args_list=['network', 'experiment_time', 'experiment_save_agent_interval',
            'experiment_seed', 'epsilon_init', 'epsilon_final',
            'epsilon_schedule_timesteps'] ):
    if isinstance(train_config_path, str):
        train_config_path = Path(train_config_path)

    ret = {}

    # Load train config file with parameters.
    train_config = configparser.ConfigParser()
    train_config.read(train_config_path)
    train_args = train_config['train_args']

    ret['network'] = train_args['network']
    ret['experiment_time']= int(train_args['experiment_time'])
    ret['save_agent_interval']= int(train_args['experiment_save_agent_interval'])

    if 'experiment_seed' in train_args.keys():
        ret['experiment_seed'] = int(train_args['experiment_seed']) 
    else:
        ret['experiment_seed'] = 0
    ret['agent_type'] = train_config["agent_type"]["agent_type"]
    # Epsilon 
    ret['epsilon_init'] = float(train_args['epsilon_init'])
    ret['epsilon_final'] = float(train_args['epsilon_final'])
    ret['epsilon_timesteps'] = float(train_args['epsilon_schedule_timesteps'])

    return SimpleNamespace(**ret)

def parse_mdp_parameters(
        config_path_or_file,
        args_list=['feature', 'action_schema', 'phases_filter', 'use_lanes']):

    if isinstance(config_path_or_file, str):  # is path
        config_path = Path(config_path_or_file)

        # Load train config file with parameters.
        config = configparser.ConfigParser()
        config.read(config_path)
        config_mdp_args = config['mdp_args']
    else: # is file without section
        config_mdp_args = config_path_or_file


    mdp_args = {}
    mdp_args['feature'] = config_mdp_args['feature']
    mdp_args['action_schema'] = config_mdp_args['action_schema']
    mdp_args['phases_filter'] = eval(config_mdp_args['phases_filter'])
    try:
        mdp_args['use_lanes'] = str2bool(config_mdp_args['use_lanes'])
    except Exception:
        import ipdb; ipdb.set_trace()


    return SimpleNamespace(**mdp_args)

def parse_env_parameters(
        config_path_or_file,
        args_list=['yellow', 'min_green', 'max_green']):

    if isinstance(config_path_or_file, str): # is path
        config_path = Path(config_path_or_file)

        # Load train config file with parameters.
        config = configparser.ConfigParser()
        config.read(config_path)
        config_env_args = config['env_args']
    else: # is file without `env_args` section
        config_env_args = config_path_or_file

    env_args = {}
    env_args['yellow'] = int(config_env_args['yellow'])
    env_args['min_green'] = int(config_env_args['min_green'])
    env_args['max_green'] = int(config_env_args['max_green'])

    return SimpleNamespace(**env_args)
    

def parse_test_config(test_config_path):

    if isinstance(test_config_path, str):
        config_path = Path(test_config_path)

    test_args = {}
    env_args = {}
    # Load test config file with parameters.
    config = configparser.ConfigParser()
    config.read(test_config_path)
    config_test_args = config['test_args']

    test_args['orig_path'] = config_test_args['run-path']
    test_args['rollout_time'] = int(config_test_args['rollout-time'])
    test_args['chkpt_num'] = int(config_test_args['chkpt-number'])
    test_args['seed'] = int(config_test_args['seed'])
    test_args['chkpt_dir_path'] = Path(test_args['orig_path']) / 'checkpoints' 
    test_args['agent_type'] = config_test_args['agent_type']
    test_args['network'] = config_test_args['network']


    # env_args['yellow'] = int(config_test_args['yellow'])
    # env_args['min_green'] = int(config_test_args['min_green'])
    # env_args['max_green'] = int(config_test_args['max_green'])
    env_args = parse_env_parameters(config_test_args)
    mdp_args = parse_mdp_parameters(config_test_args)


    return SimpleNamespace(**test_args), env_args, mdp_args 

