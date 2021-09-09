""" 
    Functions to parse configs, save and load files.
"""
from datetime import datetime
import json
from pathlib import Path
from shutil import copyfile

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

def experiment_path_create(network):
    timestamp = f'{datetime.now():%Y%m%d%H%M%S}'
    experiment_path =  Path(f'data/emissions/{network}_{timestamp}')
    Path.mkdir(experiment_path, exist_ok=True)
    print(f'Experiment: {str(experiment_path)}\n')
    return experiment_path

def experiment_config_dump(network, experiment_path, config, flow, roadnet):
    config['dir'] = f'{experiment_path.as_posix()}/'
    
    save_dir_path = Path(experiment_path) / 'config'
    save_dir_path.mkdir(exist_ok=True)
    # if not save_dir_path.exists():
    #     save_dir_path.mkdir()
    copyfile('config/train.config', save_dir_path / 'train.config')

    with (save_dir_path / 'config.json').open('w') as f: json.dump(config, f)
    with (save_dir_path / 'flow.json').open('w') as f: json.dump(flow, f)
    with (save_dir_path / 'roadnet.json').open('w') as f: json.dump(roadnet, f)
