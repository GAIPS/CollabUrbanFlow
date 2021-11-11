"""Saves a pre-trained set of weights.

    This scripts performs a shorter train simulation, evaluates the runs, averages the parameters of the best pretraining runs and saves the resulting weights on a dedicated folder:

    data/pretrain/<network>/<model>/<layers>/<model_specific_subfolder>/<n_hidden.chkpt>

    Where <model_specific_subfolder> are particular to the model, e.g, n_heads for graph attention networks.

    The pretraining runs changes the regular training run on the following characteristics:
    * experiment_time = 5 * experiment_save_agent_interval
    * epsilon_schedule_timesteps = 4.5 * experiment_save_agent_interval
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from operator import itemgetter
import shutil
import time
import json
import tempfile
import configparser
import multiprocessing
import multiprocessing.pool
# import sys
# sys.path.append(Path.cwd().as_posix())

import torch

from agents import get_agent
from environment import get_environment
from models.train  import main as pretrain
from utils.decorators import processable, benchmarked

mp = multiprocessing.get_context('spawn')
PRETRAIN_CHOICE = ('GATW', 'GATV')

class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass

class NoDaemonContext(type(multiprocessing.get_context('spawn'))):
    Process = NoDaemonProcess

class NonDaemonicPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NonDaemonicPool, self).__init__(*args, **kwargs)


def delay_train(args):
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
    return pretrain(args[1])


def get_top_ks(paths, k=3):
    """Delays execution.

        Parameters:
        -----------
        * paths: list<pathlib.Path>
        Paths to the pretrain runs.

        Returns:
        --------
        * top_ks: list<tuple<double, pathlib.Path>>
        A list of tuples containing the reward and the path to the checkpoint dir.
    """
    ret = []
    for path in paths:
        train_log_path = path / 'logs' / 'train_log.json'
        with train_log_path.open('r') as f: log = json.load(f)

        interval = len(log['rewards']) // 5
        rewards = [log['rewards'][i * interval: (i + 1) * interval] for i in range(0, 5)] 

        rewards = [sum([sum(d.values()) for d in data]) for data in rewards]

        if rewards[-1] > rewards[0]:
            chkpt_paths = [chkpt for chkpt in (path / 'checkpoints').rglob('*.chkpt')]
            chkpt_paths = sorted(chkpt_paths, key=lambda x: int(x.parent.stem))
            # return argmax of list k suppose 4
            imax = max(enumerate(rewards),key=itemgetter(1))[0]
            max_reward = rewards[imax]
            chkpt_max = chkpt_paths[imax]
            ret.append((max_reward, chkpt_max)) 
    return ret

def get_pretrain_path(network, agent_type):
    env = get_environment(network)
    model = get_agent(agent_type, env, 0.0, 0.0, 3600)
    path = model.pretrain_path

    # Creates if does not exist
    sub_folders = path.parents 

    # the last folder is `.`
    for f in range(len(sub_folders) - 2, -1, -1):
        sf = sub_folders[f]
        sf.mkdir(exist_ok=True)

    assert sf.exists()
    return path
    
def model_pooling(paths):
    state_dict = torch.load(paths[0])
    n = 1
    for path in paths[1:]:
        other_state_dict = torch.load(path)
        assert state_dict.keys() == other_state_dict.keys()
        for key, value in state_dict.items():
            state_dict[key] = n * value + other_state_dict[key] / (n + 1)
            n += 1
    return state_dict

def pretrain_batch():

    print('\nRUNNING jobs/pretrain.py\n')

    # Read script arguments from run.config file.
    run_config = configparser.ConfigParser()
    run_config.read('config/run.config')

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args","train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        ' must match the number of runs (`num_runs`) argument.')
    train_config = configparser.ConfigParser()
    train_config.read('config/train.config')

    # Pretrain during a diminished period w.r.t train.
    agent_type = train_config['agent_type']['agent_type']
    network = train_config['train_args']['network']
    save_agent_interval = train_config['train_args']['experiment_save_agent_interval']
    experiment_time = str(5 * int(save_agent_interval))

    epsilon_schedule_timesteps = str(4.5 * int(save_agent_interval))

    print('\nArguments (jobs/pretrain.py):')
    print('-----------------------------')
    if agent_type not in PRETRAIN_CHOICE: 
        print(f'Agent:{agent_type} doesnt require pretraining')
        print(f'Skipping procedure ...')
        return
    print('Number of runs: {0}'.format(num_runs))
    print('Number of processors: {0}'.format(num_processors))
    print('Train seeds: {0}'.format(train_seeds))
    print(f'Agent: {agent_type}\n')
    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'WARNING: Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
    print(f'Experiment timestamp: {timestamp}\n')

    with tempfile.TemporaryDirectory() as tempdir:

        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        tempdir_path = Path(tempdir)
        pretrain_configs = []
        for seed in train_seeds:

            # Determine target path.
            pretrain_tempfile_path = tempdir_path / f'pretrain-{seed}.config'
            pretrain_configs.append(pretrain_tempfile_path.as_posix())

            # Setup train seed.
            pretrain_tempfile_config = deepcopy(train_config)
            pretrain_tempfile_config.set("train_args", "experiment_seed", str(1000 * (seed + 1) -1))
            pretrain_tempfile_config.set("train_args", "experiment_time", str(experiment_time))
            pretrain_tempfile_config.set("train_args", "experiment_save_agent_interval", str(save_agent_interval))
            pretrain_tempfile_config.set("train_args", "epsilon_schedule_timesteps", str(epsilon_schedule_timesteps))
            

            # Write temporary train config file.
            with pretrain_tempfile_path.open('w') as f:
                pretrain_tempfile_config.write(f)
            # tmp_cfg_file = open(train_tmp_path, "w")
            # train_config.write(tmp_cfg_file)
            # tmp_cfg_file.close()

        # Run.
        # rvs: directories' names holding experiment data
        if num_processors > 1:
            train_args = zip(range(num_runs), pretrain_configs)
            pool = NonDaemonicPool(num_processors, maxtasksperchild=1)
            rvs = pool.map(delay_train, train_args)
            pool.close()
            pool.join()
        else:
            rvs = []
            for cfg in pretrain_configs:
                rvs.append(delay_train((0.0, cfg)))

    # Create a directory and move newly created files.
    paths = [Path(f) for f in rvs]
    # sort by max_reward desc
    # keep K at most
    # load checkpoints and average then
    k = 3
    top_ks = get_top_ks(paths, k=k)
    # TODO: if top_ks is empty log a warning message
    if not any(top_ks):
        print('------------------------------------')
        print('WARNING! No pretrain run successful.')
        print('------------------------------------')
        return 1
    top_ks = sorted(top_ks, key=itemgetter(0))
    if k < len(top_ks): top_ks = top_ks[:k]

    # save pretrain runs
    pretrain_path = get_pretrain_path(network, agent_type)

    _, top_k_paths = zip(*top_ks)
    pooled_state_dict = model_pooling(top_k_paths)
    torch.save(pooled_state_dict, pretrain_path)
    for path in paths: shutil.rmtree(path)

    return 0

@processable
def pretrain_job():
    # Suppress textual output.
    return pretrain_batch()

if __name__ == '__main__':
    pretrain_batch() # Use this line for textual output.
    # pretrain_job()
