"""
    jobs/train.py
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import sys
import time
import json
import tempfile
import configparser
import multiprocessing
import multiprocessing.pool


import sys
sys.path.append(Path.cwd().as_posix())
from models.train  import main as train
from utils.decorators import processable, benchmarked

mp = multiprocessing.get_context('spawn')


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


@benchmarked
def benchmarked_train(*args, **kwargs):
    return train(*args, **kwargs)


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
    return benchmarked_train(args[1])


def train_batch():

    print('\nRUNNING jobs/train.py\n')

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

    print('\nArguments (jobs/train.py):')
    print('------------------------')
    print('Number of runs: {0}'.format(num_runs))
    print('Number of processors: {0}'.format(num_processors))
    print('Train seeds: {0}'.format(train_seeds))
    print('Agent: {0}\n'.format(train_config['agent_type']['agent_type']))
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

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        tmpdir_path = Path(tmp_dir)
        train_configs = []
        for seed in train_seeds:

            # Determine target path.
            train_tmp_path = tmpdir_path / f'train-{seed}.config'
            train_configs.append(train_tmp_path.as_posix())

            # Setup train seed.
            train_tmpcfg = deepcopy(train_config)
            train_tmpcfg.set("train_args", "experiment_seed", str(seed))

            # Write temporary train config file.
            with train_tmp_path.open('w') as f:
                train_tmpcfg.write(f)
            # tmp_cfg_file = open(train_tmp_path, "w")
            # train_config.write(tmp_cfg_file)
            # tmp_cfg_file.close()

        # Run.
        # rvs: directories' names holding experiment data
        if num_processors > 1:
            train_args = zip(range(num_runs), train_configs)
            pool = NonDaemonicPool(num_processors, maxtasksperchild=1)
            rvs = pool.map(delay_train, train_args)
            pool.close()
            pool.join()
        else:
            rvs = []
            for cfg in train_configs:
                rvs.append(delay_train((0.0, cfg)))

        # Create a directory and move newly created files.
        paths = [Path(f) for f in rvs]
        commons = [p.parent for p in paths]
        if len(set(commons)) > 1:
            raise ValueError(f'Directories {set(commons)} must have the same root')
        dirpath = commons[0]
        batchpath = dirpath / timestamp
        if not batchpath.exists():
            batchpath.mkdir()

        # Move files
        for src in paths:
            dst = batchpath / src.parts[-1]
            src.replace(dst)

    sys.stdout.write(str(batchpath))

    return str(batchpath)

@processable
def train_job():
    # Suppress textual output.
    return train_batch()

if __name__ == '__main__':
    # train_batch() # Use this line for textual output.
    train_job()
