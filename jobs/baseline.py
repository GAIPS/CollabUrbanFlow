"""Runs baseline pipeline

    * Runs a selected baseline.
        Uses command-line arguments for baseline.
        Uses train.config for network and flow information.
        Uses run.config for seed and processors information.
        Uses rollouts for experiment time information.
    * Runs experimental plots.
    * Compresses file.

"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import os
import sys
import time
import json
import tempfile
import shutil
import argparse
import configparser

import multiprocessing
import multiprocessing.pool


import sys
sys.path.append(Path.cwd().as_posix())
from models.baselines  import main as baseline
from jobs.rollouts import concat
from plots.test_plots import main as test_plots 

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


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This scripts runs recursively a rollout for every checkpoint stored
            on the experiment path. If test is set to True only the last checkpoints
            will be used.
        """
    )
    parser.add_argument('baseline', type=str,
                        nargs='?', default='max_pressure',
                        help='a baseline experiment')

    flags = parser.parse_args()

    return flags

def delay_baseline(args):
    """Delays execution.

        Parameters:
        -----------
        * args: tuple
            Position 0: execution delay of the process.
            Position 1: store the baseline config file.

        Returns:
        -------
        * fnc : function
            An anonymous function to be executed with a given delay
    """
    delay, cfg_path = args
    time.sleep(delay)
    return baseline(cfg_path)

# defines the baseline pipeline
def main():
    # 1) Execute the baseline rollouts
    baseline_root_path = baseline_batch()

    # 2) Create plots with metrics plots for the baseline.
    test_plots(baseline_root_path, config_filename='baseline.config')

    # 3) Cleaning emissions.
    print('\nCleaning and compressing files...\n')
    baseline_root_path = Path(baseline_root_path)
    for csv_path in baseline_root_path.rglob('emission_log.json'):
        Path(csv_path).unlink()

    shutil.make_archive(baseline_root_path,
                    'gztar',
                    os.path.dirname(baseline_root_path),
                    baseline_root_path.name)
    shutil.rmtree(baseline_root_path)

    print('Experiment folder: {0}'.format(baseline_root_path))
    
def baseline_batch(baseline=None):

    print('\nRUNNING jobs/baseline.py\n')

    args = get_arguments()
    if baseline is None: baseline = args.baseline

    # Read script arguments from run.config file.
    run_config = configparser.ConfigParser()
    run_config.read('config/run.config')
    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args","train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        ' must match the number of runs (`num_runs`) argument.')

    # Read script arguments from run.config file.
    test_config = configparser.ConfigParser()
    test_config.read('config/test.config')
    rollout_time = int(test_config.get('test_args', 'rollout-time'))
    num_rollouts = int(test_config.get('test_args', 'num-rollouts'))

    # Computes rollout seeds from train.config and test.config.
    baseline_seeds = []
    for num_seed, train_seed in enumerate(train_seeds):
        base_seed = max(train_seeds) + train_seed * num_seed
        for num_rollout in range(num_rollouts):
            seed = base_seed + num_rollout + 1
            baseline_seeds.append(seed)
    


    print('\nArguments (jobs/baseline.py):')
    print('------------------------')
    print('Number of runs: {0}'.format(num_runs))
    print('Number of processors: {0}'.format(num_processors))
    print('Baseline seeds: {0}\n'.format(baseline_seeds))

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'WARNING: Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.
    train_config = configparser.ConfigParser()
    train_config.read('config/train.config')
    network = train_config.get('train_args', 'network')
    demand_type = train_config.get('train_args', 'demand_type')
    demand_mode = train_config.get('train_args', 'demand_mode')

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
    print(f'Experiment timestamp: {timestamp}\n')

    with tempfile.TemporaryDirectory() as f:

        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        tmpdir_path = Path(f)
        baseline_cfg_paths = []
        for seed in baseline_seeds:

            baseline_config = configparser.ConfigParser()
            baseline_config['baseline_args'] = {
                'network': network,
                'ts_type': baseline,
                'demand_type': demand_type,
                'demand_mode': demand_mode,
                'seed': seed,
                'rollout-time': rollout_time,
            }
            # Determine target path.
            cfg_path = tmpdir_path / f'{baseline}-{seed}.config'
            # Setup train seed.
            with cfg_path.open('w') as ft:
                baseline_config.write(ft)
            baseline_cfg_paths.append(str(cfg_path))

        # Run.
        # rvs: directories' names holding experiment data
        # beware of filename clashes for num_processors > 1
        if num_processors > 1:
            pool = NonDaemonicPool(num_processors, maxtasksperchild=1)
            gen = enumerate(baseline_cfg_paths)
            baseline_args = [
                (_i + 1, _p) for _i, _p in gen
            ]
            rvs = pool.map(delay_baseline, baseline_args)
            pool.close()
            pool.join()
        else:
            rvs = []
            for cfg in baseline_cfg_paths:
                rvs.append(delay_baseline((0.0, cfg)))

        paths, data = zip(*rvs)

        # Create a directory and move newly created files.
        # Make a batch path
        batch_path = Path('data/emissions') / timestamp
        if not batch_path.exists():
            batch_path.mkdir()

        # Move files
        for src in paths:
            dst = batch_path / src.parts[-1]
            src.replace(dst)

        # Create rollouts
        data = concat(data)
        filename = f'rollouts_test.json'
        target_path = batch_path / filename
        with target_path.open('w') as f:
            json.dump(data, f)


    sys.stdout.write(str(batch_path))

    return str(batch_path)

if __name__ == '__main__':
    # baseline_batch(baseline=None) # Use this line for textual output.
    main()
