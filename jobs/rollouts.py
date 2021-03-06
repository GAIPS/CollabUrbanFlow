"""
    jobs/rollouts.py
"""

from pathlib import Path
import itertools
import time
import sys
import json
import tempfile
import argparse
import configparser
import multiprocessing
import multiprocessing.pool
from collections import defaultdict

from utils.decorators import processable
from models.rollouts import main as roll
from utils import str2bool
from utils.utils import concat

mp = multiprocessing.get_context('spawn')


TRAIN_CONFIG_PATH = 'config/train.config'


def get_train_path(batch_path):
    train_cfg_paths = [p for p in batch_path.rglob('train.config')]
    if len(train_cfg_paths) == 0: return TRAIN_CONFIG_PATH
    return train_cfg_paths[0]

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
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='''A directory which it\'s subdirectories are train runs.''')

    parser.add_argument('--test', '-t', dest='test', type=str2bool,
                        default=False, nargs='?',
                        help='If true only the last checkpoints will be used.')

    parsed = parser.parse_args()
    sys.argv = [sys.argv[0]]

    return parsed


def delay_roll(args):
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
    return roll(args[1])



def rollout_batch(test=False, experiment_dir=None):

    print('\nRUNNING jobs/rollouts.py\n')

    if not experiment_dir:

        # Read script arguments.
        args = get_arguments()
        # Clear command line arguments after parsing.

        batch_path = Path(args.experiment_dir)
        test = args.test

    else:
        batch_path = Path(experiment_dir)

    chkpt_pattern = 'checkpoints'

    # Get names of train runs.
    experiment_names = list({p.parents[0] for p in batch_path.rglob(chkpt_pattern)})

    # Get checkpoints numbers.
    chkpts_dirs = [p for p in batch_path.rglob(chkpt_pattern)]
    chkpts_nums = [int(n.name) for n in chkpts_dirs[0].iterdir()]
    if len(chkpts_nums) == 0:
        raise ValueError('No checkpoints found.')

    chkpts_nums = sorted(chkpts_nums)
    # If test then pick only the last checkpoints.
    if test:
        chkpts_nums = [chkpts_nums[-1]]
        rollouts_paths = list(itertools.product(experiment_names, chkpts_nums))
    
        print('jobs/rollouts.py (test mode): using checkpoints'
                ' number {0}'.format(chkpts_nums[0]))
    else:
        rollouts_paths = list(itertools.product(experiment_names, chkpts_nums))

    # print(rollouts_paths)

    run_config = configparser.ConfigParser()
    run_config.read('config/run.config')

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        'must match the number of runs (`num_runs`) argument.')

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')


    # Get agent_type and network
    train_cfg_path = get_train_path(batch_path)
    train_config = configparser.ConfigParser()
    train_config.read(train_cfg_path)
    agent_type = train_config.get('agent_type', 'agent_type')
    network = train_config.get('train_args', 'network')

    # Override rollouts config files with test.config file parameters.
    test_config = configparser.ConfigParser()
    test_config.read('config/test.config')

    num_rollouts = int(test_config.get('test_args', 'num-rollouts'))
    rollout_time = test_config.get('test_args', 'rollout-time')

    print(f'\nArguments (jobs/test.py):')
    print('-------------------------')
    print(f'Experiment dir: {batch_path}')
    print(f'Number of processors: {num_processors}')
    print(f'Num. rollout files: {len(rollouts_paths)}')
    print(f'Num. rollout repetitions: {num_rollouts}')
    print(f'Num. rollout total: {len(rollouts_paths) * num_rollouts}')
    print(f'Rollout time: {rollout_time}\n')

    # Allocate seeds.
    custom_configs = []
    for rn, rp in enumerate(rollouts_paths):
        base_seed = max(train_seeds) + num_rollouts * rn
        for rr in range(num_rollouts):
            seed = base_seed + rr + 1
            custom_configs.append((rp, seed))

    with tempfile.TemporaryDirectory() as f:

        tmp_path = Path(f)
        # Create a config file for each rollout
        # with the respective seed. These config
        # files are stored in a temporary directory.
        rollouts_cfg_paths = []
        cfg_key = "test_args"
        for cfg in custom_configs:
            run_path, chkpt_num = cfg[0]
            seed = cfg[1]

            # Setup custom rollout settings.
            test_config.set(cfg_key, "run-path", str(run_path))
            test_config.set(cfg_key, "chkpt-number", str(chkpt_num))
            test_config.set(cfg_key, "seed", str(seed))
            test_config.set(cfg_key, "network", network)
            test_config.set(cfg_key, "agent_type", agent_type)

            # Write temporary train config file.
            cfg_path = tmp_path / f'rollouts-{run_path.name}-{chkpt_num}-{seed}.config'
            rollouts_cfg_paths.append(str(cfg_path))
            with cfg_path.open('w') as fw:
                test_config.write(fw)

        # rvs: directories' names holding experiment data
        if num_processors > 1:
            pool = NonDaemonicPool(num_processors)
            rvs = pool.map(delay_roll, [(delay, [cfg])
                            for (delay, cfg) in zip(range(len(rollouts_cfg_paths)), rollouts_cfg_paths)])
            pool.close()
        else:
            rvs = []
            for cfg in rollouts_cfg_paths:
                rvs.append(delay_roll((0.0, [cfg])))

    res = concat(rvs)
    filepart = 'test' if test else 'eval'
    filename = f'rollouts_{filepart}.json'
    target_path = batch_path / filename
    with target_path.open('w') as fj:
        json.dump(res, fj)

    sys.stdout.write(str(batch_path))

    return str(batch_path)

@processable
def rollout_job(test=False):
    # Suppress textual output.
    return rollout_batch(test=test)

if __name__ == '__main__':
    #rollout_job()
    rollout_batch() # Use this line for textual output.
