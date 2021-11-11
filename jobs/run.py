"""
    Python script to run full pipeline:

        1) jobs/pretrain.py: Initialize weights.

        2) jobs/train.py: train agent(s).

        3) analysis/train_plots.py: create training plots.

        4) Execute rollouts with last saved checkpoints (test).

        5) Create plots with metrics plots for final agent.

        5) Clean up and compress the experiment folder in
            order to optimize disk space usage.

"""
import os
import shutil
from pathlib import Path
import argparse
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
print(sys.path)

from utils import str2bool
from jobs.train import train_batch as train
from jobs.pretrain import pretrain_batch as pretrain
from jobs.rollouts import rollout_batch as rollouts
from analysis.train_plots import main as train_plots
from analysis.test_plots import main as test_plots

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This scripts runs recursively a rollout for every checkpoint stored
            on the experiment path. If test is set to True only the last checkpoints
            will be used.
        """
    )
    parser.add_argument('pretrain', type=str2bool, default=False, nargs='?',
                        help='Estimates initial weights.')
    flags = parser.parse_args()

    return flags

if __name__ == '__main__':
    # 1) Estimates initial weights
    flags = get_arguments()

    if flags.pretrain: pretrain()

    # 2) Train agent(s).
    experiment_root_path = train()

    # 3) Create train plots.
    train_plots(experiment_root_path)

    # 4) Execute rollouts with last saved checkpoints (test).
    rollouts(test=True, experiment_dir=experiment_root_path)

    # 5) Create plots with metrics plots for final agent.
    test_plots(experiment_root_path)

    # 6) Cleaning emissions.
    print('\nCleaning and compressing files...\n')
    experiment_root_path = Path(experiment_root_path)
    for csv_path in experiment_root_path.rglob('emission_log.json'):
        Path(csv_path).unlink()

    shutil.make_archive(experiment_root_path,
                    'gztar',
                    os.path.dirname(experiment_root_path),
                    experiment_root_path.name)
    shutil.rmtree(experiment_root_path)

    print('Experiment folder: {0}'.format(experiment_root_path))
