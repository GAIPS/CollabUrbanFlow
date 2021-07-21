"""

    Python script to run full pipeline:

        1) jobs/train.py: train agent(s).

        2) analysis/train_plots.py: create training plots.

        5) Clean up and compress the experiment folder in
            order to optimize disk space usage.

"""
import os
import shutil
from pathlib import Path

from jobs_train import train_batch as train
from jobs_rollouts import rollout_batch as rollouts
from train_plots import main as train_plots
from test_plots import main as test_plots

if __name__ == '__main__':
    # 1) Train agent(s).
    experiment_root_path = train()
    import ipdb; ipdb.set_trace()
    # 2) Create train plots.
    # experiment_root_path = Path('data/20210709210526.429916/')
    train_plots(experiment_root_path)

    # 3) Execute rollouts with last saved checkpoints (test).
    # rollouts(test=True, experiment_dir=experiment_root_path)

    # 4) Create plots with metrics plots for final agent.
    # test_plots(experiment_root_path)


    # 5) Cleaning emissions.
    # print('\nCleaning and compressing files...\n')
    # experiment_root_path = Path(experiment_root_path)
    # for csv_path in experiment_root_path.rglob('*-emission.csv'):
    #     Path(csv_path).unlink()

    # shutil.make_archive(experiment_root_path,
    #                 'gztar',
    #                 os.path.dirname(experiment_root_path),
    #                 experiment_root_path.name)
    # shutil.rmtree(experiment_root_path)

    print('Experiment folder: {0}'.format(experiment_root_path))
