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
import sys
  
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
print(sys.path)

from jobs.train import train_batch as train
from jobs.rollouts import rollout_batch as rollouts
from plots.train_plots import main as train_plots
from plots.test_plots import main as test_plots

if __name__ == '__main__':
    # 1) Train agent(s).
    experiment_root_path = train()
    # 2) Create train plots.
    train_plots(experiment_root_path)


    # 3) Execute rollouts with last saved checkpoints (test).
    rollouts(test=True, experiment_dir=experiment_root_path)

    # 4) Create plots with metrics plots for final agent.
    test_plots(experiment_root_path)


    # 5) Cleaning emissions.
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
