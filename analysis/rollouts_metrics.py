'''Rollouts: consistent with average_duration computation from Wei, et al. 2019 '''
from pathlib import Path
import argparse
import json
import pandas as pd
from utils import flatten2, reduce_mean

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates evaluation plots, given an experiment folder path.
            (To be used with RL-algorithms)
        """
    )
    parser.add_argument('experiment_root_folder', type=str, nargs='?',
                        help='Experiment root folder.')

    return parser.parse_args()

def merge_metrics(durations_per_round, inflow_per_round, outflow_per_round):
    validation_period = 10
    validation_duration = np.array(durations_per_round[-validation_period:])
    inflow = np.array(inflow_per_round[-validation_period:])
    outflow = np.array(outflow_per_round[-validation_period:])


def main(experiment_root_folder=None):

    print('\nRUNNING analysis/rollouts_stats.py\n')

    if not experiment_root_folder:
        args = get_arguments()
        print_arguments(args)
        experiment_root_folder = args.experiment_root_folder

    # Prepare output folder.
    rollout_path = Path(experiment_root_folder) / 'rollouts_eval.json'
    print('Rollout file: {0}\n'.format(rollout_path))
    with rollout_path.open('r') as f:
        rollouts = json.load(f)

    
    res = {}
    res['checkpoints'] = flatten2(rollouts['duration'].keys())
    res['duration'] = reduce_mean(rollouts['duration'].values())
    res['inflow'] = reduce_mean(rollouts['inflow'].values())
    res['outflow'] = reduce_mean(rollouts['outflow'].values())
    
    rollout_path = Path(experiment_root_folder) / 'rollouts_stats.csv'
    df = pd.DataFrame.  \
            from_dict(data=res).set_index('checkpoints')

    print(df.mean())

    df.to_csv(rollout_path, sep=',')
    print(f'Rollouts saved: {rollout_path.as_posix()}')

