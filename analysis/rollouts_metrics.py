'''Rollouts: consistent with average_duration computation from Wei, et al. 2019 '''
from collections import defaultdict
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

# TODO: Deprecate
# def merge_metrics(durations_per_round, inflow_per_round, outflow_per_round):
#     validation_period = 10
#     validation_duration = np.array(durations_per_round[-validation_period:])
#     inflow = np.array(inflow_per_round[-validation_period:])
#     outflow = np.array(outflow_per_round[-validation_period:])


def main(experiment_root_folder=None):


    if not experiment_root_folder:
        args = get_arguments()
        experiment_root_folder = args.experiment_root_folder

    print('\nRUNNING analysis/rollouts_stats.py\n')
    print('\tExperiment root folder: {0}\n'.format(experiment_root_folder))

    # Rollouts path.
    # Seek rollouts json

    # Get all *.json files from experiment root folder.
    log_files = sorted([p for p in Path(experiment_root_folder).rglob('rollouts_log.json')])
    print('Number of json files found: {0}'.format(len(log_files)))

    res = defaultdict(list)
    for log_file in log_files:

        print(f'Processing JSON file: {log_file.as_posix()}')


        chkpt_num = log_file.parent.parent.stem
        df = pd.read_json(log_file.as_posix()).\
             set_index('id', inplace=False)

        df['duration'] = df['finish'] - df['start']
        duration = df.groupby(by='vehicle_id')['duration'].sum()
        
        res['chkpt_num'].append(int(chkpt_num))
        res['duration'].append(duration.mean())



    # TODO: Deprecate
    # # Prepare output folder.
    # rollout_path = Path(experiment_root_folder) / 'rollouts_eval.json'
    # print('Rollout file: {0}\n'.format(rollout_path))
    # with rollout_path.open('r') as f:
    #     rollouts = json.load(f)

    # 
    # res = {}
    # res['checkpoints'] = flatten2(rollouts['duration'].keys())
    # res['duration'] = reduce_mean(rollouts['duration'].values())
    # res['inflow'] = reduce_mean(rollouts['inflow'].values())
    # res['outflow'] = reduce_mean(rollouts['outflow'].values())
    
    rollout_path = Path(experiment_root_folder) / 'rollouts_stats.csv'
    df = pd.DataFrame.  \
            from_dict(data=res). \
            sort_values(by='chkpt_num'). \
            set_index('chkpt_num', inplace=False)
    print(df)

    df.to_csv(rollout_path, sep=',')
    print(f'Rollouts saved: {rollout_path.as_posix()}')

