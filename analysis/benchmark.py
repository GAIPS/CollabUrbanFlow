import os
import tarfile
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from functools import reduce

import sys
sys.path.append(Path.cwd().as_posix())

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.emissions import str2bool

# Those files will be used for aggregation 
BENCHMARK_FILES = ['processed_data.csv', 'speed_stats.csv', 'travel_time_stats.csv']

plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates evaluation plots that allow comparisons between different experiments.
            
        """
    )

    parser.add_argument('paths', type=str, nargs='+', help='List of paths to experiments.')
    parser.add_argument('-l','--labels', nargs="+", help='List of experiments\' labels.', required=False)
    parser.add_argument('-o', '--use_parent_output', nargs=1, default=True, type=str2bool,
                        help='Uses parent directory (common ancestor) as output folder.', required=False)

    return parser.parse_args()

def print_arguments(args, output_folder_path):

    print('Arguments (analysis/compare.py):')
    print('\tExperiments: {0}\n'.format(args.paths))
    print('\tExperiments labels: {0}\n'.format(args.labels))
    print('\tOutput folder: {0}\n'.format(output_folder_path))

def get_common_path(paths):
    '''returns the common ancestor path

    Parameters:
    ----------
    * paths: list<pathlib.Path>
    experiment paths

    Returns:
    --------
    * pathlib.Path object
    output folder
    '''
    path_parts = map(lambda x: Path(x).parts, paths)

    def fn(x, y):
        return tuple([xx for xx, yy in zip(x, y) if xx == yy])
    common_folder_parts = reduce(fn, path_parts)
    common_folder_path = Path.cwd().joinpath(*common_folder_parts)
    return common_folder_path
    
def main():

    print('\nRUNNING analysis/compare.py\n')

    args = get_arguments()

    output_folder_path = Path('analysis/plots/benchmark/')
    if args.use_parent_output:
        output_folder_path = get_common_path(args.paths) / 'benchmark'
        
    print_arguments(args, output_folder_path)

    # Prepare output folder.
    os.makedirs(output_folder_path.as_posix(), exist_ok=True)


    # Open dataframes.
    processed_dfs = {}
    travel_time_dfs = {}
    speed_dfs = {}
    # FIXME: waiting_time and vehicles are missing.
    # The inner_path of the directory the files reside
    for exp_path in args.paths:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            exp_name = Path(exp_path).name.split('.')[0] + \
                        '.' + Path(exp_path).name.split('.')[1]

            # BEWARE: This section is not yet tested.
            tar = tarfile.open(exp_path)

            tar_file_path = Path(exp_path) / exp_name / 'plots' / 'test' / 'processed_data.csv'
            tar_file = tar.extractfile(tar_file_path.as_posix())
            processed_dfs[exp_name] = pd.read_csv(tar_file, header=[0, 1], index_col=0)

            tar_file_path = Path(exp_path) / exp_name / 'plots' / 'test' / 'travel_time_stats.csv'
            tar_file = tar.extractfile(tar_file_path.as_posix())
            travel_time_dfs[exp_name] = pd.read_csv(tar_file, header=False, index_col=0)

            tar_file_path = Path(exp_path) / exp_name / 'plots' / 'test' / 'speed_stats.csv'
            tar_file = tar.extractfile(tar_file_path.as_posix())
            speed_dfs[exp_name] = pd.read_csv(tar_file, header=False, index_col=0)

        else:
            # Uncompressed file (experiment_folder).
            exp_name = Path(exp_path).name

            file_path = Path(exp_path) / 'plots' / 'test' / 'processed_data.csv'
            processed_dfs[exp_name] = pd.read_csv(file_path.as_posix(), header=[0, 1], index_col=0)

            file_path = Path(exp_path) / 'plots' / 'test' / 'travel_time_stats.csv'
            travel_time_dfs[exp_name] = pd.read_csv(file_path.as_posix(), header=None, index_col=0)

            file_path = Path(exp_path) / 'plots' / 'test' / 'speed_stats.csv'
            speed_dfs[exp_name] = pd.read_csv(file_path.as_posix(), header=None, index_col=0)

    if args.labels:
        lbls = args.labels # Custom labels.
    else:
        lbls = processed_dfs.keys()  # Default labels.

    # """
    #     waiting_time_hist_kde
    # """
    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # for l, df in zip(lbls, processed_dfs.values()):
    #     plt.plot(df['waiting_time_hist_kde', 'x'],
    #             df['waiting_time_hist_kde', 'y'],
    #             label=l, linewidth=3)

    # plt.xlabel('Waiting time (s)')
    # plt.legend()
    # plt.ylabel('Density')
    # # plt.title('Waiting time')
    # 
    # plt.savefig('analysis/plots/compare/compare_waiting_time_hist.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig('analysis/plots/compare/compare_waiting_time_hist.png', bbox_inches='tight', pad_inches=0)
    # 
    # plt.close()

    """
        travel_time_hist_kde
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, processed_dfs.values()):
        plt.plot(df['travel_time_hist_kde', 'x'],
                 df['travel_time_hist_kde', 'y'],
                 label=l, linewidth=3)

    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.legend()
    # plt.title('Travel time')
    fig_path = output_folder_path  / 'joint_travel_time_hist.pdf' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    fig_path = output_folder_path  / 'joint_travel_time_hist.png' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        speed_hist_kde
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, processed_dfs.values()):
        plt.plot(df['speed_hist_kde', 'x'],
                 df['speed_hist_kde', 'y'],
                 label=l, linewidth=3)

    plt.xlabel('Average speed (m/s)')
    plt.ylabel('Density')
    plt.legend()
    # plt.title('Vehicles\' speed')
    
    fig_path = output_folder_path  / 'joint_speed_hist.pdf' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    fig_path = output_folder_path  / 'joint_speed_hist.png' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        waiting_time_per_cycle
    """
    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # for l, df in zip(lbls, processed_dfs.values()):
    #     plt.plot(df['waiting_time_per_cycle', 'x'],
    #              df['waiting_time_per_cycle', 'y'],
    #              label=l)

    # plt.xlabel('Timestep (10 s)')
    # plt.ylabel('Average waiting time (s)')
    # plt.legend()
    # # plt.title('Waiting time')
    # 
    # plt.savefig('analysis/plots/compare/compare_waiting_time.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig('analysis/plots/compare/compare_waiting_time.png', bbox_inches='tight', pad_inches=0)
    # 
    # plt.close()

    # """
    #     travel_time_per_cycle
    # """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for lbl, df in zip(lbls, processed_dfs.values()):
        plt.plot(df['travel_time_per_cycle', 'x'],
                 df['travel_time_per_cycle', 'y'],
                 label=lbl)

    plt.xlabel('Timestep (10 s)')
    plt.ylabel('Average travel time (s)')
    plt.legend()
    # plt.title('Travel time')

    fig_path = output_folder_path  / 'joint_travel_time.pdf' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    fig_path = output_folder_path  / 'joint_travel_time.png' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        throughput_per_cycle
    """
    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # for l, df in zip(lbls, processed_dfs.values()):
    #     plt.plot(df['throughput_per_cycle', 'x'],
    #              df['throughput_per_cycle', 'y'],
    #              label=l)

    # plt.xlabel('Timestep (10 s)')
    # plt.ylabel('Number of vehicles')
    # plt.legend()
    # # plt.title('Throughput')

    # plt.savefig('analysis/plots/compare/compare_throughput.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig('analysis/plots/compare/compare_throughput.png', bbox_inches='tight', pad_inches=0)

    # plt.close()

    # """
    #     vehicles_per_cycle
    # """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for lbl, df in zip(lbls, processed_dfs.values()):
        plt.plot(df['vehicles_per_cycle', 'x'],
                 df['vehicles_per_cycle', 'y'],
                 label=lbl)
    plt.xlabel('Timestep (10 seconds)')
    plt.ylabel('Number of vehicles')
    # plt.title('Number of vehicles')
    plt.legend()

    fig_path = output_folder_path  / 'joint_vehicles.pdf' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    fig_path = output_folder_path  / 'joint_vehicles.png' 
    plt.savefig(fig_path.as_posix(), bbox_inches='tight', pad_inches=0)
    # plt.savefig('analysis/plots/compare/compare_vehicles.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig('analysis/plots/compare/compare_vehicles.png', bbox_inches='tight', pad_inches=0)

    plt.close()

    # """
    #     velocities_per_cycle
    # """
    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # for l, df in zip(lbls, processed_dfs.values()):
    #     plt.plot(df['velocities_per_cycle', 'x'],
    #              df['velocities_per_cycle', 'y'],
    #              label=l)

    # plt.xlabel('Timestep (10 s)')
    # plt.ylabel('Average velocity (m/s)')
    # # plt.title('Vehicles\' velocities')
    # plt.legend()

    # plt.savefig('analysis/plots/compare/compare_velocities.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig('analysis/plots/compare/compare_velocities.png', bbox_inches='tight', pad_inches=0)

    # plt.close()

    # TODO: Refactor those two blocks.
    ''' Travel time stats ''' 
    dataframes = []
    for lbl, df in zip(lbls, travel_time_dfs.values()):
        df = df.T
        df['model'] = lbl
        df = df.set_index('model')
        dataframes.append(df)
    travel_time_df = pd.concat(dataframes, axis=0) \
                        .sort_values('mean', ascending=True)
    leaderboard_path = (output_folder_path / 'leaderboard_travel_time.csv')
    travel_time_df.to_csv(leaderboard_path)

    ''' Speed stats.''' 
    dataframes = []
    for lbl, df in zip(lbls, speed_dfs.values()):
        df = df.T
        df['model'] = lbl
        df = df.set_index('model')
        dataframes.append(df)
    speed_df = pd.concat(dataframes, axis=0). \
                sort_values('mean', ascending=False)
    leaderboard_path = (output_folder_path / 'leaderboard_speed.csv')
    speed_df.to_csv(leaderboard_path.as_posix())

if __name__ == "__main__":
    main()
