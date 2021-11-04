"""
    Python script to produce the following (global metrics) train plots:
        - reward per cycle (with mean, std and smoothed curve)
        - number of vehicles per cycle (with mean, std and smoothed curve)
        - vehicles' velocity per cycle (with mean, std and smoothed curve)

    Given the path to the experiment root folder (-p flag), the script
    searches recursively for all train_log.json files and produces the
    previous plots by averaging over all json files.

    The output plots will go into a folder named 'plots', created inside
    the given experiment root folder.
"""
from collections import defaultdict
import os
import json
import argparse
import numpy as np
from pathlib import Path
import configparser
from utils.file_io import parse_train_config

import pandas as pd

import statsmodels.api as sm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


FIGURE_X = 6.0
FIGURE_Y = 4.0

STD_CURVE_COLOR = (0.88,0.70,0.678)
MEAN_CURVE_COLOR = (0.89,0.282,0.192)
SMOOTHING_CURVE_COLOR = (0.33,0.33,0.33)


def get_arguments():

    parser = argparse.ArgumentParser(
        description="""
            Python script to produce the following (global metrics) train plots:
                - reward per cycle (with mean, std and smoothed curve)
                - number of vehicles per cycle (with mean, std and smoothed curve)
                - vehicles' velocity per cycle (with mean, std and smoothed curve)

            Given the path to the experiment root folder (-p flag), the script
            searches recursively for all train_log.json files and produces the
            previous plots by averaging over all json files.

            The output plots will go into a folder named 'plots', created inside
            the given experiment root folder.
        """
    )

    parser.add_argument('path', type=str, nargs='?',
                help='Path to the experiment root folder')

    return parser.parse_args()


def resample(data, column, freq=6, to_records=True):
   """ Resample dataframe 

    Expect ticks to be 5 to 5 seconds. Aggregate 6
    10 to 10 seconds yielding minute data.

    Params:
    ------
    * data: dict
        info from training or testing aggregated 
        by 5 seconds (decision time).

    * column: str
        a metric from info dict 
        choice = ('actions', 'velocities', 'rewards', 'vehicles')

    * freq: int
        aggregation period (6*10 --> 60)

    * to_records: Boolean
        if records is true returns a list else
        a numpy array.

   Returns:
   -------
   * ret: list or numpy array
     resampled data

   """ 
   df = pd.DataFrame.from_dict(data[column])
   index = pd.DatetimeIndex(df.index)
   df.index = index

   if column in ('rewards',):
       df = df.resample(f'{freq}n').sum()
   elif column in ('actions', 'velocities', 'vehicles'):
       df = df.resample(f'{freq}n').mean()
   else:
       raise ValueError

   if to_records:
       if column in ('vehicles', 'velocities'):
           return df.to_dict(orient='list')[0]
       return df.to_dict(orient='records')
   else:
       return  np.sum(df.values, axis=1)

def episodic_breakdown(data, total_episodes, num_series=3): 
    """Breaks data down into evenly spaced episodes"""
    episode = int(len(data['rewards']) / total_episodes)
    for k, v in data.items():
        d = []
        for eps in range(total_episodes):
            start = eps * episode
            finish = start + episode
            if eps in (0, int(total_episodes / 2), total_episodes-1):
                d += v[start:finish]
        data[k] = v
        
def main(experiment_root_folder=None):

    print('\nRUNNING analysis/train_plots.py\n')

    if not experiment_root_folder:
        args = get_arguments()
        experiment_root_folder = args.path

    print('Input files:')
    # Get all train_log.json files from experiment root folder.
    train_files = list(Path(experiment_root_folder).rglob('train_log.json'))

    # Prepare output folder.
    output_folder_path = os.path.join(experiment_root_folder, 'plots/train')
    print('\nOutput folder:\n{0}\n'.format(output_folder_path))
    os.makedirs(output_folder_path, exist_ok=True)

    # Get episode_time and total_episodes.
    train_config_path = list(Path(experiment_root_folder).rglob('train.config'))[0]
    args = parse_train_config(train_config_path)

    experiment_time = args['experiment_time']
    episode_time = args['experiment_save_agent_interval']
    total_episodes = int(experiment_time / episode_time)
    agent_type = args['agent_type']

    actions = []
    rewards = []
    rewards1 = []
    rewards2 = []
    vehicles = []
    velocities = []

    # Concatenate data for all runs.
    for run_name in train_files:

        print('Processing JSON file: {0}'.format(run_name))

        # Load JSON data.
        with open(run_name) as f:
            data = json.load(f)
        # episodic_breakdown(data, total_episodes)

        # Rewards per time-step.
        rewards1.append(resample(data, 'rewards', to_records=False))

        # aggregate data
        rewards2.append(resample(data, 'rewards'))

        # Number of vehicles per time-step.
        vehicles.append(resample(data, 'vehicles'))

        # Vehicles' velocity per time-step.
        velocities.append(resample(data, 'velocities'))

        # Agent's actions.
        actions.append(resample(data, 'actions'))

    """
        Rewards per cycle.
        (GLOBAL: sum of the reward for all intersections).
    """
    rewards = np.array(rewards1)
    episode = int(rewards.shape[1] / total_episodes)


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    X = np.linspace(1, rewards.shape[1], rewards.shape[1])
    # for eps in range(0, 3):
    #     start = eps * episode
    #     finish = start + episode
    #     xx = rewards[:, start:finish]
    Y = np.average(rewards, axis=0)
    Y_std = np.std(rewards, axis=0)
    # print(fn(eps))


    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    # plt.plot(X,Y, label=f'Mean {fn(eps)}', c=MEAN_CURVE_COLOR)
    plt.plot(X,Y, label=f'Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if rewards.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Minute')
    plt.ylabel('Reward')
    plt.legend(loc=4)

    file_name = '{0}/rewards.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/rewards.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        Rewards per intersection.
    """

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    # for eps in range(0, 3):
    #     start = eps * episode
    #     finish = start + episode
        
    dfs_rewards = [pd.DataFrame(r) for r in rewards2]

    df_concat = pd.concat(dfs_rewards)

    by_row_index = df_concat.groupby(df_concat.index)
    df_rewards = by_row_index.mean()


    window_size = min(len(df_rewards)-1, 20)

    for col in df_rewards.columns:
        plt.plot(df_rewards[col].rolling(window=window_size).mean(), label=col)

    plt.xlabel('Minutes')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.savefig('{0}/rewards_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/rewards_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    """ 
        Number of vehicles per cycle.
        (GLOBAL: For all vehicles in simulation)
    """


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    # for eps in range(0, 3):
    #     start = eps * episode
    #     finish = start + episode
        
    vehs = np.array([v for v in vehicles])


    Y = np.average(vehs, axis=0)
    Y_std = np.std(vehs, axis=0)

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    # plt.plot(X,Y, label=f'Mean {fn(eps)}', c=MEAN_CURVE_COLOR)
    plt.plot(X,Y, label=f'Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if vehs.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Minutes')
    plt.ylabel('Average Vehicles')
    plt.legend(loc=4)

    file_name = '{0}/vehicles.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/vehicles.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """ 
        Vehicles' velocity per cycle.
        (GLOBAL: For all vehicles in simulation)
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # for eps in range(0, 3):
    #     start = eps * episode
    #     finish = start + episode

    vels = np.array([v for v in velocities])

    Y = np.average(vels, axis=0)
    Y_std = np.std(vels, axis=0)

    # Replace NaNs.
    Y_lowess = np.where(np.isnan(Y), 0, Y)

    lowess = sm.nonparametric.lowess(Y_lowess, X, frac=0.10)

    # plt.plot(X,Y, label=f'Mean {fn(eps)}', c=MEAN_CURVE_COLOR)
    plt.plot(X,Y, label=f'Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if vels.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Minute')
    plt.ylabel('Average Velocity (m/s)')
    plt.legend(loc=4)

    file_name = '{0}/velocities.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/velocities.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """ 
        Actions per intersection.

        WARNING: This might require different processing here. As an example,
                 the actions taken by the DQN actions (discrete action agent)
                 differ from the ones taken by the DDPG agent (continuous action
                 agent).
    """

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    # for eps in range(0, 3):
    #     # Discrete action-schema.
    #     start = eps * episode
    #     finish = start + episode

    dfs_a = [pd.DataFrame(a) for a in actions]

    df_concat = pd.concat(dfs_a)

    by_row_index = df_concat.groupby(df_concat.index)
    df_actions = by_row_index.mean()


    window_size = min(len(df_actions)-1, 40)

    for col in df_actions.columns:
        plt.plot(df_actions[col].rolling(window=window_size).mean(), label=col)

    plt.xlabel('Minute')
    plt.ylabel('Average Action')
    plt.legend()

    plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()
        

if __name__ == '__main__':
    main()
