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


def from_json_to_dataframe(data):
    ''' Converts the json into a multi-index column. 

    FIXME:
    ------
    * Convert into episodic.
    * Make batch.

    Params:
    ------
    * data: dict<str, list<dict<str, object>>
    Example: > data
                          rewards  velocities  vehicles   actions    
    0   {'247123161': -0.3281}    9.191060        10     {'247123161': 0}  
    1   {'247123161': -4.3606}    6.466366        13     {'247123161': 0}  
    2   {'247123161': -8.0186}    3.801833        12     {'247123161': 1}  
    3  {'247123161': -10.2113}    3.338875        16     {'247123161': 1}  
    4  {'247123161': -12.3408}    1.936568        16     {'247123161': 1}  
 
        
    Returns:
    -------
    * df: multi-column dataframe.
    Example: > df.head()
    ipdb> df.head()
    tl_id  247123161    network          247123161
    metric   rewards velocities vehicles   actions
    0        -0.3281   9.191060       10         0
    1        -4.3606   6.466366       13         0
    2        -8.0186   3.801833       12         1
    3       -10.2113   3.338875       16         1
    4       -12.3408   1.936568       16         1

    
    ''' 
    cols = ['rewards', 'velocities', 'vehicles', 'actions']
    names = ['tl_id', 'metric']
    data1 = defaultdict(lambda : [])

    for col in cols:
        for d in data[col]:
            if isinstance(d, dict):
                for k, v in d.items():
                    data1[(k, col)].append(v)
            else:
                data1[('network', col)].append(d)

    df = pd.DataFrame.from_dict(data1)
    df.columns.names = names


    return df

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

    # Get agent_type.
    # train_config_path = list(Path(experiment_root_folder).rglob('train.config'))[0]
    # train_config = configparser.ConfigParser()
    # train_config.read(train_config_path)
    agent_type = 'A_CAT'

    actions = []
    rewards = []
    rewards1 = []
    rewards2 = []
    vehicles = []
    velocities = []

    sampled_rewards_2 = []

    sampled_vehicles = []
    sampled_velocities = []
    # Concatenate data for all runs.
    for run_name in train_files:

        print('Processing JSON file: {0}'.format(run_name))

        # Load JSON data.
        with open(run_name) as f:
            json_data = json.load(f)

        # Rewards per time-step.
        r = json_data['rewards']
        r = pd.DataFrame(r)
        rewards1.append(np.sum(r.values, axis=1))

        # aggregate data
        rewards2.append(json_data['rewards'])

        # Number of vehicles per time-step.
        vehicles.append(json_data['vehicles'])

        # Vehicles' velocity per time-step.
        velocities.append(json_data['velocities'])

        # Agent's actions.
        actions.append(json_data['actions'])

        # df = from_json_to_dataframe(json_data)
    """
        Rewards per cycle.
        (GLOBAL: sum of the reward for all intersections).
    """
    rewards = np.array(rewards1)


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(rewards, axis=0)
    Y_std = np.std(rewards, axis=0)
    X = np.linspace(1, rewards.shape[1], rewards.shape[1])


    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if rewards.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Decision Step')
    plt.ylabel('Reward')
    # plt.title('Train rewards ({0} runs)'.format(len(train_files)))
    plt.legend(loc=4)

    file_name = '{0}/rewards.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/rewards.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        Rewards per intersection.
    """
    dfs_rewards = [pd.DataFrame(r) for r in rewards2]
    # dfs_rewards = [pd.DataFrame(r) for r in sampled_rewards_2]

    df_concat = pd.concat(dfs_rewards)

    by_row_index = df_concat.groupby(df_concat.index)
    df_rewards = by_row_index.mean()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    window_size = min(len(df_rewards)-1, 20)

    for col in df_rewards.columns:
        plt.plot(df_rewards[col].rolling(window=window_size).mean(), label=col)

    plt.xlabel('Decision Step')
    plt.ylabel('Reward')
    # plt.title('Rewards per intersection')
    plt.legend()

    plt.savefig('{0}/rewards_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/rewards_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    """ 
        Number of vehicles per cycle.
        (GLOBAL: For all vehicles in simulation)
    """
    vehicles = np.array(vehicles)
    # vehicles = np.array(sampled_vehicles)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(vehicles, axis=0)
    Y_std = np.std(vehicles, axis=0)
    X = np.linspace(1, vehicles.shape[1], vehicles.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if vehicles.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Decision Step')
    plt.ylabel('Number of vehicles')
    # plt.title('Number of vehicles ({0} runs)'.format(len(train_files)))
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
    velocities = np.array(velocities)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(velocities, axis=0)
    Y_std = np.std(velocities, axis=0)
    try:
        X = np.linspace(1, velocities.shape[1], velocities.shape[1])
    except Exception:
        import ipdb; ipdb.set_trace()

    # Replace NaNs.
    Y_lowess = np.where(np.isnan(Y), 0, Y)

    lowess = sm.nonparametric.lowess(Y_lowess, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if velocities.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Decision Step')
    plt.ylabel('Velocity (m/s)')
    # plt.title('Train: Velocity of the vehicles ({0} runs)'.format(len(train_files)))
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
    if agent_type in ('DDPG', 'MPO'):
        # Continuous action-schema.

        # TODO: This only works for two-phased intersections.
        dfs_a = [pd.DataFrame([{i: a[0] for (i, a) in t.items()}
                                for t in run])
                                    for run in actions]

        df_concat = pd.concat(dfs_a)

        by_row_index = df_concat.groupby(df_concat.index)
        df_actions = by_row_index.mean()

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        window_size = min(len(df_actions)-1, 40)

        for col in df_actions.columns:
            plt.plot(df_actions[col].rolling(window=window_size).mean(), label=col)

        plt.xlabel('Decision Step')
        plt.ylabel('Action (Phase-1 allocation)')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.savefig('{0}/actions_per_intersection_smoothed.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions_per_intersection_smoothed.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for col in df_actions.columns:
            plt.plot(df_actions[col], label=col)

        plt.xlabel('Decision Step')
        plt.ylabel('Action (Phase-1 allocation)')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()

    else:
        # Discrete action-schema.
        dfs_a = [pd.DataFrame(run) for run in actions]

        df_concat = pd.concat(dfs_a)

        by_row_index = df_concat.groupby(df_concat.index)
        df_actions = by_row_index.mean()

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        window_size = min(len(df_actions)-1, 40)

        for col in df_actions.columns:
            plt.plot(df_actions[col].rolling(window=window_size).mean(), label=col)

        plt.ylim(-0.2,6.2)
        plt.yticks(ticks=[0,1,2,3,4,5,6], labels=['(30,70)', '(36,63)', '(43,57)', '(50,50)', '(57,43)', '(63,37)', '(70,30)'])

        plt.xlabel('Decision Step')
        plt.ylabel('Action')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()
        

if __name__ == '__main__':
    main()
