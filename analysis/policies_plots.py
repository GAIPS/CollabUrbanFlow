"""Creates a heatmap policy plot w.r.t to the delay feature for each agent.

    TODO:
    ----
    * Consolidate w.r.t multiple policies, e.g, top 3.
    * Aggregate the results of multiple training runs.
    * Make it available for multiple models: DQN, etc. 
"""
import sys
from pathlib import Path
import argparse
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())

import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import operator

import json
import torch
import torch.nn.functional as F
from agents.gatw import GATW
from agents.dqn2 import DQN2

from agents import load_agent
from utils.network import get_adjacency_from_roadnet, get_capacity_from_roadnet

ROOT_PATH = '20211103135429.895808/arterial_20211103135431'
OUTPUT_DIR = 'data/policies/'
AGENT_TYPE = 'GATW'
AGENT_CHOICES = ('GATW', 'DQN2', 'DQN3', 'DQN4')


# Signal plan constraints.
min_green = 10
max_green = 90

def fmt(hparam_key): return hparam_key.split('.')[-1]
# TODO: Refactor this
def load_chkpt(chkpt_dir_path, agent_type):

    chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)

    
    state_dict = torch.load(chkpt_path / f'{agent_type}.chkpt')
    kwargs = {fmt(k):v for k, v in state_dict.items() if 'hparams' in k} 
    # in_features = state_dict['hparams.in_features']
    # n_embeddings = state_dict['hparams.n_embeddings']
    # n_hidden = state_dict['hparams.n_hidden']
    # out_features = state_dict['hparams.out_features']
    # n_heads = state_dict['hparams.n_heads']
    # n_layers = state_dict['hparams.n_layers']

    net_class = DQN2 if 'DQN' in agent_type else eval(agent_type)
    net = net_class(**kwargs)
    net.load_state_dict(state_dict, strict=False)
    return net

def batchfy(states):
    """Converts states list into batch like data

    Parameters:
    ----------
    * states: list<dict<str, list<float>>
        list<float>: is a list representing the internal state of an agent.
        str: is the tlid of an agent.
        list<dict<...>>: Experiment timesteps.

    Returns:
    --------
    * batch: np.array<B, N, F>
        B is the batch_size i.e len(states)
        N is the number of agents e.g arterial N=3
        F is the size of the features F=4
    """
    return np.stack([
        np.concatenate([
            np.array(sv).reshape(1, -1) for sv in state.values()
        ], axis=0) for state in states
    ])


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
        Creates a heatmap policy plot w.r.t to the delay feature for each agent.

        TODO:
        ----
            * Consolidate w.r.t multiple policies, e.g, top 3.
            * Aggregate the results of multiple training runs.
            * Make it available for multiple models: DQN, etc. """
    )
    parser.add_argument('source_folder', type=str, nargs='?',
                        help='Experiment train folder (network_timestamp)')

    return parser.parse_args()

def main(source_folder):

    # 1) Prepare paths.
    source_path = Path(source_folder)
    target_path = Path(OUTPUT_DIR)

    target_path.mkdir(exist_ok=True)
    # FIXME: This is hard-coded but should
    # be the first parent.
    subfolders = (str(source_path.parent), source_path.stem, 'heatmaps')
    for subfolder in subfolders:
        target_path = target_path / subfolder
        target_path.mkdir(exist_ok=True)


    # 2) Determine agent_type.
    config_path = source_path / 'config' / 'train.config'
    train_config = configparser.ConfigParser()
    train_config.read(config_path.as_posix())
    agent_type = train_config['agent_type']['agent_type']
    

    # 3) Load checkpoint
    chkpt_path = source_path / 'checkpoints'
    net = load_chkpt(chkpt_path, agent_type)

    # 4) Load adjacency matrix && delay capacities
    config_path = source_path / 'config' / 'roadnet.json'
    with config_path.open('r') as f: roadnet  = json.load(f)

    capacities = get_capacity_from_roadnet(roadnet)

    # 5) Load the network states during training.
    log_path = source_path / 'logs' / 'train_log.json'
    with log_path.open('r') as f: states  = batchfy(json.load(f)['states'])


    if agent_type not in AGENT_CHOICES: raise KeyError('Only agent_type GATW accepted.')
    batch_size = states.shape[0]
    if agent_type == 'GATW':
        adj = np.array(get_adjacency_from_roadnet(roadnet))
        adj = np.tile(adj, (batch_size, 1, 1))

        with torch.no_grad():
            x = torch.tensor(states).type(torch.FloatTensor)
            adj = torch.tensor(adj)
            q_values = net(x, adj)
            probs = F.softmax(q_values, dim=-1).numpy()
        
    else:
        n_agents = states.shape[1] 
        with torch.no_grad():
            x = torch.tensor(states).type(torch.FloatTensor)
            xs = torch.tensor_split(x, n_agents, dim=1)
            probs = []
            for n_a in range(n_agents):
                q_values = net(xs[n_a], n_a)
                probs.append(F.softmax(q_values, dim=-1).numpy())
            probs = np.concatenate(probs, axis=1)

    # aggregate w.r.t green time
    # segregate w.r.t tlid, phase
    # gets the data from the experiments
    labels = ['phase', 'time', 'delay_0', 'delay_1', 'keep', 'change']
    for tlnum, tlcap  in enumerate(capacities.items()):
        tlid, tlph = tlcap
        assert len(tlph) == 2
        # cx, cy = tuple(tlph.values())

        data = (states[:, tlnum, :], probs[:, tlnum, :])
        data = np.concatenate(data, axis=1)
        df = pd.DataFrame(data=data, columns=labels)  
        cx, cy = df[['delay_0', 'delay_1']].max()

        fig, axes = plt.subplots(nrows=1, ncols=len(tlph))
        fig.suptitle(f'{AGENT_TYPE} {tlid}: Switch action')
        for phnum in tlph:
            ax = axes.flat[phnum]
            ph_df = df[df['phase'] == float(phnum+1)]

            xyz = ph_df[['delay_0', 'delay_1', 'change']].values

            # Set the grid to interpolate to.
            xcoord, ycoord = np.linspace(0, cx, 50), np.linspace(0, cy, 50)
            xcoord, ycoord = np.meshgrid(xcoord, ycoord)
            zcoord = griddata(xyz[:, :2], xyz[:, -1], (xcoord, ycoord), method='cubic') 

            im = ax.pcolormesh(xcoord, ycoord, zcoord, cmap=cm.jet,
                               shading='gouraud', vmin=0, vmax=1)
            ax.set_title(f'Phase {phnum+1}')
            ax.set_xlabel('Delay 1')
            ax.set_ylabel('Delay 2')

        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        file_path = target_path / f'{tlid}.png'
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.show()

if __name__ == '__main__':
    args = get_arguments()
    main(args.source_folder)
