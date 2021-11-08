"""Performs a regression tree over the features to explain the policy.

    Converting a tree.dot to png
    >>> dot -Tpng tree.dot -o tree.png
"""
import sys
from pathlib import Path
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
# import the regressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import operator
import graphviz

import json
import torch
import torch.nn.functional as F
from agents.gatw import GATW
from utils import str2bool
from utils.network import get_adjacency_from_roadnet, get_capacity_from_roadnet


ROOT_PATH = '20211103135429.895808/arterial_20211103135431'
OUTPUT_DIR = 'data/policies/'
AGENT_TYPE = 'GATW'
AGENT_CHOICES = ('GATW',)


# Signal plan constraints.
min_green = 10
max_green = 90

# TODO: Refactor this
def load_chkpt(chkpt_dir_path, agent_type):

    chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
    chkpt_path = chkpt_dir_path / str(chkpt_num)
    print("Loading checkpoint: ", chkpt_path)

    state_dict = torch.load(chkpt_path / f'{agent_type}.chkpt')
    in_features = state_dict['hparams.in_features']
    n_embeddings = state_dict['hparams.n_embeddings']
    n_hidden = state_dict['hparams.n_hidden']
    out_features = state_dict['hparams.out_features']
    n_heads = state_dict['hparams.n_heads']
    n_layers = state_dict['hparams.n_layers']

    net = GATW(in_features=in_features, n_embeddings=n_embeddings,
               n_hidden=n_hidden, out_features=out_features,
               n_heads=n_heads, n_layers=n_layers)
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
    * batch: np.array<B, N, F> B is the batch_size i.e len(states)
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
        Creates a regression tree to explain the probability of switching.

     """)
    parser.add_argument('source_folder', type=str, nargs='?',
                        help='Experiment train folder (network_timestamp)')

    parser.add_argument('--render', type=str2bool, nargs='?', default=False,
                        help='Experiment train folder (network_timestamp)')
    return parser.parse_args()

def main(source_folder, render_tree):

    # 1) Prepare paths.
    source_path = Path(source_folder)
    target_path = Path(OUTPUT_DIR)

    target_path.mkdir(exist_ok=True)
    # FIXME: This is hard-coded but should
    # be the first parent.
    subfolders = (str([src for src in source_path.parents][-1]), source_path.stem, 'trees')
    for subfolder in subfolders:
        target_path = target_path / subfolder
        target_path.mkdir(exist_ok=True)

    # 2) Load checkpoint
    chkpt_path = source_path / 'checkpoints'
    net = load_chkpt(chkpt_path, 'GATW')

    # 3) Load adjacency matrix && delay capacities
    config_path = source_path / 'config' / 'roadnet.json'
    with config_path.open('r') as f: roadnet  = json.load(f)

    adj = np.array(get_adjacency_from_roadnet(roadnet))
    capacities = get_capacity_from_roadnet(roadnet)

    # 4) Load the network states during training.
    log_path = source_path / 'logs' / 'train_log.json'
    with log_path.open('r') as f: states  = batchfy(json.load(f)['states'])

    batch_size = states.shape[0]
    adj = np.tile(adj, (batch_size, 1, 1))

    if AGENT_TYPE != 'GATW': raise KeyError('Only AGENT_TYPE GATW accepted.')
    with torch.no_grad():
        x = torch.tensor(states).type(torch.FloatTensor)
        adj = torch.tensor(adj)
        q_values = net(x, adj)
        probs = F.softmax(q_values, dim=-1).numpy()
        
        
    # aggregate w.r.t green time
    # segregate w.r.t tlid, phase
    # gets the data from the experiments
    labels = ['phase', 'time', 'delay_0', 'delay_1']
    for tlnum, tlcap  in enumerate(capacities.items()):
        tlid, tlph = tlcap
        assert len(tlph) == 2

        data = (states[:, tlnum, :], probs[:, tlnum, :])

        X, Y = states[:, tlnum, :], (probs[:, tlnum, 1] > 0.5).astype(int)

        index1 = X[:, 0] == 1.0
        X1, Y1 = X[index1, :], Y[index1]
        # create a regressor object
        classifier = DecisionTreeClassifier(random_state = 0, max_depth=3)

        # fit the regressor with X and Y data
        classifier.fit(X, Y)

        tree_path = target_path /  f'tree_{tlid}.dot'
        export_graphviz(
            classifier,
            out_file = tree_path.as_posix(),
            feature_names=labels,
            class_names=['keep', 'change']
        )
        if render_tree:
            tree_data = export_graphviz(
                classifier,
                out_file = None,
                feature_names=labels,
                class_names=['keep', 'change']
            )
            graph = graphviz.Source(tree_data)
            graph.render(tree_path.stem, view=True)

if __name__ == '__main__':
    args = get_arguments()
    main(args.source_folder, args.render)
