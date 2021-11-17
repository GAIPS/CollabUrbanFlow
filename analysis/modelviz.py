''' Visualize models

    References:
    -----------
    * https://github.com/szagoruyko/pytorchviz

    Examples:
    ---------
    >>> model = nn.Sequential()
    >>> model.add_module('W0', nn.Linear(8, 16))
    >>> model.add_module('tanh', nn.Tanh())
    >>> model.add_module('W1', nn.Linear(16, 1))

    >>> x = torch.randn(1, 8)
    >>> y = model(x)

    >>> make_dot(y.mean(), params=dict(model.named_parameters()))
'''
from pathlib import Path
import sys
sys.path.append(Path.cwd().as_posix())
import torch
from torchviz import make_dot
import graphviz

from tqdm import tqdm
from agents import get_agent
from environment import get_environment

name = 'intersection'
print(f'Loading environment: {name}')
env = get_environment(name)

name  = 'DQN3'
print(f'Populating model: {name}')
model = get_agent(name, env, 0.0, 0.0, 3600)

net = model.net
agent = model.agent
# adj = model.adjacency_matrix
n_agents = len(env.tl_ids)

print(f': {model}')
timesteps = 100
for _ in tqdm(range(timesteps)):
    agent.play_step(net, epsilon=0.0)

x = torch.tensor(agent.state).reshape((n_agents, -1))
# model_dot = make_dot(net(x, adj), params=dict(net.named_parameters()))
model_dot = make_dot(net(x, 0), params=dict(net.named_parameters()))

graph_path = Path.cwd() / 'data' 
subfolders = ('visualizations', 'models')
for subfolder in subfolders: 
    graph_path = graph_path / subfolder
    graph_path.mkdir(exist_ok=True)
graph_path = graph_path / f'{name.lower()}.dot'

model_dot.render(graph_path, view=True)

