''' Distributed version of the actor-critic algorithm

    References:
    -----------
    `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`

    Zhang, et al. 2018
'''
from copy import deepcopy
from operator import itemgetter
from collections import defaultdict
from pathlib import Path

import numpy as np

# Uncoment to run stand alone script.
import sys
sys.path.append(Path.cwd().as_posix())

from utils import flatten as ufl

# Helpful one-liners
def seq(gen): return [val for val in ufl(gen)]
def vectorize(a_dict): return np.stack(seq(a_dict.values()))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), keepdims=True, axis=-1)
def replace(x, pos, elem): x[pos] = elem; return x
def take_column(x, y): return np.take_along_axis(x, y, 1)
def unsqueeze(x): return np.expand_dims(x, 1)
def gather(x, y): return take_column(x, unsqueeze(y)).squeeze(-1)
def tile(x, n, k): return np.tile(x, (n, k, 1)).T
def clip(x): return np.minimum(np.maximum(x, -1e-8), 50)

# x is 1dim array should be [i, j, k]
def tiler(x, i=None, j=None, k=None):
    if j is not None and k is not None: return np.tile(x, (j, k, 1)).T
    if i is not None and j is not None: return np.tile(x, (i, j, 1))
    if k is not None: return np.tile(x, (k, 1)).T
    if i is not None: return np.tile(x, (i, 1))

class DistributedActorCritic(object):
    def __init__(
            self,
            phases,
            adjacency_matrix,
            epsilon_init,
            epsilon_final,
            epsilon_timesteps,
            decision_step=10,
            alpha=0.90,
            beta=0.3,
            gamma=0.98):


        # Network
        self.tl_ids = [k for k in phases.keys()]
        self.phases = phases
        self.n_agents = len(self.tl_ids)
        self.n_inputs = self.n_agents * 4
        self.n_actions = 2
        self.c = adjacency_matrix

        # Exploration & Exploitation
        assert epsilon_init >= epsilon_final
        assert epsilon_timesteps > 0

        self._eps_init = epsilon_init
        self._eps_dec = \
               (epsilon_final - epsilon_init) * decision_step / epsilon_timesteps
        self._eps_final = epsilon_final
        self._eps_explore = True

        # Hyperparameters
        self.alpha = alpha # critic stepsize
        self.beta = beta # actor stepsize
        self.gamma = gamma # discount factor

        # Dac parameters
        n_parameters = (self.n_agents, self.n_actions, self.n_inputs)
        self.mu = np.random.randn((self.n_agents))
        self.w = np.random.randn(*n_parameters) 
        self.theta = np.random.randn(*n_parameters) 

        self.reset()

    @property
    def eps(self):
        if not self._explore: return 0.0
        return max(self._eps_final, self._eps_init - self._eps_dec * self.n_steps)

    def stop(self):
        self._explore = False

    def reset(self):
        self.n_steps = 0

    def act(self, state):
        if isinstance(state, dict): state = vectorize(state) 
        return self.policy(state, choice=True)

    # TODO: move flatten operation to a decorator.
    def update(self, state, actions, reward, next_state):
        state = vectorize(state)       # [n_inputs]
        actions = vectorize(actions)   # [n_agent]
        reward = vectorize(reward)     # [n_agent]
        next_state = vectorize(next_state)

        ii = np.arange(self.n_agents)
        jj = actions

        # auxiliary variables
        next_mu = np.zeros((self.n_agents,), dtype=float) # next_step_mu
        # w = np.zeros_like(self.w) # weights before consensus

        # 1. Observe state next_state and reward
        # Update mu and execute actions.
        # Execute this line before calling self.update(.)
        # next_mu = (1 - self.alpha) * self.mu + self.alpha * reward

        # 2. Observe joint actions
        next_actions = self.act(next_state)

        # actuate on environment
        # 3. Update 
        # [n_agents] --> [n_agents, n_actions, n_inputs]
        delta = reward - self.mu + self.q(next_state, next_actions) - self.q(state, actions) 
        delta = tiler(delta, j=self.n_inputs, k=self.n_actions)

        # Critic step
        weights = self.w + self.alpha * delta * self.grad_q(state)

        # Actor step
        adv = unsqueeze(self.advantage(state, actions))    # [n_agents, 1]
        ksi = self.grad_policy(state, actions)             # [n_agents, n_inputs]
        self.theta[ii, jj, :] += (self.beta * adv * ksi)   # [n_agents, n_inputs]

        # Consensus step: broadcast weights
        self.w = np.einsum('ij, jkl -> ikl', self.c, weights)
        self.mu = (1 - self.alpha) * self.mu + self.alpha * reward
        self.n_steps += 1

    def q(self, state, actions=None):
        # [n_agents, n_actions, n_inputs] * [n_inputs]
        qval = self.w @ state  # [n_agents, n_actions]
        if actions is None: return qval
        # [n_agents]
        return gather(qval, actions)

    def grad_q(self, state):
        '''Gradient of the Q-function

            Parameters:
            -----------
            * state: np.array(<float>)
            1-dim np.array n_inputs
            
            Returns:
            --------
            * state: np.array(<float>)
            3-dim np.array(<n_agents, n_actons, n_inputs>)
        '''
        return tiler(state, i=self.n_agents, j=self.n_actions)

    def policy(self, state, choice=False):
        '''Policy pi(a | s)

            Parameters:
            -----------
            * state: np.array(<float>)
            1-dim np.array n_inputs
            * choice: bool
             if True selects an action, else outputs a distribution.
            
            Returns:
            --------
            * state: np.array(<float>)
            3-dim np.array(<n_agents, n_actons, n_inputs>)
        '''
        # gibbs distribution / Boltzman policies.
        # vectorize state
        if isinstance(state, dict): state = vectorize(state) 

        # [n_agent, n_actions, n_inputs]
        phi_s = tiler(state, i=self.n_agents, j=self.n_actions)

        # [n_agents, n_actions]
        x = clip(np.sum(self.theta * phi_s, axis=-1))

        # [n_agents, n_actions]
        probs = softmax(x)
        return np.array([np.random.choice(self.n_actions, p=p) for p in probs])

    def grad_policy(self, state, actions):
        # [n_agents]
        prob = gather(self.policy(state), actions)

        # [n_agents, n_states]
        prob = tiler(prob, k=self.n_inputs) 

        # [n_agents, n_states]
        phi_s = tiler(state, i=self.n_agents)

        # [n_agents, n_inputs]
        return (1 - prob) * phi_s

    def advantage(self, state, actions):
        # [n_agents, n_actions]
        probs = self.policy(state)

        # [n_agents]
        return self.q(state, actions) - np.sum(probs * self.q(state), axis=-1)

if __name__ == '__main__':
    phases = {
        'intersection_1_1': {
            0: ['-238059324_0', '383432312_0'],
            1: ['-238059328_0', '-238059328_1', '309265401_0', '309265401_1']
        },
        'intersection_2_1': {
            0: ['22941893_0'],
            1: ['-309265401_0', '-309265401_1', '309265400_0', '309265400_1']
        },
        'intersection_3_1': {
            0: ['23148196_0'],
            1: ['-309265400_0', '-309265400_1', '309265402_0', '309265402_1']
        }
    }

    states = {
        'intersection_1_1': [1, 20, 0.0488, 6.5335],
        'intersection_2_1': [1, 20, 0.0647, 4.8532],
        'intersection_3_1': [1, 20, 0.0256, 4.5858]
    }
    actions = {
        'intersection_1_1': 0,
        'intersection_2_1': 0,
        'intersection_3_1': 1
    }
    rewards = {
        'intersection_1_1': -7.3541,
        'intersection_2_1': -7.0,
        'intersection_3_1': -5.5807
    }
    next_states = {
        'intersection_1_1': [1, 30, 0.037, 7.3171],
        'intersection_2_1': [1, 30, 0.0, 7.0],
        'intersection_3_1': [2, 10, 2.0517, 3.529]
    }
    adj = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
    ])

    epsilon_init = 0.8
    epsilon_final = 0.01
    epsilon_timesteps = 3600

    dac = DistributedActorCritic(
        phases, adj, epsilon_init,
        epsilon_final, epsilon_timesteps
    )
    distributed_actions = dac.policy(states, choice=False)

    next_actions = dac.policy(states, choice=True)

    advantage = dac.advantage(vectorize(states), vectorize(actions))

    ksi = dac.grad_policy(vectorize(states), vectorize(actions))

    
    # for i in range(10000):
    dac.update(states, actions, rewards, next_states)
    
