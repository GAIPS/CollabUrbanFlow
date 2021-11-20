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
from utils import flatten

def actor_init(): return defaultdict(list)
def policy_init(n_input): return Actor(n_input)
def critic_init(n_input): return defaultdict(policy_init(n_input))

# class Actor:
#     def __init__(self, n_input, beta):
#         self.theta = np.random.randn((n_input, 1)) 
# 
#     def __call__(self, states):
#         return np.round(self.theta.dot(states))
# 
#     def step(self, fn_q,  

def numpfy(a_dict): return np.array(flatten(a_dict.values()))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=-1)
def compl(x, elem, pos): x[pos] = elem; return x

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
        self.mu = np.random.randn((self.n_agents))
        self.w = np.random.randn((self.n_agents, self.n_actions, self.n_inputs)) 
        self.theta = np.random.randn((self.n_agents, n_actions, self.n_inputs)) 

        self.reset()

    @property
    def value(self):
        return self._value

    @property
    def eps(self):
        if not self._explore: return 0.0
        return max(self._eps_final, self._eps_init - self._eps_dec * self.n_steps)

    def stop(self):
        self._explore = False

    def reset(self):
        self.n_steps = 0

    def act(self, state):
        if isinstance(state, dict): state = numpfy(state) 
        return self.policy(state, choice=True)

    # TODO: move flatten operation to a decorator.
    def update(self, state, actions, reward, next_state)
        state = numpfy(state)       # [n_inputs]
        actions = numpfy(actions)   # [n_agent]
        reward = numpfy(reward)     # [n_agent]
        next_state = numpfy(next_state)

        # auxiliary variables
        next_mu = np.zeros((self.n_agents,), dtype=float) # next_step_mu
        w = np.zeros_like(self.w) # weights before consensus

        # 1. Observe state next_state and reward
        # Update mu and execute actions.
        # for ntl in range(self.n_agents):
        next_mu = (1 - self.alpha) * self.mu + self.alpha * reward

        # 2. Observe joint actions
        next_actions = self.act(next_state) 
        # actuate on environment

        # 3. Update 
        # for ntl in range(self.n_agents):
        delta = reward - self.mu + self.q(next_state) - self.q(state)  
        # Critic step
        w = self.w + self.alpha * delta * self.grad_q(state)

        # Actor step
        adv = self.advantage(state, actions)
        ksi = self.grad_policy(state, actions)
        self.theta += self.beta * adv * ksi

        # TODO: consensus step.
        self.num_steps += 1
        self._eps -= self._eps_decrement
            

    def q(self, state, actions=None):
        # [n_agents, n_actions, n_inputs] * [n_inputs]
        q = self.w @ state  # [n_agents, n_actions]
        if actions is None: return q
        # [n_agents]
        return q[:, actions]

    def grad_q(self, state):
        return np.tile(state, (self.n_agents, 1))

    def policy(self, state, choice=False):
        # gibbs distribution / Boltzman policies.
        tol = 1e-8

        # [n_agent, n_actions, n_inputs]
        phis = np.tile(state, (self.n_agents, self.n_actions, 1)) 
        # [n_agent, n_actions, n_inputs]
        thetas = self.theta

        # [n_agents, n_actions]
        x = np.maximum(np.sum(thetas * phis, axis=-1), tol)

        # [n_agents, n_actions]
        probs = softmax(x)
        if not choice: return probs
        return np.choice(self.n_actions, replace=True, size=self.n_agents, p=probs)

    def grad_policy(self, state, actions):
        # [n_agents, n_actions]
        phis = self.theta @ state

        # [n_agents, n_actions]
        probs = self.policy(state)

        # [n_agents]
        return phis[:, actions] - np.sum(phis * probs, axis=-1)

    def advantage(self, state, actions):
        # [n_agents, n_actions]
        probs = self.policy(state)

        # [n_agents]
        return self.q(state, actions) - np.sum(probs * self.q(state), axis=-1)

