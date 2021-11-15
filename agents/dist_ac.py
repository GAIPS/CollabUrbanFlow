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
class DistActorCritic(object):
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
        self.mu0 = np.random.randn((self.n_agents))
        self.mu1 = np.random.randn((self.n_agents))
        self.w0 = np.random.randn((self.n_agents, self.n_inputs)) 
        self.w1 = np.random.randn((self.n_agents, self.n_inputs)) 
        self.theta = np.random.randn((self.n_agents, n_actions, self.n_inputs)) 

        self.reset()

    @property
    def value(self):
        return self._value

    @property
    def policy(self):
        return self._policy

    @property
    def eps(self):
        if not self._explore: return 0.0
        return max(self._eps_final, self._eps_init - self._eps_dec * self.n_steps)

    def stop(self):
        self._explore = False

    def reset(self):
        self.n_steps = 0

    def act(self, state):
        state = flatten(state.values())
        actions = [self.policy[tl_id](state) for tl_id in self.tl_ids]
        return actions

    # TODO: move flatten operation to a decorator.
    def update(self, state, actions, reward, next_state)
        state = numpfy(state)
        actions = numpfy(actions)
        reward = numpfy(reward)
        next_state = numpfy(next_state)

        # Update mu and execute actions.
        for ntl in range(self.n_agents):
            self.mu1[ntl] = (1 - self.alpha) * self.mu0[ntl] + self.alpha * reward[ntl]
            # TODO: implement policy
            next_actions[tl] = self.policy[tl](state)

        # Update 
        for ntl in range(self.n_agents):
            delta = reward[ntl] - self.mu0[tl] + self.q(next_state, ntl) - self.q(state, ntl)  
            # Critic step
            self.w1[ntl] = self.w0[ntl] + self.alpha * delta * self.grad_q(state, ntl)

            # Actor step
            adv = self.advantage(state, action)
            ksi = self.grad_policy(state, action, ntl)

        self.num_steps += 1
        self._eps -= self._eps_decrement
            

    def q(self, state, n_agent):
        return self.w0[n_agent].dot(state)

    def grad_q(self, state, n_agent):
        return state

    def policy(self, state, ntl):
        # gibbs distribution / Boltzman policies.
        es = []
        s_cum = 0
        for n_a in range(self.n_actions):
            theta = self.theta[ntl, n_a, :]
            es.append(np.exp(np.max(theta_0.dot(state), 1e-8)))
            s_cum += es[-1]  
        return np.array(es) / s_cum

    def grad_policy(self, state, action, ntl):
        pol = self.policy(state, ntl)
        pol_cum = np.zeros((self.n_inputs,), dtype=float)
        for n_a in range(self.n_agents):
            pol_cum += pol[n_a] * self.theta[ntl, n_a, :]
        ret = self.theta[ntl, action, :] - pol_cum
        return ret

    def advantage(self, state, action):
        q_cum = 0 
        pol = self.policy(state, ntl)
        for atl in (0, 1):
            _action = action; _action[ntl] = atl
            q_cum += pol[atl]  * self.q(state, _action)
        return self.q(state, action) - q_cum

