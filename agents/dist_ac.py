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

from utils import flatten
def actor_init(): return defaultdict(list)
def critic_init(): return defaultdict(actor_init)

class DistActorCritic(object):
    def __init__(
            self,
            phases,
            epsilon_init,
            epsilon_final,
            epsilon_timesteps,
            decision_step=10,
            alpha=0.90,
            beta=0.3,
            decay=0.55,
            gamma=0.98):

        # Network
        self._tl_ids = [k for k in phases.keys()]
        self._phases = phases

        # Exploration & Exploitation
        assert epsilon_init >= epsilon_final
        assert epsilon_timesteps > 0

        self._eps = epsilon_init
        self._eps_decrement = \
                (epsilon_final - epsilon_init) * decision_step / epsilon_timesteps
        self._eps_explore = True
        self._eps_final = epsilon_final

        # Params
        self._alpha = alpha
        self._beta = beta
        self._decay = decay
        self._gamma = gamma

        # Critic
        self._value = make_critic_dict()
        self._critic_trace = make_trace_dict()

        # Actor
        self._policy = make_actor_dict()
        self._actor_trace = make_trace_dict()

    @property
    def tl_ids(self):
        return self._tl_ids

    @property
    def phases(self):  
        return self._phases

    @property
    def value(self):
        return self._value

    @property
    def policy(self):
        return self._policy

    @property
    def eps(self):
        return self._eps

    def stop(self):
        self._eps_explore = False

    def act(self, state):
        state = flatten(state.values())
        actions = [self.policy[tl_id](state) for tl_id in self.tl_ids]
        return actions

    # TODO: move flatten operation to a decorator.
    def update(self, state, actions, reward, next_state)
        state = flatten(state.values())
        actions = flatten(actions.values())
        reward = flatten(reward.values())
        next_state = flatten(next_state.values())


        next_mu = {}
        next_actions = {}
        for ntl, tl in enumerate(self.tl_ids):
            next_mu[tl] = (1 - self.beta) * self.mu[tl] + self.beta * reward[ntl]
            next_actions[tl] = self.policy[tl](state)

        
        for ntl, tl in enumerate(self.tl_ids):
            self.delta[tl] = reward[ntl] - self.mu[tl] + 
            

