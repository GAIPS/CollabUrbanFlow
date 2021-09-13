""" Actor-critic

    References:
    ---
    "Adaptive traffic signal control with actor-critic methods in a real-world traffic network with different traffic disruption events"
    Aslani, et al. 2017
"""
import ipdb
from copy import deepcopy
from operator import itemgetter
from collections import defaultdict
from pathlib import Path

import dill
import numpy as np

def make_zero_dict():
    return defaultdict(lambda : 0)

def make_trace_dict():
    return defaultdict(lambda: make_zero_dict())
make_critic_dict = make_trace_dict

def make_actor_dict():
    return defaultdict(lambda: make_trace_dict())

class ACAT(object):
    """ Actor critic with eligibilty traces. 

        TODO:
            * Store internal state.
    """
    def __init__(
            self,
            phases,
            epsilon_init,
            epsilon_final,
            epsilon_timesteps,
            decision_step=5,
            alpha=0.90,
            beta=0.25,
            decay=0.9,
            gamma=0.9):

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

    """ Agent act: supports rollouts"""
    def act(self, state):
        actions = {}
        for _id, _state in state.items(): 
            policy = self.policy[_id][_state] 
            if any(policy): 
                action, _ = max(policy.items(), key=itemgetter(1))
                if np.random.rand() < self._eps and self._eps_explore:
                    action = np.random.choice([a for a in self.phases[_id] if a != action])
            else:
                action = np.random.choice([a for a in self.phases[_id]])
            actions[_id] = int(action)
        if self.eps + self._eps_decrement > self._eps_final:
            self._eps += self._eps_decrement
        return actions 

    """ Agent update: with eligibility traces"""
    def update(self, s_prev, a_prev, r_next, s_next):
        # Update actor-critic condition

        for tl_id in self.tl_ids:
            # Create aliases
            value = self.value[tl_id]
            policy = self.policy[tl_id]
            actor_trace = self._actor_trace[tl_id]
            critic_trace = self._critic_trace[tl_id]
            reward = r_next[tl_id]
            state_prev = s_prev[tl_id]
            state_next = s_next[tl_id]
            action_prev = a_prev[tl_id]

            delta = reward + self._gamma * value[state_next] - value[state_prev]
            # Update trace
            for s in value:
                critic_trace[s] *= self._gamma * self._decay
                if s == state_prev: critic_trace[s] += 1

                for a in (0, 1):
                    actor_trace[(s, a)] *= self._gamma * self._decay
                    if state_prev == s and a == action_prev: actor_trace[(s, a)] += 1
                    # Update actor
                    policy[s][a] += self._beta * delta * actor_trace[(s, a)]
                # Update critic
                value[s] += self._alpha * delta  * critic_trace[s]

     
    """ Reset eligibilty traces """
    def reset(self):
        self._critic_trace = make_trace_dict()
        self._actor_trace = make_trace_dict()

    """ Serialization """
    # Serializes the object's copy -- sets get_wave to null.
    def save_checkpoint(self, chkpt_dir_path, chkpt_num):
        class_name = type(self).__name__.lower()
        file_path = Path(chkpt_dir_path) / chkpt_num / f'{class_name}.chkpt'  
        file_path.parent.mkdir(exist_ok=True)
        cpy = deepcopy(self)
        with open(file_path, mode='wb') as f:
            dill.dump(cpy, f)

    # deserializes object -- except for get_wave.
    @classmethod
    def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
        class_name = cls.__name__.lower()
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{class_name}.chkpt'  
        with file_path.open(mode='rb') as f:
            new_instance = dill.load(f)
        return new_instance

if __name__ == '__main__':
    # TODO: Implement a main that tells us how to use the object.
    pass
