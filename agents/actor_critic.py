""" Actor-critic

    References:
    ---
    "Adaptive traffic signal control with actor-critic methods in a real-world traffic network with different traffic disruption events"
    Aslani, et al. 2017
"""
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
            * Eps decay rate.
    """
    def __init__(
            self,
            phases,
            alpha=0.15,
            beta=0.15,
            decay=0.9,
            eps=0.9,
            gamma=0.99):

        # Network
        self._tl_ids = [k for k in phases.keys()]
        self._phases = phases

        # Params
        self._alpha = alpha
        self._beta = beta
        self._decay = decay
        self._eps = eps
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

    @eps.setter 
    def eps(self, eps):
        self._eps = eps

    """ Agent act: supports rollouts"""
    def act(self, state, exclude_actions=set({})):
        actions = {}
        for _id, _state in state.items(): 
            if _id in exclude_actions:
                action = _state[0]
            else:
                policy = self.policy[_id][_state] 
                if any(policy): 
                    action, _ = max(policy.items(), key=itemgetter(1))
                    if np.random.rand() < self._eps:
                        action = np.random.choice([a for a in self.phases[_id] if a != action])
                else:
                    action = np.random.choice([a for a in self.phases[_id]])
            actions[_id] = int(action)
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


            delta = reward + self._gamma * value[state_next] - value[state_prev]
            # Update trace
            for s in value:
                critic_trace[s] *= self._gamma * self._decay
                if s == state_next: critic_trace[s] += 1

                for a in self.phases[tl_id]:
                    actor_trace[(s, a)] *= self._gamma * self._decay
                    if state_next == s and a == a_prev: actor_trace[(s, a)] += 1
                    # Update actor
                    policy[s][a] += self._beta * delta * actor_trace[(s, a)]
                # Update critic
                value[s] += self._alpha * delta  * critic_trace[s]

        
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

def epsilon_decay(num_episodes):
    if num_episodes > 0 and num_episodes <= 25: return -round((0.8 / 25), 4)
    if num_episodes == 26: return -0.1
    return 0


if __name__ == '__main__':
    # TODO: Implement a main that tells us how to use the object.
    pass
