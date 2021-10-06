""" Marlin

    References:
    ---
    "Multiagent Reinforcement Learning for Integrated Network of Adaptive Traffic Signal Controllers (MARLIN-ATSC): Methodology and Large-Scale Application on Downtown Toronto"
    El-Tantawy, et al. 2013
"""
from copy import deepcopy
from operator import itemgetter
from collections import defaultdict
from pathlib import Path
import json
import dill
import numpy as np

import numpy as np

def make_zero_dict():
    return defaultdict(lambda : 0)


# access over [agent_i][agent_j][state_i][state_j][action_j]
def make_policy_dict():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: make_zero_dict()))))

# access over [agent_i][agent_j][state_i][state_j][action_i][action_j]
def make_q_values_dict():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: make_zero_dict())))))


class MARLIN(object):
    """ MultiAgent Reinforcement Learning for Integrated Network
     of adaptive traffic signal controllers platform
    """

    def __init__(
            self,
            phases,
            epsilon_init,
            epsilon_final,
            epsilon_timesteps,
            network,
            decision_step=10,
            learning_rate=0.9,
            discount_factor=0.98):

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
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor



        with open('data/networks/' + network + "/edges.json") as f:
            self.edges = json.load(f)

        self._policy_estimate = make_policy_dict()
        self._q_values = make_q_values_dict()


    @property
    def tl_ids(self):
        return self._tl_ids

    @property
    def phases(self):
        return self._phases

    @property
    def eps(self):
        return self._eps

    def stop(self):
        self._eps_explore = False

    def get_policy_estimate(self, id1, id2, state1, state2, action2):
        visited_sum = sum(self._policy_estimate[id1][id2][state1][state2].values())
        if visited_sum == 0:
            return 0
        else:
            return self._policy_estimate[id1][id2][state1][state2][action2] \
               / visited_sum

    def get_q_value_sum(self, id1, id2, action1, state1, state2):
        _sum = 0
        for action2 in self.phases[id2]:
            _sum += self._q_values[id1][id2][state1][state2][action1][action2] * \
                    self.get_policy_estimate(id1, id2, state1, state2, action2)
        return _sum

    def get_max_expected_q(self, id1, id2, state1, state2):
        _max = (float("-inf"), -1)
        for action1 in self.phases[id1]:
            _sum = self.get_q_value_sum(id1, id2, action1, state1, state2)
            if _sum > _max[0]:
                _max = (_sum, action1)
        return _max[0]

    def choose_greedy_action(self, id1, state):
        _max = (float("-inf"), -1)
        for action1 in self.phases[id1]:
            _sum = 0
            for id2 in self.edges[id1]:
                _sum += self.get_q_value_sum(id1, id2, action1, state[id1], state[id2])
            if _sum > _max[0]:
                _max = (_sum, action1)
        return _max[1]


    """ Agent act: supports rollouts"""

    def act(self, state):
        actions = {}
        for id in self.tl_ids:
            action = self.choose_greedy_action(id, state)
            if np.random.rand() < self._eps and self._eps_explore:
                action = np.random.choice([a for a in self.phases[id] if a != action])

            actions[id] = int(action)

        if self.eps + self._eps_decrement > self._eps_final:
            self._eps += self._eps_decrement
        return actions


    """ Agent update"""

    def update(self, s_prev, a_prev, r_next, s_next):

        for id1 in self.tl_ids:
            for id2 in self.edges[id1]:

                self._policy_estimate[id1][id2][s_prev[id1]][s_prev[id2]][a_prev[id2]] += 1
                max_expected_q = self.get_max_expected_q(id1, id2, s_next[id1], s_next[id2])

                self._q_values[id1][id2][s_prev[id1]][s_prev[id2]][a_prev[id1]][a_prev[id2]] = \
                    (1 - self._learning_rate) * self._q_values[id1][id2][s_prev[id1]][s_prev[id2]][a_prev[id1]][a_prev[id2]] \
                    + self._learning_rate * (r_next[id1] + self._discount_factor * max_expected_q)



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
