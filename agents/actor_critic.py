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

A = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)

def gen_0():
    return 0

def gen_0_dict():
    return defaultdict(gen_0)

class ACAT(object):
    """ """
    def __init__(
            self,
            tl_id,
            num_phases,
            wave,
            approx_calc=None,
            yellow_time=5,
            alpha=0.15,
            beta=0.15,
            decay=0.9,
            eps=0.9,
            gamma=0.99):


        # Simulation control
        self._episode = 24 * 3600
        self._episode_time = 0
        self._time = 0

        # Network
        self._num_phases = num_phases
        self._tl_id = tl_id
        self._yellow_time=yellow_time

        # Params
        self._alpha=alpha
        self._beta=beta
        self._decay=decay
        self._eps=eps
        self._gamma=gamma

        # Approximator
        self.get_wave = wave
        self._approx_calc = approx_calc

        # Critic
        self._value = defaultdict(gen_0)
        self._critic_trace = defaultdict(gen_0)

        # Actor
        self._policy = defaultdict(gen_0_dict)
        self._actor_trace = defaultdict(gen_0)

        # Output
        self._rewards = []
        self._experience = {}
        self.reset()

    def compute(self):
        # states
        wave = self.approx(self.get_wave())
        reward = -sum(wave)
        wave = (self.phase,) + wave

        # Update actor-critic condition
        if self._prev_wave is not None:
            delta = reward + self._gamma * self._value[wave] - self._value[self._prev_wave]
            # Update trace
            for s in self._value:
                self._critic_trace[s] *= self._gamma * self._decay
                if s == wave: self._critic_trace[s] += 1

                for a in A:
                    self._actor_trace[(s, a)] *= self._gamma * self._decay
                    if wave == s and a == self._curr_action: self._actor_trace[(s, a)] += 1
                    # Update actor
                    self._policy[s][a] += self._beta * delta * self._actor_trace[(s, a)]
                # Update critic
                self._value[s] += self._alpha * delta  * self._critic_trace[s]


        # Register updates.
        ret = (wave, self.action, reward)

        # Select action.
        if  self._episode_time == self._next_phase_time:
            act = self.act(wave, self._episode_time, approx=False)
            if np.random.rand() < self._eps:
                act = np.random.choice([a for a in A if a != act])
            self._curr_action = act
            self._curr_phase  = (self._curr_phase + 1) % self._num_phases
            self._next_phase_time  = self._episode_time + act + self._yellow_time - 1

        self._prev_wave = wave
        self._episode_time += 1

        # Global simulation time.
        if self._episode_time % self._episode==0:
            self.reset()
        return ret

    # Should not be called from without.
    def reset(self):
        self._curr_phase = 0
        self._curr_action = np.random.choice(A).astype(int)
        self._next_phase_time = self._curr_action + self._yellow_time - 1
        self._prev_wave = None
        self._time += self._episode_time
        self._eps += epsilon_decay(self._time / self._episode)
        self._episode_time = 0

    @property
    def phase(self):
        return int(self._curr_phase)

    @property
    def next_phase_time(self):
        return int(self._next_phase_time)

    @property
    def phase_ctrl(self):
        return self.phase, self.next_phase_time

    @property
    def num_phases(self):   # non-yellow phases
        return int(self._num_phases)

    @property
    def tl_id(self):   # non-yellow phases
        return self._tl_id

    @property
    def value(self):
        return self._value

    @property
    def policy(self):
        return self._policy

    @property
    def action(self):
        return int(self._curr_action)

    """ Agent act: supports rollouts"""
    def act(self, state, episode_time, approx=True):
        wave = (self.phase,) + self.approx(state) if approx else state
        act, _ = max(self.policy[wave].items(), key=itemgetter(1))
        return int(act)

    """ Function approximation"""
    def approx(self, wave):
        if self._approx_calc is None: return wave
        npywv = self._approx_calc.map(wave)
        return tuple(npywv[0].tolist())

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
    if num_episodes > 0 and num_episodes <= 15: return -(0.8 / 15)
    if num_episodes == 16: return -0.1
    return 0


if __name__ == '__main__':
    # TODO: Implement a main that tells us how to use the object.
    pass
