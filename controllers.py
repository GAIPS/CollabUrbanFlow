"""Implementation of classic adaptive controllers and methods

    * Reactive agents.
    * TODO: Extend to multi-phase scenarios.
"""
import copy
import json
from operator import itemgetter
from collections import defaultdict

import numpy as np

from utils.network import get_capacity_from_roadnet, get_capacity_from_roadnet2

# returns a list of the maximum elements.
def argmax_k(x): return [ii for ii, xx in enumerate(x) if xx == max(x)]
# returns the index of the second hightest element.
def argmax2(x): return sorted(enumerate(x), reverse=True, key=itemgetter(1))[1][0]
def exclude(x, y): return [ii for ii, xx in enumerate(x) if ii != y]

class MaxPressure:
    """Adaptive rule based controller based of ATSC

        Reference:
        ----------
        * Wade Genders and Saiedeh Razavi, 2019
            An Open Source Framework for Adaptive Traffic Signal Control.

        See also:
        ---------
        * Wei, et al, 2018
            PressLight: Learning Max Pressure Control to Coordinate Traffic
                        Signals in Arterial Network.

        * Pravin Varaiya, 2013
            Max pressure control of a network of signalized intersections
    """
    def __init__(self, env_args,  mdp_args, n_actions):
        # Validate input arguments
        assert env_args.max_green > env_args.min_green, 'min_green must be lesser than max_green'

        self.min_green = env_args.min_green
        self.max_green = env_args.max_green
        self.yellow = env_args.yellow

        self.ts_type = 'max_pressure'
        # controller configuration parameters
        if mdp_args.feature != 'pressure': raise ValueError('feature <> pressure')
        self.feature = 'pressure'

        if mdp_args.use_lanes: raise ValueError('use_lanes must be False')
        self.action_schema = mdp_args.action_schema
        self.n_actions = n_actions

    def act(self, state):
        """Decides to switch if the next phase's pressure is greater than the next

          Parameters:
          -----------
          * state: dict<str, tuple<int>>
                ts_id --> <green_phase_id, active_phase_time, phase_0, phase_1, ...>

          Returns:
          --------
          * actions: dict<str, int>
                ts_id --> 0: keep or 1: to change
        """
        actions = {}
        for ts_id, ts_state in state.items():
            current_phase, current_time, *pressure = ts_state
            current_pressure = pressure[current_phase - 1]

            if current_time < self.yellow + self.min_green:
                actions[ts_id] = 0
            elif current_time == self.max_green:
                if self.action_schema == 'next':
                    actions[ts_id] = 1
                else:
                    actions[ts_id] =  int(np.argmax(pressure))
                    if actions[ts_id] == current_phase:
                        actions[ts_id] = int(argmax2(pressure))
            else:
                if self.action_schema == 'next':
                    next_phase = current_phase % len(pressure) 
                    actions[ts_id] = \
                        int(pressure[next_phase] > pressure[current_phase])
                elif self.action_schema == 'set':
                    if max(pressure) == current_pressure:
                        actions[ts_id] = int(current_phase - 1)
                    else:
                        # one or more phases may share the same pressure.
                        # ties are broken arbitrarely.
                        actions[ts_id]  = int(np.random.choice(argmax_k(pressure)))
        return actions

class Random:
    """Randomly selects a phase. """
    def __init__(self, env_args,  n_actions):

        assert env_args.max_green > env_args.min_green, 'min_green must be lesser than max_green'

        self.min_green = env_args.min_green
        self.max_green = env_args.max_green
        self.yellow = env_args.yellow

        self.ts_type = 'random'
        self.n_actions = n_actions


    def act(self, state):
        actions = {}
        for ts_id, ts_state in state.items():
            current_phase, current_time = ts_state[:2]
            choice = exclude(ts_state[2:], current_phase - 1) if current_time == self.max_green else self.n_actions
            actions[ts_id] = int(np.random.choice(choice))
        return actions


class Static:
    """Fixed signal plan.

        * Distributes the cycle_time evenly accoring to the phases' capacity.
    """
    def __init__(self, env_args, mdp_args, config_folder, cycle=90):
        self.min_green = env_args.min_green
        self.max_green = env_args.max_green
        self.yellow = env_args.yellow
        if self.yellow > 0: raise NotImplementedError
        self.ts_type = 'static'

        phf = mdp_args.phases_filter
        self.action_schema = mdp_args.action_schema
        if self.action_schema != 'set': raise NotImplementedError

        # shares cycle according to capacities.
        with open(config_folder / 'roadnet.json', 'r') as f:
            roadnet = json.load(f)
        capacities = get_capacity_from_roadnet(roadnet)
        capacities = {
            tl: {ph: cap for ph, cap in capacity.items() if ph in phf}
            for tl, capacity in capacities.items()
        }
        # assign phase_times according to capacities 
        cycles = {
            tl: {ph: int(cycle * (cap / sum(capacity.values()))) for ph, cap in capacity.items()}
            for tl, capacity in capacities.items()
        }
        # round phase_times according to reach cycle
        for tl, phase_cycles in cycles.items():
            res = cycle - sum(phase_cycles.values())
            max_time = max(phase_cycles.values())
            for ph, ph_time in phase_cycles.items():
                if res > 0 and ph_time == max_time:
                    cycles[tl][ph] += 1
                    res -= 1
        self.plans = cycles
        self.active_times = {
            tl: (0, 0) for tl in self.plans
        }
        
    def act(self, timestep):
        actions = {}
        for ts, phase_time in self.active_times.items():
            
            ph, time = phase_time
            # a signal plan might change (Adaptive Webster)
            if (timestep - time) >= self.plans[ts][ph]:
                next_phase = (ph + 1) % len(self.plans[ts])
                actions[ts] =  next_phase
                self.active_times[ts] = (next_phase, timestep)
        return actions


class Webster(Static):
    """Adaptive rule-based controller based on Webster's rule.

        Reference:
        ----------
        * Wade Genders and Saiedeh Razavi, 2019
            An Open Source Framework for Adaptive Traffic Signal Control.

    """
    def __init__(self, env_args, mdp_args, config_folder, cycle_start=90, min_cycle=60, max_cycle=120, aggregation_period=60):

        super(Webster, self).__init__(env_args, mdp_args, config_folder, cycle=cycle_start)
        self.ts_type = 'webster'

        # shares cycle according to capacities.
        with open(config_folder / 'roadnet.json', 'r') as f:
            roadnet = json.load(f)
        capacities, _ = get_capacity_from_roadnet2(roadnet)
        self.capacities = {
            tl: {ph: cap for ph, cap in capacity.items() if ph in mdp_args.phases_filter}
            for tl, capacity in capacities.items()
        }
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.aggregation_period = aggregation_period
        # websters missing time: yellow * n_phases
        self.missing = self.yellow * len(next(iter(self.capacities.values())))
        self.vehicles = defaultdict(list)

    def act(self, timestep):
        """Decides next phase based on the plan."""
        self.timestep = timestep
        return super(Webster, self).act(timestep)

    def update(self, vehicles):
        """Gathers flow data -- updates signal plans accordingly."""

        if (self.timestep + 1) % self.aggregation_period == 0:

            def ratio(ln, cap): return len(set(self.vehicles[ln])) / cap

            def max_ratio(lncaps):
                return max([ratio(ln, cap) for ln, cap in lncaps])

            def clip(x): return min(max(self.min_cycle, x), self.max_cycle)
            # change signal plan -- according to websters.
            for tl, capacity in self.capacities.items():
                # saturation ratios
                srs = {
                    ph: max_ratio(lncaps) for ph, lncaps in capacity.items()
                }
                # under saturated conditions
                if sum(srs.values()) < 1:
                    cycle_time = ((1.5 * self.missing) + 5) / (1 - sum(srs.values()))
                    cycle_time = clip(cycle_time)
                else:
                    cycle_time = self.min_cycle
                green_time = cycle_time - self.missing

                self.plans[tl]  = {
                    ph: int(green_time * sr / sum(srs.values())) if
                        sum(srs.values()) > 0 else (green_time / len(srs))
                    for ph, sr in srs.items()
                }
                    
                # allocates left-over green time
                res = cycle_time - sum(self.plans[tl].values())
                while res > 0:
                    for ph, ph_time in self.plans[tl].items():
                        if res > 0 and ph_time == max(self.plans[tl].values()):
                            self.plans[tl][ph] += 1
                            res -= 1
            # change vehicles.
            self.vehicles = vehicles
        else:
            for ln, vehs in vehicles.items():
                self.vehicles[ln] += vehs
