"""Implementation of classic adaptive controllers and methods
    * Reactive agents
"""
import copy

import numpy as np


class MaxPressure:
    """Adaptive rule based controller based of OSFATSC

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
    def __init__(self, min_green, max_green, yellow):
        # Validate input arguments
        assert max_green > min_green, 'min_green must be lesser than max_green'

        # controller configuration parameters
        self.ts_type = 'max_pressure'
        self.min_green = min_green
        self.max_green = max_green
        self.yellow = yellow
        self.feature = 'pressure'

    def act(self, state):
        """Decides to switch if the next phase's pressure is greater than the next

          Parameters:
          ----------
          * state: dict<str, tuple<int>>
                ts_id --> <green_phase_id, active_phase_time, phase_0, phase_1, ...>

          Returns:
          * actions: dict<str, int>
                ts_id --> 0: keep or 1: to change
        """

        actions = {}
        for ts_id, ts_state in state.items():
            current_phase, current_time = ts_state[:2]
            if current_time < self.yellow + self.min_green:
                actions[ts_id] = 0
            elif current_time == self.max_green:
                actions[ts_id] = 1
            else:
                next_phase = (current_phase + 1) % (len(ts_state) - 2)
                actions[ts_id] = \
                    int(ts_state[2:][next_phase] > ts_state[2:][current_phase])
        return actions
