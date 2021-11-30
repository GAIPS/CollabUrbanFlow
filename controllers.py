"""Implementation of classic adaptive controllers and methods
    * Reactive agents
"""
import copy
import json
import random

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
        self.step_size = 5

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


class Random:

    def __init__(self, config_folder):
        self.ts_type = 'random'
        self.feature = 'delay'
        self.yellow = 5
        self.step_size = 5

        with open(config_folder / 'roadnet.json', 'r') as f:
            network = json.load(f)
        self.intersections = [item for item in network['intersections'] if not item['virtual']]

    def act(self, state):
        actions = {}
        for i, intersection in enumerate(self.intersections):
            phases = [inter for inter in intersection['trafficLight']['lightphases'] if
                      len(inter['availableRoadLinks']) > 0]
            actions[intersection["id"]] = random.choice(range(len(phases)))
        return actions


class Static:

    def __init__(self, config_folder):
        self.ts_type = 'static'
        self.feature = 'delay'
        self.yellow = 6
        self.step = 0
        self.step_size = 1

        with open(config_folder / 'roadnet.json', 'r') as f:
            network = json.load(f)

        self.intersections = [item for item in network['intersections'] if not item['virtual']]
        self.phasectl = [(-1, 0)] * len(self.intersections)

    def act(self, state):
        actions = {}
        for i, intersection in enumerate(self.intersections):
            phaseid, next_change = self.phasectl[i]
            tl = [inter for inter in intersection['trafficLight']['lightphases'] if
                  len(inter['availableRoadLinks']) > 0]
            if next_change == self.step:
                actions[intersection['id']] = 1
                phaseid = (phaseid + 1) % len(tl)

                yellow = self.yellow if self.step != 0 else 0
                next_change = self.step + tl[phaseid]['time'] + yellow
                self.phasectl[i] = (phaseid, next_change)
            else:
                actions[intersection['id']] = 0
        self.step += 1
        return actions



class Webster:
    """
        Adaptive webster method.
    """

    def __init__(self, config_folder, cycle_time=60, aggregation_period=600, yellow=6, env=None):

        self._ts_type = 'webster'
        self.yellow = yellow
        self.feature = 'delay'
        self.step_size = 1
        with open(config_folder / 'roadnet.json', 'r') as f:
            network = json.load(f)


        self.intersections = [item for item in network['intersections'] if not item['virtual']]
        self.phasectl = [(-1, 0)] * len(self.intersections)

        # Internalise parameters.
        self._aggregation_period = aggregation_period
        self._cycle_time = cycle_time

        #Otherwise set_tls_phases(tls_phases) needs to be called after init
        if env is not None:
            self.set_env(env)
            self.init_data()


        # Internal counter.
        self._time_counter = 1

    def init_uniform_timings(self):
        # Calculate uniform timings.
        self._uniform_timings = {}
        for tid in self._tls_phases:
            timings = []

            n_phases = len(self._tls_phases[tid])

            # Calculate ratios.
            ratios = [1 / n_phases for p in range(n_phases)]

            # Calculate phases durations given allocation ratios.
            phases_durations = [np.around(r * (self._cycle_time - 6.0 * n_phases)) for r in ratios]

            # Calculate timings.
            counter = 0
            timings = []
            for p in range(n_phases):
                timings.append(counter + phases_durations[p])
                timings.append(counter + phases_durations[p] + 6.0)
                counter += phases_durations[p] + 6.0

            timings[-1] = self._cycle_time
            timings[-2] = self._cycle_time - 6.0

            self._uniform_timings[tid] = timings
        self._webster_timings = copy.deepcopy(self._uniform_timings)  # Initialize webster with uniform timings.
        self._next_signal_plan = copy.deepcopy(self._uniform_timings)

    def init_vehicles(self):
        # Initialise vehicles counts data structure.
        self._vehicles_counts = self.build_vehicles_counts(self._tls_phases)
        # Uncomment below for (static) Webster timings calculation.
        self._global_counts = self.build_vehicles_counts(self._tls_phases)

    def set_env(self, env):
        self.env = env
        self.set_tls_phases()
        self.init_data()

    def set_tls_phases(self):
        self._tls_phases = self.env.phases

    def init_data(self):
        self.init_vehicles()
        self.init_uniform_timings()

    def build_vehicles_counts(self, cf_phases):
        vehicles_counts = {}
        for tid in cf_phases.keys():
            for index, lanes in cf_phases[tid].items():
                for lane in lanes:
                    lane_name = lane.split("_")[0]
                    lane_index = lane.split("_")[1]
                    if tid not in vehicles_counts:
                        vehicles_counts[tid] = {}
                    if index not in vehicles_counts[tid]:
                        vehicles_counts[tid][index] = {}
                    if lane_name not in vehicles_counts[tid][index]:
                        vehicles_counts[tid][index][lane_name] = {}
                    vehicles_counts[tid][index][lane_name][lane_index] = []

        return vehicles_counts

    def update_vehicle_count(self, veh_count_per_lane):
        for tid in self._vehicles_counts:
            for phase in self._vehicles_counts[tid]:
                for road in self._vehicles_counts[tid][phase]:
                    for lane in self._vehicles_counts[tid][phase][road]:
                        lane_name = road + "_" + lane
                        veh_count = veh_count_per_lane[lane_name]
                        if len(veh_count) > 0:
                            for vehicle in veh_count:
                                if vehicle not in self._vehicles_counts[tid][phase][road][lane]:
                                    self._vehicles_counts[tid][phase][road][lane].append(vehicle)

                                if vehicle not in self._global_counts[tid][phase][road][lane]:
                                    self._global_counts[tid][phase][road][lane].append(vehicle)

    @property
    def ts_type(self):
        return self._ts_type



    def act(self, state):
        veh_count_per_lane = self.env.vehicles
        self.update_vehicle_count(veh_count_per_lane)

        if (self._time_counter % self._aggregation_period == 0) and self._time_counter > 1:
            # Calculate new signal plan.

            for tls_id in self._vehicles_counts.keys():
                max_counts = []
                for p in self._vehicles_counts[tls_id].keys():
                    max_count = -1
                    for edge in self._vehicles_counts[tls_id][p].keys():
                        for l in self._vehicles_counts[tls_id][p][edge].keys():
                            lane_count = len(self._vehicles_counts[tls_id][p][edge][l])
                            max_count = max(max_count, lane_count)
                    max_counts.append(max_count)

                n_phases = len(max_counts)

                if min(max_counts) < 2:

                    # Use global counts to calculate timings.
                    max_counts = []
                    for p in self._global_counts[tls_id].keys():
                        max_count = -1
                        for edge in self._global_counts[tls_id][p].keys():
                            for l in self._global_counts[tls_id][p][edge].keys():
                                lane_count = len(self._global_counts[tls_id][p][edge][l])
                                max_count = max(max_count, lane_count)
                        max_counts.append(max_count)

                    # Calculate ratios.
                    ratios = [p/sum(max_counts) for p in max_counts]

                    # Calculate phases durations given allocation ratios.
                    phases_durations = [np.around(r*(self._cycle_time-6.0*n_phases)) for r in ratios]

                    # Calculate timings.
                    counter = 0
                    timings = []
                    for p in range(n_phases):
                        timings.append(counter + phases_durations[p])
                        timings.append(counter + phases_durations[p] + 6.0)
                        counter += phases_durations[p] + 6.0

                    timings[-1] = self._cycle_time
                    timings[-2] = self._cycle_time - 6.0

                    self._next_signal_plan[tls_id] = timings

                else:
                    # Use counts from the aggregation period to calculate timings.

                    # Calculate ratios.
                    ratios = [p/sum(max_counts) for p in max_counts]

                    # Calculate phases durations given allocation ratios.
                    phases_durations = [np.around(r*(self._cycle_time-6.0*n_phases)) for r in ratios]

                    # Calculate timings.
                    counter = 0
                    timings = []
                    for p in range(n_phases):
                        timings.append(counter + phases_durations[p])
                        timings.append(counter + phases_durations[p] + 6.0)
                        counter += phases_durations[p] + 6.0

                    timings[-1] = self._cycle_time
                    timings[-2] = self._cycle_time - 6.0

                    self._next_signal_plan[tls_id] = timings

            # Reset counters.
            self._reset_counts()

        if (self._time_counter % self._cycle_time == 0) and self._time_counter > 1:
            # Update current signal plan.
            self._webster_timings = copy.deepcopy(self._next_signal_plan)

        # Increment internal counter.
        self._time_counter += 1

        # Transform timings to keep/change action based on current engine timestep
        timings = {intersection: [timing[0], timing[2] - timing[1]] for intersection, timing in
                   self._webster_timings.items()}  # Ignore yellows

        if (self.env.timestep + 1) % self._aggregation_period == 0:  # When plan changes recalculate next shift
            for i, intersection in enumerate(self.env.phases):
                self.phasectl[i] = (0, self.last_step + int(timings[intersection][0]) + self.yellow)

        # Apply webster timings to keep/change structure
        actions = {}
        for i, intersection in enumerate(self.env.phases):
            phaseid, next_change = self.phasectl[i]
            tl = {phaseid: int(timings[intersection][int(phaseid)]) for phaseid in self.env.phases[intersection]}
            if next_change == self.env.timestep:
                actions[intersection] = 1
                phaseid = (phaseid + 1) % len(tl)

                yellow = self.yellow if self.env.timestep != 0 else 0
                next_change = self.env.timestep + tl[phaseid] + yellow
                self.last_step = self.env.timestep
                self.phasectl[i] = (phaseid, next_change)
            else:
                actions[intersection] = 0

        return actions

    def _reset_counts(self):
        self._vehicles_counts = self.build_vehicles_counts(self._tls_phases)

    def terminate(self):
        self._reset_counts()

        # Uncomment below for (static) Webster timings calculation.
        global_timings = {}

        for tls_id in self._global_counts.keys():
            max_counts = []
            for p in self._global_counts[tls_id].keys():
                max_count = -1
                for edge in self._global_counts[tls_id][p].keys():
                    for l in self._global_counts[tls_id][p][edge].keys():
                        lane_count = len(self._global_counts[tls_id][p][edge][l])
                        max_count = max(max_count, lane_count)
                max_counts.append(max_count)

            n_phases = len(max_counts)

            # Calculate ratios.
            ratios = [p/sum(max_counts) for p in max_counts]

            # Calculate phases durations given allocation ratios.
            phases_durations = [np.around(r*(self._cycle_time-6.0*n_phases)) for r in ratios]

            # Calculate timings.
            counter = 0
            timings = []
            for p in range(n_phases):
                timings.append(counter + phases_durations[p])
                timings.append(counter + phases_durations[p] + 6.0)
                counter += phases_durations[p] + 6.0

            timings[-1] = self._cycle_time
            timings[-2] = self._cycle_time - 6.0

            global_timings[tls_id] = timings

        print('Global timings (Webster method):', global_timings)
