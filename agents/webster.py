import copy

import numpy as np


class WEBSTER:
    """
        Adaptive webster method.
    """

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

    def __init__(self, env, cycle_time=60, aggregation_period=300):

        self._ts_type = 'webster'

        # Internalise parameters.
        self._aggregation_period = aggregation_period
        self._cycle_time = cycle_time
        # self._tls_phases = tls_phases

        self._tls_phases = env.phases

        # Initialise vehicles counts data structure.
        self._vehicles_counts = self.build_vehicles_counts(self._tls_phases)

        # Uncomment below for (static) Webster timings calculation.
        self._global_counts = self.build_vehicles_counts(self._tls_phases)

        # Calculate uniform timings.
        self._uniform_timings = {}
        for tid in self._tls_phases:
            timings = []

            num_phases = len(self._tls_phases[tid])

            # Calculate ratios.
            ratios = [1/num_phases for p in range(num_phases)]

            # Calculate phases durations given allocation ratios.
            phases_durations = [np.around(r*(cycle_time-6.0*num_phases)) for r in ratios]

            # Calculate timings.
            counter = 0
            timings = []
            for p in range(num_phases):
                timings.append(counter + phases_durations[p])
                timings.append(counter + phases_durations[p] + 6.0)
                counter += phases_durations[p] + 6.0

            timings[-1] = self._cycle_time
            timings[-2] = self._cycle_time - 6.0

            self._uniform_timings[tid] = timings

        self._webster_timings = copy.deepcopy(self._uniform_timings) # Initialize webster with uniform timings.
        self._next_signal_plan = copy.deepcopy(self._uniform_timings)

        # Internal counter.
        self._time_counter = 1

    @property
    def ts_type(self):
        return self._ts_type

    def act(self, veh_count_per_lane):

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

                num_phases = len(max_counts)

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
                    phases_durations = [np.around(r*(self._cycle_time-6.0*num_phases)) for r in ratios]

                    # Calculate timings.
                    counter = 0
                    timings = []
                    for p in range(num_phases):
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
                    phases_durations = [np.around(r*(self._cycle_time-6.0*num_phases)) for r in ratios]

                    # Calculate timings.
                    counter = 0
                    timings = []
                    for p in range(num_phases):
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

        return self._webster_timings

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

            num_phases = len(max_counts)

            # Calculate ratios.
            ratios = [p/sum(max_counts) for p in max_counts]

            # Calculate phases durations given allocation ratios.
            phases_durations = [np.around(r*(self._cycle_time-6.0*num_phases)) for r in ratios]

            # Calculate timings.
            counter = 0
            timings = []
            for p in range(num_phases):
                timings.append(counter + phases_durations[p])
                timings.append(counter + phases_durations[p] + 6.0)
                counter += phases_durations[p] + 6.0

            timings[-1] = self._cycle_time
            timings[-2] = self._cycle_time - 6.0

            global_timings[tls_id] = timings

        print('Global timings (Webster method):', global_timings)
