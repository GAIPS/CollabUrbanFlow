'''Environment: Wrapper around engine and feature (delay) producer.

    * Converts microsimulator data into features.
    * Keeps the agent's view from the traffic light.
    * Converts agent's actions into control actions.
    * Observes traffic data and transforms into features.
    * Logs past observations

'''
import ipdb
from functools import lru_cache

from copy import deepcopy
import numpy as np
from cityflow import Engine

def make_initial_state(phases):
    return {
        tl_id: tuple([0] * (len(tl_phases) + 2))
        for tl_id, tl_phases in phases.items()
    }

class Environment(object):

    def __init__(self,  roadnet, engine=None, yellow=5, min_green=5, max_green=90, step_size=5):
        '''Environment constructor method.
            Params:
            -------
            intersection: dict
                An non virtual intersection from roadnet.json file.

            engine: cityflow.Engine object
                The microsimulator engine

            Returns:
            --------
            Converter object
        '''
        self.yellow = yellow
        self.min_green = min_green
        self.max_green = max_green
        self.step_size = step_size
        self.tl_ids = []
        self.phases = {}
        self.max_speeds = {}
        self.log = {}

        phases_per_edges = {}
        edges_max_speeds = {}
        roads = roadnet['roads']
        intersections = [intr for intr in roadnet['intersections'] if not intr['virtual']]
        for intersection in intersections:
            lightphases = intersection['trafficLight']['lightphases']
            p = 0
            for linkids in lightphases:
                if any(linkids['availableRoadLinks']):
                    linkids = linkids['availableRoadLinks']
                    edges = []
                    for linkid in linkids:
                        # startRoad should be the incoming links.
                        edgeid = intersection['roadLinks'][linkid]['startRoad']
                        lanes = [lane 
                            for road in roads if road['id'] == edgeid
                            for lane in road['lanes']]
                        num_lanes = len(lanes)
                        
                        for lid in range(num_lanes):
                            _edgeid = f'{edgeid}_{lid}'
                            if _edgeid not in edges:
                                edges.append(_edgeid)
                                edges_max_speeds[_edgeid] = lanes[lid]['maxSpeed']
                    phases_per_edges[p] = edges
                    p += 1
            self.phases[intersection['id']] = phases_per_edges 
            self.max_speeds[intersection['id']] = edges_max_speeds
            self.tl_ids.append(intersection['id']) 
        if engine is not None: self.engine = engine
        self.log[0] = make_initial_state(self.phases)

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def is_decision_step(self):
        return self.timestep % self.step_size == 0

    @property
    def timestep(self):
        return int(self.engine.get_current_time()) 


    """ Dynamic properties are cached""" 
    @property
    def vehicles(self):
        return self._get_lane_vehicles(self.timestep)


    #TODO: cache this!
    @lru_cache
    def _get_lane_vehicles(self, timestep):
        return self.engine.get_lane_vehicles()

    @property
    def speeds(self):
        return self._get_vehicle_speed(self.timestep)

    #TODO: cache this!
    @lru_cache
    def _get_vehicle_speed(self, timestep):
        return self.engine.get_vehicle_speed()

    def reset(self):
        # Agents' state
        self.log = {}
        self.log[0] = make_initial_state(self.phases)

        # Environment's state
        for tl_id in self.tl_ids:
            self.engine.set_tl_phase(tl_id, 0)
        self.engine.reset()
        
    #TODO: Make a decorator to log.
    def observe(self):
        observations = {}

        ids = self.vehicles
        vels = self.speeds
        min_green = self.min_green
        max_green  = self.max_green
        yellow = self.yellow

        # Obtain s_prev
        t_prev, s_prev = max(self.log.items(), key=lambda x: x[0])
         

        # Enforce actions constraints.
        # 1) Prevent switching to a new phase before min_green.
        # 2) Prevent keeping a phase after max_green.
        def keep(x):
            return int(s_prev[x][1]) <= (self.min_green + self.yellow)
        def switch(x):
            return int(s_prev[x][1]) >= (self.max_green + self.yellow)

        for tl_id, phases  in self.phases.items():
            delays = []
            max_speeds = self.max_speeds[tl_id]
            active_phase, active_time = s_prev[tl_id][:2] 

            # Adjust to active condition.
            if keep(tl_id) or switch(tl_id): 
                if switch(tl_id):
                    active_phase = (active_phase + 1) % len(phases)
                    active_time  = 0
                else:
                    active_time += self.timestep - t_prev
            else:
                active_time += self.timestep - t_prev
                
            for phs, edges in phases.items():
                phase_delays = []
                for edge in edges:
                    max_speed = max_speeds[edge]
                    edge_vels = [vels[idv] for idv in ids[edge]]
                    phase_delays += [delay(vel / max_speed) for vel in edge_vels]
                delays.append(round(float(sum(phase_delays)), 4))
            observations[tl_id] = (active_phase, active_time) + tuple(delays)
        self.log[self.timestep] = deepcopy(observations)
        return observations


    def step(self, actions={}):
        # Handle controller actions
        # KEEP or SWITCH phase
        # Maps agent action to controller action
        # G -> Y -> G -> Y
        if self.is_decision_step: 
            self._phase_ctl(actions)
        self.engine.next_step()

    """Performs phase control""" 
    def _phase_ctl(self, actions):
        controller_actions = {}
        _observations = self.log[self.timestep]  # self.timestep key must exist.
        for tlid, obs in _observations.items():
            phases = self.phases[tlid]
            current_phase, current_time = obs[:2]
            current_action = actions[tlid]
            if current_time == self.yellow and self.timestep > 5:
                # transitions to green
                controller_actions[tlid] = current_phase * 2

            elif (current_time > self.yellow + self.min_green and current_action == 1) or \
                    (current_time == self.max_green):
                # transitions to yellow
                controller_actions[tlid] = (current_phase * 2 + 1) % (2 * len(phases))

                # adjust log
                next_phase = (current_phase + 1) % len(phases) 
                self.log[self.timestep][tlid] = (next_phase, 0) + obs[:2]


        for tl_id, tl_phase_id in controller_actions.items():
            self.engine.set_tl_phase(tlid, tl_phase_id)

def delay(x):
    return np.exp(-5 * x)
