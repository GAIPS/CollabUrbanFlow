'''Delay: decreasing exponential with respect to maximum speed.

    * Converts microsimulator data into features.
    * Controls the agent's view from the traffic light.
    * Observes traffic data and transforms into features.
    * Logs past observations

'''
from copy import deepcopy
import numpy as np

def make_initial_state(phases):
    return {
        tl_id: tuple([0] * (len(tl_phases) + 2))
        for tl_id, tl_phases in phases.items()
    }

class DelayConverter(object):
    def __init__(self,  roadnet, engine=None, yellow=5, min_green=5, max_green=90):
        '''DelayCoverter constructor method.
            TODO: fill dictionary
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

    def reset(self):
        self.log = {}
        self.log[0] = make_initial_state(self.phases)
        
    #TODO: Make a decorator to log.
    # Call this right before agent#act
    def convert(self):
        state = {}
        exclude_actions = set({})

        step_counter = int(self.engine.get_current_time())
        ids = self.engine.get_lane_vehicles()
        vels = self.engine.get_vehicle_speed() 
        min_green = self.min_green
        max_green  = self.max_green
        yellow = self.yellow

        # Obtain s_prev
        t_prev, s_prev = max(self.log.items(), key=lambda x: x[0])
         

        # Enforce actions constraints.
        # 1) Prevent switching to a new phase before min_green.
        # 2) Prevent keeping a phase after max_green.
        def keep(x):
            return int(s_prev[x][1]) <= (min_green + yellow)
        def switch(x):
            return int(s_prev[x][1]) >= (max_green + yellow)

        for tl_id, phases  in self.phases.items():
            delays = []
            max_speeds = self.max_speeds[tl_id]
            active_phase, active_time = s_prev[tl_id][:2] 

            # Adjust to active condition.
            if keep(tl_id) or switch(tl_id): 
                exclude_actions = exclude_actions.union(set({tl_id}))

                if switch(tl_id):
                    active_phase = (active_phase + 1) % len(phases)
                    active_time  = 0
                else:
                    active_time += step_counter - t_prev
            else:
                active_time += step_counter - t_prev
                
            for phs, edges in phases.items():
                phase_delays = []
                for edge in edges:
                    max_speed = max_speeds[edge]
                    edge_vels = [vels[idv] for idv in ids[edge]]
                    phase_delays += [delay(vel / max_speed) for vel in edge_vels]
                delays.append(np.round(sum(phase_delays), 4))
            state[tl_id] = (active_phase, active_time) + tuple(delays)
        self.log[step_counter] = deepcopy(state)
        return state, exclude_actions

    # Call this right after agent#act
    # updates traffic conditions due to agent action.
    def update(self, actions):
        # Obtain s_prev
        step_counter = int(self.engine.get_current_time())
        s_prev = self.log[step_counter]  # step_counter key must exist.
        for tl_id, action in actions.items():
            if action != s_prev[tl_id][0]:
                s_prev[tl_id] = (action, 0) + s_prev[tl_id][2:]
        self.log[step_counter] = s_prev


def delay(x):
    return np.exp(-5 * x)

