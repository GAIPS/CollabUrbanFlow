'''Environment: Wrapper around engine and feature (delay) producer.

    * Converts microsimulator data into features.
    * Keeps the agent's view from the traffic light.
    * Converts agent's actions into control actions.
    * Observes traffic data and transforms into features.
    * Logs past observations

'''
from functools import lru_cache
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from utils.file_io import engine_create, engine_load_config

def g_list(): return []
def g_dict(): return defaultdict(g_list)

def get_environment(network, episode_timesteps=3600, seed=0, thread_num=4):
    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 

    return  EnvironmentGymWrapper(roadnet, eng, episode_timesteps=episode_timesteps)

class Environment(object):
    def __init__(self,  roadnet, engine=None, yellow=5, min_green=5, max_green=90, step_size=5, **kwargs):
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

        roads = roadnet['roads']
        intersections = [intr for intr in roadnet['intersections'] if not intr['virtual']]
        for intersection in intersections:
            lightphases = intersection['trafficLight']['lightphases']
            phases_per_edges = {}
            edges_max_speeds = {}
            p = 0
            for linkids in lightphases:
                if any(linkids['availableRoadLinks']):
                    linkids = linkids['availableRoadLinks']
                    edges = []
                    for linkid in linkids:
                        # startRoad should be the incoming links.
                        edgeid = intersection['roadLinks'][linkid]['startRoad']
                        lanes = [lane for road in roads if road['id'] == edgeid
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

    @lru_cache(maxsize=1)
    def _get_lane_vehicles(self, timestep):
        return self.engine.get_lane_vehicles()

    @property
    def speeds(self):
        return self._get_vehicle_speed(self.timestep)

    @lru_cache(maxsize=1)
    def _get_vehicle_speed(self, timestep):
        return self.engine.get_vehicle_speed()

    def reset(self):
        self._active_phases = {tl_id: (0, 0) for tl_id in self.tl_ids}
        for tl_id in self.tl_ids:
            self.engine.set_tl_phase(tl_id, 0)
        self.engine.reset()

    @property
    def observations(self):
        return self._observations(self.timestep)

    @lru_cache(maxsize=1)
    def _observations(self, timestep):
        active_phases = self._update_active_phases()
        features = self._update_features()
        return {_id: active_phases[_id] + features[_id] for _id in self.tl_ids}

    # TODO: include switch
    def _update_active_phases(self):
        for tl_id, internal  in self._active_phases.items():
            active_phase, active_time = internal

            active_time += self.step_size if self.timestep > 0 else 0 
            self._active_phases[tl_id] = (active_phase, active_time)
        return self._active_phases

    def _update_features(self):
        observations = {}

        ids = self.vehicles
        vels = self.speeds

        for tl_id, phases  in self.phases.items():
            delays = []
            max_speeds = self.max_speeds[tl_id]
                
            for phs, edges in phases.items():
                phase_delays = []
                for edge in edges:
                    max_speed = max_speeds[edge]
                    edge_vels = [vels[idv] for idv in ids[edge]]
                    phase_delays += [delay(vel / max_speed) for vel in edge_vels]
                delays.append(float(round(sum(phase_delays), 4)))
            observations[tl_id] = tuple(delays)
        return observations


    def loop(self, num_steps):
        # Before
        self.reset()
        for eps in tqdm(range(num_steps)):
            if self.is_decision_step:
                actions = yield self.observations
            else:
                yield
            self.step(actions)
        return 0
        
    
    def step(self, actions={}):
        # Handle controller actions
        # KEEP or SWITCH phase
        # Maps agent action to controller action
        # G -> Y -> G -> Y
        if self.is_decision_step: self._phase_ctl(actions)
        self.engine.next_step()

    """Performs phase control""" 
    def _phase_ctl(self, actions):
        for tl_id, active_phases in self._active_phases.items():
            phases = self.phases[tl_id]
            current_phase, current_time = active_phases
            current_action = actions[tl_id]
            phase_ctrl = None
            if current_time == self.yellow and self.timestep > 5:
                # transitions to green
                phase_ctrl = current_phase * 2

            elif (current_time > self.yellow + self.min_green and current_action == 1) or \
                    (current_time == self.max_green):
                # transitions to yellow
                phase_ctrl = (current_phase * 2 + 1) % (2 * len(phases))

                # adjust log
                next_phase = (current_phase + 1) % len(phases) 
                self._active_phases[tl_id] = (next_phase, 0)

            if phase_ctrl is not None:
                self.engine.set_tl_phase(tl_id, phase_ctrl)

class EnvironmentGymWrapper(Environment):
    ''' Makes it behave like a gym environment'''

    def __init__(self, *args, **kwargs):
        super(EnvironmentGymWrapper, self).__init__(*args, **kwargs)

        self.decision_step = 5 
        if 'episode_timesteps' not in kwargs:
            raise ValueError('episode timesteps must be provided')
        else:
            self.episode_timesteps = kwargs['episode_timesteps']
        self.info_dict = g_dict()

    def step(self, actions):
        if not isinstance(actions, dict):
            if len(self.tl_ids) == 1:
                actions = {'247123161': actions}
            else:
                raise NotImplementedError

        for _ in range(self.decision_step):
            super(EnvironmentGymWrapper, self).step(actions)

        new_state = self.observations
        reward = {
            tl_id: float(np.round(-sum(observ[2:]), 4))
            for tl_id, observ in  new_state.items() 
        }

        done = self.timestep >= self.episode_timesteps
        self.log(actions)
        return new_state, reward, done, None

    def reset(self):
        super(EnvironmentGymWrapper, self).reset()
        # Case its a single intersection: simplify outputs
        return self.observations

    def log(self, actions):
        r_next = {_id: -float(sum(_obs[2:])) for _id, _obs in self.observations.items()}

        sum_speeds = sum(([float(vel) for vel in self.speeds.values()]))
        num_vehicles = len(self.speeds)
        self.info_dict["rewards"].append(r_next)
        self.info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
        self.info_dict["vehicles"].append(num_vehicles)
        self.info_dict["observation_spaces"].append(self.observations) # No function approximation.
        self.info_dict["actions"].append({k: int(v) for k, v in actions.items()})
        self.info_dict["states"].append(self.observations)

""" features computation """
# TODO: Compute features from data seperately
def delay(x):
    return np.exp(-5 * x)
