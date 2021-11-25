'''Environment: Wrapper around engine and feature producer.

    * Converts microsimulator data into features.
    * Keeps the agent's view from the traffic light.
    * Converts agent's actions into control actions.
    * Observes traffic data and transforms into features.
    * Logs past observations
    * Produces features: Delay and pressure.
    
    Limitations:
    ------------
    * Supports intersections with fixed number of phases.

'''
from functools import lru_cache
from collections import defaultdict
import os

import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
from functools import cached_property

from features import compute_delay, compute_pressure
from utils.network import get_phases
from utils.file_io import engine_create, engine_load_config

FEATURE_CHOICE = ('delay', 'pressure')

def simple_hash(x): return hash(x) % (11 * 255)

def get_environment(network, episode_timesteps=3600, seed=0, thread_num=4):
    eng = engine_create(network, seed=seed, thread_num=thread_num)
    config, flows, roadnet = engine_load_config(network) 

    return Environment(network, roadnet, eng, episode_timesteps=episode_timesteps)

class Environment(object):
    def __init__(self,
                 network, 
                 roadnet,
                 engine=None,
                 yellow=5,
                 min_green=5,
                 max_green=90,
                 step_size=5,
                 feature='delay',
                 episode_timesteps=-1,
                 emit=False,
                 **kwargs):
        '''Environment constructor method.
            Params:
            -------
            intersection: dict
                An non virtual intersection from roadnet.json file.agent.model

            engine: cityflow.Engine object
                The microsimulator engine

            Returns:
            --------
            Converter object
        '''
        # Network id
        self.network = network
        # Signal plans regulation
        self.yellow = yellow
        self.min_green = min_green
        self.max_green = max_green
        self.step_size = step_size

        # Roadnet
        self._inc, self._out, self._lim = get_phases(roadnet, filter_phases=[0, 1, 2, 3])

        # Loop control
        self._episode_timestep = episode_timesteps

        # Emissions
        self.emit = emit
        self._emissions = []
        self.info_dict = defaultdict(list)

        if feature not in FEATURE_CHOICE:
            raise ValueError(f'feature {feature} must be in {FEATURE_CHOICE}')
        self.feature = feature


        if engine is not None: self.engine = engine

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def is_observation_step(self):
        return self.timestep % self.step_size == 0

    @property
    def is_update_step(self):
        return self.timestep % 10 == 0

    @property
    def timestep(self):
        return int(self.engine.get_current_time())

    @property
    def done(self):
        if (self._episode_timestep == -1): return False
        return self.timestep >= self._episode_timestep

    @property
    def emissions(self):
        return self._emissions

    @cached_property
    def tl_ids(self):
        return sorted(self.phases.keys())

    @cached_property
    def phases(self): return self._inc

    @cached_property
    def num_phases(self):
        # It should be fixed
        assert len(set([len(phase) for phase in self.phases.values()])) == 1
        return len(next(iter(self.phases.values())))

    @cached_property
    def incoming_roadlinks(self): return self._inc

    @cached_property
    def outgoing_roadlinks(self): return self._out

    @cached_property
    def max_speeds(self): return self._lim

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
        # self._emissions = [] 
        self._active_phases = {tl_id: (1, 0) for tl_id in self.tl_ids}
        for tl_id in self.tl_ids:
            self.engine.set_tl_phase(tl_id, 0)
        self.engine.reset()
        if self.emit: self._update_emissions()
        return self.observations

    @property
    def observations(self):
        return self._observations(self.timestep)

    @lru_cache(maxsize=1)
    def _observations(self, timestep):
        active_phases = self._update_active_phases()
        features = self._update_features()
        return {_id: active_phases[_id] + features[_id] for _id in self.tl_ids}

    @property
    def reward(self):
        return {_id: -float(sum(_obs[2:])) for _id, _obs in self.observations.items()}

    # TODO: include switch
    def _update_active_phases(self):
        for tl_id, internal in self._active_phases.items():
            active_phase, active_time = internal

            active_time += self.step_size if self.timestep > 0 else 0
            self._active_phases[tl_id] = (active_phase, active_time)
        return self._active_phases

    def _update_features(self):
        if self.feature == 'delay':
            return compute_delay(self.phases, self.vehicles,
                                 self.speeds, self.max_speeds)
        return compute_pressure(self.incoming_roadlinks,
                                self.outgoing_roadlinks, self.vehicles)


    def _update_emissions(self):
        """Builds sumo like emission file"""
        if not any(self.emissions) or self.timestep > self.emissions[-1]['time']:
            for veh_id in self.engine.get_vehicles(include_waiting=False):
                data = self.engine.get_vehicle_info(veh_id)

                self._emissions.append({
                    'time': self.timestep,
                    'id': veh_id,
                    'lane': data['drivable'],
                    'pos': float(data['distance']),
                    'route': simple_hash(data['route']),
                    'speed': float(data['speed']),
                    'type': 'human',
                    'x': 0,
                    'y': 0
                })

    def loop(self, num_steps):
        # Before
        self.reset()
        experience = self.observations, self.reward, self.done, None
        for eps in tqdm(range(num_steps)):
            if self.is_update_step:
                actions = yield experience
            else:
                yield
            experience = self.step(actions)
        return 0

    def step(self, actions={}):
        # Handle controller actions
        # KEEP or SWITCH phase
        # Maps agent action to controller action
        # G -> Y -> G -> Y
        if self.is_update_step and self.timestep > 5: self.log(actions)
        if self.is_observation_step:
            self._phase_ctl(actions)
        self.engine.next_step()
        if self.emit: self._update_emissions()
        if self.is_observation_step:
            return self.observations, self.reward, self.done, None
        return None

    """Performs phase control"""
    def _phase_ctl(self, actions):
        for tl_id, active_phases in self._active_phases.items():
            phases = self.phases[tl_id]
            current_phase, current_time = active_phases
            current_action = actions[tl_id]
            phase_ctrl = None
            if (self.yellow > 0 and current_time == self.yellow) and self.timestep > 5:
                # transitions to green: y -> G
                phase_ctrl = 2 * (current_phase - 1)

            elif (current_time >= self.yellow + self.min_green and current_action == 1) or \
                    (current_time == self.max_green):
                # transitions to yellow: G -> y
                # if yellow is zero; go to next green.
                phase_ctrl = 2 * (current_phase - 1) + int(self.yellow > 0)

                # adjust log
                next_phase = (current_phase % len(phases))
                self._active_phases[tl_id] = (next_phase + 1, 0)

            if phase_ctrl is not None:
                # phase_ctrl 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0 
                # phase      1 -> 2 -> 2 -> 3 -> 3 -> 4 -> 4 -> 1 -> 1
                self.engine.set_tl_phase(tl_id, phase_ctrl)

    def log(self, actions):

        sum_speeds = sum(([float(vel) for vel in self.speeds.values()]))
        num_vehicles = len(self.speeds)
        self.info_dict["rewards"].append(self.reward)
        self.info_dict["velocities"].append(0 if num_vehicles == 0 else sum_speeds / num_vehicles)
        self.info_dict["vehicles"].append(num_vehicles)
        self.info_dict["observation_spaces"].append(self.observations) # No function approximation.
        self.info_dict["actions"].append({k: int(v) for k, v in actions.items()})
        self.info_dict["states"].append(self.observations)
        self.info_dict["timesteps"].append(self.timestep)


