'''Environment: Wrapper around engine and feature producer.

    * Converts microsimulator data into features.
    * Keeps the agent's view from the traffic light.
    * Converts agent's actions into control actions.
    * Observes traffic data and transforms into features.
    * Logs past observations
    * Produces features: Delay, pressure and WAVE.
    
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

from features import compute_delay, compute_pressure, compute_wave
from utils.network import get_phases, get_lanes
from utils.utils import flatten2
from utils.file_io import engine_create, engine_load_config

FEATURE_CHOICE = ('delay', 'wave', 'pressure')
# Colight phase feature
FEATURE_PHASE = {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
}

def simple_hash(x): return hash(x) % (11 * 255)
def lval(x): return len(next(iter(x.values()))) 
def toph(x, y): return tuple(FEATURE_PHASE[int(x)] + [y])

def get_environment(network, episode_timesteps=3600, seed=0, thread_num=4):
    eng = engine_create(network, seed=seed, thread_num=thread_num)
    config, flows, roadnet = engine_load_config(network) 

    return Environment(network, roadnet, eng, episode_timesteps=episode_timesteps)

class Environment(object):
    def __init__(self,
                 network, 
                 roadnet,
                 env_args,
                 mdp_args, 
                 engine=None,
                 step_size=5,
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
        self.yellow = env_args.yellow
        self.min_green = env_args.min_green
        self.max_green = env_args.max_green
        self.step_size = step_size

        # mdp args
        # TODO: Implement `set` phase
        # TODO: Implement `use_lanes` False
        if mdp_args.feature not in FEATURE_CHOICE:
            raise ValueError(f'feature {feature} must be in {FEATURE_CHOICE}')
        if not mdp_args.action_schema in ('next', 'set'): raise NotImplementedError
        self.mdp_args = mdp_args
        self.feature = mdp_args.feature
        self.action_schema = mdp_args.action_schema
        self.phases_filter = mdp_args.phases_filter
        self.use_lanes = mdp_args.use_lanes


        # Roadnet
        inc, out, lim = get_phases(roadnet, phases_filter=self.phases_filter)
        self.phases_incoming = inc
        self.phases_outgoing = out
        self.max_speeds = lim

        inc, out  = get_lanes(roadnet)
        self.lanes_incoming = inc
        self.lanes_outgoing = out

        # Loop control
        self._episode_timestep = episode_timesteps

        # Emissions
        self.emit = emit
        self._emissions = []
        self.info_dict = defaultdict(list)

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
    def phases(self): return self.phases_incoming

    @cached_property 
    def lanes(self): return self.lanes_incoming

    @cached_property
    def n_phases(self):
        # They all should have the same number of phases.
        assert len(set([len(phase) for phase in self.phases.values()])) == 1
        return lval(self.phases)

    @cached_property
    def n_features(self): # <--> edges or phases
        if not self.use_lanes: return self.n_phases + 2
        # n_features are the number of roads per intersec.
        assert len(set([len(lanes) for lanes in self.lanes.values()])) == 1
        return lval(self.lanes) + lval(FEATURE_PHASE) + 1

    @cached_property
    def n_actions(self): # <--> edges or phases
        if self.action_schema == 'set': return self.n_phases 
        if self.action_schema == 'next': return 2
        raise NotImplementedError

        


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
        if self.use_lanes: # expand phases
            active_phases = {tl: toph(*phd) for tl, phd in active_phases.items()}
        features = self._update_features()
        return {tl: active_phases[tl] + features[tl] for tl in self.tl_ids}

    @property
    def reward(self):
        n_sum = 9 if self.use_lanes else 2
        return {tl: -float(sum(obs[n_sum:])) for tl, obs in self.observations.items()}

    # TODO: include next
    def _update_active_phases(self):
        for tl_id, internal in self._active_phases.items():
            active_phase, active_time = internal

            active_time += self.step_size if self.timestep > 0 else 0
            self._active_phases[tl_id] = (active_phase, active_time)
        return self._active_phases

    def _update_features(self):
        if self.feature == 'delay':
            if self.use_lanes:
                return compute_delay(self.lanes, self.vehicles,
                                     self.speeds, self.max_speeds, self.use_lanes)
            else:
                return compute_delay(self.phases, self.vehicles,
                                     self.speeds, self.max_speeds, self.use_lanes)
        if self.feature == 'wave':
            if self.use_lanes:
                return compute_wave(self.lanes, self.vehicles, self.use_lanes)
            else:
                return compute_wave(self.phases, self.vehicles, self.use_lanes)

        return compute_pressure(self.phases_incoming,
                                self.phases_outgoing, self.vehicles)


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

        def fn(current_phase, current_time, current_action):
            if current_time >= self.max_green: return True
            if current_time >= self.yellow + self.min_green:
                if self.action_schema == 'next': return current_action == 1
                if self.action_schema == 'set': return (current_action != (current_phase - 1))
            return False

        for tl_id, active_phases in self._active_phases.items():
            phases = self.phases[tl_id]
            current_phase, current_time = active_phases
            current_action = actions[tl_id]
            phase_ctrl = None
            if (self.yellow > 0 and current_time == self.yellow) and self.timestep > 5:
                # transitions to green: y -> G
                phase_ctrl = 2 * (current_phase - 1)

            elif fn(current_phase, current_time, current_action):

                # adjust log
                if self.action_schema == 'next':
                    # transitions to yellow: G -> y
                    # if yellow is zero; go to next green.
                    phase_ctrl = 2 * (current_phase - 1) + int(self.yellow > 0)

                    next_phase = (current_phase % len(phases))
                    self._active_phases[tl_id] = (next_phase + 1, 0)

                elif self.action_schema == 'set':
                    # transitions to yellow: G -> y
                    # if yellow is zero; go to next green.
                    phase_ctrl = 2 * current_action + int(self.yellow > 0)

                    self._active_phases[tl_id] = (current_action + 1, 0)

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
        self.info_dict["observation_spaces"].append(self._sanitize_obs()) 
        self.info_dict["actions"].append({k: int(v) for k, v in actions.items()})
        self.info_dict["states"].append(self._sanitize_obs())
        self.info_dict["timesteps"].append(self.timestep)

    # typecast numpy types to python
    def _sanitize_obs(self):
        return {k:float(v) for k, values in self.observations.items() for v in values} 


