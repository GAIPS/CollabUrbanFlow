'''Environment: Wrapper around engine and feature producer.

    * Converts microsimulator data into features.
    * Keeps the agent's view from the traffic light.
    * Converts agent's actions into control actions.
    * Observes traffic data and transforms into features.
    * Logs past observations
    * Produces features: Delay and pressure.

'''
from functools import lru_cache
from collections import defaultdict

import numpy as np
from tqdm import tqdm


from features import compute_delay, compute_pressure
from utils.network import get_phases
from utils.file_io import engine_create, engine_load_config

FEATURE_CHOICE = ('delay', 'pressure')

def get_environment(network, episode_timesteps=3600, seed=0, thread_num=4):
    eng = engine_create(network, seed=seed, thread_num=4)
    config, flows, roadnet = engine_load_config(network) 

    return  EnvironmentGymWrapper(roadnet, eng, episode_timesteps=episode_timesteps)

class Environment(object):
    def __init__(self,
                 roadnet,
                 engine=None,
                 yellow=5,
                 min_green=5,
                 max_green=90,
                 step_size=5,
                 feature='delay',
                 **kwargs):
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

        _inc, _out, _lmt = get_phases(roadnet)
        self._incoming_roadlinks = _inc
        self._outgoing_roadlinks = _out
        self._speed_limit = _lmt

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
    def tl_ids(self):
        return sorted(self.phases.keys())

    @property
    def phases(self):
        return self._incoming_roadlinks

    @property
    def incoming_roadlinks(self):
        return self._incoming_roadlinks

    @property
    def outgoing_roadlinks(self):
        return self._outgoing_roadlinks

    @property
    def max_speeds(self):
        return self._speed_limit

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

    def loop(self, num_steps):
        # Before
        self.reset()
        for eps in tqdm(range(num_steps)):
            if self.is_observation_step:
                obs = self.observations
            if self.is_update_step:
                actions = yield obs
            else:
                yield
            self.step(actions)
        return 0

    def step(self, actions={}):
        # Handle controller actions
        # KEEP or SWITCH phase
        # Maps agent action to controller action
        # G -> Y -> G -> Y
        if self.is_observation_step: self._phase_ctl(actions)
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

            elif (current_time >= self.yellow + self.min_green and current_action == 1) or \
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
        self.info_dict = defaultdict(list)

    def step(self, actions):
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
