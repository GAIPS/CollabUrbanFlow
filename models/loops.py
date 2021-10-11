'''Loops: defines common training and rollout routines. 

    TODO:
    * move update_emissions to object of class.
    * deprecate dependencies.

'''
from functools import lru_cache
from collections import defaultdict
import os

import numpy as np
from tqdm import tqdm
from tqdm.auto import trange

from features import compute_delay, compute_pressure
from utils.network import get_phases
from utils.file_io import engine_create, engine_load_config, expr_logs_dump

def train_loop(env, agent, approx, experiment_time, episode_time, chkpt_dir, seed):
    # 1) Seed everything
    np.random.seed(seed)    
    num_episodes = int(experiment_time / episode_time)

    s_prev = None
    a_prev = None

    for eps in trange(num_episodes, position=0):
        gen = env.loop(episode_time)

        try:
            while True:
                experience = next(gen)
                if experience is not None:
                    observations, reward = experience[:2]
                    state = approx.approximate(observations)
                    actions = agent.act(state)

                    if s_prev is None and a_prev is None:
                        s_prev = state
                        a_prev = actions

                    else:
                        agent.update(s_prev, a_prev, reward, state)
                        
                    s_prev = state
                    a_prev = actions
                    gen.send(actions)

        except StopIteration as e:
            result = e.value

            chkpt_num = str(eps * episode_time)
            os.makedirs(chkpt_dir, exist_ok=True)
            agent.save_checkpoint(chkpt_dir, chkpt_num)

            s_prev = None
            a_prev = None
            agent.reset()
    return env.info_dict

# TODO: Move emissions to environment
def rollback_loop(env, agent, approx, rollout_time, target_path, seed):
    # Makes sure the environment will generate emissions.
    env.emit = True
    gen = env.loop(rollout_time)
    np.random.seed(seed)

    try:
        while True:
            experience = next(gen)
            if experience is not None:
                observations = experience[0]
                state = approx.approximate(observations)
                actions = agent.act(state)

                gen.send(actions)
    except StopIteration as e:
        result = e.value
    expr_logs_dump(target_path, 'emission_log.json', env.emissions)
    
    return env.info_dict
