import ipdb
from tqdm.auto import trange

from agents.DQN_Lightning import Agent
from utils.file_io import expr_logs_dump

def train_loop(agent):
    agent.trainer.fit(agent.model)

    return agent.model.agent.env.info_dict

def rollback_loop(env , nets, target_path, rollout_time):

    env.emit = True
    agent = Agent(env)
    # TODO: Get device
    # TODO: Move emissions to a separate module.
    # play_step runs 10 timesteps at a time, hence rollout_time/10
    for timestep in trange(rollout_time//10, position=1):
        agent.play_step(nets, epsilon=0.0)

    expr_logs_dump(target_path, 'emission_log.json', env.emissions)

    return agent.env.info_dict
