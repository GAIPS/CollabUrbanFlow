from agents.actor_critic import ACAT
from agents.marlin import MARLIN
from agents.dqn import DQN_MODEL

def load_agent(env, agent_type, chkpt_dir_path, chkpt_num, rollout_time, network):
    if agent_type == 'ACAT': return ACAT.load_checkpoint(chkpt_dir_path, chkpt_num)
    if agent_type == 'MARLIN': return MARLIN.load_checkpoint(chkpt_dir_path, chkpt_num)
    if agent_type == 'DQN': return DQN_MODEL.load_checkpoint(env, chkpt_dir_path, rollout_time, network)
    raise ValueError(f'{agent_type} not defined.')

def get_agent(agent_type, env, epsilon_init, epsilon_final,
              epsilon_timesteps, network, save_agent_interval, experiment_time, chkpt_dir, seed):

    if agent_type == 'ACAT':
        return ACAT(env.phases, epsilon_init,
                    epsilon_final, epsilon_timesteps)
    if agent_type == 'MARLIN':
        return MARLIN(env.phases, epsilon_init,
                      epsilon_final, epsilon_timesteps, network)
    if agent_type == 'DQN':
        return DQN_MODEL(epsilon_init, epsilon_final, epsilon_timesteps, network, save_agent_interval, experiment_time, chkpt_dir, seed)

    raise ValueError(f'{agent_type} not defined.')
