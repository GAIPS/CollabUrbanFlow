
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything

from tqdm.auto import trange
from utils.file_io import expr_logs_dump

def train_loop(env, model, train_args, chkpt_dir, seed):
    experiment_time = train_args.experiment_time
    save_agent_interval = train_args.save_agent_interval 
    model.save_path = chkpt_dir
    seed_everything(seed)
    # FIXME: RLDataLoader apperantly has an irregular
    # number of epochs.
    # * 2456 are the first 5 episodes
    # * 197 epochs for each episode thereafter
    # for 80 eps:
    # num_epochs = 2456 + 197 * 75
    num_episodes = experiment_time / save_agent_interval
    num_epochs = 2456 + 197 * (num_episodes-5)

    # 10000 of populate. Since each trainer step runs 10 env timesteps, we need to divide max_steps by 10.
    assert experiment_time > save_agent_interval
    # DQN
    # max_trainer_steps = (experiment_time-10000)/10
    # GAT
    # max_trainer_steps = (experiment_time + 10000) /10
    max_trainer_steps = experiment_time /10
    model.episode_timesteps = save_agent_interval

    # kwargs = {
    #     'max_steps':max_trainer_steps,
    #     'log_every_n_steps':100,
    #     'val_check_interval':100
    # }
    # if torch.cuda.is_available(): kwargs['on_gpu'] = 1
    trainer = Trainer(
        max_steps=max_trainer_steps,
        log_every_n_steps=100,
        val_check_interval=100,
        gpus= int(torch.cuda.is_available())
    )
    trainer.fit(model)

    return env.info_dict

def rollback_loop(env, agent, nets, rollout_time, target_path, seed):

    env.emit = True
    seed_everything(seed)
    # TODO: Get device
    # TODO: Move emissions to a separate module.
    # play_step runs 10 timesteps at a time, hence rollout_time/10
    for timestep in trange(rollout_time//10, position=1):
        agent.play_step(nets, epsilon=0.0)

    expr_logs_dump(target_path, 'emission_log.json', env.emissions)

    return env.info_dict
