
import numpy as np
import torch
from pytorch_lightning import Trainer

from tqdm.auto import trange
from utils.file_io import expr_logs_dump

def train_loop(env, model, experiment_time, save_agent_interval, chkpt_dir, seed):

    # This should come from agent
    # parser = self.add_model_specific_args()
    # args = parser.parse_args()

    # args.epsilon_init = epsilon_init
    # args.epsilon_final = epsilon_final
    # args.epsilon_timesteps = epsilon_timesteps
    # args.network = network
    # args.episode_timesteps = save_agent_interval
    model.save_path = chkpt_dir
    torch.manual_seed(seed)
    np.random.seed(seed)
    # model = DQNLightning(**vars(args))


    # FIXME: RLDataLoader apperantly has an irregular
    # number of epochs.
    # * 2456 are the first 5 episodes
    # * 197 epochs for each episode thereafter
    # for 80 eps:
    # num_epochs = 2456 + 197 * 75
    num_episodes = experiment_time / save_agent_interval
    num_epochs = 2456 + 197 * (num_episodes-5)
    # self.trainer = pl.Trainer(
    #     max_epochs=self.num_epochs,
    #     log_every_n_steps=500,
    #     val_check_interval=100)


    # 10000 of populate. Since each trainer step runs 10 env timesteps, we need to divide max_steps by 10.
    assert experiment_time > save_agent_interval
    max_trainer_steps = (experiment_time-10000)/10

    trainer = Trainer(
        max_steps=max_trainer_steps,
        log_every_n_steps=500,
        val_check_interval=100
    )
    trainer.fit(model)

    return model.env.info_dict

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
