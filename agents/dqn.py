"""Deep Reinforcement Learning for Multi-agent system.

* Deep Q-networks

* Using Deep Q-networks for independent learners.

* TODO:
    Creates a MAS class that controls agents.
    Separate buffers.

To run the template, just run:
`python dqn.py`

`tensorboard --logdir lightning_logs`

References:
-----------
* Deep Q-network (DQN)

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""
import argparse

import numpy as np
import torch

from environment import get_environment
from models.DQN_Lightning import DQNLightning, DQN
import pytorch_lightning as pl

TRAIN_CONFIG_PATH = 'config/train.config'
RUN_CONFIG_PATH = 'config/run.config'


class DQN_MODEL:

    def __init__(self, epsilon_init, epsilon_final, epsilon_timesteps, network, save_agent_interval, experiment_time, save_path, seed):

        parser = self.add_model_specific_args()
        args = parser.parse_args()
        args.epsilon_init = epsilon_init
        args.epsilon_final = epsilon_final
        args.epsilon_timesteps = epsilon_timesteps
        args.network = network
        args.episode_timesteps = save_agent_interval
        args.save_path = save_path
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = DQNLightning(**vars(args))


        # FIXME: RLDataLoader apperantly has an irregular
        # number of epochs.
        # * 2456 are the first 5 episodes
        # * 197 epochs for each episode thereafter
        # for 80 eps:
        # num_epochs = 2456 + 197 * 75
        self.num_episodes = experiment_time / save_agent_interval
        self.num_epochs = 2456 + 197 * (self.num_episodes-5)
        # self.trainer = pl.Trainer(
        #     max_epochs=self.num_epochs,
        #     log_every_n_steps=500,
        #     val_check_interval=100)


        # 10000 of populate. Since each trainer step runs 10 env timesteps, we need to divide max_steps by 10.
        assert experiment_time > save_agent_interval
        max_trainer_steps = (experiment_time-10000)/10

        self.trainer = pl.Trainer(
            max_steps=max_trainer_steps,
            log_every_n_steps=500,
            val_check_interval=100)



    @staticmethod
    def load_checkpoint(chkpt_dir_path, rollout_time, network, chkpt_num=None):

        if chkpt_num == None:
            chkpt_num = max(int(folder.name) for folder in chkpt_dir_path.iterdir())
        chkpt_path = chkpt_dir_path / str(chkpt_num)
        print("Loading checkpoint: ", chkpt_path)
        env = get_environment(network, episode_timesteps=rollout_time)

        nets = []
        for tl_id in env.tl_ids:
            dqn = DQN()
            dqn.load_state_dict(torch.load(chkpt_path / f'{tl_id}.chkpt'))
            nets.append(dqn)

        return env, nets

    @staticmethod
    def add_model_specific_args():  # pragma: no-cover
        parent_parser = argparse.ArgumentParser(add_help=False)
        parser = parent_parser.add_argument_group("DQNLightning")
        parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
        parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
        parser.add_argument("--network", type=str, default="intersection", help="roadnet name")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--sync_rate", type=int, default=500,
                            help="how many frames do we update the target network")
        parser.add_argument("--replay_size", type=int, default=50000, help="capacity of the replay buffer")
        parser.add_argument("--warm_start_steps", type=int, default=1000,
                            help="how many samples do we use to fill our buffer at the start of training",
                            )
        parser.add_argument("--epsilon_timesteps", type=int, default=3500,
                            help="what frame should epsilon stop decaying")
        parser.add_argument("--epsilon_init", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--epsilon_final", type=float, default=0.01, help="final value of epsilon")
        parser.add_argument("--episode_timesteps", type=int, default=3600, help="max length of an episode")

        parser.add_argument("--save_path", type=str, default=None, help="Directory to save experiments.")
        return parent_parser

# def main(args, train_config_path=TRAIN_CONFIG_PATH, seed=0):
#     # Setup config parser path.
#     print(f'Loading train parameters from: {train_config_path}')
#
#     train_args = parse_train_config(train_config_path)
#     args.network = train_args['network']
#     experiment_time = train_args['experiment_time']
#     args.episode_timesteps = train_args['experiment_save_agent_interval']
#     # num_epochs = experiment_time // 10
#     num_steps = train_args['experiment_time']
#
#     # Epsilon
#     args.epsilon_init = train_args['epsilon_init']
#     args.epsilon_final = train_args['epsilon_final']
#     args.epsilon_timesteps = train_args['epsilon_schedule_timesteps']
#
#     if args.save_path is None:
#         expr_path = expr_path_create(args.network)
#         args.save_path = expr_path / 'checkpoints'
#         Path(args.save_path).mkdir(exist_ok=True)
#
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     model = DQNLightning(**vars(args))
#
#     # FIXME: RLDataLoader apperantly has an irregular
#     # number of epochs.
#     # * 2456 are the first 5 episodes
#     # * 197 epochs for each episode thereafter
#     # num_epochs = 2456 + 197 * 75
#     num_epochs = 2456
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         log_every_n_steps=500,
#         val_check_interval=100)
#     trainer.fit(model)
#     # TODO: Move this routine to env or create a delegate chain.
#     expr_logs_dump(args.save_path, 'train_log.json', model.agent.env.info_dict)
#
#     # 2) Create train plots.
#     load_path = Path('data/emissions/arterial_20211007235041/checkpoints/')
#     train_plots(load_path.parent)
#
#     # Load checkpoint for testing
#     rollout_timesteps = 21600
#     loaded_nets = []
#     chkpt_num = max(int(folder.name) for folder in load_path.iterdir())
#     chkpt_path = load_path / str(chkpt_num)
#     env = get_environment('arterial', episode_timesteps=rollout_timesteps)
#
#     net = []
#     for tl_id in env.tl_ids:
#         dqn = DQN()
#         dqn.load_state_dict(torch.load(chkpt_path / f'{tl_id}.chkpt'))
#         net.append(dqn)
#
#     num_rollouts = 2
#     rollout_timesteps = args.episode_timesteps
#     rollout_list = []
#     for num_rollout in trange(num_rollouts, position=0):
#         agent = Agent(env)
#         target_path = expr_path_test_target(load_path.parent, network=args.network)
#
#         # TODO: Get device
#         # TODO: Move emissions to a separate module.
#         # TODO: Refactor emissions -- separate Log?
#         emissions = []
#         for timestep in trange(rollout_timesteps, position=1):
#             agent.play_step(net, epsilon=0.0)
#             update_emissions(env.engine, emissions)
#         info_dict = agent.env.info_dict
#         info_dict['id'] = chkpt_num
#         rollout_list.append(info_dict)
#
#         expr_logs_dump(target_path, 'emission_log.json', emissions)
#
#         env = get_environment('arterial', episode_timesteps=rollout_timesteps)
#     # expr_logs_dump(args.save_path, 'train_log.json', model.agent.env.info_dict)
#
#     res = concat(rollout_list)
#     filename = f'rollouts_test.json'
#     target_path = batch_path / filename
#     with target_path.open('w') as fj:
#         json.dump(res, fj)
#
#     test_plots(load_path)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(add_help=False)
#     parser = DQNLightning.add_model_specific_args(parser)
#     args = parser.parse_args()
#
#     main(args)
