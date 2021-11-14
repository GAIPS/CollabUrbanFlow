# from pytorch_lightning import LightningModule
# TODO: Replace for LightningModule
# from agents.dqn import DQN_MODEL

import models.loops as lp
import models.lightning_loops as ll
def get_loop(agent_type, train=True):
    # if isinstance(agent, LightningModule):
    # Place here other deep learning.
    if agent_type in ('DQN', 'DQN2', 'DQN3', 'DQN4', 'GATV', 'GATW'):
        if train: return ll.train_loop
        return ll.rollback_loop
    if train: return lp.train_loop
    return lp.rollback_loop
