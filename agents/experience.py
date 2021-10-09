"""Experience: Provides common data management methods and experience definition for RL-agents

    * Experience: (s, a, r, s')
    * ReplayBuffer manages 
    * RLIterator torch module that makes batches from online dataset
"""
from collections import deque, namedtuple, OrderedDict

import numpy as np
from torch.utils.data.dataset import IterableDataset


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"]
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    >>> ReplayBuffer(5)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.ReplayBuffer object at ...>
    """

    def __init__(self, capacity):
        """
        Parameters:
        -----------
            capacity: size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """Add experience to the buffer.

        Parameters:
        -----------
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Draes a random sample for experience.

        Parameters:
        -----------
            batch_size maximum number of experience
        """
        sample_size = min(len(self.buffer), batch_size)
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>
    """

    def __init__(self, buffer, sample_size):
        """
        Parameters:
        -----------
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

