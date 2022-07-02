import math
import torch
import numpy as np

class EpsDecayGreedyQPolicy:
    """Implement the epsilon greedy policy
        Eps Greedy policy either:
        - takes a random action with probability epsilon
        - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps_start, eps_end, eps_decay):

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, q_values, steps_num=0, is_test=False):
        """Return the selected action
            # Arguments
                q_values (np.ndarray): List of the estimations of Q for each action
                steps_num (int): Number of steps so far
        # Returns
            Selected action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        # action_np = np.random.randint(0, nb_actions)
        # action = torch.tensor(action_np, device=self.device)
        # return action

        eps_curr = self._get_eps_threshold(steps_num)

        if np.random.uniform() < eps_curr and not is_test:
            action_np = np.random.randint(0, nb_actions)
            action = torch.tensor(action_np, device=self.device)
        else:
            action = torch.argmax(q_values).clone().detach()
        return action


    def _get_eps_threshold(self, steps_num=0):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * steps_num / self.eps_decay)
        