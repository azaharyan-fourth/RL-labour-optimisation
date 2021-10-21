import random
import numpy as np

class Agent:
    def __init__(self, epsilon=0.5):
        self.action_space = [-1,0,1]
        self.epsilon = epsilon
        # plug in model
        
    def act(self, state):
        #exploration
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)

        # exploitation
        # evaluate Q-value for each state-action pair
        # return argmax of all q-values