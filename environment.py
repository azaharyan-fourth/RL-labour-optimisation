import numpy as np
from forecasting_models.prophet_model import ProphetModel

class Environment:
    def __init__(self, dataset, window=30):
        self.dataset = dataset
        self.window = window
        self.t = window-1 #timestep of the series

        self.model_sales = ProphetModel.load_model('sales_model.json')
        self.model_hours = ProphetModel.load_model('hours_model.json')

    def get_state(self, index):
        # for eact time step(value) form the state of the environment
        # => it could be a fixed window size
        # return state
        
        state = self.dataset.dataset_train[index:index+self.window].copy()
        return state

    def step(self, action):
        # make step in the environment and return the next state and the reward
        # return (nex_state, reward)
        pass

    def reset(self):
        self.t = self.window-1

    def iter_train_dataset(self):
        for value in self.dataset.dataset_train[self.window+1:]:
            yield value