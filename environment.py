import numpy as np
import torch
from torch._C import dtype
from forecasting_models.prophet_model import ProphetModel
from torch_standard_scaler import TorchStandardScaler

class Environment:
    def __init__(self, dataset, window=30):
        self.dataset = dataset
        self.window = window
        self.action_space = [-1, 0, 1]
        self.t = window #timestep of the series

        self.model_sales = ProphetModel.load_model('sales_model.json')

        self.COST_HOUR = 8.3

    def get_state(self, index=None):
        # for eact time step(value) form the state of the environment
        # => it could be a fixed window size
        # return state

        if index is None:
            index = self.t
        
        state = self.dataset.dataset_train[index-self.window:index+1].copy()
        return state

    def step(self, action_idx):
        # make step in the environment and return the next state and the reward
        # return (nex_state, reward)

        current_state = self.get_state()

        action_value = torch.tensor(self.action_space[action_idx])

        current_state.loc[self.t, 'HoursWorked'] += action_value.numpy()
        current_state.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
        forecast = self.model_sales.predict(current_state)

        forecast_profit = forecast.iloc[-1]['yhat'] - \
                        current_state.loc[self.t]['HoursWorked']*self.COST_HOUR

        actual_profit = self.dataset.dataset_train.iloc[self.t]['sales'] - \
                self.dataset.dataset_train.iloc[self.t]['HoursWorked']*self.COST_HOUR

        reward = forecast_profit - actual_profit

        self.t += 1

        return self.get_state(), reward

    def reset(self):
        self.t = self.window

    def iter_train_dataset(self):
        for value in self.dataset.dataset_train[self.window+1:].iterrows():
            yield value
            
    def transform_data_for_nn(self, df):
        df_transformed = df[['sales', 'HoursWorked', 'timestamp_s']]
        data = torch.tensor(df_transformed.values, dtype=torch.float)
        scaler = TorchStandardScaler()
        scaler.fit(data)

        return scaler.transform(data)