import numpy as np
import pandas as pd
import torch
from torch._C import dtype
from forecasting_models.xgboost_model import XGBoostModel
import time

class Environment:
    def __init__(self, dataset, window=30):
        self.dataset = dataset
        self.window = window
        self.action_space = [-5, 0, 5]
        self.t = window #timestep of the series

        self.model_sales = XGBoostModel()

        #self.predicted_values = pd.read_csv('./data/precomputed_forecasts_xgb_5.csv')

        self.COST_HOUR = 20

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_state(self, index=None, is_test=False):
        # for eact time step(value) form the state of the environment
        # => it could be a fixed window size
        # return state

        if index is None:
            index = self.t
        
        state = None

        if is_test:
            state = self.dataset.dataset_val.loc[index-self.window:index].copy()
        else:
            state = self.dataset.dataset_train.iloc[index-self.window:index+1].copy()

        return state

    def step(self, action_idx, is_test=False):
        # make step in the environment and return the next state and the reward
        # return (nex_state, reward)

        current_state = self.get_state(is_test=is_test)

        action_value = torch.tensor(self.action_space[action_idx])

        forecast_no_action = self.get_predicted_value(current_state.loc[self.t]['date'],
                                    current_state, 0)

        hours_no_action = current_state.loc[self.t]['HoursWorked']
        current_state.loc[self.t, 'HoursWorked'] += action_value.numpy()

        forecast_action = self.get_predicted_value(current_state.loc[self.t]['date'],
                                    current_state,
                                    action_value.numpy())

        forecast_action = self._fix_forecasts_minmax(forecast_action, forecast_no_action, action_value)

        forecast_profit = forecast_action - \
                        current_state.loc[self.t]['HoursWorked']*self.COST_HOUR

        actual_profit = forecast_no_action - \
                hours_no_action*self.COST_HOUR

        #(forecasts with action-labour with action) - (forecast with 0 - labour with 0)
        reward = forecast_profit - actual_profit

        self.t += 1

        done = self.t == len(self.dataset.dataset_train)-1
        next_state = self.get_state(is_test=is_test)

        if done:
            next_state = None

        return next_state, reward, done

    def reset(self, index=None):
        if index != None:
            self.t = index + self.window
        else:   
            self.t = self.window

    def iter_train_dataset(self):
        for value in self.dataset.dataset_train[self.window+1:].iterrows():
            yield value

    def iter_test_dataset(self):
        for value in self.dataset.dataset_val[self.window+1:].iterrows():
            yield value
            
    def transform_data_for_nn(self, df):
        df_transformed = df[['sales', 'HoursWorked', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']]
        data = torch.tensor(df_transformed.values, dtype=torch.float, device=self.device)

        return data

    def get_predicted_value(self, date, current_state, action):
        #is_predicted_empty = self.predicted_values.loc[pd.to_datetime(self.predicted_values['date']) == date].empty

        test = self.model_sales.create_features(current_state)
        return self.model_sales.test(test.tail(1))[0]

        #return self.predicted_values.loc[pd.to_datetime(self.predicted_values['date']) == date][str(action)].values[0]

    def train_environment(self):
        X_train, y_train = self.model_sales.create_features(self.dataset.dataset_train, label='sales')
        X_test, y_test = self.model_sales.create_features(self.dataset.dataset_val, label='sales')

        self.model_sales.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    def _fix_forecasts_minmax(self, forecast_action, forecast_noaction, action):
        if action > 0:
            forecast_action = max(forecast_action, forecast_noaction)
        elif action < 0:
            forecast_action = min(forecast_action, forecast_noaction)

        return forecast_action
