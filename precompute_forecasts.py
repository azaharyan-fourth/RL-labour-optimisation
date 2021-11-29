from environment import Environment
from dataset import Dataset
from forecasting_models.xgboost_model import XGBoostModel
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def compute_prophet():
    location_id = 14922
    dataset = Dataset('./data/all_sales_shifts_14922.csv', location_id)
    env = Environment(dataset)
    df = pd.DataFrame()
    for i in env.iter_train_dataset():
        row = {'date': i[1].date }
        for action in env.action_space:
            current_state = env.get_state(i[0])
            action_value = torch.tensor(action)

            current_state.loc[i[0], 'HoursWorked'] += action_value.numpy()
            current_state.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

            forecast = env.model_sales.predict(current_state).iloc[-1]['yhat']

            row[action] = forecast
        df = df.append(row, ignore_index=True)


    df.to_csv('./data/precomputed_forecasts.csv')

def compute_xgboost():

    dataset = pd.read_csv('./data/all_sales_shifts_14922.csv')
    X_train = dataset[dataset['date'] < '2020-01-01'][['HoursWorked', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']].to_numpy()
    X_test = dataset[(dataset['date'] >= '2020-01-01') & (dataset['date'] < '2020-03-01')][['HoursWorked', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']].to_numpy()

    y_train = dataset[dataset['date'] < '2020-01-01']['sales'].to_numpy()
    y_test = dataset[(dataset['date'] >= '2020-01-01') & (dataset['date'] < '2020-03-01')]['sales'].to_numpy()

    df = pd.DataFrame()
    model = XGBoostModel()
    model.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    X = dataset[dataset['date'] < '2020-03-01'][['HoursWorked', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']]
    for i in X.iterrows():
        row = { 'index': i[0] }
        row['date'] = dataset.loc[i[0], 'date']
        for action in [-5, 0, 5]:
            current_state = i[1].copy()

            current_state.loc['HoursWorked'] += action

            prediction = model.test(np.expand_dims(current_state.to_numpy(), axis=0))

            row[action] = prediction[0]
        df = df.append(row, ignore_index=True)
    
    df.to_csv('./data/precomputed_forecasts_xgb_5.csv')

def compute_xgboost_labour():
    dataset = pd.read_csv('./data/all_sales_shifts_14922.csv')
    X_train = dataset[dataset['date'] < '2020-01-01'][['sales', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']].to_numpy()
    X_test = dataset[(dataset['date'] >= '2020-01-01') & (dataset['date'] < '2020-03-01')][['sales', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']].to_numpy()

    y_train = dataset[dataset['date'] < '2020-01-01']['HoursWorked'].to_numpy()
    y_test = dataset[(dataset['date'] >= '2020-01-01') & (dataset['date'] < '2020-03-01')]['HoursWorked'].to_numpy()

    df = pd.DataFrame()
    model = XGBoostModel()
    model.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    
    X = dataset[dataset['date'] < '2020-03-01'][['sales', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']]

    for i in X.iterrows():
        row = { 'index': i[0] }
        row['date'] = dataset.loc[i[0], 'date']
        for action in [-5, 0, 5]:
            current_state = i[1].copy()

            prediction = model.test(np.expand_dims(current_state.to_numpy(), axis=0))

            row['HoursWorked_forecast'] = prediction[0]
        df = df.append(row, ignore_index=True)

    df.to_csv('./data/precomputed_forecasts_xgb_labour.csv')


if __name__ == '__main__':
    compute_xgboost_labour()